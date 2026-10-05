"""Trace a checkpoint replay and compare fresh-session learning runs."""

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
import torchvision
from PIL import __version__ as pillow_version

from . import checkpoint as checkpointing
from .config import Config
from .hypernet import HyperNetwork, build_target
from .telemetry import environment
from .trainer import UnCLe


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def runtime_settings():
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def code_identity():
    repo = Path(__file__).resolve().parent.parent
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Replay requires a clean tracked checkout. Commit the tracing code first.")
    files = git("ls-files", "uncle", "scripts", "pyproject.toml", "requirements.txt").splitlines()
    return {"commit": git("rev-parse", "HEAD"),
            "files": {name: file_hash(repo / name) for name in files}}


def data_manifest(tasks):
    """Record ordered sample identities, labels, file bytes, and transforms."""
    result = {}
    for task, splits in sorted(tasks.items()):
        result[task] = {}
        for split, dataset in sorted(splits.items()):
            rows = []
            for local_index, base_index in enumerate(dataset.indices):
                path = dataset.base.image_path(base_index)
                rows.append({"index": local_index,
                             "sample": path.relative_to(dataset.base.root).as_posix(),
                             "label": dataset.targets[local_index],
                             "sha256": file_hash(path)})
            result[task][split] = {"classes": dataset.classes,
                                   "transform": repr(dataset.base.transform), "samples": rows}
    return result


def run_replay(checkpoint, tasks, output, contract_path, *, label, new_task="9", focus_task="0"):
    """Run one learn request. Output directories are never reused.

    Session A creates a contract; session B verifies the same contract.
    PyTorch RNG is restored from the checkpoint. Python and NumPy RNG use
    the recorded config seed because historical checkpoints omit their states.
    """
    output, contract_path = Path(output), Path(contract_path)
    if label not in {"A", "B"}:
        raise ValueError("label must be A or B")
    if label == "B" and not contract_path.is_file():
        raise FileNotFoundError("Session B needs the saved session A contract")
    if label == "A" and contract_path.exists():
        raise FileExistsError("Use a new experiment ID for a new session A")
    saved = checkpointing.load(checkpoint)
    if saved is None:
        raise FileNotFoundError(checkpoint)
    history = [(row["action"], row["task"]) for row in saved["history"]]
    if history != [("learn", "3"), ("learn", "0"), ("forget", "3")]:
        raise ValueError("Expected the post-U3 checkpoint after L3, L0, U3")
    if new_task in saved["tasks_with_codes"]:
        raise ValueError("Replay task already exists")
    config = Config(**saved["config"])
    if config.torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This checkpoint replay requires a GPU runtime")
    if saved["cuda_rng"] is None:
        raise ValueError("Checkpoint has no CUDA RNG state")
    manifest = data_manifest(tasks)
    contract = {
        "schema_version": 1, "checkpoint_sha256": file_hash(checkpoint),
        "config": json.loads(json.dumps(vars(config))), "code": code_identity(),
        "data_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "settings": runtime_settings(), "new_task": new_task, "focus_task": focus_task,
        "rng_policy": "checkpoint PyTorch CPU/CUDA; Python and NumPy reset to config.seed",
    }
    if label == "B":
        frozen = json.loads(contract_path.read_text(encoding="utf-8"))
        differences = [key for key in contract if frozen.get(key) != contract[key]]
        if differences:
            raise ValueError("Replay contract differs: " + ", ".join(differences))
    output.mkdir(parents=True, exist_ok=False)
    if label == "A":
        write_json(contract_path, contract)
    write_json(output / "data_manifest.json", manifest)
    del manifest
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    target = build_target(config)
    hypernet = HyperNetwork(target, config)
    uncle = UnCLe(hypernet, config, target, tasks, progress=True)
    checkpointing.restore(saved, hypernet=hypernet, uncle=uncle)
    protected = [name for name in saved["seen"] if name != new_task]
    metadata = {"schema_version": 1, "label": label, "contract": contract,
                "environment": environment(config, hypernet, tasks),
                "cudnn_version": torch.backends.cudnn.version(),
                "numpy": np.__version__, "torchvision": torchvision.__version__,
                "pillow": pillow_version, "settings": runtime_settings()}
    write_json(output / "environment.json", metadata)
    rows = 0
    with (output / "trace.jsonl").open("x", encoding="utf-8") as stream:
        def observe(record):
            nonlocal rows
            state = {"cpu": torch.get_rng_state()}
            state.update({f"cuda{i}": value for i, value in enumerate(torch.cuda.get_rng_state_all())})
            row = {"step": record["step"], "rng_sha256": tensor_hash(state),
                   "parameters_sha256": tensor_hash(hypernet.state_dict()),
                   "buffers_sha256": tensor_hash({f"{task}/{name}": value
                        for task, buffers in uncle.task_buffers.items() for name, value in buffers.items()})}
            if record["step"] == 0:
                torch.save({**state, "python": random.getstate(), "numpy": np.random.get_state()},
                           output / "training_boundary_rng.pt")
                row["new_task_embedding_sha256"] = tensor_hash({new_task: hypernet.task_codes[new_task]})
                row["starting_accuracies"] = {task: uncle.accuracy(task) for task in (focus_task, new_task)}
            else:
                row.update({"epoch": record["epoch"], "loss": record["loss"],
                            "indices": record["indices"].tolist(),
                            "input_sha256": tensor_hash({"images": record["images"], "labels": record["labels"]})})
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            rows += 1
        losses = uncle.learn(new_task, protected, on_step=observe)
    final = {task: uncle.accuracy(task) for task in (focus_task, new_task)}
    write_json(output / "result.json", {"complete": True, "trace_rows": rows,
                                        "epoch_losses": losses, "final_accuracies": final})
    # A final checkpoint is evidence only; it must never become session B's input.
    torch.save({"hypernet": hypernet.state_dict(), "task_buffers": uncle.task_buffers},
               output / "final_model.pt")
    files = [path for path in output.iterdir() if path.is_file()]
    write_json(output / "manifest.json", {path.name: file_hash(path) for path in files})
    verify_run(output)
    return output


def verify_run(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest.items():
        if file_hash(folder / name) != digest:
            raise ValueError(f"Saved artifact hash mismatch: {name}")
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (folder / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    config = json.loads((folder / "environment.json").read_text(encoding="utf-8"))["contract"]["config"]
    data = json.loads((folder / "data_manifest.json").read_text(encoding="utf-8"))
    task = json.loads((folder / "environment.json").read_text(encoding="utf-8"))["contract"]["new_task"]
    count = len(data[task]["train"]["samples"])
    expected = config["epochs"] * ((count + config["batch_size"] - 1) // config["batch_size"]) + 1
    if not result.get("complete") or result["trace_rows"] != expected or len(rows) != expected:
        raise ValueError("Incomplete replay trace")
    if [row["step"] for row in rows] != list(range(expected)):
        raise ValueError("Trace update sequence differs")
    return rows


def compare_replays(first, second, output):
    """Report metadata differences and the earliest recorded trace mismatch."""
    traces = [verify_run(folder) for folder in (first, second)]
    metadata = [json.loads((Path(folder) / "environment.json").read_text(encoding="utf-8"))
                for folder in (first, second)]
    if [row["label"] for row in metadata] != ["A", "B"]:
        raise ValueError("Compare session A followed by session B")
    differences = {}
    for key in ("contract", "environment", "cudnn_version", "numpy", "torchvision", "pillow", "settings"):
        if metadata[0].get(key) != metadata[1].get(key):
            differences[key] = {"A": metadata[0].get(key), "B": metadata[1].get(key)}
    mismatch = None
    for a, b in zip(*traces):
        fields = [key for key in sorted(set(a) | set(b)) if a.get(key) != b.get(key)]
        if fields:
            mismatch = {"step": a["step"], "fields": fields,
                        "A": {key: a.get(key) for key in fields},
                        "B": {key: b.get(key) for key in fields}}
            break
    if mismatch is None and len(traces[0]) != len(traces[1]):
        mismatch = {"step": min(map(len, traces)), "fields": ["trace_length"]}
    final = [json.loads((Path(folder) / "result.json").read_text(encoding="utf-8"))["final_accuracies"]
             for folder in (first, second)]
    report = {"runs": {"A": str(first), "B": str(second)},
              "metadata_differences": differences, "first_trace_difference": mismatch,
              "final_accuracies": {"A": final[0], "B": final[1]},
              "interpretation": ("Historical spread was not reproduced at recorded points."
                   if mismatch is None and final[0] == final[1] else
                   "A difference was observed; the cause requires a targeted check.")}
    write_json(output, report)
    return report
