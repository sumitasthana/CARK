"""Reusable session checks and tables for checkpoint-based notebooks."""

import math
from pathlib import Path
import subprocess

import torch

from . import checkpoint as checkpointing


def inspect_checkpoint(path, *, expected_history=None, expected_accuracies=None,
                       expected_config=None):
    """Read and validate saved metadata without constructing a model.

    Only use trusted checkpoints. The returned summary contains no model
    tensors. Recorded accuracies still need verification on the restored model.
    """
    path = Path(path).expanduser().resolve()
    saved = checkpointing.load(path)
    if saved is None:
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    history = [(row["action"], row["task"]) for row in saved["history"]]
    if expected_history is not None and history != list(expected_history):
        raise ValueError(f"Unexpected checkpoint history: {history}")
    for key, expected in (expected_config or {}).items():
        actual = saved["config"].get(key)
        if actual != expected:
            raise ValueError(f"Checkpoint {key}: expected {expected!r}, found {actual!r}")
    for task, expected in (expected_accuracies or {}).items():
        actual = saved["previous"].get(str(task))
        if actual is None or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Task {task}: expected accuracy {expected}, found {actual}")
    return {
        "checkpoint": str(path), "config": dict(saved["config"]),
        "history": history, "seen": list(saved["seen"]),
        "forgotten": list(saved["forgotten"]),
        "accuracies": dict(saved["previous"]),
        "buffer_tasks": list(saved["task_buffers"]),
    }


def prepare_session(checkpoint, *, mount_drive=True, require_gpu=True,
                    expected_history=None, expected_accuracies=None,
                    expected_config=None):
    """Mount Drive, check the runtime, and print a tensor-free checkpoint summary."""
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime before preparing this experiment.")
    if mount_drive:
        from google.colab import drive
        drive.mount("/content/drive")

    repo = Path(__file__).resolve().parent.parent
    if Path.cwd().resolve() != repo:
        raise RuntimeError(f"Run the setup cell first; the working directory must be {repo}.")
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    summary = inspect_checkpoint(
        checkpoint, expected_history=expected_history,
        expected_accuracies=expected_accuracies, expected_config=expected_config,
    )
    print("Commit:", commit)
    print("PyTorch:", torch.__version__)
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name} ({gpu.total_memory / 1024**3:.1f} GiB)")
    else:
        print("Device: CPU")
    print("Checkpoint:", summary["checkpoint"])
    print("Completed requests:", summary["history"])
    print("Forgotten tasks:", summary["forgotten"])
    print("Saved accuracies:", summary["accuracies"])
    print("Tasks with buffer entries:", summary["buffer_tasks"])
    for key in ("dataset", "backbone", "seed", "chunks", "hidden", "code_dim",
                "epochs", "learning_rate", "forgetting_learning_rate", "beta",
                "gamma", "noise_samples"):
        print(f"{key}: {summary['config'].get(key)}")
    return {**summary, "git_commit": commit}


def show_diagnostic(report, *, gradient_steps=(1, 2, 5, 10), min_before=25.0,
                    target_at_most=12.0, drift_under=5.0):
    """Print accuracy screens and gradient norms from an existing report.

    Gradients and raw norms are measured before updates; accuracies are after
    updates. The screen is a debugging criterion, not evidence of deletion.
    Missing gradient measurements are reported as unavailable, never as zero.
    """
    trace = report["trace"]
    if not trace or trace[0]["step"] != 0:
        raise ValueError("The report needs a step-zero observation.")
    task = report["task"]
    retained = report["retained"]
    initial = trace[0]["accuracies"]
    valid_start = (
        report.get("initial_matches_checkpoint") is True
        and bool(retained)
        and all(initial[name] >= min_before for name in [task, *retained])
    )
    complete = report["status"] == "complete"
    print("Status:", report["status"])
    print("Commit:", report["environment"].get("git_commit"))
    print("Settings:", report["settings"])
    print("Valid starting point:", valid_start)
    print("step  target  retained_mean  max_drift  raw_norm_before  passes_screen")
    passing = []
    for row in trace:
        accuracies = row["accuracies"]
        mean = sum(accuracies[name] for name in retained) / len(retained) if retained else math.nan
        drift = max((abs(accuracies[name] - initial[name]) for name in retained), default=math.nan)
        passes = bool(complete and valid_start and row["step"] > 0
                      and accuracies[task] <= target_at_most and drift < drift_under)
        if passes:
            passing.append(row["step"])
        raw = f"{row['raw_norm']:.4g}" if "raw_norm" in row else "n/a"
        print(f"{row['step']:4d}  {accuracies[task]:6.1f}  {mean:13.1f}  "
              f"{drift:9.1f}  {raw:>15}  {passes}")

    print("\nGradient norms before each selected update (noise includes gamma):")
    print("step  group                    noise       preserve")
    for row in trace:
        if row["step"] not in gradient_steps:
            continue
        if "noise_gradient" not in row or "preserve_gradient" not in row:
            print(f"{row['step']:4d}  gradient measurements unavailable")
            continue
        noise, preserve = row["noise_gradient"], row["preserve_gradient"]
        for group in sorted(set(noise) | set(preserve)):
            n = f"{noise[group]:.4g}" if group in noise else "n/a"
            p = f"{preserve[group]:.4g}" if group in preserve else "n/a"
            print(f"{row['step']:4d}  {group:22}  {n:>10}  {p:>13}")
    print("Passing steps:", passing)
    print("Report:", report["report_path"])
    return {"valid_start": valid_start, "complete": complete, "passing_steps": passing}
