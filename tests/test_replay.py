"""CPU checks for observational learning traces and replay comparison."""
import copy
from contextlib import contextmanager
import shutil
import uuid
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from uncle.config import Config
from uncle.hypernet import HyperNetwork, build_target
from uncle.trainer import UnCLe
from uncle.replay import check_gpu_model, compare_replays, file_hash, tensor_hash, write_json


@contextmanager
def temporary_folder():
    # Python 3.14's private tempfile permissions conflict with this Windows sandbox.
    parent = Path(tempfile.gettempdir()).resolve()
    folder = parent / ("replay-test-" + uuid.uuid4().hex)
    folder.mkdir()
    try:
        yield folder
    finally:
        if folder.resolve().parent != parent:
            raise ValueError("Test cleanup escaped its temporary parent")
        shutil.rmtree(folder)


class ReplayTests(unittest.TestCase):
    def test_gpu_check_accepts_matching_model_and_rejects_different_model(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.current_device", return_value=0), \
             patch("torch.cuda.get_device_name", return_value="A100"):
            self.assertEqual(check_gpu_model("A100"), "A100")
            self.assertEqual(check_gpu_model("A100", first_run_gpu="A100"), "A100")
            with self.assertRaisesRegex(ValueError, "No training has started"):
                check_gpu_model("Blackwell")
            with self.assertRaisesRegex(ValueError, "Keep REQUIRED_GPU unchanged"):
                check_gpu_model("Blackwell", first_run_gpu="A100")
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Training needs a GPU"):
                check_gpu_model("A100")

    def test_observer_preserves_updates_order_and_rng(self):
        config = Config(tasks=("A",), requests=(("learn", "A"),), backbone="cnn",
                        chunks=4, hidden=(8,), code_dim=4, epochs=2, batch_size=2, device="cpu")
        torch.manual_seed(15)
        dataset = TensorDataset(torch.rand(5, 1, 28, 28), torch.tensor([0, 1, 2, 3, 4]))
        target = build_target(config)
        initial = HyperNetwork(target, config)
        rng = torch.get_rng_state()
        runs = []
        observations = []
        for tracing in (False, True):
            model = copy.deepcopy(initial)
            trainer = UnCLe(model, config, copy.deepcopy(target), {"A": {"train": dataset}})
            torch.set_rng_state(rng)
            def observe(row):
                observations.append({"step": row["step"],
                                     "indices": row.get("indices", torch.tensor([])).tolist()})
                torch.rand(13)  # Observational RNG consumption must be isolated.
                model.eval()
            losses = trainer.learn("A", [], on_step=observe if tracing else None)
            runs.append((tensor_hash(model.state_dict()), losses, torch.get_rng_state()))
        self.assertEqual(runs[0][:2], runs[1][:2])
        self.assertTrue(torch.equal(runs[0][2], runs[1][2]))
        self.assertTrue(model.training)
        self.assertEqual([row["step"] for row in observations], list(range(7)))
        for start in (1, 4):
            indices = sum([row["indices"] for row in observations[start:start + 3]], [])
            self.assertEqual(sorted(indices), list(range(5)))

    def _saved_run(self, folder, label, changed=False):
        folder.mkdir()
        contract = {"config": {"epochs": 1, "batch_size": 1}, "new_task": "9"}
        write_json(folder / "environment.json", {
            "label": label, "contract": contract, "environment": {"gpu": "synthetic"},
            "cudnn_version": None, "numpy": "test", "settings": {}})
        write_json(folder / "data_manifest.json", {"9": {"train": {"samples": [0, 1]}}})
        rows = [{"step": 0, "parameters_sha256": "same"},
                {"step": 1, "loss": 1.0, "input_sha256": "changed" if changed else "same"},
                {"step": 2, "loss": 0.5}]
        (folder / "trace.jsonl").write_text(''.join(json.dumps(row) + '\n' for row in rows))
        write_json(folder / "result.json", {"complete": True, "trace_rows": 3,
                                             "final_accuracies": {"0": 40, "9": 30}})
        write_json(folder / "manifest.json", {p.name: file_hash(p) for p in folder.iterdir()})

    def test_first_divergence_and_agreement(self):
        with temporary_folder() as directory:
            root = Path(directory)
            self._saved_run(root / "A", "A")
            self._saved_run(root / "B", "B", changed=True)
            report = compare_replays(root / "A", root / "B", root / "report.json")
            self.assertEqual(report["first_trace_difference"]["step"], 1)
            self.assertEqual(report["first_trace_difference"]["fields"], ["input_sha256"])
            self._saved_run(root / "C", "B")
            report = compare_replays(root / "A", root / "C", root / "report.json")
            self.assertIsNone(report["first_trace_difference"])
            (root / "C" / "trace.jsonl").write_text('')
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                compare_replays(root / "A", root / "C", root / "report.json")


if __name__ == "__main__":
    unittest.main()
