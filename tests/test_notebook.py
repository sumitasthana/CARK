"""Notebook helpers on synthetic checkpoints and mocked Colab services."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from uncle.notebook import inspect_checkpoint, prepare_session, show_diagnostic


spec = importlib.util.spec_from_file_location(
    "colab_setup", Path(__file__).resolve().parents[1] / "scripts" / "colab_setup.py"
)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "before.pt"
        torch.save({
            "format": 1, "config": {"backbone": "cnn"},
            "history": [{"action": "learn", "task": "3"}],
            "seen": ["3"], "forgotten": [], "previous": {"3": 26.0},
            "task_buffers": {"3": {}}, "hypernet": {"weight": torch.ones(2)},
        }, self.path)

    def test_inspection_validates_metadata_without_retaining_or_changing_weights(self):
        before = hashlib.sha256(self.path.read_bytes()).digest()
        result = inspect_checkpoint(
            self.path, expected_history=[("learn", "3")],
            expected_accuracies={"3": 26.0}, expected_config={"backbone": "cnn"},
        )
        self.assertNotIn("hypernet", result)
        self.assertEqual(result["accuracies"], {"3": 26.0})
        self.assertEqual(before, hashlib.sha256(self.path.read_bytes()).digest())

    def test_wrong_checkpoint_is_rejected(self):
        for arguments in (
            {"expected_history": [("forget", "3")]},
            {"expected_accuracies": {"3": 44.6}},
            {"expected_accuracies": {"missing": 26.0}},
            {"expected_config": {"backbone": "resnet50"}},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                inspect_checkpoint(self.path, **arguments)
        with self.assertRaises(FileNotFoundError):
            inspect_checkpoint(self.path.with_name("missing.pt"))

    def test_cpu_runtime_fails_before_mounting_drive(self):
        with patch("uncle.notebook.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "GPU runtime"):
                prepare_session(self.path)

    def report(self):
        return {
            "status": "complete", "task": "3", "retained": ["0"],
            "initial_matches_checkpoint": True, "environment": {"git_commit": "test"},
            "settings": {"gamma": 5e-6}, "report_path": "report.json",
            "trace": [
                {"step": 0, "accuracies": {"3": 26.0, "0": 44.6}},
                {"step": 1, "accuracies": {"3": 10.0, "0": 44.0},
                 "raw_norm": 1.0, "noise_gradient": {"trunk": 2.0},
                 "preserve_gradient": {"trunk": 0.0}},
            ],
        }

    def test_display_uses_recorded_gradients_and_does_not_mutate_report(self):
        report = self.report()
        before = copy.deepcopy(report)
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            summary = show_diagnostic(report)
        self.assertEqual(summary["passing_steps"], [1])
        self.assertIn("trunk", text.getvalue())
        self.assertEqual(report, before)

    def test_failed_control_or_incomplete_run_cannot_pass_screen(self):
        for changes in ({"initial_matches_checkpoint": False}, {"status": "failed"},
                        {"retained": []}):
            report = {**self.report(), **changes}
            with contextlib.redirect_stdout(io.StringIO()):
                summary = show_diagnostic(report)
            self.assertEqual(summary["passing_steps"], [])

    def test_old_report_does_not_invent_gradients(self):
        report = self.report()
        del report["trace"][1]["noise_gradient"]
        del report["trace"][1]["preserve_gradient"]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            show_diagnostic(report)
        self.assertIn("gradient measurements unavailable", output.getvalue())


class SetupTests(unittest.TestCase):
    def test_loaded_package_is_rejected_before_any_command(self):
        with patch.object(setup.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "before importing uncle"):
                setup.setup_repo()
            run.assert_not_called()

    def existing_repo(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = Path(tmp.name)
        (repo / ".git").mkdir()
        return repo

    def test_dirty_clone_is_not_updated(self):
        repo = self.existing_repo()
        with patch.object(setup.sys, "modules", {}), \
             patch.object(setup.subprocess, "check_output", side_effect=[setup.REMOTE, " M work.py"]), \
             patch.object(setup.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "Local changes"):
                setup.setup_repo(repo)
            run.assert_not_called()

    def test_different_branch_is_not_updated(self):
        repo = self.existing_repo()
        with patch.object(setup.sys, "modules", {}), \
             patch.object(setup.subprocess, "check_output", side_effect=[setup.REMOTE, "", "work"]), \
             patch.object(setup.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "not on main"):
                setup.setup_repo(repo)
            run.assert_not_called()

    def test_clean_clone_fetches_and_fast_forwards_without_replacing_gpu_packages(self):
        repo = self.existing_repo()
        with patch.object(setup.sys, "modules", {}), \
             patch.object(setup.subprocess, "check_output", side_effect=[setup.REMOTE, "", "main", "abc"]), \
             patch.object(setup.subprocess, "run") as run, \
             patch.object(setup.os, "chdir"), patch.object(setup.sys, "path", []), \
             contextlib.redirect_stdout(io.StringIO()):
            run.return_value.returncode = 0
            self.assertEqual(setup.setup_repo(repo), repo.resolve())
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0][-3:], ["fetch", "origin", "main"])
        self.assertEqual(commands[2][-3:], ["merge", "--ff-only", "FETCH_HEAD"])
        self.assertIn("--no-deps", commands[3])
        self.assertNotIn("--upgrade", commands[3])


class NotebookFlowTests(unittest.TestCase):
    def test_all_five_cells_with_a_local_cpu_checkpoint(self):
        # Adapt only Colab services, paths, and compute size. Execute the exact
        # notebook cell text against the real checkpoint and diagnostic APIs.
        import test_diagnostics
        from uncle import diagnose_forgetting

        fixture = test_diagnostics.DiagnosticTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        before = hashlib.sha256(fixture.path.read_bytes()).digest()
        notebook = Path(__file__).resolve().parents[1] / "notebooks" / "04_gradient_diagnostics.ipynb"
        cells = ["".join(c["source"]) for c in json.loads(notebook.read_text(encoding="utf-8"))["cells"]
                 if c["cell_type"] == "code"]
        namespace = {}

        def local_prepare(checkpoint, **kwargs):
            return prepare_session(checkpoint, mount_drive=False, require_gpu=False, **kwargs)

        def local_diagnostic(checkpoint, **kwargs):
            kwargs.update(device="cpu", download=False, tasks=fixture.tasks)
            return diagnose_forgetting(checkpoint, **kwargs)

        with patch("urllib.request.urlretrieve"), patch("runpy.run_path"), \
             patch("uncle.notebook.prepare_session", side_effect=local_prepare), \
             patch("uncle.diagnose_forgetting", side_effect=local_diagnostic), \
             contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cells[0], "setup_cell", "exec"), namespace)
            exec(compile(cells[1], "config_cell", "exec"), namespace)
            namespace.update(
                CHECKPOINT=fixture.path, DATA=fixture.root, OUTPUT=fixture.root / "reports",
                SETTINGS=dict(task="3", forgetting_lr=1e-5, gamma=5e-6, steps=2),
                EXPECTED=dict(expected_history=[("learn", "3"), ("learn", "0")],
                              expected_config={"backbone": "cnn"}),
            )
            for i, code in enumerate(cells[2:], 2):
                exec(compile(code, f"cell_{i}", "exec"), namespace)

        report = namespace["report"]
        self.assertTrue(namespace["screen"]["complete"])
        self.assertTrue(report["initial_matches_checkpoint"])
        self.assertEqual([r["step"] for r in report["trace"]], [0, 1, 2])
        self.assertIn("noise_gradient", report["trace"][1])
        self.assertIn("gradient_cosine", report["trace"][1])
        self.assertIn("adam_update_norm", report["trace"][1])
        self.assertTrue(Path(report["report_path"]).is_file())
        self.assertEqual(before, hashlib.sha256(fixture.path.read_bytes()).digest())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
