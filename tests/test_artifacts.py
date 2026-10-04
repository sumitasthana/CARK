"""Checks for the shared saved-run reader."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uncle.artifacts import RunArtifactError, load_run_artifacts


class ArtifactTests(unittest.TestCase):
    def write(self, folder, kind, value):
        (folder / f"{kind}_seq1_seed0.json").write_text(json.dumps(value), encoding="utf-8")

    def test_reads_complete_run_without_opening_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write(folder, "history", [
                {"index": 0, "action": "learn", "task": "3"},
                {"index": 1, "action": "forget", "task": "3"},
            ])
            self.write(folder, "origin", {"config": {"requests": [
                ["learn", "3"], ["forget", "3"]]}})
            self.write(folder, "summary", {"retain_accuracy": 40.0})
            (folder / "checkpoint_seq1_seed0.pt").write_bytes(b"not a PyTorch file")
            run = load_run_artifacts(
                folder, required=("origin", "history", "summary"),
                require_checkpoint=True)
            self.assertTrue(run["sequence_matches_config"])
            self.assertEqual(run["files"]["summary"]["retain_accuracy"], 40.0)

    def test_reports_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write(folder, "history", [{"index": 0, "action": "learn", "task": "0"}])
            with self.assertRaises(RunArtifactError) as caught:
                load_run_artifacts(folder, required=("history", "summary"))
            self.assertIn("summary_seq1_seed0.json", caught.exception.missing[0])

    def test_marks_changed_request_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write(folder, "history", [
                {"index": 0, "action": "learn", "task": "0"}])
            self.write(folder, "origin", {"config": {"requests": [
                ["learn", "3"]]}})
            run = load_run_artifacts(folder, required=("origin", "history"))
            self.assertFalse(run["sequence_matches_config"])

    def test_rejects_corrupt_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write(folder, "history", [{"index": 1, "action": "learn", "task": "0"}])
            with self.assertRaisesRegex(RunArtifactError, "Invalid request index"):
                load_run_artifacts(folder, required=("history",))
