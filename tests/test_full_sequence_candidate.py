"""A full request sequence can branch safely from a learned checkpoint."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import TensorDataset

from uncle.config import Config
from uncle import checkpoint as checkpointing
from uncle.experiments import run_experiment
from uncle.telemetry import environment


class FullSequenceCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.manual_seed(7)
        self.tasks = {
            name: {"train": TensorDataset(torch.randn(8, 1, 28, 28), torch.arange(8)),
                   "test": TensorDataset(torch.randn(8, 1, 28, 28), torch.arange(8))}
            for name in ("3", "0", "9")
        }
        self.prefix = Config(
            tasks=("3", "0", "9"), requests=(("learn", "3"), ("learn", "0")),
            backbone="cnn", hidden=(4,), chunks=4, epochs=1, batch_size=4,
            eval_batch_size=4, burn_in=3, noise_samples=2, device="cpu",
        )
        self.full = replace(
            self.prefix,
            requests=(*self.prefix.requests, ("forget", "3"), ("learn", "9")),
            gamma=5e-6, forgetting_learning_rate=1e-5,
            burn_in=3, burn_in_min=3, burn_in_decay=1.0,
        )

    def run_case(self, config, folder, **kwargs):
        with patch("uncle.experiments.build_tasks", return_value=self.tasks):
            return run_experiment(config=config, sequence=1, output=self.root / folder,
                                  verbose=False, progress=False, **kwargs)

    def test_branch_matches_uninterrupted_run_and_preserves_source(self):
        source = self.run_case(self.prefix, "source")
        source_path = self.root / "source" / "checkpoint_seq1_cnn_seed0.pt"
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        whole = self.run_case(self.full, "whole")
        without_forget = replace(
            self.full, requests=(("learn", "3"), ("learn", "0"), ("learn", "9")))
        control = self.run_case(without_forget, "control", starting_checkpoint=source_path)
        self.assertEqual(control["history"],
                         self.run_case(without_forget, "control_whole")["history"])
        branch = self.run_case(self.full, "branch", starting_checkpoint=source_path,
                               snapshot_after_requests=(2,))
        self.assertEqual(branch["history"], whole["history"])
        self.assertEqual(branch["history"][:2], source["history"])
        self.assertEqual(branch["history"][2]["burn_in"], 3)
        self.assertEqual(branch["numbers"], whole["numbers"])
        self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), digest)
        snapshot = (self.root / "branch" /
                    "checkpoint_seq1_cnn_seed0_after_request02_forget3.pt")
        self.assertEqual(len(checkpointing.load(snapshot)["history"]), 3)
        self.assertEqual(len(checkpointing.load(
            self.root / "branch" / "checkpoint_seq1_cnn_seed0.pt")["history"]), 4)
        origin = json.loads((self.root / "branch" / "origin_seq1_cnn_seed0.json").read_text())
        self.assertEqual(origin["starting_checkpoint_sha256"], digest)
        post_forget = self.run_case(self.full, "post_forget_branch",
                                    starting_checkpoint=snapshot)
        self.assertEqual(post_forget["history"], whole["history"])
        longer = replace(self.full, requests=(*self.full.requests, ("forget", "9")))
        self.run_case(longer, "longer_source", snapshot_after_requests=(2,))
        long_snapshot = (self.root / "longer_source" /
                         "checkpoint_seq1_cnn_seed0_after_request02_forget3.pt")
        short_branch = self.run_case(self.full, "short_branch",
                                     starting_checkpoint=long_snapshot)
        self.assertEqual(short_branch["history"], whole["history"])
        with self.assertRaisesRegex(ValueError, "forgetting settings must match"):
            self.run_case(replace(self.full, gamma=1e-4), "changed_past_forget",
                          starting_checkpoint=snapshot)
        resumed = self.run_case(self.full, "branch", starting_checkpoint=source_path,
                                snapshot_after_requests=(2,))
        self.assertEqual(resumed["history"], branch["history"])

        environment_path = self.root / "branch" / "environment_seq1_cnn_seed0.json"
        original_environment = json.loads(environment_path.read_text())
        with patch("uncle.experiments.environment",
                   return_value={**environment(self.full), "git_commit": "later-session",
                                 "hypernetwork_parameters": 1, "generated_parameters": 1}):
            self.run_case(self.full, "branch", starting_checkpoint=source_path)
        after_resume = json.loads(environment_path.read_text())
        self.assertEqual(after_resume["git_commit"], original_environment["git_commit"])
        self.assertEqual(after_resume["resume_sessions"][-1]["after_requests"], 4)
        self.assertEqual(after_resume["resume_sessions"][-1]["environment"]["git_commit"],
                         "later-session")

    def test_branch_rejects_changed_training_and_request_history(self):
        self.run_case(self.prefix, "source")
        source = self.root / "source" / "checkpoint_seq1_cnn_seed0.pt"
        with self.assertRaisesRegex(ValueError, "training settings"):
            self.run_case(replace(self.full, beta=0.7), "wrong_beta",
                          starting_checkpoint=source)
        wrong_order = replace(self.full, requests=(("learn", "0"), ("learn", "3"),
                                                  ("forget", "3")))
        with self.assertRaisesRegex(ValueError, "prefix differs"):
            self.run_case(wrong_order, "wrong_order", starting_checkpoint=source)
        with self.assertRaisesRegex(ValueError, "must differ"):
            self.run_case(self.full, "source", starting_checkpoint=source)


if __name__ == "__main__":
    unittest.main()
