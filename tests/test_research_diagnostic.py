"""Checks for the model-neutral research diagnostic rules."""

import csv
import io
import json
import copy
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uncle.research_diagnostic import (
    archived_runs, assess_forgetting, capture_components, compare_components,
    component_share, diagonal_fisher, inspect_components, interaction_residual,
    paired_recovery, replace_components, runtime_run, select_trainable_components,
    specificity_ratio,
)


class ResearchDiagnosticTests(unittest.TestCase):
    def test_latest_transcription_matches_all_structured_rows(self):
        data = ROOT / "docs" / "experiments"
        source = (data / "sources" / "forgetting_30step_20260929.txt").read_text(
            encoding="utf-8")
        accuracy = list(csv.DictReader(io.StringIO(
            "step,target_pct,retained_pct,absolute_drift_pp,raw_norm_before,passes_screen\n"
            + source.split("step,target_pct,retained_pct,absolute_drift_pp,raw_norm_before,passes_screen\n", 1)[1]
            .split("\n\nGradient norms", 1)[0])))
        gradient = list(csv.DictReader(io.StringIO(
            "step,parameter_group,weighted_noise_gradient_norm_before,preservation_gradient_norm_before\n"
            + source.split("step,parameter_group,weighted_noise_gradient_norm_before,preservation_gradient_norm_before\n", 1)[1])))
        run = "forgetting-30step-20260929"
        def saved(name):
            with (data / name).open(encoding="utf-8", newline="") as stream:
                return [row for row in csv.DictReader(stream) if row["run"] == run]
        trace = saved("forgetting_traces.csv")
        self.assertEqual(len(trace), len(accuracy))
        for original, stored in zip(accuracy, trace):
            self.assertEqual((original["step"], original["target_pct"],
                              original["retained_pct"], original["absolute_drift_pp"],
                              original["passes_screen"]),
                             (stored["step"], stored["task3_accuracy_pct"],
                              stored["task0_accuracy_pct"], stored["task0_absolute_drift_pp"],
                              stored["passes_screen"]))
        self.assertEqual(
            [{key: value for key, value in row.items() if key != "run"}
             for row in saved("gradient_norms.csv")], gradient)
        self.assertEqual(
            [(row["step"], row["raw_norm_before"]) for row in saved("raw_output_norms.csv")],
            [(row["step"], row["raw_norm_before"]) for row in accuracy if row["step"] != "0"])

    def test_archive_accumulates_every_step_and_preserves_reported_screens(self):
        data = ROOT / "docs" / "experiments"
        registry = json.loads((data / "registry.json").read_text(encoding="utf-8"))
        records = {row["id"]: row for row in registry["experiments"]}
        reports = [assess_forgetting(run, registry["thresholds"])
                   for run in archived_runs(data / "registry.json",
                                             data / "forgetting_traces.csv")]
        self.assertEqual(len(reports), 9)
        self.assertEqual(sum(len(row["trajectory"]) for row in reports), 239)
        for report in reports:
            expected = records[report["id"]]["observations"]
            self.assertEqual(report["passing_steps"], expected["passing_steps"])
            self.assertEqual(report["source"], "transcribed table")
            self.assertIsNone(report["checkpoint_match"])

    def test_screen_keeps_failed_steps_and_rejects_mismatched_checkpoint(self):
        run = {
            "id": "toy", "source": "runtime JSON", "target": "X", "retained": ["R"],
            "status": "complete", "checkpoint_match": True,
            "trace": [
                {"step": 0, "accuracies": {"X": 26, "R": 44.6}},
                {"step": 1, "accuracies": {"X": 13, "R": 44.0}},
                {"step": 2, "accuracies": {"X": 11, "R": 40.0}},
                {"step": 3, "accuracies": {"X": 12, "R": 40.2}},
            ],
        }
        result = assess_forgetting(run)
        self.assertEqual(result["passing_steps"], [2, 3])
        self.assertEqual(result["best_step_within_retention_limit"], 2)
        self.assertEqual(len(result["trajectory"]), 4)
        run["checkpoint_match"] = False
        self.assertEqual(assess_forgetting(run)["passing_steps"], [])

    def test_runtime_report_keeps_new_measurements_and_source_unchanged(self):
        report = {
            "report_path": "/drive/forget_sample.json", "status": "complete",
            "task": "X", "retained": ["R"], "checkpoint": "/drive/checkpoint.pt",
            "initial_matches_checkpoint": True,
            "environment": {"git_commit": "abc"}, "settings": {"steps": 1},
            "trace": [
                {"step": 0, "accuracies": {"X": 26, "R": 44.6}},
                {"step": 1, "accuracies": {"X": 11, "R": 44.0},
                 "gradient_cosine": {"shared": -0.8},
                 "adam_update_norm": {"shared": 0.001}},
            ],
        }
        before = copy.deepcopy(report)
        result = assess_forgetting(runtime_run(report))
        self.assertEqual(result["passing_steps"], [1])
        self.assertEqual(result["commit"], "abc")
        self.assertEqual(result["trajectory"][1]["measurements"]["gradient_cosine"],
                         {"shared": -0.8})
        self.assertEqual(report, before)

    def test_matched_recovery_requires_baselines_and_exposes_controls(self):
        def row(condition, step, heldout, retained):
            return {
                "pair_id": "seed0-seq1-X", "task": "X", "seed": 0,
                "sequence": 1, "split_id": "sha256:example", "budget": 5,
                "step": step, "intervention": "full", "condition": condition,
                "heldout_accuracy_pct": heldout,
                "adaptation_accuracy_pct": heldout + 1,
                "retained_accuracy_pct": retained,
                "correct_class_log_probability": -2 + heldout / 100,
            }
        observations = [row("unlearned", 0, 10, 45),
                        row("never_learned", 0, 9, 46),
                        row("unlearned", 20, 25, 43),
                        row("never_learned", 20, 15, 45)]
        result = paired_recovery(observations)
        late = next(item for item in result if item["step"] == 20)
        self.assertEqual(late["recovery_advantage_pp"], 10)
        self.assertAlmostEqual(late["likelihood_gain"], 0.1)
        self.assertEqual(late["maintenance_cost_pp"],
                         {"unlearned": -2, "never_learned": -1})
        self.assertIsNone(late["pre_deletion_recovery_advantage_pp"])
        observations += [row("pre_deletion", 0, 40, 45),
                         row("pre_deletion", 20, 60, 43)]
        result = paired_recovery(observations)
        self.assertEqual(next(item for item in result if item["step"] == 20)
                         ["pre_deletion_recovery_advantage_pp"], 45)
        self.assertTrue(all(item["probe_sensitivity_decision"] is None for item in result))
        with self.assertRaisesRegex(ValueError, "Missing step-zero"):
            paired_recovery([row("unlearned", 20, 25, 43),
                             row("never_learned", 20, 15, 45)])
        observations[-1]["split_id"] = "different-split"
        with self.assertRaisesRegex(ValueError, "Missing paired"):
            paired_recovery(observations)

    def test_ratios_keep_denominators_and_undefined_values(self):
        self.assertIsNone(specificity_ratio(5, 0)["ratio"])
        self.assertEqual(specificity_ratio(6, 2)["ratio"], 3)
        self.assertIsNone(component_share(0, -1)["ratio"])
        self.assertEqual(component_share(10, 4)["ratio"], 0.6)
        self.assertEqual(interaction_residual(10, [8, 7], 3)["interaction_residual"], 2)

    def test_component_audit_detects_only_changed_roles(self):
        components = {
            "task_state": {"code": torch.tensor([1.0, 2.0])},
            "shared": {"weight": torch.tensor([3.0, 4.0])},
            "optional_buffers": {},
        }
        before = capture_components(components)
        components["shared"]["weight"][0] += 2
        result = compare_components(before, components)
        self.assertEqual(result["task_state"]["l2_change"], 0)
        self.assertEqual(result["shared"]["l2_change"], 2)
        self.assertEqual(result["shared"]["changed_tensors"], 1)
        self.assertFalse(result["optional_buffers"]["available"])
        components["shared"]["extra"] = torch.tensor([0.0])
        with self.assertRaisesRegex(ValueError, "Tensor names changed"):
            compare_components(before, components)

    def test_component_intervention_freezes_all_other_parameters(self):
        model = torch.nn.Module()
        model.register_parameter("task_code", torch.nn.Parameter(torch.tensor([1.0])))
        model.register_parameter("shared", torch.nn.Parameter(torch.tensor([2.0])))
        model.register_parameter("unmapped", torch.nn.Parameter(torch.tensor([3.0])))
        model.register_buffer("running", torch.tensor([4.0]))
        roles = {
            "task_embedding": {"task_code": model.task_code},
            "shared_layers": {"shared": model.shared},
            "buffers": {"running": model.running},
        }
        selected = select_trainable_components(model, roles, {"task_embedding"})
        self.assertEqual(selected, [model.task_code])
        self.assertTrue(model.task_code.requires_grad)
        self.assertFalse(model.shared.requires_grad)
        self.assertFalse(model.unmapped.requires_grad)
        with self.assertRaisesRegex(ValueError, "No trainable"):
            select_trainable_components(model, roles, {"buffers"})

        before = capture_components(roles)
        audit = replace_components(roles, {"buffers": {"running": torch.tensor([9.0])}})
        self.assertEqual(audit["buffers"]["l2_change"], 5)
        self.assertEqual(audit["task_embedding"]["changed_tensors"], 0)
        self.assertEqual(audit["shared_layers"]["changed_tensors"], 0)
        self.assertTrue(torch.equal(model.task_code.detach(), before["task_embedding"]["task_code"]))
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            replace_components(roles, {"buffers": {"running": torch.tensor([1])}})
        self.assertEqual(model.running.item(), 9)

    def test_live_weight_gradient_view_and_exact_diagonal_fisher(self):
        model = torch.nn.Module()
        model.register_parameter("weight", torch.nn.Parameter(torch.tensor(0.0)))
        model.register_buffer("running", torch.tensor(3.0))
        model.weight.grad = torch.tensor(7.0)
        roles = {"weights": {"weight": model.weight},
                 "batchnorm_buffers": {"running": model.running}}
        view = inspect_components(roles)
        self.assertEqual(view["weights"]["weight"]["gradient_l2"], 7)
        self.assertFalse(view["batchnorm_buffers"]["running"]["gradient_recorded"])
        self.assertEqual(view["weights"]["weight"]["value_l2"], 0)

        def logits_for(network, x):
            return torch.stack((network.weight * x, network.weight * 0))
        examples = [(torch.tensor(2.0), 0), (torch.tensor(1.0), 1)]
        for kind in ("empirical", "model_predicted"):
            result = diagonal_fisher(model, examples, roles, logits_for, kind)
            self.assertEqual(result["examples"], 2)
            self.assertAlmostEqual(result["diagonal"]["weights"]["weight"].item(), 0.625)
            self.assertAlmostEqual(result["mean_per_parameter"]["weights"], 0.625)
            self.assertNotIn("batchnorm_buffers", result["diagonal"])
            self.assertEqual(model.weight.grad.item(), 7)
            self.assertTrue(model.training)
        model.weight.requires_grad_(False)
        result = diagonal_fisher(model, examples, roles, logits_for)
        self.assertAlmostEqual(result["mean_per_parameter"]["weights"], 0.625)
        self.assertFalse(model.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
