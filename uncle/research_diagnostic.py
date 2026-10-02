"""Model-neutral measurements for forgetting diagnostics.

This module separates observations from scientific conclusions. Existing
forgetting traces can be screened and model components can be inspected now.
Matched recovery records and calculations belong with the future probe runner.
"""

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
import csv
import json
import math

import torch


DEFAULT_SCREEN = {
    "minimum_initial_accuracy_pct": 25.0,
    "target_accuracy_at_most_pct": 12.0,
    "retained_absolute_drift_strictly_under_pp": 5.0,
}


def _number(value, label):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def assess_forgetting(run, thresholds=DEFAULT_SCREEN):
    """Screen every observed step without discarding a failed trajectory.

    Required fields: id, target, retained, status, source, and trace. Each
    trace row has step and accuracies keyed by task. Optional measurement
    fields from later diagnostic versions are preserved in the trajectory.
    """
    target = str(run["target"])
    retained = [str(name) for name in run["retained"]]
    if not retained or target in retained or len(set(retained)) != len(retained):
        raise ValueError("Retained tasks must be distinct from the target")
    trace = run["trace"]
    if not trace or [row["step"] for row in trace] != list(range(len(trace))):
        raise ValueError("Trace needs consecutive steps starting at zero")
    initial = trace[0]["accuracies"]
    names = [target, *retained]
    for name in names:
        if name not in initial:
            raise ValueError(f"Missing initial accuracy for task {name}")
    valid_start = (min(_number(initial[name], name) for name in names) >= _number(
        thresholds["minimum_initial_accuracy_pct"], "minimum initial accuracy")
                   and run.get("checkpoint_match") is not False)
    passing = []
    eligible = []
    trajectory = []
    for row in trace:
        accuracies = row["accuracies"]
        if any(name not in accuracies for name in names):
            raise ValueError(f"Step {row['step']} has missing task accuracy")
        measured = {name: _number(accuracies[name], name) for name in names}
        if any(not 0 <= value <= 100 for value in measured.values()):
            raise ValueError("Accuracy must be between zero and 100 percent")
        drift = max(abs(measured[name] - float(initial[name])) for name in retained)
        retention_ok = drift < thresholds["retained_absolute_drift_strictly_under_pp"]
        target_ok = measured[target] <= thresholds["target_accuracy_at_most_pct"]
        passed = (run["status"] == "complete" and valid_start and row["step"] > 0
                  and retention_ok and target_ok)
        if passed:
            passing.append(row["step"])
        if valid_start and row["step"] > 0 and retention_ok:
            eligible.append((measured[target], row["step"]))
        trajectory.append({
            "step": row["step"], "target_accuracy_pct": measured[target],
            "retained_accuracies_pct": {name: measured[name] for name in retained},
            "maximum_retained_drift_pp": drift,
            "retention_within_screen": retention_ok,
            "target_within_screen": target_ok,
            "passes_screen": passed,
            "measurements": {key: value for key, value in row.items()
                             if key not in {"step", "accuracies"}},
        })
    best = min(eligible) if eligible else None
    return {
        "id": run["id"], "source": run["source"], "status": run["status"],
        "target": target, "retained": retained,
        "settings": run.get("settings"), "commit": run.get("commit"),
        "checkpoint": run.get("checkpoint"),
        "valid_initial_accuracy": valid_start,
        "checkpoint_match": run.get("checkpoint_match"),
        "best_step_within_retention_limit": best[1] if best else None,
        "best_target_accuracy_within_retention_limit_pct": best[0] if best else None,
        "passing_steps": passing, "trajectory": trajectory,
        "scope": "Forgetting screen only; recovery and component attribution need matched controls.",
    }


def runtime_run(report):
    """Adapt a saved diagnose_forgetting report without changing it."""
    return {
        "id": Path(report["report_path"]).stem,
        "source": "runtime JSON",
        "status": report["status"], "target": report["task"],
        "retained": report["retained"], "trace": report["trace"],
        "settings": report["settings"],
        "commit": report.get("environment", {}).get("git_commit"),
        "checkpoint": report.get("checkpoint"),
        "checkpoint_match": report.get("initial_matches_checkpoint"),
    }


def archived_runs(registry_path, traces_path):
    """Read historical transcriptions, preserving their weaker provenance."""
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    with Path(traces_path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in rows:
        groups[row["run"]].append(row)
    experiments = {row["id"]: row for row in registry["experiments"]}
    for identifier, observations in groups.items():
        if identifier not in experiments:
            raise ValueError(f"No registry record for {identifier}")
        record = experiments[identifier]
        yield {
            "id": identifier, "source": "transcribed table",
            "status": record["observations"].get("reported_status", "complete"),
            "target": registry["common_protocol"]["target_task"],
            "retained": [registry["common_protocol"]["retained_task"]],
            "trace": [{
                "step": int(row["step"]),
                "accuracies": {
                    registry["common_protocol"]["target_task"]: row["task3_accuracy_pct"],
                    registry["common_protocol"]["retained_task"]: row["task0_accuracy_pct"],
                },
            } for row in observations],
            "settings": record["settings"], "commit": record["reported_commit"],
            "checkpoint": None, "checkpoint_match": None,
        }


def capture_components(components: Mapping[str, Mapping[str, object]]):
    """Copy named tensors for a model-neutral before/after audit."""
    return {
        role: {name: tensor.detach().cpu().clone() for name, tensor in members.items()}
        for role, members in components.items()
    }


def compare_components(before, components):
    """Measure exact state changes; an empty role is explicitly unavailable."""
    if set(before) != set(components):
        raise ValueError("Component roles changed")
    results = {}
    for role, old_members in before.items():
        members = components[role]
        if set(old_members) != set(members):
            raise ValueError(f"Tensor names changed for {role}")
        squared = 0.0
        max_change = 0.0
        changed = 0
        for name, old in old_members.items():
            current = members[name].detach().cpu()
            if old.shape != current.shape or old.dtype != current.dtype:
                raise ValueError(f"Tensor shape or dtype changed for {role}.{name}")
            if old.is_complex():
                raise ValueError(f"Complex component tensor is unsupported: {role}.{name}")
            if torch.equal(old, current):
                continue
            changed += 1
            delta = current.to(torch.float64) - old.to(torch.float64)
            squared += delta.square().sum().item()
            max_change = max(max_change, delta.abs().max().item())
        results[role] = {
            "tensors": len(old_members), "changed_tensors": changed,
            "l2_change": squared ** 0.5, "maximum_absolute_change": max_change,
            "available": bool(old_members),
        }
    return results


def inspect_components(components):
    """Read live parameter values and gradients without changing the model.

    This is a per-tensor view. Buffers are reported but have no gradients.
    A missing gradient means no gradient was recorded, not a zero gradient.
    """
    results = {}
    for role, members in components.items():
        results[role] = {}
        for name, tensor in members.items():
            value = tensor.detach()
            if value.is_complex():
                raise ValueError(f"Complex component tensor is unsupported: {role}.{name}")
            gradient = tensor.grad if isinstance(tensor, torch.nn.Parameter) else None
            results[role][name] = {
                "kind": "parameter" if isinstance(tensor, torch.nn.Parameter) else "buffer",
                "shape": list(value.shape),
                "count": value.numel(),
                "value_l2": float(torch.linalg.vector_norm(value.float()).item()),
                "value_min": float(value.min().item()) if value.numel() else None,
                "value_max": float(value.max().item()) if value.numel() else None,
                "gradient_recorded": gradient is not None,
                "gradient_l2": (float(torch.linalg.vector_norm(gradient.detach().float()).item())
                                if gradient is not None else None),
            }
    return results


def diagonal_fisher(model, examples, components, logits_for, kind="empirical"):
    """Compute per-parameter diagonal Fisher from individual examples.

    examples yields (input, class_index) pairs. logits_for(model, input) must
    return one vector of class logits. The empirical form uses the true class.
    The model-predicted form sums over every class with its predicted probability.
    The full parameter-by-parameter Fisher matrix is not computed. BatchNorm
    buffers are excluded. The caller chooses and records the example set.
    """
    if kind not in {"empirical", "model_predicted"}:
        raise ValueError("Fisher kind must be empirical or model_predicted")
    model_parameters = {id(parameter) for parameter in model.parameters()}
    selected = []
    names = []
    seen = set()
    for role, members in components.items():
        for name, tensor in members.items():
            if not isinstance(tensor, torch.nn.Parameter):
                continue
            if id(tensor) in seen or id(tensor) not in model_parameters:
                raise ValueError("Fisher parameter roles must be unique and belong to the model")
            seen.add(id(tensor))
            names.append((role, name))
            selected.append(tensor)
    if not selected:
        raise ValueError("No parameters selected for Fisher")
    requires_grad_flags = [parameter.requires_grad for parameter in selected]
    totals = [torch.zeros_like(parameter, dtype=torch.float32) for parameter in selected]
    count = 0
    training_flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        for parameter in selected:
            parameter.requires_grad_(True)
        with torch.enable_grad():
            for input_value, target in examples:
                logits = logits_for(model, input_value)
                if logits.ndim != 1 or logits.numel() < 2:
                    raise ValueError("Fisher needs one vector with at least two class logits")
                if not bool(torch.isfinite(logits).all().item()):
                    raise ValueError("Fisher logits must be finite")
                target = int(target)
                if not 0 <= target < logits.numel():
                    raise ValueError("Target class is outside the logits")
                log_probabilities = torch.log_softmax(logits, dim=0)
                probabilities = log_probabilities.detach().exp()
                classes = [target] if kind == "empirical" else range(logits.numel())
                for position, class_index in enumerate(classes):
                    gradients = torch.autograd.grad(
                        log_probabilities[class_index], selected,
                        retain_graph=position < len(classes) - 1, allow_unused=True)
                    weight = (1.0 if kind == "empirical"
                              else float(probabilities[class_index].item()))
                    for total, gradient in zip(totals, gradients):
                        if gradient is not None:
                            if not bool(torch.isfinite(gradient).all().item()):
                                raise ValueError("Fisher gradients must be finite")
                            total.add_(gradient.detach().float().square(), alpha=weight)
                count += 1
    finally:
        for parameter, enabled in zip(selected, requires_grad_flags):
            parameter.requires_grad_(enabled)
        for module, training in training_flags:
            module.training = training
    if not count:
        raise ValueError("Fisher needs at least one example")
    diagonal = {}
    for (role, name), total in zip(names, totals):
        diagonal.setdefault(role, {})[name] = (total / count).cpu()
    scores = {
        role: sum(tensor.sum().item() for tensor in members.values()) /
              sum(tensor.numel() for tensor in members.values())
        for role, members in diagonal.items()
    }
    return {"kind": kind, "examples": count, "diagonal": diagonal,
            "mean_per_parameter": scores}


def uncle_components(uncle, task):
    """Map UnCLe state to general roles without changing the core analysis."""
    task = str(task)
    if task not in uncle.hypernet.task_codes or task not in uncle.task_buffers:
        raise ValueError("The task needs an embedding and stored buffers")
    parameters = dict(uncle.hypernet.named_parameters())
    return {
        "task_embedding": {f"task_codes.{task}": parameters[f"task_codes.{task}"]},
        "chunk_embeddings": {"chunk_codes": parameters["chunk_codes"]},
        "shared_layers": {name: tensor for name, tensor in parameters.items()
                          if name.startswith("trunk.")},
        "output_heads": {name: tensor for name, tensor in parameters.items()
                         if name.startswith("heads.")},
        "task_normalization_buffers": dict(uncle.task_buffers[task]),
    }
