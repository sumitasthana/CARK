"""Model-neutral measurements for forgetting and paired recovery studies.

This module separates observations from scientific conclusions. Existing
forgetting traces can be screened now. Paired recovery records use the same
schema regardless of the model that produced them.
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


def paired_recovery(observations):
    """Calculate matched recovery gaps, with controls visible in every result.

    One record describes one condition at one adaptation budget and step.
    Records require pair_id, task, seed, sequence, split_id, budget, step,
    intervention, condition, heldout_accuracy_pct, adaptation_accuracy_pct,
    retained_accuracy_pct, and correct_class_log_probability. Pairing rejects
    mismatches rather than comparing unrelated model runs.
    """
    required = ("pair_id", "task", "seed", "sequence", "split_id", "budget",
                "step", "intervention", "condition", "heldout_accuracy_pct",
                "adaptation_accuracy_pct", "retained_accuracy_pct",
                "correct_class_log_probability")
    groups = defaultdict(dict)
    for row in observations:
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Probe observation missing {missing}")
        if row["condition"] not in {"unlearned", "never_learned", "pre_deletion",
                                    "never_forgotten"}:
            raise ValueError("Unknown probe condition")
        if int(row["budget"]) < 0 or int(row["step"]) < 0:
            raise ValueError("Budget and step must be nonnegative")
        key = tuple(row[field] for field in required[:5]) + (
            int(row["budget"]), int(row["step"]), row["intervention"])
        if row["condition"] in groups[key]:
            raise ValueError("Duplicate condition at the same probe point")
        groups[key][row["condition"]] = row
    results = []
    for key, conditions in sorted(groups.items(), key=lambda item: str(item[0])):
        if not {"unlearned", "never_learned"} <= conditions.keys():
            raise ValueError(f"Missing paired unlearned or never-learned condition: {key}")
        unlearned = conditions["unlearned"]
        reference = conditions["never_learned"]
        baseline_key = (*key[:6], 0, key[7])
        baseline = groups.get(baseline_key, {})
        if not {"unlearned", "never_learned"} <= baseline.keys():
            raise ValueError(f"Missing step-zero baseline for paired probe: {key}")
        accuracy = (_number(unlearned["heldout_accuracy_pct"], "heldout accuracy")
                    - _number(reference["heldout_accuracy_pct"], "reference heldout accuracy"))
        likelihood = (_number(unlearned["correct_class_log_probability"], "log probability")
                      - _number(reference["correct_class_log_probability"], "reference log probability"))
        results.append({
            **dict(zip(required[:8], key)),
            "recovery_advantage_pp": accuracy,
            "likelihood_gain": likelihood,
            "adaptation_accuracy_pct": {
                name: _number(row["adaptation_accuracy_pct"], "adaptation accuracy")
                for name, row in conditions.items()},
            "retained_accuracy_pct": {
                name: _number(row["retained_accuracy_pct"], "retained accuracy")
                for name, row in conditions.items()},
            "maintenance_cost_pp": {
                name: (_number(row["retained_accuracy_pct"], "retained accuracy")
                       - _number(baseline[name]["retained_accuracy_pct"], "baseline retained accuracy"))
                for name, row in conditions.items() if name in baseline},
            "pre_deletion_control_present": "pre_deletion" in conditions,
            "never_forgotten_control_present": "never_forgotten" in conditions,
            "pre_deletion_recovery_advantage_pp": (
                _number(conditions["pre_deletion"]["heldout_accuracy_pct"],
                        "pre-deletion heldout accuracy")
                - _number(reference["heldout_accuracy_pct"], "reference heldout accuracy")
                if "pre_deletion" in conditions else None),
            "probe_sensitivity_decision": None,
        })
    return results


def specificity_ratio(target_advantage, control_advantage):
    """Return S with its operands; zero denominators are explicitly undefined."""
    numerator = _number(target_advantage, "target advantage")
    denominator = _number(control_advantage, "control advantage")
    return {"numerator": numerator, "denominator": denominator,
            "ratio": numerator / denominator if denominator != 0 else None}


def component_share(full_advantage, intervened_advantage):
    """Return p(c) for a specified reset or swap and its matched reference."""
    full = _number(full_advantage, "full advantage")
    changed = _number(intervened_advantage, "intervened advantage")
    return {"numerator": full - changed, "denominator": full,
            "ratio": (full - changed) / full if full != 0 else None}


def interaction_residual(full_advantage, individual_advantages, combined_advantage):
    """Compare the combined effect with the sum of separate effects."""
    full = _number(full_advantage, "full advantage")
    combined = _number(combined_advantage, "combined advantage")
    singles = [_number(value, "individual advantage") for value in individual_advantages]
    combined_effect = full - combined
    individual_effects = [full - value for value in singles]
    return {"combined_effect": combined_effect,
            "individual_effects": individual_effects,
            "interaction_residual": combined_effect - sum(individual_effects)}


def capture_components(components: Mapping[str, Mapping[str, object]]):
    """Copy named tensors for a model-neutral component intervention audit."""
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
            if not old.is_floating_point():
                if not old.equal(current):
                    changed += 1
                    delta = current.float() - old.float()
                    squared += delta.square().sum().item()
                    max_change = max(max_change, delta.abs().max().item())
                continue
            delta = current - old
            nonzero = bool(delta.count_nonzero().item())
            if nonzero:
                changed += 1
            squared += delta.square().sum().item()
            if nonzero:
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


def select_trainable_components(model, components, allowed_roles):
    """Freeze all mapped parameters except the requested semantic roles.

    Buffers are never optimizer parameters. Every parameter in the supplied
    model is frozen first, including parameters outside the named roles. Return
    the selected parameters for a fresh optimizer.
    """
    allowed = set(allowed_roles)
    unknown = allowed - set(components)
    if unknown:
        raise ValueError(f"Unknown trainable component roles: {sorted(unknown)}")
    model_parameters = list(model.parameters())
    model_ids = {id(parameter) for parameter in model_parameters}
    selected = []
    seen = set()
    for role, members in components.items():
        for tensor in members.values():
            if not isinstance(tensor, torch.nn.Parameter):
                continue
            if id(tensor) in seen:
                raise ValueError("Parameter appears in more than one component role")
            if id(tensor) not in model_ids:
                raise ValueError("Mapped parameter does not belong to the model")
            seen.add(id(tensor))
            if role in allowed:
                selected.append(tensor)
    if not selected:
        raise ValueError("No trainable parameters selected")
    for parameter in model_parameters:
        parameter.requires_grad_(False)
    for parameter in selected:
        parameter.requires_grad_(True)
    return selected


def replace_components(components, replacements):
    """Copy exact named donor tensors and audit the affected roles.

    The caller chooses the donor and its scientific meaning. Shape and dtype
    are checked before any write. A model-specific adapter remains responsible
    for making sure donor states are comparable.
    """
    if not replacements or not set(replacements) <= set(components):
        raise ValueError("Replacements must name known component roles")
    for role, members in replacements.items():
        if not members or set(members) != set(components[role]):
            raise ValueError(f"Replacement needs every named tensor in {role}")
        for name, donor in members.items():
            current = components[role][name]
            if current.shape != donor.shape or current.dtype != donor.dtype:
                raise ValueError(f"Incompatible replacement for {role}.{name}")
    before = capture_components(components)
    try:
        with torch.no_grad():
            for role, members in replacements.items():
                for name, donor in members.items():
                    components[role][name].copy_(donor)
        audit = compare_components(before, components)
        if any(item["changed_tensors"] for role, item in audit.items()
               if role not in replacements):
            raise RuntimeError("An unselected component changed")
        return audit
    except BaseException:
        with torch.no_grad():
            for role, members in before.items():
                for name, value in members.items():
                    components[role][name].copy_(value)
        raise


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
