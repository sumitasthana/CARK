"""Validate the experiment archive and render its GitHub wiki pages."""

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "experiments"


def read_csv(name):
    with (DATA / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def validate(registry, traces, gradients, norms):
    ids = [row["id"] for row in registry["experiments"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate experiment identifiers")
    thresholds = registry["thresholds"]
    for experiment in registry["experiments"]:
        for evidence in experiment["evidence"]:
            if not (ROOT / evidence["path"]).is_file():
                raise ValueError(f"Missing evidence: {evidence['path']}")
        rows = [row for row in traces if row["run"] == experiment["id"]]
        if not rows:
            continue
        steps = [int(row["step"]) for row in rows]
        complete = experiment["observations"].get("reported_status", "complete") == "complete"
        expected_length = experiment["settings"]["steps"] + 1 if complete else len(rows)
        if steps != list(range(expected_length)) or len(rows) > experiment["settings"]["steps"] + 1:
            raise ValueError(f"Missing or unordered observations: {experiment['id']}")
        initial_target = float(rows[0]["task3_accuracy_pct"])
        initial_retained = float(rows[0]["task0_accuracy_pct"])
        valid = min(initial_target, initial_retained) >= thresholds["minimum_initial_accuracy_pct"]
        passing, failures = [], []
        for row in rows:
            target = float(row["task3_accuracy_pct"])
            retained = float(row["task0_accuracy_pct"])
            drift = abs(retained - initial_retained)
            if not math.isclose(drift, float(row["task0_absolute_drift_pp"]), abs_tol=1e-9):
                raise ValueError(f"Incorrect drift: {row}")
            passed = (valid and int(row["step"]) > 0
                      and target <= thresholds["target_accuracy_at_most_pct"]
                      and drift < thresholds["retained_absolute_drift_strictly_under_pp"])
            if str(passed) != row["passes_screen"]:
                raise ValueError(f"Incorrect screen: {row}")
            if passed:
                passing.append(int(row["step"]))
            if int(row["step"]) > 0 and drift >= thresholds["retained_absolute_drift_strictly_under_pp"]:
                failures.append(int(row["step"]))
        observations = experiment["observations"]
        expected = {
            "passing_steps": passing,
            "first_retention_failure_step": failures[0] if failures else None,
            "task3_final_accuracy_pct": float(rows[-1]["task3_accuracy_pct"]),
            "task0_final_accuracy_pct": float(rows[-1]["task0_accuracy_pct"]),
            "minimum_task3_accuracy_pct": min(float(r["task3_accuracy_pct"]) for r in rows),
            "accuracy_observations": len(rows),
        }
        for key, value in expected.items():
            if observations[key] != value:
                raise ValueError(f"Registry/trace mismatch: {experiment['id']} {key}")
    for table, key_fields in ((traces, ("run", "step")),
                              (gradients, ("run", "step", "parameter_group")),
                              (norms, ("run", "step"))):
        keys = [tuple(row[key] for key in key_fields) for row in table]
        if len(keys) != len(set(keys)) or any(row["run"] not in ids for row in table):
            raise ValueError("Duplicate observations or unknown run")


def value(item):
    if item is None:
        return "Not recorded"
    if isinstance(item, (list, dict)):
        return json.dumps(item, ensure_ascii=False)
    return str(item).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(value(item) for item in row) + " |" for row in rows],
    ]) + "\n"


def render(registry, traces, gradients, norms):
    base = registry["repository"]
    link = lambda title, slug: f"[{title}]({base}/wiki/{slug})"
    source = lambda path: f"[{Path(path).name}]({base}/blob/main/{path})"
    pages = {}
    rows = registry["experiments"]
    index = table(["Record", "Category", "Date evidence", "Decision"], [
        [link(row["title"], "Experiment-" + row["id"]), row["category"],
         row["run_date"] or "Not recorded", row["decision"]] for row in rows
    ])
    pages["Home.md"] = f"""# CARK wiki

- {link('Concepts and processes: FAQ', 'Concepts-and-processes')}: general explanations of
  hypernetworks, heads and chunks, learning, forgetting, evaluation, and experiment workflow.
- {link('Run logs explained', 'run-logs-explained')}: the experiment archive, including
  results, protocol, evidence, software checks, and the record template.
- {source('docs/RESEARCH_DIAGNOSTIC.md')}: how the reusable diagnostic screens
  current runs and accepts future matched recovery probes.
- {link('Anchor paper', 'An-Unlearning-Framework-for-Continual-Learning')}: the paper
  used as the reproduction reference.
"""
    pages["Concepts-and-processes.md"] = (ROOT / "docs" / "CONCEPTS.md").read_text(encoding="utf-8")
    pages["An-Unlearning-Framework-for-Continual-Learning.md"] = (
        ROOT / "docs" / "ANCHOR_PAPER.md"
    ).read_text(encoding="utf-8")
    pages["run-logs-explained.md"] = f"""# Run logs explained

Updated {registry['updated']}. This section records completed experiments and numerical
checks. It separates reported measurements, interpretation, and proposed next work.

**Current result:** the full-sequence reproduction failed retention, and none of
the recorded checkpoint diagnostics passed the joint forgetting/retention screen.
The latest 30-step run reached 12.6% target accuracy at step 27 while retained
accuracy had drifted 3.6 points. The target threshold is 12%.

- {link('Experiment index', 'Experiment-index')}: all {len(rows)} archived records.
- {link('Protocol and decision rules', 'Protocol')}: settings, thresholds, and comparability.
- {link('Latest 30-step result', 'Experiment-forgetting-30step-20260929')}: accuracy and gradient measurements.
- {link('Evidence and downloadable files', 'Evidence-and-files')}: provenance and source data.
- {link('Software validation', 'Software-validation')}: checks kept separate from research runs.
- {link('Record template', 'Record-template')}: how to log the next experiment.
- {link('Anchor paper', 'An-Unlearning-Framework-for-Continual-Learning')}: the reproduction reference.

Original Drive artifacts are not mirrored here. Measurements transcribed from
tables retain their reported precision; missing values remain missing. The
archive includes the full-sequence run, early learning and forgetting probes,
checkpoint preparation, E08-E15, the latest 30-step run, and two numerical checks. It does not present
planned Fisher attribution or relearning experiments as completed work.

The canonical structured record is {source('docs/experiments/registry.json')}.
Pages are generated with `python scripts/build_experiment_wiki.py`.

The {source('docs/RESEARCH_DIAGNOSTIC.md')} keeps current forgetting screens
separate from future matched recovery and component tests.
"""
    pages["Experiment-index.md"] = "# Experiment index\n\n" + registry["identifier_policy"] + "\n\n" + index
    pages["Protocol.md"] = f"""# Protocol and interpretation

The short diagnostic is `L3 L0 U3`: learn task 3, learn task 0, then forget task 3
while protecting task 0. Tiny ImageNet accuracy uses its labelled validation split.
Each task has ten classes. The common intended model is ResNet50 generated in
200 chunks, with hidden widths 128/256/512 and 32-dimensional task/chunk codes.
Common defaults are intended settings, not verified metadata for every run.

The E08 checkpoint has reported starting accuracies of 26.0% for task 3 and 44.6%
for task 0. E08 onward restore that starting state for each diagnostic. Within a
forget request, keep one continuous Adam optimiser and one frozen reference.
Calling one-step forget requests repeatedly changes the experiment.

## Diagnostic screen

- Both initial accuracies must be at least 25%.
- After an update, target-task accuracy must be at most 12%.
- Absolute retained-task accuracy drift must be strictly below five percentage points.
- Step zero is a starting observation, never a passing update.

The full reproduction has separate criteria in {source('docs/PLAN.md')}. A short
screen pass would identify a candidate for validation, not establish deletion.
Learning-only and numerical checks are not judged by the forgetting screen.

## Timing and units

Accuracies are percentages; drift is in percentage points. Losses, raw-output
norms, and gradient norms are measured before their named update. Accuracy is
measured after it. Step 10's raw norm therefore describes the model after nine
updates. Reported E15 and 30-step noise-gradient norms already include gamma.

## Comparison limits

E04-E07 start from separately trained models. E10 also changes the reported GPU
and helper relative to E09. E11-E14 requested the same T4 runtime, but lack
separate per-run hardware evidence. Commit `4c08f54` introduces separate task-code
and forgetting-noise streams; E15 reports commit `e3087f0`. Matching E14's gamma
and learning rate therefore does not make E15 an exact replay of its trajectory.

The preservation penalty constrains generated weights, not accuracy directly.
At the initial snapshot its gradient is zero. Gradient magnitudes do not reveal
their relative directions or their contributions to an Adam update. Measurements
for the BatchNorm head concern generated scale and offset parameters; per-task
running means and variances are separate stored buffers.

## Next work, not yet a result

The new diagnostic code can measure gradient alignment, actual Adam update
norms, and before/after component changes, but no GPU result from that code has
been supplied. Combined gradients and raw/scaled generated-weight changes remain
unmeasured. No gradient-cancellation finding, relearning advantage, or component
storage attribution has been established by the archived runs.
"""
    files = ["registry.json", "manifest.json", "forgetting_traces.csv", "gradient_norms.csv",
             "raw_output_norms.csv", "e13_sampled_losses.csv", "e15_reported_output.txt",
             "notebook_saved_outputs.txt", "sources/full_sequence_20260921.md",
             "sources/fisher_sanity_output.txt", "sources/forgetting_30step_20260929.txt"]
    pages["Evidence-and-files.md"] = "# Evidence and files\n\n" + "\n".join(
        "- " + source("docs/experiments/" + filename) for filename in files
    ) + """

The main registry includes every archived experiment. The manifest covers
checkpoint diagnostics and records their reported artifact paths. Together the
trace files contain {len(traces)} accuracy observations, {len(gradients)} supplied
gradient rows, and {len(norms)} supplied raw-output norms. The 30-step run has
supplied gradient norms at steps 1, 2, 5, 10, 20, 25, 27, and 30. Missing
losses are blank, not inferred from norms.

`e15_reported_output.txt` is a transcription of the user's pasted table, not the
original runtime JSON. The runtime reported completion and a valid starting
point. GPU identity, runtime, and loss components were not supplied for E15.
The E15 date comes from its report filename; the original file was not opened.

Some earlier settings are only intended settings from instructions. Saved
notebook outputs corroborate E13/E14 settings, but the notebook mixes several
runs. Its setup cell is not proof of every run's device or source commit.

The full-sequence summary and Fisher output come from local historical notes.
They are archived as reported evidence. The Fisher toy check does not imply
that a forgotten classifier is confidently wrong or that Fisher locates storage.

Drive paths are reported artifact locations, not public download links or a
guarantee of current availability. No checkpoint weights are published here.
"""
    pages["Record-template.md"] = """# Adding an experiment

Create the record when a run starts and fill its observed results when it ends.
Preserve failed and interrupted runs. Distinguish intended settings from the
settings printed or stored by the actual run. Use null for unavailable metadata.

1. Add an entry to `docs/experiments/registry.json` with a stable identifier,
   question, category, date evidence, commit, hardware, settings, initial state,
   observations, decision, interpretation, limitations, evidence, and artifacts.
2. Preserve the original JSON report when available. If only a pasted table is
   available, archive a labelled transcription and retain its printed precision.
3. Add observations to the relevant CSV. Record whether measurements are before
   or after an update. Do not reconstruct missing gradients or losses.
4. For a checkpoint diagnostic, update `manifest.json` and derive the screen
   from the original thresholds. Do not change thresholds to make a run pass.
5. Run `python scripts/build_experiment_wiki.py`, then
   `python scripts/build_experiment_wiki.py --check`. Update the narrative log
   with the result and next question. Commit source data and generated pages.
6. Copy the generated `docs/wiki/*.md` files into the wiki checkout and push its
   default branch. Preserve any existing pages outside the generated set.

## Entry outline

- Question and the one change being tested.
- Starting checkpoint and initial measured accuracies.
- Actual configuration, code commit, GPU, seed, data split, and timing.
- Results, including failures and the full measured trajectory.
- Decision under the predeclared criterion.
- Interpretation, alternative explanations, and missing evidence.
- Next experiment or stop decision.
- Source reports and their availability.

Code tests belong on the software-validation page. Proposed experiments belong
in the plan until they have an actual run record.
"""
    pages["Software-validation.md"] = f"""# Software validation

These are software checks, not measurements of successful unlearning.

## Commit 4c08f54 review

The conversation records 17 passing core checks and eight passing diagnostic
tests, with one CUDA-only test skipped. Additional CPU checks on CNN and
ResNet18 confirmed matching losses and model states with gradient measurement
enabled and disabled, with and without protected tasks. Consuming the global
random stream between runs did not change the tested forgetting result.

## Notebook utilities at e3087f0

The conversation records 11 passing notebook-helper tests, including execution
of all five notebook cells on a small CPU checkpoint with Colab services mocked.
Before publishing, 17 core checks, seven task checks, and 18 experiment checks
passed. One task check was skipped. The earlier diagnostic run passed eight
tests with one CUDA-only skip. The guide checker had no file at its default
path, so it did not execute guide blocks in that validation pass.

These records describe separate validation sessions. Earlier logs report guide
checks using an explicitly supplied guide path; that does not mean the default
guide check ran during the e3087f0 publish. Full-scale GPU memory behavior was
not validated by the CPU tests.

Source: the recorded review and publication results in the project conversation;
earlier validation history remains in {source('docs/EXPERIMENT_LOG.md')}.
"""
    sampled_losses = read_csv("e13_sampled_losses.csv")
    for experiment in rows:
        identifier = experiment["id"]
        sections = ["# " + experiment["title"],
                    link("Experiment index", "Experiment-index"),
                    "## Question\n\n" + experiment["question"],
                    "## Record\n\n" + table(["Field", "Value"], [
                        ["Identifier", identifier], ["Category", experiment["category"]],
                        ["Date", experiment["run_date"]], ["Date evidence", experiment["date_evidence"]],
                        ["Reported source commit", experiment["reported_commit"]],
                        ["Hardware", experiment["hardware"]], ["Starting state", experiment["initial_state"]],
                    ]),
                    "## Settings\n\n" + table(["Setting", "Value"], experiment["settings"].items())
                    + "\n" + experiment["settings_evidence"],
                    "## Observations\n\n" + table(["Measurement", "Value"], [
                        [key, "None in supplied trace" if key == "first_retention_failure_step" and item is None else item]
                        for key, item in experiment["observations"].items()])]
        accuracy = [row for row in traces if row["run"] == identifier]
        if accuracy:
            norm_map = {row["step"]: row["raw_norm_before"] for row in norms if row["run"] == identifier}
            sections.append("## Accuracy trace\n\n" + table(
                ["Step", "Task 3 %", "Task 0 %", "Absolute drift (pp)", "Passes screen", "Raw norm before"],
                [[r["step"], r["task3_accuracy_pct"], r["task0_accuracy_pct"], r["task0_absolute_drift_pp"],
                  r["passes_screen"], norm_map.get(r["step"], "Not supplied")] for r in accuracy]))
            losses = [r for r in accuracy if r["weighted_noise_before"] or r["preserve_before"]]
            losses += [r for r in sampled_losses if r["run"] == identifier]
            if losses:
                sections.append("## Supplied losses before updates\n\n" + table(
                    ["Step", "Weighted noise", "Preservation", "Total"],
                    [[r["step"], r["weighted_noise_before"] or "Not supplied", r["preserve_before"] or "Not supplied",
                      r["total_loss_before"] or "Not supplied"] for r in losses]))
        group_rows = [row for row in gradients if row["run"] == identifier]
        if group_rows:
            sections.append("## Supplied gradient norms before updates\n\nNoise includes gamma. Values retain printed precision.\n\n" + table(
                ["Step", "Group", "Weighted noise gradient", "Preservation gradient"],
                [[r["step"], r["parameter_group"], r["weighted_noise_gradient_norm_before"], r["preservation_gradient_norm_before"]] for r in group_rows]))
        sections.extend([
            "## Decision\n\n" + experiment["decision"],
            "## Interpretation\n\n" + experiment["interpretation"],
            "## Limitations\n\n" + "\n".join("- " + item for item in experiment["limitations"]),
            "## Next step\n\n" + (experiment["next_step"] or "No separate next step was recorded."),
            "## Evidence\n\n" + "\n".join("- " + source(item["path"]) + ": " + item["kind"]
                + (". " + item["note"] if item["note"] else "") for item in experiment["evidence"]),
        ])
        if experiment["reported_artifacts"]:
            sections.append("## Reported artifact locations\n\n" + "\n".join(
                f"- `{item['path']}`: {item['availability']}" for item in experiment["reported_artifacts"]))
        pages["Experiment-" + identifier + ".md"] = "\n\n".join(sections) + "\n"
    pages["_Sidebar.md"] = "\n".join("- " + link(title, slug) for title, slug in [
        ("Home", "Home"), ("Concepts and processes: FAQ", "Concepts-and-processes"),
        ("Run logs explained", "run-logs-explained"),
        ("Latest 30-step result", "Experiment-forgetting-30step-20260929"),
        ("Anchor paper", "An-Unlearning-Framework-for-Continual-Learning")]) + "\n"
    for name in ["Experiment-index.md", "Protocol.md", "Evidence-and-files.md",
                 "Record-template.md", "Software-validation.md",
                 *["Experiment-" + row["id"] + ".md" for row in rows]]:
        title, body = pages[name].split("\n", 1)
        pages[name] = title + "\n\n" + link("Run logs explained", "run-logs-explained") + "\n" + body
    return pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "wiki")
    parser.add_argument("--check", action="store_true", help="Validate data and compare pages without writing")
    args = parser.parse_args()
    registry = json.loads((DATA / "registry.json").read_text(encoding="utf-8"))
    traces, gradients, norms = (read_csv(name) for name in
                               ("forgetting_traces.csv", "gradient_norms.csv", "raw_output_norms.csv"))
    validate(registry, traces, gradients, norms)
    pages = render(registry, traces, gradients, norms)
    for name, text in pages.items():
        if "\u2014" in text:
            raise ValueError(f"Em dash in generated prose: {name}")
        destination = args.output / name
        if args.check:
            if not destination.exists() or destination.read_text(encoding="utf-8") != text:
                raise ValueError(f"Page needs regeneration: {destination}")
        else:
            args.output.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
    print(f"Validated {len(registry['experiments'])} records and {len(traces)} accuracy observations; "
          f"{'checked' if args.check else 'generated'} {len(pages)} wiki pages.")


if __name__ == "__main__":
    main()
