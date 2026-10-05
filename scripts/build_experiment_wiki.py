"""Validate the experiment archive and render its GitHub wiki pages."""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "experiments"
REPORTS = ROOT / "docs" / "reports" / "trajectory"


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
    pages["Home.md"] = f"""# CARK wiki

Start here:

- {link('Experiment trajectory', 'Experiment-trajectory')}: every run so far, in order, with figures.
- {link('Concepts and processes', 'Concepts-and-processes')}: what the model does and how to read its results.
- {link('Model diagnostics', 'Model-diagnostics')}: inspect weights, gradients, and Fisher scores.
- {link('Run logs', 'run-logs-explained')}: what we ran and what happened.
- {link('Anchor paper', 'An-Unlearning-Framework-for-Continual-Learning')}: the paper we are reproducing.
"""
    # Trajectory reports are dated snapshots, one file per edition. Each keeps
    # its figures beside it in the wiki repository, so the relative image paths
    # in the source resolve there without rewriting. "Experiment-trajectory" is
    # the stable landing page listing every edition, newest first.
    editions = []
    for report in sorted(REPORTS.glob("20*.md"), reverse=True):
        text = report.read_text(encoding="utf-8")
        pages[f"Experiment-trajectory-{report.stem}.md"] = text
        editions.append((report.stem, text.split("\n", 1)[0].lstrip("# ").strip()))
    if not editions:
        raise ValueError(f"No trajectory reports found in {REPORTS}")
    pages["Experiment-trajectory.md"] = (
        "# Experiment trajectory\n\n"
        "Each edition is a snapshot of every run recorded up to its date. "
        "Editions are kept rather than replaced, so an earlier reading of the "
        "evidence stays available.\n\n"
        + table(["Date", "Edition", "Status"], [
            [date, link(title, f"Experiment-trajectory-{date}"),
             "Latest" if index == 0 else "Superseded"]
            for index, (date, title) in enumerate(editions)])
        + f"\n\nSource files live in {source('docs/reports/trajectory')}.\n")
    pages["Model-diagnostics.md"] = (
        (ROOT / "docs" / "MODEL_DIAGNOSTICS.md").read_text(encoding="utf-8")
        .replace("(../notebooks/04_gradient_diagnostics.ipynb)",
                 f"({base}/blob/main/notebooks/04_gradient_diagnostics.ipynb)")
        .replace("(../uncle/research_diagnostic.py)",
                 f"({base}/blob/main/uncle/research_diagnostic.py)")
        .replace("(PLAN.md)", f"({base}/blob/main/docs/PLAN.md)")
    )
    pages["Concepts-and-processes.md"] = (ROOT / "docs" / "CONCEPTS.md").read_text(encoding="utf-8")
    pages["An-Unlearning-Framework-for-Continual-Learning.md"] = (
        ROOT / "docs" / "ANCHOR_PAPER.md"
    ).read_text(encoding="utf-8")
    pages["run-logs-explained.md"] = f"""# Run logs explained

Updated {registry['updated']}. This is a record of runs, including failed runs.

**What have we found?** No recorded checkpoint run meets both targets. The
latest 30-step run got task 3 down to 12.6% at step 27. We need 12% or lower.
Task 0 changed by 3.6 percentage points at that step, within its limit.

- {link('Experiment trajectory', 'Experiment-trajectory')}: all {len(rows)} records, told in order.
- {link('Rules for reading results', 'Protocol')}: what counts as a pass.
- {link('Source files', 'Evidence-and-files')}: where the numbers came from.
- {link('Software checks', 'Software-validation')}: checks of the code.
- {link('Add a run', 'Record-template')}: how to keep the next record.

The original files on Drive are not copied into this wiki. Some numbers came
from tables pasted into the conversation. We keep those numbers as reported and
leave missing values blank. The {source('docs/experiments/registry.json')} is the
structured record. The {link('Model diagnostics', 'Model-diagnostics')} page
explains measurements inside the model.
"""
    pages["Protocol.md"] = f"""# Rules for reading results

The short test is `L3 L0 U3`: learn task 3, learn task 0, then try to forget
task 3 while keeping task 0. Accuracy comes from Tiny ImageNet validation images.
Each task has ten classes. The intended model is ResNet50 with a hypernetwork
that generates weights in 200 chunks. These are intended settings; older run
records do not confirm every setting.

The E08 checkpoint starts at 26.0% on task 3 and 44.6% on task 0. Later
checkpoint tests restore it before forgetting. Each test makes one continuous
forget request. Restarting the request after every step would change the test.

## What counts as a pass?

- Both tasks must start at 25% accuracy or higher.
- After at least one update, task 3 must be at 12% or lower.
- Task 0 must stay within 5 percentage points of its starting accuracy.

For example, task 0 starts at 44.6%. A result at 41.0% has changed by 3.6
points and is within the limit. A result at 39.6% has changed by 5 points and
fails. Passing this short test would only make a run worth studying further.
It would not prove that task 3's information is gone. The full study has
separate rules in {source('docs/PLAN.md')}.

## When are measurements taken?

The loss, raw weight size, and gradient size at a step are measured before its
update. Accuracy is measured after the update. In the printed tables, the raw
size at step 10 describes the model after nine updates. The noise gradient
numbers already include gamma.

## Which runs can we compare?

E04-E07 used separately trained models. E10 also changed the reported GPU and
helper relative to E09. E11-E14 requested the same T4 runtime, but we lack
separate hardware records for each run. E15 used code with separate random
streams for task codes and forgetting noise. Even with the same learning rate
and gamma, E14 and E15 are not exact replays.

The preservation term tries to hold generated weights near saved values. It
does not hold accuracy directly. Its gradient is zero at the saved starting
state. Gradient size alone cannot tell us whether two gradients oppose each
other or how Adam will update the weights. BatchNorm running statistics are
stored separately from generated BatchNorm parameters.

## What remains unknown?

New code can inspect weights, gradients, and diagonal Fisher scores. No GPU
result from that complete inspection has been reported. These measurements
do not, by themselves, show that task information was removed or locate where
it is stored. See {link('Model diagnostics', 'Model-diagnostics')}.
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
    pages["_Sidebar.md"] = "\n".join("- " + link(title, slug) for title, slug in [
        ("Home", "Home"), ("Experiment trajectory", "Experiment-trajectory"),
        ("Concepts and processes: FAQ", "Concepts-and-processes"),
        ("Model diagnostics", "Model-diagnostics"),
        ("Run logs explained", "run-logs-explained"),
        ("Anchor paper", "An-Unlearning-Framework-for-Continual-Learning")]) + "\n"
    for name in ["Protocol.md", "Evidence-and-files.md",
                 "Record-template.md", "Software-validation.md"]:
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
    # A page that is no longer generated has to leave the output directory too,
    # or it survives as an orphan that nothing links to and nobody updates.
    stale = sorted(path for path in args.output.glob("*.md") if path.name not in pages)
    if stale and args.check:
        raise ValueError("Pages no longer generated: "
                         + ", ".join(path.name for path in stale))
    for path in stale:
        path.unlink()
        print(f"removed {path.name}")

    # Trajectory pages link their figures relatively, so the figures travel
    # with them into the wiki repository.
    figures = REPORTS / "figures"
    if not args.check:
        shutil.copytree(figures, args.output / "figures", dirs_exist_ok=True)
    elif any(not (args.output / "figures" / f.relative_to(figures)).is_file()
             for f in figures.rglob("*.svg")):
        raise ValueError(f"Trajectory figures missing from {args.output}")
    print(f"Validated {len(registry['experiments'])} records and {len(traces)} accuracy observations; "
          f"{'checked' if args.check else 'generated'} {len(pages)} wiki pages.")


if __name__ == "__main__":
    main()
