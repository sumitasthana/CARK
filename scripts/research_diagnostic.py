"""Inspect all recorded forgetting steps with one reusable research screen.

Examples:
    python scripts/research_diagnostic.py --archive
    python scripts/research_diagnostic.py --archive --report /path/to/forget.json
    python scripts/research_diagnostic.py --report /path/to/forget.json --output summary.json
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uncle.research_diagnostic import assess_forgetting, archived_runs, runtime_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true",
                        help="Include the tracked historical accuracy traces")
    parser.add_argument("--report", type=Path, action="append", default=[],
                        help="Saved diagnose_forgetting JSON; may be repeated")
    parser.add_argument("--output", type=Path,
                        help="Also write the complete step-by-step assessment as JSON")
    args = parser.parse_args()
    if not args.archive and not args.report:
        parser.error("Choose --archive or at least one --report")

    runs = []
    if args.archive:
        directory = ROOT / "docs" / "experiments"
        runs.extend(archived_runs(directory / "registry.json",
                                  directory / "forgetting_traces.csv"))
    for path in args.report:
        report = json.loads(path.read_text(encoding="utf-8"))
        runs.append(runtime_run(report))
    ids = [run["id"] for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate run identifier")
    results = [assess_forgetting(run) for run in runs]
    print("run  source  steps  best within retention  step  passing steps")
    for item in results:
        best = item["best_target_accuracy_within_retention_limit_pct"]
        best_text = "n/a" if best is None else f"{best:.1f}%"
        print(f"{item['id']}  {item['source']}  {len(item['trajectory']) - 1}"
              f"  {best_text}  {item['best_step_within_retention_limit']}"
              f"  {item['passing_steps']}")
    print("These are forgetting screens, not recovery or component attribution results.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"schema_version": 1, "runs": results}, indent=2)
                               + "\n", encoding="utf-8")
        print("Saved:", args.output)


if __name__ == "__main__":
    main()
