"""Package tracked research records and saved run artifacts for sharing.

Run on CPU after mounting Drive in Colab, or point --drive-root at a local copy.
The archive contains source files unchanged, a file manifest, and a readable index.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile


DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/uncle/E08_forgetting_trace")
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz"}


def git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def tracked_files(repo: Path) -> list[Path]:
    raw = subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z"])
    return [repo / os.fsdecode(name) for name in raw.split(b"\0") if name]


def add_file(bundle: ZipFile, source: Path, archive_name: str) -> dict:
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as stream, bundle.open(archive_name, "w", force_zip64=True) as target:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            target.write(block)
            digest.update(block)
            size += len(block)
    return {"path": archive_name, "bytes": size, "sha256": digest.hexdigest()}


def known_paths(drive_root: Path) -> dict[str, bool]:
    return {
        "pre_forgetting_checkpoint": (drive_root / "before_forgetting.pt").is_file(),
        "full_sequence_history": (
            drive_root / "full_sequence_27step_candidate/history_seq1_resnet50_seed0.json"
        ).is_file(),
        "full_sequence_summary": (
            drive_root / "full_sequence_27step_candidate/summary_seq1_resnet50_seed0.json"
        ).is_file(),
        "paired_L9_root": (
            drive_root / "paired_L9_probe/same_runtime_paired"
        ).is_dir(),
    }


def readable_index(repo: Path, manifest: dict) -> str:
    registry = json.loads(
        (repo / "docs/experiments/registry.json").read_text(encoding="utf-8")
    )
    lines = [
        "# Experiment evidence bundle", "",
        f"Created (UTC): {manifest['created_at_utc']}",
        f"Repository commit: {manifest['repository_commit']}",
        f"Tracked repository files: {manifest['counts']['repository_files']}",
        f"Saved Drive files: {manifest['counts']['drive_files']}",
        f"Checkpoints included: {manifest['include_checkpoints']}", "",
        "The tracked archive includes reported or transcribed observations. Drive files",
        "are copies of saved artifacts found at export time. The manifest records a SHA-256",
        "hash for every copied source file. A file absent at export time is not reconstructed.",
        "The ZIP does not include conversational messages that were never saved to files.",
        "", "## Recorded experiments", "",
        "| Record | Type | Recorded decision |", "| --- | --- | --- |",
    ]
    for item in registry["experiments"]:
        lines.append(
            f"| {item['id']} | {item['category']} | {item['decision']} |"
        )
    lines += ["", "## Known Drive locations", ""]
    for name, present in manifest["known_drive_paths"].items():
        lines.append(f"- {name}: {'found' if present else 'not found'}")
    lines += [
        "", "## Where to look", "",
        "- `repository/docs/experiments/`: archived tables, traces, and original transcriptions.",
        "- `repository/docs/wiki/Experiment-trajectory.md`: explanations and limits for earlier runs.",
        "- `repository/notebooks/`: experiment and reporting notebooks.",
        "- `drive/`: saved run JSON, diagnostics, plots, and optional checkpoints.",
        "- `manifest.json`: file names, sizes, SHA-256 hashes, and exclusions.",
        "- `RUN_TIMELINES.md`: a readable view of each saved request history.",
        "", "## Limits", "",
        "A missing value is not a zero. Older runs may have lost runtime-local files.",
        "Recorded decisions are not new checks performed by this export.",
    ]
    excluded = manifest["excluded"]
    if excluded:
        lines += ["", f"Excluded files: {len(excluded)}. See `manifest.json` for each path and reason."]
    return "\n".join(lines) + "\n"


def readable_timelines(drive_root: Path) -> str:
    lines = [
        "# Saved request timelines", "",
        "These rows come from saved `history_*.json` files. The original JSON",
        "is also in this ZIP. Accuracy dictionaries show every task recorded at",
        "each request. Time is included only when a matching costs file has it.",
        "",
    ]
    histories = sorted(drive_root.rglob("history_*.json")) if drive_root.is_dir() else []
    if not histories:
        lines += ["No saved request histories were found at export time.", ""]
    for path in histories:
        name = path.relative_to(drive_root).as_posix()
        lines += [f"## {name}", ""]
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
            stem = path.name.removeprefix("history_")
            costs_path = path.with_name("costs_" + stem)
            costs = (json.loads(costs_path.read_text(encoding="utf-8"))
                     if costs_path.is_file() else {})
            seconds = {row["index"]: row.get("seconds")
                       for row in costs.get("requests", [])}
            for row in history:
                index = row.get("index")
                lines += [
                    f"### Request {index}: {row.get('action')} task {row.get('task')}",
                    "",
                    f"- Before: `{json.dumps(row.get('before', {}), sort_keys=True)}`",
                    f"- After: `{json.dumps(row.get('after', {}), sort_keys=True)}`",
                    f"- Seen tasks: `{json.dumps(row.get('seen', []))}`",
                    f"- Forgotten tasks: `{json.dumps(row.get('forgotten', []))}`",
                    f"- Final loss: {row.get('final_loss', 'not recorded')}",
                    f"- Forgetting steps: {row.get('burn_in', 'not applicable')}",
                    f"- Request seconds: {seconds.get(index, 'not recorded')}",
                    "",
                ]
        except (OSError, ValueError, KeyError, TypeError) as error:
            lines += [f"Could not parse this history: {error}", ""]
    return "\n".join(lines) + "\n"


def export_bundle(
    repo: Path, drive_root: Path, output: Path, include_checkpoints: bool = False
) -> dict:
    repo = repo.resolve()
    drive_root = drive_root.resolve()
    output = output.resolve()
    if not (repo / "docs/experiments/registry.json").is_file():
        raise FileNotFoundError("Repository experiment archive not found")
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing bundle: {output}")
    if drive_root.is_dir() and (output == drive_root or drive_root in output.parents):
        raise ValueError("Put the ZIP outside the scanned Drive run folder")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".writing")
    if temporary.exists():
        raise FileExistsError(temporary)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": git_text(repo, "rev-parse", "HEAD"),
        "repository_clean": not bool(git_text(repo, "status", "--porcelain")),
        "drive_root_found": drive_root.is_dir(),
        "known_drive_paths": known_paths(drive_root),
        "include_checkpoints": include_checkpoints,
        "files": [],
        "excluded": [],
        "counts": {"repository_files": 0, "drive_files": 0},
    }
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6,
                     allowZip64=True) as bundle:
            for source in tracked_files(repo):
                if not source.is_file():
                    manifest["excluded"].append({
                        "path": f"repository/{source.relative_to(repo).as_posix()}",
                        "reason": "tracked file unavailable",
                    })
                    continue
                name = f"repository/{source.relative_to(repo).as_posix()}"
                manifest["files"].append(add_file(bundle, source, name))
                manifest["counts"]["repository_files"] += 1

            if drive_root.is_dir():
                for source in sorted(drive_root.rglob("*")):
                    if not source.is_file():
                        continue
                    name = f"drive/{source.relative_to(drive_root).as_posix()}"
                    if source.is_symlink():
                        reason = "symlink skipped"
                    elif source.suffix.lower() in ARCHIVE_SUFFIXES:
                        reason = "existing archive skipped"
                    elif source.name.endswith(".writing"):
                        reason = "incomplete temporary file skipped"
                    elif source.suffix.lower() in CHECKPOINT_SUFFIXES and not include_checkpoints:
                        reason = "checkpoint excluded; use --include-checkpoints"
                    else:
                        reason = None
                    if reason:
                        manifest["excluded"].append({
                            "path": name, "bytes": source.stat().st_size,
                            "reason": reason,
                        })
                        continue
                    manifest["files"].append(add_file(bundle, source, name))
                    manifest["counts"]["drive_files"] += 1

            bundle.writestr("README.md", readable_index(repo, manifest))
            bundle.writestr("RUN_TIMELINES.md", readable_timelines(drive_root))
            bundle.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--drive-root", type=Path, default=DEFAULT_DRIVE_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-checkpoints", action="store_true")
    args = parser.parse_args()
    output = args.output or (
        args.drive_root.parent / "experiment_exports"
        / f"experiment_evidence_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.zip"
    )
    manifest = export_bundle(
        args.repo, args.drive_root, output, args.include_checkpoints
    )
    print("Bundle:", output)
    print("Tracked repository files:", manifest["counts"]["repository_files"])
    print("Saved Drive files:", manifest["counts"]["drive_files"])
    print("Excluded files:", len(manifest["excluded"]))
    print("Drive root found:", manifest["drive_root_found"])


if __name__ == "__main__":
    main()
