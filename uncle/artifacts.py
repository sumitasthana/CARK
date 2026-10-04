"""Read saved experiment JSON without loading model checkpoints."""

import json
from pathlib import Path


KINDS = ("origin", "environment", "history", "costs", "summary")


class RunArtifactError(ValueError):
    """A saved run is missing files or has inconsistent request records."""

    def __init__(self, message, missing=()):
        super().__init__(message)
        self.missing = list(missing)


def load_run_artifacts(folder, *, required=("history", "summary"),
                       require_checkpoint=False):
    """Return one run's JSON and paths, checking its request history.

    `required` lets readers accept partial archives. The checkpoint is only
    checked for presence and size; its tensors are never loaded.
    """
    folder = Path(folder)
    unknown = set(required) - set(KINDS)
    if unknown:
        raise ValueError(f"Unknown artifact kinds: {sorted(unknown)}")
    histories = sorted(folder.glob("history_*.json")) if folder.is_dir() else []
    if len(histories) != 1:
        raise RunArtifactError(
            f"Expected one history file in {folder}", [f"one history file in {folder}"])
    stem = histories[0].stem.removeprefix("history_")
    paths = {kind: folder / f"{kind}_{stem}.json" for kind in KINDS}
    checkpoint = folder / f"checkpoint_{stem}.pt"
    missing = [str(paths[kind]) for kind in required if not paths[kind].is_file()]
    if require_checkpoint and (not checkpoint.is_file() or checkpoint.stat().st_size == 0):
        missing.append(str(checkpoint))
    if missing:
        raise RunArtifactError("Missing run artifacts", missing)

    files = {}
    for kind, path in paths.items():
        if path.is_file():
            try:
                files[kind] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RunArtifactError(f"Cannot read {path}: {exc}") from exc
    history = files["history"]
    if not isinstance(history, list) or not history:
        raise RunArtifactError(f"Empty or invalid history: {paths['history']}")
    for index, row in enumerate(history):
        if not isinstance(row, dict) or row.get("index") != index:
            raise RunArtifactError(f"Invalid request index at position {index}")
        if row.get("action") not in ("learn", "forget") or not isinstance(row.get("task"), str):
            raise RunArtifactError(f"Invalid request at position {index}")

    matches_config = None
    if "origin" in files:
        try:
            configured = files["origin"]["config"]["requests"]
            matches_config = (
                len(history) == len(configured)
                and all([row["action"], row["task"]] == request
                        for row, request in zip(history, configured))
            )
        except (KeyError, TypeError):
            raise RunArtifactError(f"Invalid request config: {paths['origin']}")

    return {
        "folder": folder, "stem": stem, "files": files, "paths": paths,
        "checkpoint": checkpoint, "sequence_matches_config": matches_config,
    }
