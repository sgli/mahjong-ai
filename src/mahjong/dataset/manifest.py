"""Dataset manifest / metadata (DATA_SPEC sections 10 & 14)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_commit(repo_dir: Path | None = None) -> str | None:
    """Return the current git commit hash, or ``None`` if unavailable."""
    cwd = str(repo_dir) if repo_dir else None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def build_manifest(
    *,
    dataset_version: str,
    action_schema_version: str,
    feature_version: str,
    config: dict,
    counts: dict,
    git_commit_hash: str | None,
    rules: str | None = None,
) -> dict:
    """Assemble the manifest dict (JSON-serialisable)."""
    manifest = {
        "dataset_version": dataset_version,
        "action_schema_version": action_schema_version,
        "feature_version": feature_version,
        "git_commit": git_commit_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "counts": counts,
    }
    if rules is not None:
        manifest["rules"] = rules
    return manifest


def write_manifest(path: Path, manifest: dict) -> None:
    """Write the manifest JSON file (atomic-ish: write temp then replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)
