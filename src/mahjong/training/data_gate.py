"""Bind a completed legality audit to the current dataset and converter sources."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AUDIT_SOURCES = (
    "src/mahjong/dataset/legality.py", "src/mahjong/dataset/schema.py",
    "src/mahjong/rules/legal.py", "src/mahjong/rules/scoring.py", "src/mahjong/rules/yaku.py",
    "src/mahjong/rules/decompose.py", "src/mahjong/training/fast_encode.py",
    "src/mahjong/features/action_space_v2.py", "src/mahjong/features/encoder_v2.py",
    "scripts/audit_action_feature_v2.py",
)


def source_hashes():
    names = {p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.py")}
    names.update(AUDIT_SOURCES)
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(names)}


def dataset_identity(dataset_dir):
    root = Path(dataset_dir)
    inventory = []
    for split in ("train", "validation", "test"):
        for path in sorted((root / split).glob("*.parquet")):
            stat = path.stat()
            inventory.append((path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
    return {
        "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "shard_inventory_sha256": hashlib.sha256(json.dumps(inventory).encode()).hexdigest(),
        "shards": len(inventory),
    }


def require_full_audit(path, dataset_dir):
    audit = json.loads(Path(path).read_text(encoding="utf-8"))
    if audit.get("passed") is not True or audit.get("limit_per_split") is not None:
        raise ValueError("full training requires a passing complete contextual data audit")
    if audit.get("dataset_identity") != dataset_identity(dataset_dir):
        raise ValueError("dataset changed after audit")
    audited_sources = audit.get("source_sha256") or {}
    current_sources = source_hashes()
    changed_audit_sources = [
        name for name in AUDIT_SOURCES
        if audited_sources.get(name) != current_sources.get(name)
    ]
    if changed_audit_sources:
        raise ValueError(f"legality/data-audit source changed after audit: {changed_audit_sources}")
    manifest = json.loads((Path(dataset_dir) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != "decision-v3" or manifest.get("config", {}).get("diagnostic_only") is not False:
        raise ValueError("full training requires a completed non-diagnostic decision-v3 dataset")
    # The manifest describes the raw decision-v3 conversion which was audited
    # with feature-v2/action-v2. Runtime feature encoders may be versioned
    # independently when they are derived entirely from the preserved columns.
    if (manifest.get("feature_version"), manifest.get("action_schema_version")) != (
        "feature-v2", "action-v2",
    ):
        raise ValueError("full training requires the audited decision-v3 feature-v2/action-v2 source dataset")
    from ..rules.legal import LEGAL_MASK_VERSION
    if manifest.get("legal_mask_version") != LEGAL_MASK_VERSION:
        raise ValueError("dataset legal mask version changed")
    if audit.get("dataset_manifest") != manifest:
        raise ValueError("audit manifest content differs from dataset")
    if set(audit.get("splits", {})) != {"train", "validation", "test"}:
        raise ValueError("audit split coverage incomplete")
    for split, result in audit["splits"].items():
        if (not result.get("rows") or result.get("context_verified_rows") != result["rows"]
                or result.get("files_scanned") != result.get("files_available")):
            raise ValueError("audit does not verify all rows/shards")
        expected = manifest.get("counts", {}).get("splits", {}).get(split, {})
        if expected.get("samples") != result["rows"] or expected.get("files") != result["files_available"]:
            raise ValueError(f"audit/manifest counts disagree for {split}")
    if manifest.get("counts", {}).get("samples") != sum(r["rows"] for r in audit["splits"].values()):
        raise ValueError("audit/manifest total row count differs")
    return audit
