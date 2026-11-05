"""Verify the complete frozen DeepSpeak acquisition without changing it.

The check joins the frozen target manifest, the acquisition ledger, the 25
per-archive records, and the local FLAC files. It verifies identity, labels,
paths, SHA-256 hashes, audio format and expected counts, then writes a compact
machine-readable record and a human-readable report.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf


PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT / "outputs" / "manifests" / "target_deepspeak_v2_test.csv"
MANIFEST_RECORD = PROJECT / "outputs" / "manifests" / "manifest_record.json"
ACQUISITION = PROJECT / "outputs" / "acquisition"
LEDGER = ACQUISITION / "acquisition_ledger.csv"
SHARD_RECORDS = ACQUISITION / "shard_records"
OUT_JSON = ACQUISITION / "FULL_ACQUISITION_VERIFICATION.json"
OUT_MD = ACQUISITION / "FULL_ACQUISITION_VERIFICATION.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    frozen = json.loads(MANIFEST_RECORD.read_text(encoding="utf-8"))
    manifest = read_csv(MANIFEST)
    ledger = read_csv(LEDGER)
    failures: list[str] = []

    manifest_hash = sha256_file(MANIFEST)
    if manifest_hash != frozen["manifest_sha256"]:
        failures.append("Frozen target manifest hash changed")

    manifest_ids = [row["sample_id"] for row in manifest]
    ledger_ids = [row["sample_id"] for row in ledger]
    if len(manifest_ids) != len(set(manifest_ids)):
        failures.append("Target manifest contains duplicate sample IDs")
    if len(ledger_ids) != len(set(ledger_ids)):
        failures.append("Acquisition ledger contains duplicate sample IDs")
    missing = sorted(set(manifest_ids) - set(ledger_ids))
    unexpected = sorted(set(ledger_ids) - set(manifest_ids))
    if missing:
        failures.append(f"Ledger is missing {len(missing)} frozen samples")
    if unexpected:
        failures.append(f"Ledger contains {len(unexpected)} unexpected samples")

    manifest_by_id = {row["sample_id"]: row for row in manifest}
    shard_paths = sorted(SHARD_RECORDS.glob("deepspeak-v2-*.json"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
    if len(shard_paths) != 25:
        failures.append(f"Expected 25 shard records, found {len(shard_paths)}")
    shard_samples: list[dict] = []
    shard_summaries = []
    revisions = set()
    for path in shard_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        revisions.add(record["revision"])
        selected = record["selected_files"]
        if record["selected_count"] != len(selected):
            failures.append(f"{path.name}: selected_count does not match selected_files")
        if any(row["archive"] != record["archive"] for row in selected):
            failures.append(f"{path.name}: archive field mismatch")
        if any(row["archive_sha256"] != record["archive_sha256"] for row in selected):
            failures.append(f"{path.name}: archive SHA mismatch inside selected files")
        shard_samples.extend(selected)
        shard_summaries.append({
            "archive": record["archive"],
            "archive_size_bytes": record["archive_size_bytes"],
            "archive_sha256": record["archive_sha256"],
            "selected_count": record["selected_count"],
        })
    if len(revisions) != 1:
        failures.append(f"Shard records contain {len(revisions)} dataset revisions")
    shard_ids = [row["sample_id"] for row in shard_samples]
    if len(shard_ids) != len(set(shard_ids)):
        failures.append("Shard records contain duplicate sample IDs")
    if set(shard_ids) != set(manifest_ids):
        failures.append("Combined shard records do not exactly match the frozen manifest")

    bad_label = 0
    bad_identity = 0
    bad_hash = 0
    bad_audio = 0
    missing_file = 0
    total_duration = 0.0
    sample_rates = Counter()
    channels = Counter()
    subtype = Counter()
    ledger_by_id = {row["sample_id"]: row for row in ledger}
    for index, sample_id in enumerate(manifest_ids, 1):
        m = manifest_by_id[sample_id]
        row = ledger_by_id.get(sample_id)
        if row is None:
            continue
        if row["label"] != m["label"]:
            bad_label += 1
        if row["identity"] != m["identity"]:
            bad_identity += 1
        audio_path = Path(row["audio_path"])
        if not audio_path.is_file():
            missing_file += 1
            continue
        if sha256_file(audio_path) != row["audio_sha256"]:
            bad_hash += 1
        try:
            info = sf.info(audio_path)
            sample_rates[info.samplerate] += 1
            channels[info.channels] += 1
            subtype[info.subtype] += 1
            total_duration += float(info.duration)
            if info.samplerate != 16000 or info.channels != 1 or info.format != "FLAC":
                bad_audio += 1
        except Exception:
            bad_audio += 1
        if index % 100 == 0:
            print(f"verified {index}/{len(manifest_ids)}", flush=True)

    if bad_label:
        failures.append(f"{bad_label} ledger labels disagree with the frozen manifest")
    if bad_identity:
        failures.append(f"{bad_identity} ledger identities disagree with the frozen manifest")
    if missing_file:
        failures.append(f"{missing_file} local audio files are missing")
    if bad_hash:
        failures.append(f"{bad_hash} local audio hashes disagree with the ledger")
    if bad_audio:
        failures.append(f"{bad_audio} local audio files fail the FLAC/16 kHz/mono contract")

    label_counts = Counter(row["label"] for row in manifest)
    identity_count = len({row["identity"] for row in manifest})
    expected = frozen["counts"]
    if label_counts != Counter({"real": expected["real"], "fake": expected["fake"]}):
        failures.append(f"Label counts changed: {dict(label_counts)}")
    if identity_count != frozen["identity_count"]:
        failures.append(f"Identity count changed: {identity_count}")

    result = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "PASS" if not failures else "FAIL",
        "study_id": frozen["study_id"],
        "dataset_revision": next(iter(revisions)) if len(revisions) == 1 else sorted(revisions),
        "manifest_sha256": manifest_hash,
        "manifest_rows": len(manifest),
        "ledger_rows": len(ledger),
        "shard_record_count": len(shard_paths),
        "label_counts": dict(label_counts),
        "identity_count": identity_count,
        "total_duration_seconds": round(total_duration, 6),
        "total_duration_hours": round(total_duration / 3600.0, 6),
        "sample_rates": dict(sample_rates),
        "channels": dict(channels),
        "audio_subtypes": dict(subtype),
        "bad_label": bad_label,
        "bad_identity": bad_identity,
        "missing_file": missing_file,
        "bad_hash": bad_hash,
        "bad_audio": bad_audio,
        "failures": failures,
        "shards": shard_summaries,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Full DeepSpeak acquisition verification",
        "",
        f"- Status: **{result['status']}**",
        f"- Frozen dataset revision: `{result['dataset_revision']}`",
        f"- Frozen manifest SHA-256: `{manifest_hash}`",
        f"- Shard records: {len(shard_paths)}/25",
        f"- Manifest and ledger: {len(manifest)}/{len(ledger)} rows",
        f"- Labels: {label_counts['real']} real, {label_counts['fake']} fake",
        f"- Official test identities: {identity_count}",
        f"- Total audio duration: {total_duration / 3600.0:.3f} hours",
        f"- Audio contract: {dict(sample_rates)} Hz, {dict(channels)} channel(s), {dict(subtype)}",
        f"- File hashes checked: {len(manifest) - missing_file}; mismatches: {bad_hash}",
        "",
        "This record verifies acquisition integrity only. It is not a detector-performance result.",
    ]
    if failures:
        lines.extend(["", "## Failures", ""] + [f"- {item}" for item in failures])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "manifest_rows", "label_counts", "identity_count", "total_duration_hours", "bad_hash", "bad_audio")}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
