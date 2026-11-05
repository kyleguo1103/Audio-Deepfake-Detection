"""Build and freeze the strict DeepSpeak v2 official-test audio manifest.

This script reads only locally stored metadata. It does not download media and
does not inspect model scores. The output is deterministic for a fixed config
and fixed metadata files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "config" / "confirm_config.json"
OUTPUT_COLUMNS = [
    "sample_id",
    "set",
    "domain",
    "dataset",
    "video_file",
    "expected_zip_member",
    "label",
    "label_int",
    "identity",
    "speaker_id",
    "official_split",
    "script_type",
    "audio_provider",
    "video_engine",
    "recording_source",
    "recording_target",
    "lineage_group",
    "uncertainty_group",
    "annotation_source",
    "order_index",
]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalise_identity(value: str) -> str:
    value = str(value).strip()
    try:
        return str(int(float(value)))
    except ValueError as exc:
        raise ValueError(f"Invalid identity value: {value!r}") from exc


def verify_inputs(config: dict, metadata_dir: Path) -> dict[str, str]:
    actual: dict[str, str] = {}
    for name, expected in config["metadata_sha256"].items():
        path = metadata_dir / name
        if not path.is_file():
            raise SystemExit(f"Missing required metadata file: {path}")
        digest = sha256_file(path)
        actual[name] = digest
        if digest.lower() != expected.lower():
            raise SystemExit(
                f"Metadata hash mismatch for {name}:\n"
                f"  expected {expected}\n  actual   {digest}"
            )
    return actual


def build_rows(config: dict, metadata_dir: Path) -> list[dict[str, str | int]]:
    split_rows = read_csv(metadata_dir / "annotations-split-def.csv")
    split_by_identity = {
        normalise_identity(row["identity"]): row["split"].strip().lower()
        for row in split_rows
    }
    test_ids = {identity for identity, split in split_by_identity.items() if split == "test"}
    if len(test_ids) != config["target"]["expected_counts"]["test_identities"]:
        raise SystemExit(f"Expected 56 test identities, found {len(test_ids)}")

    selected: list[dict[str, str | int]] = []
    allowed_scripts = {"scripted-short", "unscripted"}
    allowed_fake_audio = {"elevenlabs", "playht", "speechify"}

    for row in read_csv(metadata_dir / "annotations-real.csv"):
        identity = normalise_identity(row["identity"])
        script_type = row["script-type"].strip().lower()
        if identity not in test_ids or script_type not in allowed_scripts:
            continue
        video_file = row["video-file"].strip()
        recording = Path(video_file).stem
        selected.append({
            "sample_id": f"deepspeak-v2:real:{video_file}",
            "set": "target",
            "domain": "deepspeak_v2",
            "dataset": "DeepSpeak v2.0",
            "video_file": video_file,
            "expected_zip_member": f"real/{video_file}",
            "label": "real",
            "label_int": 0,
            "identity": identity,
            "speaker_id": identity,
            "official_split": "test",
            "script_type": script_type,
            "audio_provider": "real",
            "video_engine": "real",
            "recording_source": recording,
            "recording_target": recording,
            "lineage_group": f"real:{identity}:{recording}",
            "uncertainty_group": identity,
            "annotation_source": "annotations-real.csv",
        })

    for row in read_csv(metadata_dir / "annotations-fake.csv"):
        identity_source = normalise_identity(row["identity-source"])
        identity_target = normalise_identity(row["identity-target"])
        script_type = row["script-type"].strip().lower()
        audio_provider = row["audio-config"].strip().lower()
        if (
            identity_source not in test_ids
            or identity_target not in test_ids
            or script_type not in allowed_scripts
            or audio_provider not in allowed_fake_audio
        ):
            continue
        if identity_source != identity_target:
            raise SystemExit(
                "Eligible fake row crosses identities unexpectedly: "
                f"{row['video-file']} ({identity_source} -> {identity_target})"
            )
        video_file = row["video-file"].strip()
        engine = row["engine"].strip().lower()
        recording_source = row["recording-source"].strip()
        recording_target = row["recording-target"].strip()
        selected.append({
            "sample_id": f"deepspeak-v2:fake:{video_file}",
            "set": "target",
            "domain": "deepspeak_v2",
            "dataset": "DeepSpeak v2.0",
            "video_file": video_file,
            "expected_zip_member": f"{engine}/{video_file}",
            "label": "fake",
            "label_int": 1,
            "identity": identity_target,
            "speaker_id": identity_target,
            "official_split": "test",
            "script_type": script_type,
            "audio_provider": audio_provider,
            "video_engine": engine,
            "recording_source": recording_source,
            "recording_target": recording_target,
            "lineage_group": (
                f"fake:{identity_target}:{audio_provider}:"
                f"{recording_source}:{recording_target}"
            ),
            "uncertainty_group": identity_target,
            "annotation_source": "annotations-fake.csv",
        })

    selected.sort(
        key=lambda row: (
            int(row["label_int"]),
            int(str(row["identity"])),
            str(row["video_file"]),
        )
    )
    for index, row in enumerate(selected):
        row["order_index"] = index
    return selected


def validate_rows(config: dict, rows: list[dict[str, str | int]]) -> dict:
    sample_ids = [str(row["sample_id"]) for row in rows]
    video_files = [str(row["video_file"]) for row in rows]
    members = [str(row["expected_zip_member"]).lower() for row in rows]
    for values, name in ((sample_ids, "sample_id"), (video_files, "video_file"), (members, "expected_zip_member")):
        duplicated = [item for item, count in Counter(values).items() if count > 1]
        if duplicated:
            raise SystemExit(f"Duplicate {name}: {duplicated[:3]}")

    labels = Counter(str(row["label"]) for row in rows)
    expected = config["target"]["expected_counts"]
    actual = {"real": labels["real"], "fake": labels["fake"], "total": len(rows)}
    if any(actual[key] != expected[key] for key in ("real", "fake", "total")):
        raise SystemExit(f"Unexpected manifest counts: {actual}; expected {expected}")

    identities = {str(row["identity"]) for row in rows}
    if len(identities) != expected["test_identities"]:
        raise SystemExit(f"Expected {expected['test_identities']} identities, found {len(identities)}")

    fake_rows = [row for row in rows if row["label"] == "fake"]
    return {
        "counts": actual,
        "identity_count": len(identities),
        "fake_audio_provider": dict(sorted(Counter(str(row["audio_provider"]) for row in fake_rows).items())),
        "fake_video_engine": dict(sorted(Counter(str(row["video_engine"]) for row in fake_rows).items())),
        "script_type": dict(sorted(Counter(str(row["script_type"]) for row in rows).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "manifests")
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    metadata_dir = Path(config["paths"]["metadata_dir"])
    metadata_hashes = verify_inputs(config, metadata_dir)
    rows = build_rows(config, metadata_dir)
    summary = validate_rows(config, rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "target_deepspeak_v2_test.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study_id": config["study_id"],
        "status": "frozen_before_deepspeak_media_download_or_scoring",
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "metadata_sha256": metadata_hashes,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        **summary,
        "restriction": "Local restricted-data derivative; do not publish this manifest.",
    }
    record_path = args.output_dir / "manifest_record.json"
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
