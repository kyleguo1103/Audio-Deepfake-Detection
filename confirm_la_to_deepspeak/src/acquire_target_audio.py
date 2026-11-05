"""Acquire only the frozen DeepSpeak target audio, one ZIP shard at a time.

Default behaviour is a local dry-run. Network transfer starts only with
``--execute``. Each archive is downloaded into its own temporary cache,
selected videos are decoded to mono 16 kHz FLAC one at a time, an audit record
is written, and the temporary archive cache is removed before the next shard.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "config" / "confirm_config.json"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_frozen_manifest(config: dict) -> tuple[list[dict[str, str]], dict]:
    manifest_path = PROJECT_DIR / config["target"]["manifest"]
    record_path = manifest_path.with_name("manifest_record.json")
    if not manifest_path.is_file() or not record_path.is_file():
        raise SystemExit("Frozen target manifest is missing; run build_target_manifest.py first.")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    actual = sha256_file(manifest_path)
    if actual != record["manifest_sha256"]:
        raise SystemExit(
            "Frozen target manifest changed after creation. Rebuild before any acquisition:\n"
            f"  expected {record['manifest_sha256']}\n  actual   {actual}"
        )
    config_hash = sha256_file(DEFAULT_CONFIG if Path(config.get("_config_path", DEFAULT_CONFIG)) == DEFAULT_CONFIG else Path(config["_config_path"]))
    if config_hash != record["config_sha256"]:
        raise SystemExit(
            "Configuration changed after the manifest was frozen. Treat this as a protocol amendment "
            "and rebuild the manifest before media acquisition."
        )
    rows = read_csv(manifest_path)
    if len(rows) != config["target"]["expected_counts"]["total"]:
        raise SystemExit(f"Expected 1800 frozen rows, found {len(rows)}")
    return rows, record


def find_ffmpeg() -> str:
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        found = shutil.which("ffmpeg")
        if found:
            return found
    raise SystemExit("ffmpeg was not found. Install imageio-ffmpeg or add ffmpeg to PATH.")


def safe_rmtree(path: Path, allowed_parent: Path) -> None:
    resolved = path.resolve()
    parent = allowed_parent.resolve()
    if resolved == parent or parent not in resolved.parents:
        raise RuntimeError(f"Refusing to remove path outside staging directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def download_http(
    repo_id: str,
    repo_type: str,
    filename: str,
    revision: str,
    destination: Path,
) -> Path:
    """Download one gated Hub file with an observable, resumable HTTP stream."""
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url  # type: ignore
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub before downloading media.") from exc

    hub_url = hf_hub_url(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
    )
    metadata = get_hf_file_metadata(hub_url, token=True)
    if metadata.size is None:
        raise RuntimeError(f"Remote size is unavailable for {filename}")
    expected_size = int(metadata.size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    if destination.exists():
        if destination.stat().st_size == expected_size:
            print(f"  using completed archive already in staging: {destination}", flush=True)
            return destination
        destination.unlink()

    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        partial.unlink()
        offset = 0
    if offset:
        print(f"  resuming at {offset / (1024 ** 3):.2f}/{expected_size / (1024 ** 3):.2f} GiB", flush=True)

    while True:
        pattern = re.compile(
            rf"^{re.escape(partial.name)}\.seg-(\d+)-(\d+)(?:\.tmp)?$"
        )
        candidate: Path | None = None
        for path in partial.parent.glob(f"{partial.name}.seg-{offset:012d}-*"):
            match = pattern.match(path.name)
            if match and int(match.group(1)) == offset and 0 < path.stat().st_size <= int(match.group(2)) - offset + 1:
                if candidate is None or (candidate.name.endswith(".tmp") and not path.name.endswith(".tmp")):
                    candidate = path
        if candidate is None:
            break
        recovered = candidate.stat().st_size
        with partial.open("ab") as output, candidate.open("rb") as source:
            shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
        candidate.unlink()
        offset += recovered
        print(f"  recovered contiguous range; prefix is now {offset / (1024 ** 3):.2f} GiB", flush=True)

    segment_size = 8 * 1024 * 1024
    segments: list[tuple[int, int, Path]] = []
    start = offset
    while start < expected_size:
        end = min(start + segment_size, expected_size) - 1
        segment = partial.with_name(f"{partial.name}.seg-{start:012d}-{end:012d}")
        expected_segment_size = end - start + 1
        if segment.exists() and segment.stat().st_size != expected_segment_size:
            segment.unlink()
        segments.append((start, end, segment))
        start = end + 1

    def fetch_segment(item: tuple[int, int, Path]) -> tuple[Path, int]:
        segment_start, segment_end, segment_path = item
        expected_segment_size = segment_end - segment_start + 1
        segment_location = metadata.location
        if segment_path.exists() and segment_path.stat().st_size == expected_segment_size:
            return segment_path, expected_segment_size
        temporary = segment_path.with_name(segment_path.name + ".tmp")
        if temporary.exists() and temporary.stat().st_size > expected_segment_size:
            temporary.unlink()
        for attempt in range(1, 6):
            downloaded = temporary.stat().st_size if temporary.exists() else 0
            if downloaded == expected_segment_size:
                break
            current_start = segment_start + downloaded
            request = urllib.request.Request(
                segment_location,
                headers={
                    "Range": f"bytes={current_start}-{segment_end}",
                    "User-Agent": "reit7841-deepspeak-acquisition/1.0",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        raise RuntimeError(
                            f"Server did not honour byte range {current_start}-{segment_end}: "
                            f"HTTP {response.status}"
                        )
                    with temporary.open("ab") as output:
                        while True:
                            block = response.read(8 * 1024 * 1024)
                            if not block:
                                break
                            output.write(block)
            except Exception:
                if attempt == 5:
                    raise
                refreshed = get_hf_file_metadata(hub_url, token=True)
                segment_location = refreshed.location
                continue
        actual_segment_size = temporary.stat().st_size
        if actual_segment_size != expected_segment_size:
            raise RuntimeError(
                f"Incomplete range {segment_start}-{segment_end}: "
                f"expected {expected_segment_size}, found {actual_segment_size}"
            )
        os.replace(temporary, segment_path)
        return segment_path, actual_segment_size

    if segments:
        print(f"  fetching {len(segments)} ranges with up to 4 parallel connections", flush=True)
        completed_bytes = offset
        report_step = 128 * 1024 * 1024
        next_report = completed_bytes + report_step
        with ThreadPoolExecutor(max_workers=min(4, len(segments))) as executor:
            futures = {executor.submit(fetch_segment, item): item for item in segments}
            for future in as_completed(futures):
                _, segment_bytes = future.result()
                completed_bytes += segment_bytes
                if completed_bytes >= next_report or completed_bytes == expected_size:
                    print(
                        f"  received {completed_bytes / (1024 ** 3):.2f}/{expected_size / (1024 ** 3):.2f} GiB",
                        flush=True,
                    )
                    while next_report <= completed_bytes:
                        next_report += report_step

    with partial.open("ab") as output:
        for _, _, segment in segments:
            with segment.open("rb") as source:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
            segment.unlink()

    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Incomplete download for {filename}: expected {expected_size} bytes, found {actual_size}"
        )
    os.replace(partial, destination)
    return destination


def normalise_member(name: str) -> str:
    return name.replace("\\", "/").strip("/").lower()


def manifest_match(member: str, desired: dict[str, dict[str, str]]) -> dict[str, str] | None:
    normalised = normalise_member(member)
    direct = desired.get(normalised)
    if direct is not None:
        return direct
    parts = normalised.split("/")
    return desired.get("/".join(parts[-2:])) if len(parts) >= 2 else None


def decode_to_flac(ffmpeg: str, source_video: Path, output_flac: Path) -> None:
    output_flac.parent.mkdir(parents=True, exist_ok=True)
    partial = output_flac.with_name(output_flac.stem + ".part" + output_flac.suffix)
    if partial.exists():
        partial.unlink()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "flac",
        str(partial),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {source_video.name}: {completed.stderr.strip()}")
    partial.replace(output_flac)


def audio_duration(path: Path) -> float | None:
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(str(path))
        return float(info.frames / info.samplerate)
    except Exception:
        return None


def resolve_revision(config: dict, output_dir: Path) -> str:
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub before resolving or downloading media.") from exc

    prefix = config["dataset"]["displayed_revision_prefix"]
    info = HfApi().dataset_info(config["dataset"]["repo_id"], revision=prefix)
    full_sha = str(info.sha)
    if not full_sha.startswith(prefix):
        raise SystemExit(f"Resolved revision {full_sha} does not start with expected prefix {prefix}")
    record = {
        "resolved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_id": config["dataset"]["repo_id"],
        "requested_revision_prefix": prefix,
        "resolved_full_revision": full_sha,
        "note": "Resolved before the first DeepSpeak media shard was processed.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "revision_record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return full_sha


def get_revision(config: dict, output_dir: Path, allow_network: bool) -> str:
    record_path = output_dir / "revision_record.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        revision = str(record["resolved_full_revision"])
        prefix = config["dataset"]["displayed_revision_prefix"]
        if revision.startswith(prefix):
            return revision
        raise SystemExit(f"Saved revision {revision} does not match expected prefix {prefix}")
    if not allow_network:
        return "NOT_YET_RESOLVED"
    return resolve_revision(config, output_dir)


def load_existing_records(record_dir: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not record_dir.exists():
        return records
    for path in sorted(record_dir.glob("deepspeak-v2-*.json")):
        shard = json.loads(path.read_text(encoding="utf-8"))
        for item in shard.get("selected_files", []):
            sample_id = item["sample_id"]
            if sample_id in records:
                raise SystemExit(f"Duplicate acquired sample across shard records: {sample_id}")
            records[sample_id] = item
    return records


def write_ledger(output_dir: Path, records: dict[str, dict], manifest_rows: list[dict[str, str]]) -> None:
    path = output_dir / "acquisition_ledger.csv"
    columns = [
        "sample_id", "label", "identity", "video_file", "archive", "zip_member",
        "archive_sha256", "video_sha256", "audio_path", "audio_sha256", "duration_seconds",
    ]
    order = {row["sample_id"]: int(row["order_index"]) for row in manifest_rows}
    rows = sorted(records.values(), key=lambda item: order[item["sample_id"]])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def process_archive(
    *,
    archive_path: Path,
    archive_name: str,
    archive_sha256: str,
    desired: dict[str, dict[str, str]],
    already_found: dict[str, dict],
    audio_root: Path,
    ffmpeg: str,
) -> list[dict]:
    selected: list[dict] = []
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            row = manifest_match(info.filename, desired)
            if row is None:
                continue
            sample_id = row["sample_id"]
            if sample_id in already_found or any(item["sample_id"] == sample_id for item in selected):
                raise RuntimeError(f"Frozen sample appears in more than one archive/member: {sample_id}")

            relative_audio = Path(row["label"]) / row["identity"] / (Path(row["video_file"]).stem + ".flac")
            output_audio = audio_root / relative_audio
            with tempfile.TemporaryDirectory(prefix="deepspeak_member_", dir=archive_path.parent) as temp_dir:
                temp_video = Path(temp_dir) / Path(row["video_file"]).name
                with archive.open(info, "r") as source, temp_video.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                video_hash = sha256_file(temp_video)
                decode_to_flac(ffmpeg, temp_video, output_audio)

            duration = audio_duration(output_audio)
            selected.append({
                "sample_id": sample_id,
                "label": row["label"],
                "identity": row["identity"],
                "video_file": row["video_file"],
                "archive": archive_name,
                "zip_member": info.filename,
                "archive_sha256": archive_sha256,
                "video_sha256": video_hash,
                "audio_path": str(output_audio.resolve()),
                "audio_sha256": sha256_file(output_audio),
                "duration_seconds": "" if duration is None else f"{duration:.6f}",
            })
            print(f"    extracted {len(selected):4d}: {row['video_file']}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true", help="Start authenticated shard downloads and extraction")
    parser.add_argument("--resolve-revision", action="store_true", help="Resolve and record the full repository SHA, then stop")
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int, default=24)
    parser.add_argument("--keep-shard-cache", action="store_true", help="Debug only; consumes substantial storage")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["_config_path"] = str(args.config.resolve())
    output_dir = PROJECT_DIR / "outputs" / "acquisition"
    rows, manifest_record = load_frozen_manifest(config)
    revision = get_revision(config, output_dir, allow_network=args.execute or args.resolve_revision)
    if args.resolve_revision and not args.execute:
        print(f"Resolved full revision: {revision}")
        return 0

    data_root = Path(config["paths"]["data_root"])
    staging_root = Path(config["paths"]["staging_dir"])
    audio_root = Path(config["paths"]["audio_dir"])
    free_gib = shutil.disk_usage(data_root).free / (1024 ** 3)

    print(f"Study: {config['study_id']}")
    print(f"Frozen manifest: {len(rows)} clips; SHA-256 {manifest_record['manifest_sha256']}")
    print(f"Revision: {revision}")
    print(f"Audio output: {audio_root}")
    print(f"Staging: {staging_root}")
    print(f"Free space on data volume: {free_gib:.1f} GiB")
    print("Storage policy: one archive at a time; remove its isolated cache after its records are saved.")

    if not args.execute:
        print("DRY RUN ONLY: no network access and no media download occurred. Add --execute to begin.")
        return 0
    if not (0 <= args.start_shard <= args.end_shard <= 24):
        raise SystemExit("Shard range must satisfy 0 <= start <= end <= 24")
    if free_gib < 14.0:
        raise SystemExit(f"At least 14 GiB free is required for safe one-shard staging; found {free_gib:.1f} GiB")

    staging_root.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)
    record_dir = output_dir / "shard_records"
    record_dir.mkdir(parents=True, exist_ok=True)
    existing = load_existing_records(record_dir)
    desired = {normalise_member(row["expected_zip_member"]): row for row in rows}
    ffmpeg = find_ffmpeg()

    archive_names = config["dataset"]["archives"]
    for shard_index in range(args.start_shard, args.end_shard + 1):
        archive_name = archive_names[shard_index]
        record_path = record_dir / f"deepspeak-v2-{shard_index}.json"
        if record_path.exists():
            print(f"[{shard_index:02d}/24] already complete: {archive_name}")
            continue
        shard_cache = staging_root / f"shard_{shard_index:02d}_cache"
        shard_cache.mkdir(parents=True, exist_ok=True)
        print(f"[{shard_index:02d}/24] downloading {archive_name}", flush=True)
        completed = False
        try:
            archive_path = download_http(
                repo_id=config["dataset"]["repo_id"],
                repo_type=config["dataset"]["repo_type"],
                filename=archive_name,
                revision=revision,
                destination=shard_cache / archive_name,
            )
            print(f"  hashing {archive_name}")
            archive_hash = sha256_file(archive_path)
            selected = process_archive(
                archive_path=archive_path,
                archive_name=archive_name,
                archive_sha256=archive_hash,
                desired=desired,
                already_found=existing,
                audio_root=audio_root,
                ffmpeg=ffmpeg,
            )
            payload = {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "repo_id": config["dataset"]["repo_id"],
                "revision": revision,
                "archive": archive_name,
                "archive_size_bytes": archive_path.stat().st_size,
                "archive_sha256": archive_hash,
                "selected_count": len(selected),
                "selected_files": selected,
            }
            record_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            existing.update({item["sample_id"]: item for item in selected})
            write_ledger(output_dir, existing, rows)
            completed = True
            print(f"  recorded {len(selected)} selected clips; cumulative {len(existing)}/{len(rows)}")
        finally:
            if not args.keep_shard_cache and completed:
                safe_rmtree(shard_cache, staging_root)
                print(f"  removed temporary cache for {archive_name}")
            elif not completed:
                print(f"  preserved partial cache for resumable retry: {shard_cache}")

    missing = [row["sample_id"] for row in rows if row["sample_id"] not in existing]
    if args.start_shard == 0 and args.end_shard == 24:
        if missing:
            raise SystemExit(f"Acquisition incomplete: {len(missing)} frozen samples were not found; first: {missing[0]}")
        print(f"COMPLETE: all {len(rows)} frozen target clips were acquired and audited.")
    else:
        print(f"Partial shard range complete; {len(missing)} frozen samples remain unresolved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
