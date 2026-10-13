"""Step 1: build and freeze the three manifests (source train pool, source validation, target).

Audio can come from the official zip archives or from an already-extracted copy
(e.g. --la-source D:\\7841\\LA). Files are resolved by their official relative
paths, so stray duplicate folders are ignored. The step samples the source clips
with the pre-registered seed, hashes every selected audio file and writes
outputs/manifests/*.csv plus manifest_record.json. Later steps refuse to run if
any manifest is edited, and step 2 reads audio from the locations frozen here.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

from pilot_common import (KNOWN_ARCHIVES, MANIFEST_COLUMNS, MANIFEST_FILES, SourceReader, known_archive_problem,
                          load_config, manifest_dir, now_utc,
                          sha256_bytes, sha256_file, stable_seed, write_csv, write_json)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_):
        return iterable

LA_PROTOCOL_DIR = "ASVspoof2019_LA_cm_protocols"


def choose(items: list[str], n: int, seed: int) -> list[str]:
    items = sorted(items)
    if n > len(items):
        raise SystemExit(f"cannot sample {n} from {len(items)} items")
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(items), size=n, replace=False)
    return sorted(items[i] for i in picked)


def locate(reader: SourceReader, relative: str, prefixes=("", "LA/", "release_in_the_wild/")) -> str:
    """Find an official relative path under the source root (allowing one known top folder)."""
    for prefix in prefixes:
        if reader.exists(prefix + relative):
            return prefix + relative
    hits = reader.find("/" + relative) + ([relative] if reader.exists(relative) else [])
    hits = sorted(set(hits))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one '{relative}' in {reader.path}, found {hits[:5]}")
    return hits[0]


def build_la(cfg, reader: SourceReader) -> tuple[dict[str, list[dict]], dict]:
    src = cfg["source"]
    seed = src["sampling_seed"]
    out, evidence = {}, {}
    for set_name, part_key in (("source_train", "train_pool"), ("source_val", "validation")):
        part = src[part_key]
        protocol = locate(reader, f"{LA_PROTOCOL_DIR}/{part['protocol_suffix']}")
        root_prefix = protocol[: -len(f"{LA_PROTOCOL_DIR}/{part['protocol_suffix']}")]
        protocol_bytes = reader.read(protocol)
        rows = []
        for line in protocol_bytes.decode("utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            speaker, utt, _, system, key = fields
            rows.append({"speaker": speaker, "utt": utt, "system": system, "key": key})
        evidence[set_name] = {"protocol": protocol, "protocol_sha256": sha256_bytes(protocol_bytes),
                              "protocol_rows": len(rows), "labels": dict(Counter(r["key"] for r in rows))}
        bona = [r["utt"] for r in rows if r["key"] == "bonafide"]
        by_attack: dict[str, list[str]] = {}
        for r in rows:
            if r["key"] == "spoof":
                by_attack.setdefault(r["system"], []).append(r["utt"])
        meta = {r["utt"]: r for r in rows}
        selected = choose(bona, part["n_bonafide"], stable_seed(seed, set_name, "bonafide"))
        attacks = part["attacks"]
        if "n_spoof_per_attack" in part:
            per_attack = {a: part["n_spoof_per_attack"] for a in attacks}
        else:
            base, extra = divmod(part["n_spoof_total"], len(attacks))
            per_attack = {a: base + (1 if i < extra else 0) for i, a in enumerate(attacks)}
        for attack in attacks:
            selected += choose(by_attack.get(attack, []), per_attack[attack], stable_seed(seed, set_name, attack))
        audio_dir = f"{root_prefix}ASVspoof2019_LA_{part['partition']}/flac"
        manifest = []
        for order, utt in enumerate(selected):
            r = meta[utt]
            member = f"{audio_dir}/{utt}.flac"
            if not reader.exists(member):
                raise SystemExit(f"audio for {utt} not found at {member}")
            raw_label = r["key"]
            manifest.append({
                "sample_id": f"la_{part['partition']}/{utt}", "set": set_name, "domain": "source",
                "dataset": "ASVspoof2019_LA", "zip_member": member,
                "label": src["label_map"][raw_label], "raw_label": raw_label,
                "speaker_id": r["speaker"], "attack_id": r["system"] if raw_label == "spoof" else "bonafide",
                "sha256": "", "order_index": order, "in_core_subset": 1,
            })
        if reader.is_dir:  # completeness evidence for an extracted copy
            folder = reader.path / Path(*audio_dir.split("/"))
            present = {p.stem for p in folder.glob("*.flac")}
            evidence[set_name].update({"audio_files_in_folder": len(present),
                                       "protocol_files_missing": sum(1 for r in rows if r["utt"] not in present)})
            if evidence[set_name]["protocol_files_missing"]:
                print(f"WARNING: {evidence[set_name]['protocol_files_missing']} protocol files missing in {folder} "
                      "(selected files are all present).")
        out[set_name] = manifest
    return out, evidence


def build_itw(cfg, reader: SourceReader) -> tuple[list[dict], dict]:
    tgt = cfg["target"]
    meta_member = locate(reader, "meta.csv", prefixes=("", "release_in_the_wild/"))
    audio_root = meta_member[: -len("meta.csv")]
    meta_bytes = reader.read(meta_member)
    csv_reader = csv.DictReader(io.StringIO(meta_bytes.decode("utf-8-sig")))
    cols = {c.lower().strip(): c for c in (csv_reader.fieldnames or [])}
    for needed in ("file", "speaker", "label"):
        if needed not in cols:
            raise SystemExit(f"meta.csv lacks column '{needed}': {csv_reader.fieldnames}")
    records = []
    for row in csv_reader:
        raw_label = row[cols["label"]].strip()
        if raw_label not in tgt["label_map"]:
            raise SystemExit(f"unexpected In-the-Wild label '{raw_label}' (expected {list(tgt['label_map'])})")
        name = PurePosixPath(row[cols["file"]].strip()).name
        member = audio_root + name
        records.append({"stem": PurePosixPath(name).stem, "member": member,
                        "speaker": row[cols["speaker"]].strip(), "raw_label": raw_label})
    if len({r["stem"] for r in records}) != len(records):
        raise SystemExit("duplicate file entries in meta.csv")
    missing = [r["member"] for r in records if not reader.exists(r["member"])]
    if missing:
        raise SystemExit(f"{len(missing)} files listed in meta.csv are missing (first: {missing[0]})")

    def sort_key(stem: str):
        return (0, int(stem), stem) if re.fullmatch(r"\d+", stem) else (1, 0, stem)

    core_cfg = tgt["core_subset"]
    core: set[str] = set()
    for raw_label in tgt["label_map"]:
        stems = [r["stem"] for r in records if r["raw_label"] == raw_label]
        core.update(choose(stems, min(core_cfg["n_per_label"], len(stems)), stable_seed(core_cfg["seed"], "itw_core", raw_label)))
    ordered = sorted((r for r in records if r["stem"] in core), key=lambda r: sort_key(r["stem"]))
    ordered += sorted((r for r in records if r["stem"] not in core), key=lambda r: sort_key(r["stem"]))
    manifest = []
    for order, r in enumerate(ordered):
        manifest.append({
            "sample_id": f"itw/{r['stem']}", "set": "target", "domain": "target", "dataset": "In-the-Wild",
            "zip_member": r["member"], "label": tgt["label_map"][r["raw_label"]], "raw_label": r["raw_label"],
            "speaker_id": r["speaker"], "attack_id": "", "sha256": "", "order_index": order,
            "in_core_subset": int(r["stem"] in core),
        })
    evidence = {"meta_csv": meta_member, "meta_sha256": sha256_bytes(meta_bytes), "rows": len(records),
                "labels": dict(Counter(r["raw_label"] for r in records))}
    return manifest, evidence


def hash_members(reader: SourceReader, manifest: list[dict], label: str) -> None:
    for row in tqdm(manifest, desc=f"hashing {label}", unit="file"):
        row["sha256"] = sha256_bytes(reader.read(row["zip_member"]))


def describe_source(path: Path, skip_hash: bool) -> dict:
    if path.is_dir():
        return {"path": str(path.resolve()), "type": "directory", "bytes": None, "sha256": None}
    info = {"path": str(path.resolve()), "type": "zip", "bytes": path.stat().st_size,
            "sha256": None if skip_hash else sha256_file(path, progress=True)}
    known = KNOWN_ARCHIVES.get(path.name)
    if known and os.environ.get("PILOT_SKIP_KNOWN_ARCHIVE_CHECK") != "1":
        info["official_sha256"] = known["sha256"]
        info["official_source"] = known["source"]
        info["official_sha256_match"] = None if info["sha256"] is None else info["sha256"] == known["sha256"]
        if info["official_sha256_match"] is False:
            raise SystemExit(f"{path.name}: SHA-256 differs from the official file - re-download it.")
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--data-root")
    parser.add_argument("--la-source", help="LA.zip or an extracted ASVspoof 2019 LA folder (default: config la_zip)")
    parser.add_argument("--itw-source", help="release_in_the_wild.zip or an extracted folder (default: config itw_zip)")
    parser.add_argument("--skip-zip-hash", action="store_true", help="skip SHA-256 of whole zip archives (saves a few minutes)")
    parser.add_argument("--force", action="store_true", help="overwrite frozen manifests (only before any embedding exists)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.data_root)
    out_dir = manifest_dir(cfg)
    record_path = out_dir / "manifest_record.json"
    if record_path.exists() and not args.force:
        raise SystemExit(f"Manifests are already frozen ({record_path}). Use --force only if no embedding has been extracted yet.")

    la_path = Path(args.la_source) if args.la_source else cfg["_paths"]["la_zip"]
    itw_path = Path(args.itw_source) if args.itw_source else cfg["_paths"]["itw_zip"]
    for p in (la_path, itw_path):
        if not p.exists():
            raise SystemExit(f"missing audio source: {p}")
        issue = known_archive_problem(p)
        if issue:
            raise SystemExit(issue)

    la_reader = SourceReader(la_path)
    print(f"Source audio : {la_path} ({la_reader.kind})")
    la, la_evidence = build_la(cfg, la_reader)
    for set_name in ("source_train", "source_val"):
        hash_members(la_reader, la[set_name], set_name)
    itw_reader = SourceReader(itw_path)
    print(f"Target audio : {itw_path} ({itw_reader.kind})")
    target, itw_evidence = build_itw(cfg, itw_reader)
    hash_members(itw_reader, target, "target")

    manifests = {"source_train": la["source_train"], "source_val": la["source_val"], "target": target}
    train_speakers = {r["speaker_id"] for r in manifests["source_train"]}
    val_speakers = {r["speaker_id"] for r in manifests["source_val"]}
    if train_speakers & val_speakers:
        raise SystemExit(f"speaker leakage between source train and validation: {sorted(train_speakers & val_speakers)}")
    all_hashes = Counter(r["sha256"] for rows in manifests.values() for r in rows)
    duplicates = {h: c for h, c in all_hashes.items() if c > 1}
    cross_domain_dupes = ({r["sha256"] for r in target} & {r["sha256"] for s in ("source_train", "source_val") for r in manifests[s]})

    summary = {}
    for set_name, rows in manifests.items():
        write_csv(out_dir / MANIFEST_FILES[set_name], rows, MANIFEST_COLUMNS)
        summary[set_name] = {
            "n": len(rows),
            "labels": dict(Counter(r["label"] for r in rows)),
            "speakers": len({r["speaker_id"] for r in rows}),
            "attacks": dict(Counter(r["attack_id"] for r in rows)) if set_name != "target" else None,
            "core_subset_labels": dict(Counter(r["label"] for r in rows if r["in_core_subset"] == 1)) if set_name == "target" else None,
        }

    record = {
        "created_at_utc": now_utc(),
        "study_id": cfg["study_id"],
        "config_sha256": cfg["_config_sha256"],
        "archives": {"la": describe_source(la_path, args.skip_zip_hash), "itw": describe_source(itw_path, args.skip_zip_hash)},
        "source_evidence": {"la": la_evidence, "itw": itw_evidence},
        "summary": summary,
        "speaker_disjoint_source_train_vs_val": True,
        "duplicate_audio_hashes_across_all_sets": len(duplicates),
        "identical_audio_shared_by_source_and_target": len(cross_domain_dupes),
        "manifest_sha256": {s: sha256_file(out_dir / f) for s, f in MANIFEST_FILES.items()},
        "note": "Frozen before any embedding or result exists. Later steps verify these hashes and read audio from the paths above.",
    }
    write_json(record_path, record)
    for set_name, info in summary.items():
        print(f"{set_name:13s} n={info['n']:6d} labels={info['labels']} speakers={info['speakers']}")
    if duplicates:
        print(f"NOTE: {len(duplicates)} audio hashes occur more than once across sets (recorded).")
    if cross_domain_dupes:
        print(f"WARNING: {len(cross_domain_dupes)} identical audio files appear in both source and target.")
    print(f"Frozen manifests written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
