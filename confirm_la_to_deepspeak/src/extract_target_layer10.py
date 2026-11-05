"""Extract frozen XLS-R layer-10 mean-pooled features for DeepSpeak target audio."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT = Path(__file__).resolve().parents[1]
PILOT_SRC = PROJECT.parent / "pilot_la_to_itw" / "src"
sys.path.insert(0, str(PILOT_SRC))
from pilot_common import normalise, preprocess  # noqa: E402

CONFIG = PROJECT / "config" / "confirm_config.json"
MANIFEST = PROJECT / "outputs" / "manifests" / "target_deepspeak_v2_test.csv"
LEDGER = PROJECT / "outputs" / "acquisition" / "acquisition_ledger.csv"
VERIFY = PROJECT / "outputs" / "acquisition" / "FULL_ACQUISITION_VERIFICATION.json"
OUT_DIR = Path(os.environ.get(
    "DEEPSPEAK_LAYER10_DIR",
    "D:/7841_data/confirm_la_to_deepspeak/embeddings/target/layer10",
))
SHARD_SIZE = 250


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    verification = json.loads(VERIFY.read_text(encoding="utf-8"))
    if verification["status"] != "PASS" or verification["manifest_rows"] != 1800:
        raise SystemExit("Full acquisition verification has not passed")
    manifest = sorted(read_csv(MANIFEST), key=lambda r: int(r["order_index"]))
    ledger = {row["sample_id"]: row for row in read_csv(LEDGER)}
    if any(row["sample_id"] not in ledger for row in manifest):
        raise SystemExit("Acquisition ledger is incomplete")

    import torch
    from transformers import Wav2Vec2Model

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    fx = cfg["feature_extractor"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2Model.from_pretrained(fx["model_id"], revision=fx["resolved_revision"], local_files_only=True)
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved != fx["resolved_revision"]:
        raise SystemExit(f"XLS-R revision mismatch: {resolved} != {fx['resolved_revision']}")
    model.eval().to(device)
    use_amp = device.type == "cuda"
    layer = int(fx["fixed_hidden_state_layer"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    processed = 0
    n_shards = (len(manifest) + SHARD_SIZE - 1) // SHARD_SIZE
    for shard in range(n_shards):
        npy_path = OUT_DIR / f"shard_{shard:04d}.npy"
        csv_path = OUT_DIR / f"shard_{shard:04d}.csv"
        if npy_path.exists() and csv_path.exists():
            print(f"target/layer10 shard {shard + 1}/{n_shards}: already complete")
            continue
        rows = manifest[shard * SHARD_SIZE:(shard + 1) * SHARD_SIZE]
        vectors = []
        records = []
        for row in rows:
            acquired = ledger[row["sample_id"]]
            signal, sr = sf.read(acquired["audio_path"], dtype="float32", always_2d=True)
            signal = signal.mean(axis=1).astype(np.float32)
            signal, info = preprocess(signal, int(sr), cfg)
            signal = normalise(signal)
            with torch.inference_mode():
                x = torch.from_numpy(signal).unsqueeze(0).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    hidden = model(x, output_hidden_states=True).hidden_states[layer]
                pooled = hidden.float().mean(dim=1)[0].cpu().numpy().astype(np.float16)
            if not np.isfinite(pooled.astype(np.float32)).all():
                raise RuntimeError(f"Non-finite embedding for {row['sample_id']}")
            vectors.append(pooled)
            records.append({
                "row": len(records),
                "sample_id": row["sample_id"],
                "label": row["label"],
                "identity": row["identity"],
                "script_type": row["script_type"],
                "audio_provider": row["audio_provider"],
                "video_engine": row["video_engine"],
                "audio_sha256": acquired["audio_sha256"],
                **info,
            })
            processed += 1
            if processed % 50 == 0:
                rate = processed / max(time.time() - started, 1e-6)
                left = (len(manifest) - processed) / max(rate, 1e-6) / 60.0
                print(f"\rprocessed {processed}/{len(manifest)} at {rate:.1f} clips/s, about {left:.1f} min left", end="", flush=True)
        tmp_npy = npy_path.with_name(npy_path.stem + ".tmp.npy")
        tmp_csv = csv_path.with_suffix(".tmp")
        np.save(tmp_npy, np.stack(vectors))
        write_csv(tmp_csv, records)
        tmp_npy.replace(npy_path)
        tmp_csv.replace(csv_path)
        print(f"\ntarget/layer10 shard {shard + 1}/{n_shards}: complete ({len(rows)} clips)")

    matrices = []
    all_records = []
    for shard in range(n_shards):
        matrices.append(np.load(OUT_DIR / f"shard_{shard:04d}.npy"))
        all_records.extend(read_csv(OUT_DIR / f"shard_{shard:04d}.csv"))
    matrix = np.concatenate(matrices, axis=0)
    if matrix.shape != (1800, 1024):
        raise SystemExit(f"Unexpected target embedding shape {matrix.shape}")
    if [row["sample_id"] for row in all_records] != [row["sample_id"] for row in manifest]:
        raise SystemExit("Target embedding order differs from frozen manifest")
    record = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": fx["model_id"],
        "resolved_revision": resolved,
        "hidden_state_layer": layer,
        "pooling": fx["pooling"],
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "n_clips": len(all_records),
        "embedding_shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "minutes_this_run": round((time.time() - started) / 60.0, 3),
        "source": "FULL_ACQUISITION_VERIFICATION.json PASS",
    }
    (OUT_DIR / "extraction_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
