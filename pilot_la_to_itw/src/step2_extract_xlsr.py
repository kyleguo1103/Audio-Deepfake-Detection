"""Step 2: extract mean-pooled XLS-R hidden states for every manifest clip.

Jobs (set / variant):
  source_train / clean, codec, noise, channel   (augmented copies of the SAME clips)
  source_val   / clean
  target       / clean   (In-the-Wild, pre-declared core subset first, then the rest)

Output: <embedding_dir>/<set>/<variant>/shard_XXXX.npy  float16 [n, 25, 1024]
        + shard_XXXX.csv with per-clip preprocessing / augmentation records.
Resumable: finished shards are skipped, so the script can be stopped and rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from augment import augment
from pilot_common import (SourceReader, decode_audio, ffmpeg_encoders, ffmpeg_exe, load_config, load_manifest, normalise,
                          now_utc, preprocess, source_paths_from_record, verify_manifest_hashes, write_json)

JOBS = [
    ("source_val", "clean"),
    ("source_train", "clean"),
    ("source_train", "codec"),
    ("source_train", "noise"),
    ("source_train", "channel"),
    ("target", "clean"),
]
SHARD_COLUMNS = ["row", "sample_id", "label", "orig_sr", "raw_seconds", "lead_silence_s", "trail_silence_s",
                 "trim_fallback", "used_seconds", "aug_params"]


class ClipDataset:
    """Map-style dataset usable by torch DataLoader (picklable for Windows workers)."""

    def __init__(self, rows, source_path, variant, cfg, ffmpeg, encoders):
        self.rows = rows
        self.reader = SourceReader(source_path)
        self.variant = variant
        self.cfg = cfg
        self.ffmpeg = ffmpeg
        self.encoders = encoders

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        signal, sr = decode_audio(self.reader.read(row["zip_member"]), self.ffmpeg)
        signal, info = preprocess(signal, sr, self.cfg)
        params = {}
        if self.variant != "clean":
            signal, params = augment(signal, self.variant, row["sample_id"], self.cfg, self.ffmpeg, self.encoders)
        info["aug_params"] = json.dumps(params, sort_keys=True)
        return row["sample_id"], row["label"], normalise(signal), info


def first_item(batch):
    return batch[0]


class XlsrEncoder:
    def __init__(self, cfg, device_name: str):
        import torch
        from transformers import Wav2Vec2Model

        self.torch = torch
        fx = cfg["feature_extractor"]
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        kwargs = {"revision": fx["revision"]} if fx.get("revision") else {}
        self.model = Wav2Vec2Model.from_pretrained(fx["model_id"], **kwargs)
        self.model.eval().to(self.device)
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.use_amp = self.device.type == "cuda"
        self.description = {
            "model_id": fx["model_id"], "resolved_commit": self.revision, "device": str(self.device),
            "gpu": torch.cuda.get_device_name(0) if self.device.type == "cuda" else None,
            "amp_fp16": self.use_amp, "torch": torch.__version__,
        }
        try:
            import transformers

            self.description["transformers"] = transformers.__version__
        except Exception:  # noqa: BLE001
            pass

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(wav).unsqueeze(0).to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                out = self.model(x, output_hidden_states=True)
            pooled = torch.stack([h.float().mean(dim=1)[0] for h in out.hidden_states])  # [25, 1024]
        return pooled.cpu().numpy().astype(np.float16)


class FakeEncoder:
    """Cheap spectral stand-in used ONLY by the software self-test (no torch needed)."""

    description = {"model_id": "FAKE_SPECTRAL_ENCODER_FOR_SELF_TEST", "resolved_commit": None, "device": "cpu"}
    revision = None

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        frame, hop = 400, 160
        if len(wav) < frame:
            wav = np.pad(wav, (0, frame - len(wav)))
        n = 1 + (len(wav) - frame) // hop
        frames = np.stack([wav[i * hop: i * hop + frame] for i in range(n)]) * np.hanning(frame)
        power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        bands = np.array_split(np.arange(power.shape[1]), 32)
        logband = np.log(np.stack([power[:, b].mean(axis=1) for b in bands], axis=1) + 1e-8)
        layers = [logband.mean(0), logband.std(0), np.diff(logband, axis=0).std(0) if n > 1 else np.zeros(32),
                  np.percentile(logband, 90, axis=0), np.percentile(logband, 10, axis=0)]
        return np.stack(layers).astype(np.float16)


def iter_items(dataset, workers: int):
    if workers > 0:
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers, collate_fn=first_item,
                            persistent_workers=False, prefetch_factor=4)
        yield from loader
    else:
        for i in range(len(dataset)):
            yield dataset[i]


def write_shard(npy_path: Path, csv_path: Path, vectors: list, records: list) -> None:
    matrix = np.stack(vectors)
    if not np.isfinite(matrix.astype(np.float32)).all():
        raise RuntimeError(f"non-finite embedding values in {npy_path}")
    tmp_npy = npy_path.with_name(npy_path.stem + ".tmp.npy")
    np.save(tmp_npy, matrix)
    tmp_csv = csv_path.with_suffix(".tmp")
    with tmp_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHARD_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    tmp_npy.replace(npy_path)
    tmp_csv.replace(csv_path)


def run_job(cfg, set_name, variant, encoder, args, ffmpeg, encoders, embedding_dir: Path) -> dict:
    rows = sorted(load_manifest(cfg, set_name), key=lambda r: int(r["order_index"]))
    if args.limit:
        rows = rows[: args.limit]
    sources = source_paths_from_record(cfg)
    source_path = sources["source"] if set_name.startswith("source") else sources["target"]
    out_dir = embedding_dir / set_name / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_size = int(cfg["feature_extractor"]["shard_size"])
    n_shards = (len(rows) + shard_size - 1) // shard_size
    pending = [s for s in range(n_shards)
               if not ((out_dir / f"shard_{s:04d}.npy").exists() and (out_dir / f"shard_{s:04d}.csv").exists())]
    start = time.time()
    done_clips = 0
    if pending:
        todo_rows = [r for s in pending for r in rows[s * shard_size: (s + 1) * shard_size]]
        dataset = ClipDataset(todo_rows, source_path, variant, cfg, ffmpeg, encoders)
        shard_iter = iter(pending)
        current = next(shard_iter)
        current_rows = rows[current * shard_size: (current + 1) * shard_size]
        vectors, records = [], []
        for sample_id, label, wav, info in iter_items(dataset, args.workers):
            k = len(vectors)
            if sample_id != current_rows[k]["sample_id"]:
                raise RuntimeError("loader order mismatch")
            vectors.append(encoder(wav))
            records.append({"row": k, "sample_id": sample_id, "label": label, **{c: info[c] for c in SHARD_COLUMNS[3:]}})
            done_clips += 1
            if done_clips % 50 == 0:
                rate = done_clips / max(time.time() - start, 1e-6)
                left = (len(todo_rows) - done_clips) / max(rate, 1e-6) / 60
                print(f"\r  {set_name}/{variant}: {done_clips}/{len(todo_rows)}  {rate:5.1f} clips/s  ~{left:5.1f} min left", end="", flush=True)
            if len(vectors) == len(current_rows):
                write_shard(out_dir / f"shard_{current:04d}.npy", out_dir / f"shard_{current:04d}.csv", vectors, records)
                vectors, records = [], []
                current = next(shard_iter, None)
                if current is not None:
                    current_rows = rows[current * shard_size: (current + 1) * shard_size]
    elapsed = time.time() - start
    complete = all((out_dir / f"shard_{s:04d}.npy").exists() for s in range(n_shards))
    print(f"\r  {set_name}/{variant}: {'complete' if complete else 'INCOMPLETE'} ({len(rows)} clips, {elapsed / 60:.1f} min this run)                    ")
    return {"set": set_name, "variant": variant, "n_clips": len(rows), "n_shards": n_shards, "complete": complete,
            "minutes_this_run": round(elapsed / 60, 2), "clips_processed_this_run": done_clips}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--data-root")
    parser.add_argument("--only", default="", help="comma list like source_val/clean,target/clean (default: all jobs)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=2, help="audio loading processes (0 = main process only)")
    parser.add_argument("--limit", type=int, default=0, help="debug: only first N clips per job")
    parser.add_argument("--fake-model", action="store_true", help="software self-test only; NOT a research run")
    args = parser.parse_args()
    cfg = load_config(args.config, args.data_root)
    manifest_hashes = verify_manifest_hashes(cfg)
    embedding_dir = cfg["_paths"]["embedding_dir"]
    if args.limit:
        embedding_dir = embedding_dir.with_name(embedding_dir.name + "_debug_limit")
        print(f"DEBUG MODE (--limit {args.limit}): writing to {embedding_dir}, never used by step 3.")

    ffmpeg = ffmpeg_exe()
    encoders = ffmpeg_encoders(ffmpeg)
    encoder = FakeEncoder() if args.fake_model else XlsrEncoder(cfg, args.device)
    print(f"Encoder: {encoder.description}")
    jobs = JOBS
    if args.only:
        wanted = {tuple(item.split("/")) for item in args.only.split(",")}
        jobs = [j for j in JOBS if j in wanted]
    record_path = embedding_dir / "extraction_record.json"
    history = []
    if record_path.exists():
        history = json.loads(record_path.read_text(encoding="utf-8")).get("runs", [])
    entry = {
        "started_utc": now_utc(), "finished_utc": None, "encoder": encoder.description,
        "fake_model": args.fake_model, "limit": args.limit, "jobs": [],
        "manifest_sha256": manifest_hashes, "config_sha256": cfg["_config_sha256"],
        "codec_encoders_available": sorted(e for e in encoders if e in {"libmp3lame", "aac", "libopus"}),
        "python": sys.version.split()[0],
    }
    history.append(entry)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    write_json(record_path, {"runs": history})  # written up-front so an interrupted run is still on record
    results = entry["jobs"]
    for set_name, variant in jobs:
        results.append(run_job(cfg, set_name, variant, encoder, args, ffmpeg, encoders, embedding_dir))
        write_json(record_path, {"runs": history})
    entry["finished_utc"] = now_utc()
    write_json(record_path, {"runs": history})
    if not all(r["complete"] for r in results):
        print("Some jobs are incomplete - rerun the same command to resume.")
        return 1
    print("All requested extraction jobs complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
