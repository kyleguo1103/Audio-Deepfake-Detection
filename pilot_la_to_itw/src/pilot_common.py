"""Shared helpers for the ASVspoof 2019 LA -> In-the-Wild augmentation pilot.

Everything that touches labels, paths, audio decoding and preprocessing lives
here so that every step applies exactly the same rules.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "config" / "pilot_config.json"
LABEL_TO_INT = {"real": 0, "fake": 1}


# --------------------------------------------------------------------------- config

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024, progress: bool = False) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
            done += len(block)
            if progress and total:
                print(f"\r  hashing {path.name}: {100.0 * done / total:5.1f}%", end="", flush=True)
    if progress:
        print()
    return digest.hexdigest()


def load_config(path: str | os.PathLike | None = None, data_root: str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    raw = config_path.read_bytes()
    cfg = json.loads(raw.decode("utf-8"))
    cfg["_config_path"] = str(config_path.resolve())
    cfg["_config_sha256"] = sha256_bytes(raw)
    root = data_root or os.environ.get("PILOT_DATA_ROOT") or cfg["paths"]["data_root"]
    resolved = {}
    for key, value in cfg["paths"].items():
        if key == "data_root":
            resolved[key] = Path(root)
        else:
            resolved[key] = Path(value.replace("{data_root}", str(root)).replace("{project_dir}", str(PROJECT_DIR)))
    cfg["_paths"] = resolved
    return cfg


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False, default=str)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


MANIFEST_COLUMNS = [
    "sample_id", "set", "domain", "dataset", "zip_member", "label", "raw_label",
    "speaker_id", "attack_id", "sha256", "order_index", "in_core_subset",
]
MANIFEST_FILES = {
    "source_train": "source_train_pool.csv",
    "source_val": "source_validation.csv",
    "target": "target_itw.csv",
}


def manifest_dir(cfg: dict[str, Any]) -> Path:
    return cfg["_paths"]["output_dir"] / "manifests"


def load_manifest(cfg: dict[str, Any], set_name: str) -> list[dict[str, str]]:
    return read_csv(manifest_dir(cfg) / MANIFEST_FILES[set_name])


def verify_manifest_hashes(cfg: dict[str, Any]) -> dict[str, str]:
    """Refuse to continue if a frozen manifest was edited after step 1."""
    record_path = manifest_dir(cfg) / "manifest_record.json"
    if not record_path.exists():
        raise SystemExit("manifest_record.json not found - run step1_build_manifests.py first")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    current = {}
    for set_name, filename in MANIFEST_FILES.items():
        digest = sha256_file(manifest_dir(cfg) / filename)
        expected = record["manifest_sha256"][set_name]
        if digest != expected:
            raise SystemExit(f"Frozen manifest {filename} changed after freezing ({digest} != {expected}).")
        current[set_name] = digest
    return current


# --------------------------------------------------------------------------- audio

def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        found = shutil.which("ffmpeg")
        if found:
            return found
    raise RuntimeError("ffmpeg not found: pip install imageio-ffmpeg")


def ffmpeg_encoders(exe: str) -> set[str]:
    out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False).stdout
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "AVS.":
            names.add(parts[1])
    return names


def decode_audio(data: bytes, ffmpeg: str | None = None) -> tuple[np.ndarray, int]:
    """Decode bytes (wav/flac) to mono float32. soundfile first, ffmpeg fallback."""
    try:
        import soundfile as sf  # type: ignore

        signal, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return signal.mean(axis=1).astype(np.float32), int(sr)
    except Exception:
        exe = ffmpeg or ffmpeg_exe()
        proc = subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
            input=data, capture_output=True, check=True,
        )
        return np.frombuffer(proc.stdout, dtype=np.float32).copy(), 16000


def resample(signal: np.ndarray, sr: int, target: int = 16000) -> np.ndarray:
    if sr == target:
        return signal.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    divisor = gcd(sr, target)
    return resample_poly(signal, target // divisor, sr // divisor).astype(np.float32)


def trim_silence(signal: np.ndarray, sr: int, top_db: float, frame: int, hop: int) -> tuple[int, int]:
    """Return [start, end) sample indices of the non-silent region (energy relative to clip max)."""
    if len(signal) < frame:
        return 0, len(signal)
    count = 1 + (len(signal) - frame) // hop
    starts = hop * np.arange(count)
    # cumulative-sum RMS per frame (memory-light)
    squared = np.concatenate([[0.0], np.cumsum(signal.astype(np.float64) ** 2)])
    energy = (squared[starts + frame] - squared[starts]) / frame
    rms = np.sqrt(energy + 1e-12)
    db = 20.0 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)
    voiced = np.flatnonzero(db > -top_db)
    if voiced.size == 0:
        return 0, len(signal)
    return int(starts[voiced[0]]), int(min(len(signal), starts[voiced[-1]] + frame))


def preprocess(signal: np.ndarray, sr: int, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    pp = cfg["preprocessing"]
    target_sr = int(pp["sample_rate_hz"])
    signal = resample(signal, sr, target_sr)
    info: dict[str, Any] = {"orig_sr": sr, "raw_seconds": round(len(signal) / target_sr, 4)}
    trim_cfg = pp["trim_silence"]
    start, end = 0, len(signal)
    fallback = False
    if trim_cfg["enabled"]:
        start, end = trim_silence(signal, target_sr, trim_cfg["top_db"], trim_cfg["frame_length"], trim_cfg["hop_length"])
        if (end - start) / target_sr < trim_cfg["min_kept_seconds"]:
            start, end, fallback = 0, len(signal), True
    info["lead_silence_s"] = round(start / target_sr, 4)
    info["trail_silence_s"] = round((len(signal) - end) / target_sr, 4)
    info["trim_fallback"] = int(fallback)
    signal = signal[start:end]
    max_len = int(round(pp["max_seconds"] * target_sr))
    signal = signal[:max_len]
    info["used_seconds"] = round(len(signal) / target_sr, 4)
    return signal.astype(np.float32), info


def normalise(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float32)
    return (signal - signal.mean()) / np.sqrt(signal.var() + 1e-7)


class SourceReader:
    """Reads audio either from a .zip archive or from an extracted folder.

    Members are POSIX-style relative paths. The zip handle is opened lazily so the
    object is safe to pass to DataLoader worker processes on Windows.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.is_dir = self.path.is_dir()
        self._zip: zipfile.ZipFile | None = None
        self._names: set[str] | None = None

    @property
    def kind(self) -> str:
        return "directory" if self.is_dir else "zip"

    def _zipfile(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.path)
        return self._zip

    def read(self, member: str) -> bytes:
        if self.is_dir:
            return (self.path / Path(*member.split("/"))).read_bytes()
        return self._zipfile().read(member)

    def exists(self, member: str) -> bool:
        if self.is_dir:
            return (self.path / Path(*member.split("/"))).is_file()
        if self._names is None:
            self._names = set(self._zipfile().namelist())
        return member in self._names

    def find(self, suffix: str) -> list[str]:
        """Members whose path ends with `suffix` (zip: all members; folder: recursive search)."""
        if self.is_dir:
            name = suffix.split("/")[-1]
            hits = [p.relative_to(self.path).as_posix() for p in self.path.rglob(name) if p.is_file()]
            return sorted(h for h in hits if h.endswith(suffix) and not h.split("/")[-1].startswith("._"))
        return sorted(n for n in self._zipfile().namelist()
                      if n.endswith(suffix) and not n.split("/")[-1].startswith("._"))

    def __getstate__(self):
        return {"path": self.path, "is_dir": self.is_dir, "_zip": None, "_names": None}

    def __setstate__(self, state):
        self.__dict__.update(state)


# Official distributor values (Hugging Face LFS record for mueller91/In-The-Wild, checked 2026-10-10).
KNOWN_ARCHIVES = {
    "release_in_the_wild.zip": {
        "bytes": 8161489023,
        "sha256": "46665f30a6758b45642c659c7e0e80da2ba3d121b064716824a56a010ddf9e1a",
        "source": "huggingface.co/datasets/mueller91/In-The-Wild (LFS oid)",
    },
}


def known_archive_problem(path: Path) -> str | None:
    """Return a message if a known archive has the wrong size (e.g. an unfinished download)."""
    if os.environ.get("PILOT_SKIP_KNOWN_ARCHIVE_CHECK") == "1" or path.is_dir() or not path.exists():
        return None
    known = KNOWN_ARCHIVES.get(path.name)
    if known and path.stat().st_size != known["bytes"]:
        have, want = path.stat().st_size, known["bytes"]
        return (f"{path.name} is {have:,} bytes but the official file is {want:,} bytes "
                f"({100 * have / want:.1f}%) - the download is incomplete or the file is different")
    return None


def source_paths_from_record(cfg: dict[str, Any]) -> dict[str, Path]:
    """Audio source locations frozen by step 1 (zip or extracted folder)."""
    record = json.loads((manifest_dir(cfg) / "manifest_record.json").read_text(encoding="utf-8"))
    return {"source": Path(record["archives"]["la"]["path"]), "target": Path(record["archives"]["itw"]["path"])}


# --------------------------------------------------------------------------- metrics

def eer_and_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """EER with fake=1 as the positive class; higher score = more likely fake."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    order = np.argsort(-score, kind="mergesort")
    s_sorted = score[order]
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    # keep last index of each distinct threshold
    distinct = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s_sorted) - 1]
    tpr = np.r_[0.0, tp[distinct] / n_pos]
    fpr = np.r_[0.0, fp[distinct] / n_neg]
    thresholds = np.r_[np.inf, s_sorted[distinct]]
    fnr = 1.0 - tpr
    diff = fnr - fpr  # decreasing from 1 to -1
    idx = int(np.flatnonzero(diff <= 0)[0])
    if idx == 0:
        return 0.0, float(thresholds[1])
    d0, d1 = diff[idx - 1], diff[idx]
    w = d0 / (d0 - d1) if d0 != d1 else 0.0
    eer = (fpr[idx - 1] + w * (fpr[idx] - fpr[idx - 1]) + fnr[idx - 1] + w * (fnr[idx] - fnr[idx - 1])) / 2.0
    threshold = thresholds[idx] if np.isfinite(thresholds[idx]) else thresholds[idx + 1]
    return float(eer), float(threshold)


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    from scipy.stats import rankdata

    y = np.asarray(y).astype(int)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def full_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    """Metrics from a real-valued fake score (classifier logit). Ranking metrics use the score directly,
    so saturated probabilities never create artificial ties; log-loss uses sigmoid(score)."""
    from scipy.special import expit
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                                 f1_score, log_loss, precision_score, recall_score)

    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)
    pred = (score >= threshold).astype(int)
    eer, _ = eer_and_threshold(y, score)
    prob = np.clip(expit(score), 1e-7, 1 - 1e-7)
    return {
        "n": int(len(y)),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "eer": eer,
        "roc_auc": roc_auc(y, score),
        "pr_auc_fake": float(average_precision_score(y, score)),
        "log_loss": float(log_loss(y, prob, labels=[0, 1])),
        "threshold_logit": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_fake": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "f1_real": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "precision_fake": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "recall_real": float(recall_score(y, pred, pos_label=0, zero_division=0)),
    }


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
