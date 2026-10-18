"""Detection engine for the XLS-R deepfake detector dashboard.

The engine reuses the pilot code in ../pilot_la_to_itw/src (or the thesis-workspace equivalent), so audio decoding, silence
trimming, the 6-second crop, normalisation, the XLS-R-53 encoder and the logistic-regression probes are
exactly the ones behind run_20261011_113206:

  * probes are refit on the cached source embeddings (ASVspoof 2019 LA, seed-42 training subset, the
    (layer, C) chosen on source validation) and use the source-validation EER point as threshold;
  * nothing about In-the-Wild is used to build or tune the probes; In-the-Wild clips are only scored.

Run `python detector.py --check` for a console self-check without the web interface.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import pickle
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

APP_DIR = Path(__file__).resolve().parent


def _default_pilot_dir() -> Path:
    """Find the pilot in either the public-repository or thesis-workspace layout."""
    candidates = [
        APP_DIR.parent / "pilot_la_to_itw",
        APP_DIR.parents[1] / "06_Implementation" / "pilot_la_to_itw",
    ]
    for candidate in candidates:
        if (candidate / "src" / "pilot_common.py").exists():
            return candidate
    return candidates[0]


DEFAULT_PILOT_DIR = _default_pilot_dir()

CONDITIONS = ["none", "codec", "noise", "channel", "mix", "all"]
CONDITION_NAMES = {
    "none": "No augmentation",
    "codec": "Codec",
    "noise": "Noise",
    "channel": "Channel",
    "mix": "Mix (pre-registered)",
    "all": "All 3 families",
}
PROFILE_CONDITIONS = ("none", "noise", "mix")
DEMO_SEED_NOTE = "seed-42 training subset (the pilot's selection seed)"
CACHE_VERSION = 1


def log(msg: str) -> None:
    print(f"[dashboard] {msg}", flush=True)


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


# ----------------------------------------------------------------------------- data classes

@dataclass
class Probe:
    condition: str
    layer: int
    C: float
    model: Any
    threshold: float
    val_eer: float
    n_train: int

    def logit(self, embedding: np.ndarray) -> float:
        """embedding: [n_layers, dim] (float16/32). Returns the classifier logit (higher = more fake)."""
        x = embedding[self.layer].astype(np.float32)[None, :]
        return float(self.model.decision_function(x)[0])


@dataclass
class ModelScore:
    condition: str
    name: str
    layer: int
    C: float
    logit: float
    threshold: float
    margin: float
    fake: bool
    score: float  # sigmoid(margin): 0.5 = decision threshold, not a calibrated probability
    pilot_target_eer: float | None
    pilot_bal_acc: float | None
    recorded_logit: float | None = None


@dataclass
class Analysis:
    source: str
    duration_s: float
    info: dict[str, Any]
    waveform: np.ndarray  # 16 kHz mono, before trimming
    used_start: int
    used_end: int
    scores: dict[str, ModelScore]
    layer_margins: dict[str, dict[int, float]]
    seconds_encoder: float
    ground_truth: dict[str, Any] | None = None


# ----------------------------------------------------------------------------- engine

class Engine:
    def __init__(self, pilot_dir: str | os.PathLike | None = None, data_root: str | None = None,
                 device: str = "auto", encoder: Any = None, retrain: bool = False,
                 progress: Callable[[str], None] = log):
        self.progress = progress
        self.pilot_dir = Path(pilot_dir or os.environ.get("PILOT_DIR") or DEFAULT_PILOT_DIR).resolve()
        src = self.pilot_dir / "src"
        if not (src / "pilot_common.py").exists():
            raise SystemExit(f"Pilot code not found at {src}. Pass --pilot-dir pointing to pilot_la_to_itw.")
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        import pilot_common as pc  # noqa: E402  (pilot module, imported from its own folder)

        self.pc = pc
        self.cfg = pc.load_config(self.pilot_dir / "config" / "pilot_config.json", data_root)
        self.data_root = Path(self.cfg["_paths"]["data_root"])
        if (self.data_root / "hf_cache").is_dir():
            os.environ.setdefault("HF_HOME", str(self.data_root / "hf_cache"))  # where step 2 cached XLS-R
        self.out_dir = Path(self.cfg["_paths"]["output_dir"])
        self.run_id = (self.out_dir / "LATEST_RUN.txt").read_text(encoding="utf-8").strip()
        self.run_dir = self.out_dir / self.run_id
        self.selected = json.loads((self.run_dir / "selected_configuration.json").read_text(encoding="utf-8"))
        self.summary = json.loads((self.run_dir / "summary.json").read_text(encoding="utf-8"))
        self.run_record = json.loads((self.run_dir / "run_record.json").read_text(encoding="utf-8"))
        self.layers = list(self.cfg["selection"]["candidate_layers"])
        self.cond_summary = {c["condition"]: c for c in self.summary["conditions"]}
        self.per_layer = self._read_per_layer()
        self.manifests_ok = self._verify_manifests()

        self.probes: dict[str, Probe] = {}
        self.layer_probes: dict[tuple[str, int], Probe] = {}
        self.probe_source = ""
        self._load_or_train_probes(retrain)

        self.itw = ItwSampler(self)
        self.encoder = encoder if encoder is not None else self._load_encoder(device)
        self.encoder_desc = getattr(self.encoder, "description", {"model_id": type(self.encoder).__name__})
        self._itw_files: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ setup helpers

    def _read_per_layer(self) -> dict[tuple[str, int], dict[str, float]]:
        path = self.run_dir / "exploratory_fixed_layer" / "per_layer_summary.csv"
        out: dict[tuple[str, int], dict[str, float]] = {}
        if not path.exists():
            return out
        for row in self.pc.read_csv(path):
            out[(row["condition"], int(row["layer"]))] = {
                "C": float(row["C"]),
                "target_eer": float(row["target_eer_mean"]),
                "delta": float(row["delta_vs_none_same_layer"]),
                "val_eer": float(row["val_eer_mean"]),
            }
        return out

    def _verify_manifests(self) -> bool:
        try:
            self.pc.verify_manifest_hashes(self.cfg)
            return True
        except SystemExit as exc:
            self.progress(f"WARNING: {exc}")
            return False

    def _cache_path(self) -> Path:
        return self.data_root / "demo_cache" / f"probes_{self.run_id}_{self.cfg['_config_sha256'][:8]}.pkl"

    def _load_or_train_probes(self, retrain: bool) -> None:
        import sklearn

        cache = self._cache_path()
        if cache.exists() and not retrain:
            try:
                with cache.open("rb") as handle:
                    payload = pickle.load(handle)
                if (payload.get("cache_version") == CACHE_VERSION and payload.get("sklearn") == sklearn.__version__
                        and payload.get("config_sha256") == self.cfg["_config_sha256"]
                        and payload.get("selected") == self.selected):
                    self.probes = {c: Probe(**d) for c, d in payload["probes"].items()}
                    self.layer_probes = {(d["condition"], d["layer"]): Probe(**d) for d in payload["layer_probes"]}
                    self.probe_source = f"loaded from cache ({cache.name})"
                    self.progress(f"Probes {self.probe_source}")
                    return
                self.progress("Probe cache is out of date (different scikit-learn or config) - retraining.")
            except Exception as exc:  # noqa: BLE001
                self.progress(f"Could not read probe cache ({exc}) - retraining.")
        self._train_probes()
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as handle:
            pickle.dump({"cache_version": CACHE_VERSION, "sklearn": sklearn.__version__,
                         "config_sha256": self.cfg["_config_sha256"], "selected": self.selected,
                         "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "probes": {c: vars(p) for c, p in self.probes.items()},
                         "layer_probes": [vars(p) for p in self.layer_probes.values()]}, handle)
        self.probe_source = f"trained now and cached to {cache}"

    def _train_probes(self) -> None:
        from augment import FAMILIES
        from step3_train_eval import TrainData, align, load_embeddings, make_model, stratified_subset

        pc, cfg = self.pc, self.cfg
        emb_dir = Path(cfg["_paths"]["embedding_dir"])
        t0 = time.time()
        self.progress(f"Training probes from cached source embeddings in {emb_dir} (first run only) ...")
        pool = pc.load_manifest(cfg, "source_train")
        val_manifest = pc.load_manifest(cfg, "source_val")
        train_sets = {}
        for variant in ("clean",) + tuple(FAMILIES):
            loaded = load_embeddings(emb_dir, "source_train", variant)
            if loaded is None:
                raise SystemExit(f"Missing source embeddings: {emb_dir / 'source_train' / variant}. "
                                 "The dashboard needs the pilot's step-2 features.")
            train_sets[variant] = loaded
        val_set = load_embeddings(emb_dir, "source_val", "clean")
        if val_set is None:
            raise SystemExit(f"Missing source validation embeddings in {emb_dir}")
        data = TrainData(cfg, pool, train_sets)
        val_rows = align(val_manifest, val_set, "source_val")
        y_val = np.array([pc.LABEL_TO_INT[r["label"]] for r in val_manifest])
        seed = cfg["selection"]["seed_used_for_selection"]
        subset = stratified_subset(data.labels, cfg["train_subset_fraction"], seed)
        fitted: dict[tuple[str, int, float], Probe] = {}

        def fit(cond: str, layer: int, C: float) -> Probe:
            key = (cond, layer, float(C))
            if key not in fitted:
                X, y = data.build(cond, subset, layer)
                model = make_model(C, cfg).fit(X, y)
                pv = model.decision_function(val_set.layer(layer, val_rows))
                val_eer, thr = pc.eer_and_threshold(y_val, pv)
                fitted[key] = Probe(cond, layer, float(C), model, float(thr), float(val_eer), int(len(y)))
            return fitted[key]

        jobs = [(c, self.selected[c]["layer"], self.selected[c]["C"], "selected") for c in CONDITIONS if c in self.selected]
        for cond in PROFILE_CONDITIONS:
            for layer in self.layers:
                C = self.per_layer.get((cond, layer), {}).get("C", self.selected[cond]["C"])
                jobs.append((cond, layer, C, "layer"))
        for i, (cond, layer, C, kind) in enumerate(jobs, 1):
            probe = fit(cond, layer, C)
            if kind == "selected":
                self.probes[cond] = probe
            else:
                self.layer_probes[(cond, layer)] = probe
            self.progress(f"  [{i}/{len(jobs)}] {cond:8s} layer {layer:2d} C={C:<6} "
                          f"source-val EER {probe.val_eer * 100:5.2f}%  ({time.time() - t0:5.1f}s)")
        del train_sets, val_set, data

    def _load_encoder(self, device: str):
        from step2_extract_xlsr import XlsrEncoder

        cfg = copy.deepcopy(self.cfg)
        commit = self.run_record.get("encoder", {}).get("resolved_commit")
        if commit and not cfg["feature_extractor"].get("revision"):
            cfg["feature_extractor"]["revision"] = commit  # same weights as the pilot run
        self.progress(f"Loading {cfg['feature_extractor']['model_id']} (revision {commit or 'default'}) ...")
        encoder = XlsrEncoder(cfg, device)
        self.progress(f"Encoder ready on {encoder.description.get('gpu') or encoder.description.get('device')}")
        return encoder

    # ------------------------------------------------------------------ status

    def status(self) -> dict[str, Any]:
        desc = self.encoder_desc
        return {
            "run_id": self.run_id,
            "config_sha": self.cfg["_config_sha256"][:8],
            "manifests_ok": self.manifests_ok,
            "encoder": desc.get("model_id"),
            "encoder_commit": (desc.get("resolved_commit") or "")[:7],
            "device": desc.get("gpu") or desc.get("device", "cpu"),
            "probe_source": self.probe_source,
            "itw_available": self.itw.available,
            "itw_clips": len(self.itw.rows),
            "itw_reason": self.itw.reason,
        }

    def model_choices(self) -> list[tuple[str, str]]:
        out = []
        for cond in CONDITIONS:
            if cond in self.probes:
                p = self.probes[cond]
                out.append((f"{CONDITION_NAMES[cond]} · layer {p.layer}", cond))
        return out

    # ------------------------------------------------------------------ analysis

    SOUNDFILE_SUFFIXES = {".wav", ".flac", ".ogg", ".oga", ".aif", ".aiff", ".mp3"}

    def read_file(self, path: str | os.PathLike) -> tuple[np.ndarray, int]:
        signal, sr, _ = self.decode(path)
        return signal, sr

    def decode(self, path: str | os.PathLike) -> tuple[np.ndarray, int, str]:
        """Decode an audio file to mono float32; returns (signal, sample_rate, decoder used).

        WAV/FLAC/OGG/MP3 go through soundfile, exactly as in the pilot. Everything else (M4A/AAC, MP4, WebM, ...)
        goes straight to ffmpeg with the file path. soundfile is not trusted for other formats: on Windows its
        MP3 reader can "open" an M4A file and return zero samples instead of raising an error. If soundfile
        returns nothing, ffmpeg is tried as well.
        """
        path = Path(path)
        if path.suffix.lower() in self.SOUNDFILE_SUFFIXES:
            try:
                import soundfile as sf  # type: ignore

                data, sr = sf.read(str(path), dtype="float32", always_2d=True)
                if data.shape[0] > 0:
                    return data.mean(axis=1).astype(np.float32), int(sr), "soundfile"
            except Exception:  # noqa: BLE001  (unreadable for soundfile, or soundfile missing)
                pass
        signal = self._ffmpeg_decode(path)
        return signal, 16000, "ffmpeg"

    def _ffmpeg_decode(self, path: Path) -> np.ndarray:
        import subprocess

        exe = self.pc.ffmpeg_exe()
        proc = subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn", "-f", "f32le", "-ac", "1",
             "-ar", "16000", "pipe:1"],
            capture_output=True, check=False)
        samples = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        if proc.returncode != 0 or samples.size == 0:
            detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
            raise ValueError(f"{path.name} could not be decoded as audio"
                             + (f" (ffmpeg: {detail[-1][:160]})" if detail else "") + ".")
        return samples

    def analyse_path(self, path: str | os.PathLike) -> Analysis:
        signal, sr, decoder = self.decode(path)
        name = Path(path).name
        meta = self._itw_files.get(Path(path).name)  # Gradio copies files into its cache but keeps the name
        result = self.analyse(signal, sr, meta["sample_id"] if meta else name, ground_truth=meta)
        result.info["decoder"] = decoder
        return result

    def analyse(self, signal: np.ndarray, sr: int, source: str, ground_truth: dict[str, Any] | None = None) -> Analysis:
        pc = self.pc
        if signal.size == 0:
            raise ValueError("The audio file is empty.")
        wav16 = pc.resample(signal.astype(np.float32), sr, 16000)
        processed, info = pc.preprocess(signal.astype(np.float32), sr, self.cfg)
        if processed.size < 400:
            raise ValueError("The audio is too short to analyse (less than 25 ms after trimming).")
        start = 0 if info.get("trim_fallback") else int(round(info["lead_silence_s"] * 16000))
        end = min(len(wav16), start + len(processed))
        t0 = time.time()
        embedding = np.asarray(self.encoder(pc.normalise(processed)), dtype=np.float32)
        seconds = time.time() - t0

        recorded = (ground_truth or {}).get("recorded", {})
        scores: dict[str, ModelScore] = {}
        for cond, probe in self.probes.items():
            logit = probe.logit(embedding)
            margin = logit - probe.threshold
            summ = self.cond_summary.get(cond, {})
            scores[cond] = ModelScore(
                condition=cond, name=CONDITION_NAMES[cond], layer=probe.layer, C=probe.C, logit=logit,
                threshold=probe.threshold, margin=margin, fake=margin >= 0, score=sigmoid(margin),
                pilot_target_eer=summ.get("target_eer_mean"), pilot_bal_acc=summ.get("target_balanced_accuracy_mean"),
                recorded_logit=recorded.get(cond))
        layer_margins: dict[str, dict[int, float]] = {}
        for (cond, layer), probe in self.layer_probes.items():
            layer_margins.setdefault(cond, {})[layer] = probe.logit(embedding) - probe.threshold
        return Analysis(source=source, duration_s=len(wav16) / 16000, info=info, waveform=wav16,
                        used_start=start, used_end=end, scores=scores, layer_margins=layer_margins,
                        seconds_encoder=seconds, ground_truth=ground_truth)

    def draw_itw(self, label: str = "any", speaker: str | None = None, rng: random.Random | None = None) -> tuple[str, dict[str, Any]]:
        """Copy a random In-the-Wild clip to a temp file; returns (path, metadata with ground truth)."""
        meta, raw = self.itw.draw(label, speaker, rng)
        suffix = Path(meta["zip_member"]).suffix or ".wav"
        folder = Path(tempfile.gettempdir()) / "xlsr_dashboard_itw"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"itw_{meta['sample_id'].replace('/', '_')}{suffix}"
        path.write_bytes(raw)
        self._itw_files[path.name] = meta
        return str(path), meta


class ItwSampler:
    """Random access to the In-the-Wild clips listed in the pilot's frozen target manifest."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.rows: list[dict[str, str]] = []
        self.available = False
        self.reason = ""
        self._reader = None
        self._recorded: dict[str, dict[str, float]] | None = None
        try:
            self.rows = engine.pc.load_manifest(engine.cfg, "target")
            path = engine.pc.source_paths_from_record(engine.cfg)["target"]
            if not Path(path).exists():
                self.reason = f"In-the-Wild archive not found at {path}"
                return
            self._reader = engine.pc.SourceReader(Path(path))
            self.available = bool(self.rows)
            self.reason = "" if self.available else "target manifest is empty"
        except Exception as exc:  # noqa: BLE001
            self.reason = f"In-the-Wild sampling unavailable: {exc}"

    def speakers(self) -> list[str]:
        return sorted({r["speaker_id"] for r in self.rows})

    def recorded_scores(self) -> dict[str, dict[str, float]]:
        """Seed-42 logits written by the pilot run, keyed by sample_id."""
        if self._recorded is None:
            self._recorded = {}
            path = self.engine.run_dir / "predictions_target.csv"
            if path.exists():
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    for row in csv.DictReader(handle):
                        self._recorded[row["sample_id"]] = {
                            c: float(row[f"score_{c}_seed42"]) for c in CONDITIONS if f"score_{c}_seed42" in row}
        return self._recorded

    def draw(self, label: str = "any", speaker: str | None = None, rng: random.Random | None = None) -> tuple[dict[str, Any], bytes]:
        if not self.available or self._reader is None:
            raise RuntimeError(self.reason or "In-the-Wild clips are not available")
        pool = self.rows
        if label in ("real", "fake"):
            pool = [r for r in pool if r["label"] == label]
        if speaker:
            pool = [r for r in pool if r["speaker_id"] == speaker]
        if not pool:
            raise ValueError("No In-the-Wild clip matches that filter.")
        row = (rng or random).choice(pool)
        raw = self._reader.read(row["zip_member"])
        meta = {"sample_id": row["sample_id"], "speaker_id": row["speaker_id"], "label": row["label"],
                "zip_member": row["zip_member"], "recorded": self.recorded_scores().get(row["sample_id"], {})}
        return meta, raw


# ----------------------------------------------------------------------------- console check

def main() -> int:
    parser = argparse.ArgumentParser(description="Console self-check for the dashboard engine")
    parser.add_argument("--pilot-dir")
    parser.add_argument("--data-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--check", action="store_true", help="score a few random In-the-Wild clips")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()
    engine = Engine(args.pilot_dir, args.data_root, args.device, retrain=args.retrain)
    print(json.dumps(engine.status(), indent=2))
    if not args.check:
        return 0
    if not engine.itw.available:
        print(f"In-the-Wild check skipped: {engine.itw.reason}")
        return 0
    rng = random.Random(7841)
    worst = 0.0
    for _ in range(args.n):
        path, meta = engine.draw_itw(rng=rng)
        result = engine.analyse_path(path)
        line = [f"{meta['sample_id']:>10s} {meta['label']:4s}"]
        for cond, s in result.scores.items():
            rec = s.recorded_logit
            if rec is not None:
                worst = max(worst, abs(rec - s.logit))
            line.append(f"{cond}={s.logit:+7.3f}" + (f" (run {rec:+7.3f})" if rec is not None else ""))
        print("  ".join(line))
    print(f"Largest |live - recorded| logit difference: {worst:.4f}")
    print("CHECK PASSED" if worst < 0.25 else "CHECK WARNING: live scores differ from the recorded pilot scores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
