"""Run the frozen none-vs-noise, fixed-layer DeepSpeak confirmation test."""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
PILOT = PROJECT.parent / "pilot_la_to_itw"
PILOT_SRC = PILOT / "src"
sys.path.insert(0, str(PILOT_SRC))
from pilot_common import eer_and_threshold, full_metrics, roc_auc  # noqa: E402

CONFIG = PROJECT / "config" / "confirm_config.json"
TARGET_MANIFEST = PROJECT / "outputs" / "manifests" / "target_deepspeak_v2_test.csv"
SOURCE_TRAIN = PILOT / "outputs" / "manifests" / "source_train_pool.csv"
SOURCE_VAL = PILOT / "outputs" / "manifests" / "source_validation.csv"
SOURCE_EMB = Path(os.environ.get("PILOT_EMBEDDINGS_DIR", "D:/7841_data/pilot_la_to_itw/embeddings"))
TARGET_EMB = Path(os.environ.get(
    "DEEPSPEAK_LAYER10_DIR",
    "D:/7841_data/confirm_la_to_deepspeak/embeddings/target/layer10",
))
OUTPUTS = PROJECT / "outputs"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_source_variant(set_name: str, variant: str, layer: int) -> tuple[np.ndarray, list[dict[str, str]]]:
    folder = SOURCE_EMB / set_name / variant
    matrices, records = [], []
    for npy_path in sorted(folder.glob("shard_[0-9][0-9][0-9][0-9].npy")):
        matrix = np.load(npy_path, mmap_mode="r")
        if matrix.ndim != 3 or layer >= matrix.shape[1]:
            raise SystemExit(f"Unexpected source embedding shape {matrix.shape} in {npy_path}")
        matrices.append(np.asarray(matrix[:, layer, :], dtype=np.float32))
        records.extend(read_csv(npy_path.with_suffix(".csv")))
    return np.concatenate(matrices, axis=0), records


def load_target() -> tuple[np.ndarray, list[dict[str, str]]]:
    matrices, records = [], []
    for npy_path in sorted(TARGET_EMB.glob("shard_[0-9][0-9][0-9][0-9].npy")):
        matrices.append(np.asarray(np.load(npy_path), dtype=np.float32))
        records.extend(read_csv(npy_path.with_suffix(".csv")))
    if not matrices:
        raise SystemExit("Target layer-10 embeddings are missing")
    return np.concatenate(matrices, axis=0), records


def align(manifest: list[dict[str, str]], records: list[dict[str, str]], matrix: np.ndarray, name: str) -> np.ndarray:
    index = {row["sample_id"]: i for i, row in enumerate(records)}
    missing = [row["sample_id"] for row in manifest if row["sample_id"] not in index]
    if missing:
        raise SystemExit(f"{name}: {len(missing)} samples have no embedding")
    return matrix[np.array([index[row["sample_id"]] for row in manifest], dtype=np.int64)]


def stratified_subset(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    for value in np.unique(labels):
        idx = np.flatnonzero(labels == value)
        parts.append(np.sort(rng.choice(idx, size=int(round(fraction * len(idx))), replace=False)))
    return np.sort(np.concatenate(parts))


def make_model(C: float, cfg: dict):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    clf = cfg["classifier"]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, class_weight=clf["class_weight"], solver=clf["solver"], max_iter=clf["max_iter"])),
    ])


def bootstrap(y: np.ndarray, scores_none: np.ndarray, scores_noise: np.ndarray, identities: np.ndarray, reps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    identity_list = np.unique(identities)
    by_identity = {identity: np.flatnonzero(identities == identity) for identity in identity_list}
    none_values = np.empty(reps, dtype=np.float64)
    noise_values = np.empty(reps, dtype=np.float64)
    used = 0
    for rep in range(reps):
        selected = rng.choice(identity_list, size=len(identity_list), replace=True)
        idx = np.concatenate([by_identity[item] for item in selected])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        none_values[used] = np.mean([eer_and_threshold(yy, row[idx])[0] for row in scores_none])
        noise_values[used] = np.mean([eer_and_threshold(yy, row[idx])[0] for row in scores_noise])
        used += 1
        if (rep + 1) % 1000 == 0:
            print(f"bootstrap {rep + 1}/{reps}", flush=True)
    none_values = none_values[:used]
    noise_values = noise_values[:used]
    delta = noise_values - none_values
    return {
        "reps_requested": reps,
        "reps_used": used,
        "none_eer_ci": [float(np.percentile(none_values, 2.5)), float(np.percentile(none_values, 97.5))],
        "noise_eer_ci": [float(np.percentile(noise_values, 2.5)), float(np.percentile(noise_values, 97.5))],
        "delta_eer_ci": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
        "p_delta_eer_ge_0": float(np.mean(delta >= 0)),
    }


def main() -> int:
    import sklearn

    t0 = time.time()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    layer = int(cfg["feature_extractor"]["fixed_hidden_state_layer"])
    C = float(cfg["classifier"]["fixed_C_for_both_conditions"])
    seeds = [int(item) for item in cfg["source"]["seeds"]]
    fraction = float(cfg["source"]["train_subset_fraction"])
    train_manifest = sorted(read_csv(SOURCE_TRAIN), key=lambda r: int(r["order_index"]))
    val_manifest = sorted(read_csv(SOURCE_VAL), key=lambda r: int(r["order_index"]))
    target_manifest = sorted(read_csv(TARGET_MANIFEST), key=lambda r: int(r["order_index"]))
    label_to_int = {"real": 0, "fake": 1}
    y_train = np.array([label_to_int[row["label"]] for row in train_manifest], dtype=np.int64)
    y_val = np.array([label_to_int[row["label"]] for row in val_manifest], dtype=np.int64)
    y_target = np.array([label_to_int[row["label"]] for row in target_manifest], dtype=np.int64)
    identities = np.array([row["identity"] for row in target_manifest])

    train_clean_raw, train_clean_records = load_source_variant("source_train", "clean", layer)
    train_noise_raw, train_noise_records = load_source_variant("source_train", "noise", layer)
    val_raw, val_records = load_source_variant("source_val", "clean", layer)
    target_raw, target_records = load_target()
    train_clean = align(train_manifest, train_clean_records, train_clean_raw, "source train clean")
    train_noise = align(train_manifest, train_noise_records, train_noise_raw, "source train noise")
    X_val = align(val_manifest, val_records, val_raw, "source validation")
    X_target = align(target_manifest, target_records, target_raw, "DeepSpeak target")
    if X_target.shape != (1800, 1024):
        raise SystemExit(f"Unexpected target feature shape {X_target.shape}")

    run_dir = OUTPUTS / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics = []
    target_scores = {"none": [], "noise": []}
    val_scores = {"none": [], "noise": []}
    thresholds = {"none": [], "noise": []}
    for seed in seeds:
        subset = stratified_subset(y_train, fraction, seed)
        for condition in ("none", "noise"):
            if condition == "none":
                X = train_clean[subset]
                y = y_train[subset]
            else:
                X = np.concatenate([train_clean[subset], train_noise[subset]], axis=0)
                y = np.concatenate([y_train[subset], y_train[subset]], axis=0)
            model = make_model(C, cfg).fit(X, y)
            score_val = model.decision_function(X_val)
            _, threshold = eer_and_threshold(y_val, score_val)
            score_target = model.decision_function(X_target)
            target_scores[condition].append(score_target)
            val_scores[condition].append(score_val)
            thresholds[condition].append(threshold)
            val_metric = full_metrics(y_val, score_val, threshold)
            target_metric = full_metrics(y_target, score_target, threshold)
            metrics.append({"condition": condition, "seed": seed, "split": "source_validation", "layer": layer, "C": C, "n_train": len(y), "threshold": threshold, **val_metric})
            metrics.append({"condition": condition, "seed": seed, "split": "target", "layer": layer, "C": C, "n_train": len(y), "threshold": threshold, **target_metric})
            print(f"{condition} seed {seed}: target EER {target_metric['eer']:.4f}, AUC {target_metric['roc_auc']:.4f}")
    write_csv(run_dir / "metrics_per_seed.csv", metrics)
    target_scores = {key: np.stack(value) for key, value in target_scores.items()}
    val_scores = {key: np.stack(value) for key, value in val_scores.items()}

    predictions = []
    for i, row in enumerate(target_manifest):
        item = {
            "sample_id": row["sample_id"],
            "identity": row["identity"],
            "label": row["label"],
            "script_type": row["script_type"],
            "audio_provider": row["audio_provider"],
            "video_engine": row["video_engine"],
        }
        for condition in ("none", "noise"):
            for k, seed in enumerate(seeds):
                item[f"score_{condition}_seed{seed}"] = f"{target_scores[condition][k, i]:.8f}"
        predictions.append(item)
    write_csv(run_dir / "predictions_target.csv", predictions)

    reps = int(cfg["evaluation"]["bootstrap_reps"])
    boot = bootstrap(y_target, target_scores["none"], target_scores["noise"], identities, reps, int(cfg["evaluation"]["bootstrap_seed"]))
    (run_dir / "bootstrap_identity_cluster.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")

    condition_summary = {}
    for condition in ("none", "noise"):
        rows = [row for row in metrics if row["condition"] == condition and row["split"] == "target"]
        condition_summary[condition] = {}
        for metric in ("eer", "roc_auc", "pr_auc_fake", "balanced_accuracy", "macro_f1", "accuracy", "recall_fake", "recall_real"):
            values = np.array([float(row[metric]) for row in rows])
            condition_summary[condition][f"{metric}_mean"] = float(values.mean())
            condition_summary[condition][f"{metric}_sd"] = float(values.std(ddof=1))
    delta = condition_summary["noise"]["eer_mean"] - condition_summary["none"]["eer_mean"]
    lo, hi = boot["delta_eer_ci"]
    verdict = "improvement_confirmed" if hi < 0 else ("harm_confirmed" if lo > 0 else "inconclusive")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study_id": cfg["study_id"],
        "status": "completed_without_target_driven_changes",
        "fixed_layer": layer,
        "fixed_C": C,
        "seeds": seeds,
        "n_source_train_pool": len(train_manifest),
        "n_source_validation": len(val_manifest),
        "n_target": len(target_manifest),
        "target_counts": {"real": int((y_target == 0).sum()), "fake": int((y_target == 1).sum()), "identities": int(len(np.unique(identities)))},
        "conditions": condition_summary,
        "delta_eer_noise_minus_none": delta,
        "delta_eer_ci_identity_cluster": [lo, hi],
        "decision_rule": cfg["evaluation"]["decision_rule"],
        "primary_verdict": verdict,
        "bootstrap": boot,
        "model_revision": cfg["feature_extractor"]["resolved_revision"],
        "dataset_revision": "dd0d22e5546e6ed26c9c2599d8dc345d2760ad7c",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "minutes": round((time.time() - t0) / 60.0, 3),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    delta_pp = 100.0 * delta
    lines = [
        "# Confirmatory ASVspoof LA to DeepSpeak v2 result",
        "",
        f"- Frozen XLS-R layer: {layer}",
        f"- Fixed logistic-regression C: {C}",
        f"- Target: 1,800 DeepSpeak v2 test clips from 56 identities",
        f"- No augmentation mean EER: {100 * condition_summary['none']['eer_mean']:.2f}%",
        f"- Additive noise mean EER: {100 * condition_summary['noise']['eer_mean']:.2f}%",
        f"- Difference, noise minus none: {delta_pp:+.2f} percentage points",
        f"- 95% identity-cluster bootstrap CI: [{100 * lo:+.2f}, {100 * hi:+.2f}] percentage points",
        f"- Pre-specified decision: **{verdict}**",
        "",
        "The result uses the frozen target manifest once. DeepSpeak labels did not influence training, layer selection, C, preprocessing or thresholds.",
    ]
    (run_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTPUTS / "LATEST_RUN.txt").write_text(run_dir.name, encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "delta_eer": delta, "delta_ci": [lo, hi], "verdict": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
