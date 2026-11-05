"""Run the bounded post-confirmation fixed-layer augmentation follow-up.

This script intentionally labels every output exploratory. DeepSpeak was already
observed in the confirmatory none-vs-noise run, so these additional conditions
cannot constitute a new independent validation.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]
PILOT = PROJECT.parent / "pilot_la_to_itw"
PILOT_SRC = PILOT / "src"
sys.path.insert(0, str(PILOT_SRC))

from pilot_common import eer_and_threshold, full_metrics, roc_auc  # noqa: E402
from step3_train_eval import (  # noqa: E402
    TrainData,
    align as align_pilot,
    load_embeddings,
    make_model,
    stratified_subset,
)


FOLLOWUP_CONFIG = PROJECT / "config" / "exploratory_followup_2026-11-04.json"
CONFIRM_CONFIG = PROJECT / "config" / "confirm_config.json"
PILOT_CONFIG = PILOT / "config" / "pilot_config.json"
TARGET_MANIFEST = PROJECT / "outputs" / "manifests" / "target_deepspeak_v2_test.csv"
SOURCE_TRAIN = PILOT / "outputs" / "manifests" / "source_train_pool.csv"
SOURCE_VAL = PILOT / "outputs" / "manifests" / "source_validation.csv"
SOURCE_EMB = Path(os.environ.get("PILOT_EMBEDDINGS_DIR", "D:/7841_data/pilot_la_to_itw/embeddings"))
TARGET_EMB = Path(os.environ.get(
    "DEEPSPEAK_LAYER10_DIR",
    "D:/7841_data/confirm_la_to_deepspeak/embeddings/target/layer10",
))
OUTPUTS = PROJECT / "outputs"
CONDITIONS = ("none", "codec", "noise", "channel", "mix", "all")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_target() -> tuple[np.ndarray, list[dict[str, str]]]:
    matrices, records = [], []
    for npy_path in sorted(TARGET_EMB.glob("shard_[0-9][0-9][0-9][0-9].npy")):
        matrices.append(np.asarray(np.load(npy_path), dtype=np.float32))
        records.extend(read_csv(npy_path.with_suffix(".csv")))
    if not matrices:
        raise SystemExit("Target layer-10 embeddings are missing")
    return np.concatenate(matrices, axis=0), records


def align_target(manifest: list[dict[str, str]], records: list[dict[str, str]], matrix: np.ndarray) -> np.ndarray:
    index = {row["sample_id"]: i for i, row in enumerate(records)}
    missing = [row["sample_id"] for row in manifest if row["sample_id"] not in index]
    if missing:
        raise SystemExit(f"DeepSpeak target: {len(missing)} samples have no embedding")
    return matrix[np.array([index[row["sample_id"]] for row in manifest], dtype=np.int64)]


def paired_identity_bootstrap(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    identities: np.ndarray,
    reps: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    identity_list = np.unique(identities)
    by_identity = {identity: np.flatnonzero(identities == identity) for identity in identity_list}
    eer_values = {condition: np.empty(reps, dtype=np.float64) for condition in CONDITIONS}
    auc_values = {condition: np.empty(reps, dtype=np.float64) for condition in CONDITIONS}
    used = 0
    for rep in range(reps):
        selected = rng.choice(identity_list, size=len(identity_list), replace=True)
        idx = np.concatenate([by_identity[item] for item in selected])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        for condition in CONDITIONS:
            eer_values[condition][used] = np.mean(
                [eer_and_threshold(yy, per_seed[idx])[0] for per_seed in scores[condition]]
            )
            auc_values[condition][used] = np.mean(
                [roc_auc(yy, per_seed[idx]) for per_seed in scores[condition]]
            )
        used += 1
        if (rep + 1) % 1000 == 0:
            print(f"bootstrap {rep + 1}/{reps}", flush=True)

    result = {"reps_requested": reps, "reps_used": used, "conditions": {}}
    base_eer = eer_values["none"][:used]
    base_auc = auc_values["none"][:used]
    for condition in CONDITIONS:
        eers = eer_values[condition][:used]
        aucs = auc_values[condition][:used]
        delta_eer = eers - base_eer
        delta_auc = aucs - base_auc
        result["conditions"][condition] = {
            "eer_ci": [float(np.percentile(eers, 2.5)), float(np.percentile(eers, 97.5))],
            "auc_ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
            "delta_eer_vs_none_ci": [
                float(np.percentile(delta_eer, 2.5)),
                float(np.percentile(delta_eer, 97.5)),
            ],
            "delta_auc_vs_none_ci": [
                float(np.percentile(delta_auc, 2.5)),
                float(np.percentile(delta_auc, 97.5)),
            ],
            "p_delta_eer_ge_0": float(np.mean(delta_eer >= 0)),
        }
    return result


def make_figures(run_dir: Path, summary: dict) -> None:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = ["None", "Codec", "Noise", "Channel", "Mix", "All"]
    colors = ["#26334f", "#3178d4", "#1e9f75", "#ef6a32", "#8b5fbf", "#566573"]
    means = np.array([100 * summary["conditions"][c]["eer_mean"] for c in CONDITIONS])
    cis = np.array([summary["bootstrap"]["conditions"][c]["eer_ci"] for c in CONDITIONS]) * 100

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    err = np.vstack([means - cis[:, 0], cis[:, 1] - means])
    ax.bar(labels, means, color=colors, yerr=err, capsize=4)
    ax.axhline(50, color="#8a8f98", linestyle="--", linewidth=1, label="Chance-level EER")
    ax.set_ylabel("DeepSpeak target EER (%)")
    ax.set_title("EXPLORATORY fixed-layer follow-up (layer 10, C = 0.1)")
    ax.set_ylim(0, max(55, float(np.max(cis[:, 1]) + 3)))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    for i, value in enumerate(means):
        ax.text(i, value + 0.8, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_target_eer_by_condition.png", dpi=180)
    plt.close(fig)

    delta = np.array([100 * summary["conditions"][c]["delta_eer_vs_none"] for c in CONDITIONS[1:]])
    delta_ci = np.array(
        [summary["bootstrap"]["conditions"][c]["delta_eer_vs_none_ci"] for c in CONDITIONS[1:]]
    ) * 100
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    y_pos = np.arange(len(CONDITIONS) - 1)
    xerr = np.vstack([delta - delta_ci[:, 0], delta_ci[:, 1] - delta])
    ax.errorbar(delta, y_pos, xerr=xerr, fmt="o", color="#26334f", ecolor="#3178d4", capsize=4)
    ax.axvline(0, color="#ef6a32", linestyle="--", linewidth=1.2)
    ax.set_yticks(y_pos, labels[1:])
    ax.invert_yaxis()
    ax.set_xlabel("EER difference versus none (percentage points; negative is better)")
    ax.set_title("Paired 95% identity-cluster intervals — exploratory only")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_delta_eer_vs_none.png", dpi=180)
    plt.close(fig)


def main() -> int:
    t0 = time.time()
    followup_cfg = json.loads(FOLLOWUP_CONFIG.read_text(encoding="utf-8"))
    confirm_cfg = json.loads(CONFIRM_CONFIG.read_text(encoding="utf-8"))
    pilot_cfg = json.loads(PILOT_CONFIG.read_text(encoding="utf-8"))
    layer = int(followup_cfg["fixed_settings"]["hidden_state_layer"])
    C = float(followup_cfg["fixed_settings"]["classifier_C"])
    seeds = [int(value) for value in followup_cfg["fixed_settings"]["source_seeds"]]
    fraction = float(followup_cfg["fixed_settings"]["source_subset_fraction"])

    train_manifest = sorted(read_csv(SOURCE_TRAIN), key=lambda row: int(row["order_index"]))
    val_manifest = sorted(read_csv(SOURCE_VAL), key=lambda row: int(row["order_index"]))
    target_manifest = sorted(read_csv(TARGET_MANIFEST), key=lambda row: int(row["order_index"]))
    label_to_int = {"real": 0, "fake": 1}
    y_train = np.array([label_to_int[row["label"]] for row in train_manifest], dtype=np.int64)
    y_val = np.array([label_to_int[row["label"]] for row in val_manifest], dtype=np.int64)
    y_target = np.array([label_to_int[row["label"]] for row in target_manifest], dtype=np.int64)
    identities = np.array([row["identity"] for row in target_manifest])

    train_sets = {}
    for variant in ("clean", "codec", "noise", "channel"):
        loaded = load_embeddings(SOURCE_EMB, "source_train", variant)
        if loaded is None:
            raise SystemExit(f"Missing source embeddings: {variant}")
        train_sets[variant] = loaded
    data = TrainData(pilot_cfg, train_manifest, train_sets)
    val_set = load_embeddings(SOURCE_EMB, "source_val", "clean")
    if val_set is None:
        raise SystemExit("Missing source validation embeddings")
    val_rows = align_pilot(val_manifest, val_set, "source validation")
    X_val = val_set.layer(layer, val_rows)
    target_raw, target_records = load_target()
    X_target = align_target(target_manifest, target_records, target_raw)
    if X_target.shape != (1800, 1024):
        raise SystemExit(f"Unexpected target feature shape {X_target.shape}")

    run_dir = OUTPUTS / f"exploratory_followup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics = []
    target_scores: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}
    thresholds: dict[str, list[float]] = {condition: [] for condition in CONDITIONS}

    for seed in seeds:
        subset = stratified_subset(y_train, fraction, seed)
        for condition in CONDITIONS:
            X_train, yy = data.build(condition, subset, layer)
            model = make_model(C, confirm_cfg).fit(X_train, yy)
            score_val = model.decision_function(X_val)
            _, threshold = eer_and_threshold(y_val, score_val)
            score_target = model.decision_function(X_target)
            target_scores[condition].append(score_target)
            thresholds[condition].append(float(threshold))
            val_metric = full_metrics(y_val, score_val, threshold)
            target_metric = full_metrics(y_target, score_target, threshold)
            metrics.append(
                {
                    "classification": "EXPLORATORY_POST_CONFIRMATION",
                    "condition": condition,
                    "seed": seed,
                    "split": "source_validation",
                    "layer": layer,
                    "C": C,
                    "n_train": len(yy),
                    "threshold": threshold,
                    **val_metric,
                }
            )
            metrics.append(
                {
                    "classification": "EXPLORATORY_POST_CONFIRMATION",
                    "condition": condition,
                    "seed": seed,
                    "split": "target",
                    "layer": layer,
                    "C": C,
                    "n_train": len(yy),
                    "threshold": threshold,
                    **target_metric,
                }
            )
            print(
                f"{condition:7s} seed {seed}: target EER {target_metric['eer']:.4f}, "
                f"AUC {target_metric['roc_auc']:.4f}",
                flush=True,
            )
    write_csv(run_dir / "metrics_per_seed.csv", metrics)
    scores = {condition: np.stack(values) for condition, values in target_scores.items()}

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
        for condition in CONDITIONS:
            for k, seed in enumerate(seeds):
                item[f"score_{condition}_seed{seed}"] = f"{scores[condition][k, i]:.8f}"
        predictions.append(item)
    write_csv(run_dir / "predictions_target.csv", predictions)

    bootstrap = paired_identity_bootstrap(
        y_target,
        scores,
        identities,
        int(followup_cfg["fixed_settings"]["bootstrap_reps"]),
        int(followup_cfg["fixed_settings"]["bootstrap_seed"]),
    )
    (run_dir / "bootstrap_identity_cluster.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )

    condition_summary = {}
    for condition in CONDITIONS:
        rows = [row for row in metrics if row["condition"] == condition and row["split"] == "target"]
        entry = {"condition": condition, "layer": layer, "C": C, "n_train": int(rows[0]["n_train"])}
        for metric in (
            "eer",
            "roc_auc",
            "pr_auc_fake",
            "balanced_accuracy",
            "macro_f1",
            "accuracy",
            "recall_fake",
            "recall_real",
        ):
            values = np.array([float(row[metric]) for row in rows])
            entry[f"{metric}_mean"] = float(values.mean())
            entry[f"{metric}_sd"] = float(values.std(ddof=1))
        condition_summary[condition] = entry
    base = condition_summary["none"]
    for condition in CONDITIONS:
        condition_summary[condition]["delta_eer_vs_none"] = (
            condition_summary[condition]["eer_mean"] - base["eer_mean"]
        )
        condition_summary[condition]["delta_auc_vs_none"] = (
            condition_summary[condition]["roc_auc_mean"] - base["roc_auc_mean"]
        )

    best_condition = min(CONDITIONS, key=lambda condition: condition_summary[condition]["eer_mean"])
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study_id": followup_cfg["study_id"],
        "classification": "EXPLORATORY_POST_CONFIRMATION",
        "target_already_observed": True,
        "fixed_layer": layer,
        "fixed_C": C,
        "seeds": seeds,
        "n_target": len(target_manifest),
        "target_counts": {
            "real": int((y_target == 0).sum()),
            "fake": int((y_target == 1).sum()),
            "identities": int(len(np.unique(identities))),
        },
        "conditions": condition_summary,
        "bootstrap": bootstrap,
        "best_mean_eer_condition": best_condition,
        "interpretation_rule": followup_cfg["interpretation_rule"],
        "model_revision": confirm_cfg["feature_extractor"]["resolved_revision"],
        "dataset_revision": "dd0d22e5546e6ed26c9c2599d8dc345d2760ad7c",
        "minutes": round((time.time() - t0) / 60.0, 3),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_figures(run_dir, summary)

    lines = [
        "# Exploratory fixed-layer augmentation follow-up",
        "",
        "> **Status: EXPLORATORY_POST_CONFIRMATION.** DeepSpeak had already been evaluated before this comparison was chosen. These results explain the failure and generate hypotheses; they are not a new independent validation.",
        "",
        f"- Fixed XLS-R layer: {layer}",
        f"- Fixed logistic-regression C: {C}",
        "- Source seeds: 42, 1337, 2026",
        "- Target: 1,800 DeepSpeak v2 test clips from 56 identities",
        "- Paired uncertainty: 10,000 identity-cluster bootstrap replications",
        "",
        "## Target results",
        "",
        "| Condition | Mean EER | Delta vs none | 95% paired CI for delta | Mean AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        entry = condition_summary[condition]
        ci = bootstrap["conditions"][condition]["delta_eer_vs_none_ci"]
        lines.append(
            f"| {condition} | {100 * entry['eer_mean']:.2f}% | "
            f"{100 * entry['delta_eer_vs_none']:+.2f} pp | "
            f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}] pp | {entry['roc_auc_mean']:.3f} |"
        )
    best = condition_summary[best_condition]
    best_ci = bootstrap["conditions"][best_condition]["delta_eer_vs_none_ci"]
    lines.extend(
        [
            "",
            "## Bounded interpretation",
            "",
            f"The lowest mean EER was produced by **{best_condition}** "
            f"({100 * best['eer_mean']:.2f}%; delta versus none "
            f"{100 * best['delta_eer_vs_none']:+.2f} pp, paired 95% CI "
            f"[{100 * best_ci[0]:+.2f}, {100 * best_ci[1]:+.2f}] pp).",
            "",
            "Even if a paired interval excludes zero, the finding remains exploratory because the DeepSpeak target result was already visible. It must not be described as pre-registered or independently confirmed.",
            "",
            "## Reusable evidence",
            "",
            "- `metrics_per_seed.csv`",
            "- `predictions_target.csv`",
            "- `bootstrap_identity_cluster.json`",
            "- `summary.json`",
            "- `figures/fig1_target_eer_by_condition.png`",
            "- `figures/fig2_delta_eer_vs_none.png`",
        ]
    )
    (run_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTPUTS / "LATEST_EXPLORATORY_FOLLOWUP.txt").write_text(run_dir.name, encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "best_mean_eer_condition": best_condition}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
