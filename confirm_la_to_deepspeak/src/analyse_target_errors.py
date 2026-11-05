"""Exploratory error analysis for a saved DeepSpeak evaluation run.

The script never retrains or changes a model. It reads saved sample-level
scores and source-validation thresholds, then describes where target errors
occur. Outputs containing DeepSpeak sample identifiers remain local and must
not be uploaded with a public reproducibility package.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT / "outputs"
SCORE_PATTERN = re.compile(r"^score_(.+)_seed(\d+)$")


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


def discover_scores(columns: list[str]) -> tuple[list[str], list[int]]:
    pairs = []
    for column in columns:
        match = SCORE_PATTERN.match(column)
        if match:
            pairs.append((match.group(1), int(match.group(2))))
    conditions = sorted({condition for condition, _ in pairs}, key=lambda x: (x != "none", x))
    seeds = sorted({seed for _, seed in pairs})
    expected = {(condition, seed) for condition in conditions for seed in seeds}
    if set(pairs) != expected:
        raise SystemExit("Prediction columns do not form a complete condition-by-seed grid")
    return conditions, seeds


def safe_rate(mask: np.ndarray, errors: np.ndarray) -> float:
    return float(errors[mask].mean()) if int(mask.sum()) else float("nan")


def mean_target_metrics(metric_rows: list[dict[str, str]], conditions: list[str]) -> dict[str, dict[str, float]]:
    output = {}
    for condition in conditions:
        rows = [row for row in metric_rows if row["split"] == "target" and row["condition"] == condition]
        output[condition] = {
            "eer": float(np.mean([float(row["eer"]) for row in rows])),
            "roc_auc": float(np.mean([float(row["roc_auc"]) for row in rows])),
            "balanced_accuracy": float(np.mean([float(row["balanced_accuracy"]) for row in rows])),
            "recall_fake": float(np.mean([float(row["recall_fake"]) for row in rows])),
            "recall_real": float(np.mean([float(row["recall_real"]) for row in rows])),
        }
    return output


def plot_overall(out_dir: Path, conditions: list[str], overall: list[dict]) -> None:
    aggregated = {}
    for condition in conditions:
        rows = [row for row in overall if row["condition"] == condition]
        aggregated[condition] = {
            "real_fpr": float(np.mean([float(row["real_false_positive_rate"]) for row in rows])),
            "fake_fnr": float(np.mean([float(row["fake_false_negative_rate"]) for row in rows])),
        }
    x = np.arange(len(conditions))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(
        x - width / 2,
        [100 * aggregated[c]["real_fpr"] for c in conditions],
        width,
        label="Real clips predicted fake",
        color="#3178d4",
    )
    ax.bar(
        x + width / 2,
        [100 * aggregated[c]["fake_fnr"] for c in conditions],
        width,
        label="Fake clips predicted real",
        color="#ef6a32",
    )
    ax.set_xticks(x, conditions)
    ax.set_ylabel("Error rate at source-validation threshold (%)")
    ax.set_title("Target operating-point errors reveal threshold transfer failure")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_class_error_rates.png", dpi=180)
    plt.close(fig)


def plot_provider(out_dir: Path, conditions: list[str], subgroup_rows: list[dict]) -> None:
    providers = sorted(
        {row["group_value"] for row in subgroup_rows if row["group_type"] == "fake_audio_provider"}
    )
    if not providers:
        return
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    x = np.arange(len(providers))
    width = 0.8 / len(conditions)
    for i, condition in enumerate(conditions):
        values = []
        for provider in providers:
            rows = [
                row
                for row in subgroup_rows
                if row["condition"] == condition
                and row["group_type"] == "fake_audio_provider"
                and row["group_value"] == provider
            ]
            values.append(100 * float(np.mean([float(row["error_rate"]) for row in rows])))
        ax.bar(x - 0.4 + width / 2 + i * width, values, width, label=condition)
    ax.set_xticks(x, providers)
    ax.set_ylabel("Fake false-negative rate (%)")
    ax.set_title("Exploratory fake-audio-provider error rates")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=min(3, len(conditions)))
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_fake_errors_by_provider.png", dpi=180)
    plt.close(fig)


def plot_score_scatter(
    out_dir: Path,
    rows: list[dict[str, str]],
    conditions: list[str],
    seeds: list[int],
    best: str,
) -> float:
    if best == "none":
        return 1.0
    none = np.array(
        [np.mean([float(row[f"score_none_seed{seed}"]) for seed in seeds]) for row in rows]
    )
    other = np.array(
        [np.mean([float(row[f"score_{best}_seed{seed}"]) for seed in seeds]) for row in rows]
    )
    y = np.array([row["label"] == "fake" for row in rows])
    corr = float(np.corrcoef(none, other)[0, 1])
    rng = np.random.default_rng(7841)
    idx = np.sort(rng.choice(len(rows), size=min(900, len(rows)), replace=False))
    fig, ax = plt.subplots(figsize=(6.1, 5.4))
    ax.scatter(none[idx][~y[idx]], other[idx][~y[idx]], s=10, alpha=0.45, label="real", color="#3178d4")
    ax.scatter(none[idx][y[idx]], other[idx][y[idx]], s=10, alpha=0.55, label="fake", color="#ef6a32")
    ax.set_xlabel("None: mean target score across seeds")
    ax.set_ylabel(f"{best}: mean target score across seeds")
    ax.set_title(f"Scores remain strongly related (Pearson r = {corr:.3f})")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_score_scatter.png", dpi=180)
    plt.close(fig)
    return corr


def plot_identity(out_dir: Path, conditions: list[str], identity_rows: list[dict]) -> None:
    values = []
    for condition in conditions:
        values.append(
            [100 * float(row["mean_error_rate"]) for row in identity_rows if row["condition"] == condition]
        )
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.boxplot(values, tick_labels=conditions, showfliers=False)
    ax.set_ylabel("Per-identity error rate (%)")
    ax.set_title("Performance varies substantially across DeepSpeak identities")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_identity_error_distribution.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Saved confirmatory or exploratory run directory")
    args = parser.parse_args()
    run_dir = Path(args.run).resolve()
    predictions = read_csv(run_dir / "predictions_target.csv")
    metric_rows = read_csv(run_dir / "metrics_per_seed.csv")
    conditions, seeds = discover_scores(list(predictions[0]))
    thresholds = {
        (row["condition"], int(row["seed"])): float(row["threshold"])
        for row in metric_rows
        if row["split"] == "target"
    }
    if set(thresholds) != {(condition, seed) for condition in conditions for seed in seeds}:
        raise SystemExit("Missing source-validation thresholds in target metric rows")

    y = np.array([1 if row["label"] == "fake" else 0 for row in predictions], dtype=np.int64)
    identities = np.array([row["identity"] for row in predictions])
    out_dir = run_dir / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    errors: dict[tuple[str, int], np.ndarray] = {}
    overall = []
    subgroup_rows = []
    identity_rows = []
    for condition in conditions:
        for seed in seeds:
            scores = np.array([float(row[f"score_{condition}_seed{seed}"]) for row in predictions])
            pred = (scores >= thresholds[(condition, seed)]).astype(np.int64)
            error = pred != y
            errors[(condition, seed)] = error
            overall.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "n": len(y),
                    "error_rate": float(error.mean()),
                    "real_false_positive_rate": safe_rate(y == 0, error),
                    "fake_false_negative_rate": safe_rate(y == 1, error),
                    "source_validation_threshold": thresholds[(condition, seed)],
                }
            )

            group_specs: list[tuple[str, np.ndarray]] = [
                ("label", np.array([row["label"] for row in predictions])),
                ("script_type", np.array([row["script_type"] for row in predictions])),
            ]
            fake = y == 1
            for group_type, field in (
                ("fake_audio_provider", "audio_provider"),
                ("fake_video_engine", "video_engine"),
                ("fake_provider_engine", None),
            ):
                if field:
                    values = np.array([row[field] for row in predictions])
                else:
                    values = np.array(
                        [f"{row['audio_provider']}+{row['video_engine']}" for row in predictions]
                    )
                for value in sorted(set(values[fake])):
                    mask = fake & (values == value)
                    subgroup_rows.append(
                        {
                            "condition": condition,
                            "seed": seed,
                            "group_type": group_type,
                            "group_value": value,
                            "label_scope": "fake",
                            "n": int(mask.sum()),
                            "error_rate": safe_rate(mask, error),
                        }
                    )
            for group_type, values in group_specs:
                for value in sorted(set(values)):
                    mask = values == value
                    subgroup_rows.append(
                        {
                            "condition": condition,
                            "seed": seed,
                            "group_type": group_type,
                            "group_value": value,
                            "label_scope": "mixed" if group_type == "script_type" else value,
                            "n": int(mask.sum()),
                            "error_rate": safe_rate(mask, error),
                        }
                    )

        for identity in sorted(set(identities)):
            mask = identities == identity
            per_seed = np.array([errors[(condition, seed)][mask].mean() for seed in seeds])
            real_mask = mask & (y == 0)
            fake_mask = mask & (y == 1)
            identity_rows.append(
                {
                    "condition": condition,
                    "identity": identity,
                    "n": int(mask.sum()),
                    "n_real": int(real_mask.sum()),
                    "n_fake": int(fake_mask.sum()),
                    "mean_error_rate": float(per_seed.mean()),
                    "real_error_rate": float(
                        np.mean([safe_rate(real_mask, errors[(condition, seed)]) for seed in seeds])
                    ),
                    "fake_error_rate": float(
                        np.nanmean([safe_rate(fake_mask, errors[(condition, seed)]) for seed in seeds])
                    ) if int(fake_mask.sum()) else "",
                }
            )

    hard_cases = []
    for i, row in enumerate(predictions):
        item = {
            "sample_id": row["sample_id"],
            "identity": row["identity"],
            "label": row["label"],
            "script_type": row["script_type"],
            "audio_provider": row["audio_provider"],
            "video_engine": row["video_engine"],
        }
        for condition in conditions:
            count = int(sum(bool(errors[(condition, seed)][i]) for seed in seeds))
            item[f"wrong_seed_count_{condition}"] = count
        hard_cases.append(item)

    write_csv(out_dir / "overall_threshold_errors.csv", overall)
    write_csv(out_dir / "subgroup_errors.csv", subgroup_rows)
    write_csv(out_dir / "identity_errors.csv", identity_rows)
    write_csv(out_dir / "hard_cases_LOCAL_RESTRICTED.csv", hard_cases)

    target_metrics = mean_target_metrics(metric_rows, conditions)
    best = min(conditions, key=lambda condition: target_metrics[condition]["eer"])
    plot_overall(out_dir, conditions, overall)
    plot_provider(out_dir, conditions, subgroup_rows)
    corr = plot_score_scatter(out_dir, predictions, conditions, seeds, best)
    plot_identity(out_dir, conditions, identity_rows)

    aggregated_overall = {}
    for condition in conditions:
        rows = [row for row in overall if row["condition"] == condition]
        aggregated_overall[condition] = {
            "error_rate_mean": float(np.mean([float(row["error_rate"]) for row in rows])),
            "real_false_positive_rate_mean": float(
                np.mean([float(row["real_false_positive_rate"]) for row in rows])
            ),
            "fake_false_negative_rate_mean": float(
                np.mean([float(row["fake_false_negative_rate"]) for row in rows])
            ),
            **target_metrics[condition],
        }
    always_wrong = {
        condition: {
            "all_three_seeds": int(
                sum(int(row[f"wrong_seed_count_{condition}"]) == len(seeds) for row in hard_cases)
            ),
            "majority_of_seeds": int(
                sum(int(row[f"wrong_seed_count_{condition}"]) >= 2 for row in hard_cases)
            ),
        }
        for condition in conditions
    }
    provider_means = defaultdict(dict)
    for condition in conditions:
        provider_values = sorted(
            {row["group_value"] for row in subgroup_rows if row["group_type"] == "fake_audio_provider"}
        )
        for provider in provider_values:
            rows = [
                row
                for row in subgroup_rows
                if row["condition"] == condition
                and row["group_type"] == "fake_audio_provider"
                and row["group_value"] == provider
            ]
            provider_means[condition][provider] = float(
                np.mean([float(row["error_rate"]) for row in rows])
            )

    summary = {
        "classification": "EXPLORATORY_ERROR_ANALYSIS",
        "source_run": run_dir.name,
        "conditions": conditions,
        "seeds": seeds,
        "best_mean_eer_condition": best,
        "overall": aggregated_overall,
        "always_wrong_counts": always_wrong,
        "fake_provider_false_negative_rates": provider_means,
        "score_correlation_none_vs_best": corr,
        "restricted_output_warning": "hard_cases_LOCAL_RESTRICTED.csv contains DeepSpeak sample identifiers and must not be uploaded publicly.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    best_row = aggregated_overall[best]
    none_row = aggregated_overall["none"]
    provider_range = provider_means[best]
    low_provider = min(provider_range, key=provider_range.get)
    high_provider = max(provider_range, key=provider_range.get)
    lines = [
        "# Exploratory DeepSpeak error analysis",
        "",
        "> This analysis explains saved results; it does not retrain models or create a new confirmatory claim.",
        "",
        f"- Source run: `{run_dir.name}`",
        f"- Conditions analysed: {', '.join(conditions)}",
        f"- Target samples: {len(predictions)}; identities: {len(set(identities))}",
        "- Operating point: each condition and seed uses its source-validation EER threshold.",
        "",
        "## Main findings",
        "",
        f"1. The lowest mean target EER in this run is **{best}** at {100 * best_row['eer']:.2f}%, compared with {100 * none_row['eer']:.2f}% for none.",
        f"2. At the transferred source threshold, {best} still misclassifies {100 * best_row['real_false_positive_rate_mean']:.1f}% of real clips and {100 * best_row['fake_false_negative_rate_mean']:.1f}% of fake clips. Its balanced accuracy is {100 * best_row['balanced_accuracy']:.1f}%.",
        f"3. Mean none and {best} target scores remain strongly related (Pearson r = {corr:.3f}); augmentation changes ranking only modestly rather than creating a clean domain-invariant separation.",
        f"4. For {best}, fake false-negative rates vary from {100 * provider_range[low_provider]:.1f}% ({low_provider}) to {100 * provider_range[high_provider]:.1f}% ({high_provider}) across audio providers.",
        f"5. {always_wrong[best]['all_three_seeds']} of {len(predictions)} clips are wrong for {best} under all three source seeds, showing a persistent subset rather than only seed noise.",
        "",
        "## Interpretation for the thesis",
        "",
        "A lower EER or higher AUC describes ranking quality across possible thresholds. The transferred source threshold is a separate deployment problem. The results show that both representation shift and threshold/calibration shift matter on DeepSpeak. Target-label calibration would change the research protocol, so it is not applied here.",
        "",
        "## Files",
        "",
        "- `overall_threshold_errors.csv`",
        "- `subgroup_errors.csv`",
        "- `identity_errors.csv`",
        "- `hard_cases_LOCAL_RESTRICTED.csv` (local only; do not upload)",
        "- `summary.json`",
        "- `fig1_class_error_rates.png`",
        "- `fig2_fake_errors_by_provider.png`",
        "- `fig3_score_scatter.png`",
        "- `fig4_identity_error_distribution.png`",
    ]
    (out_dir / "ERROR_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out_dir), "best_mean_eer_condition": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
