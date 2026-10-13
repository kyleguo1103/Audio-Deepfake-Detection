"""Step 4: figures + RESULTS.md for one run directory (called automatically by step 3).

Usage (to rebuild): python src/step4_report.py [--run outputs/run_YYYYmmdd_HHMMSS]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pilot_common import LABEL_TO_INT, eer_and_threshold, load_config, read_csv

# reference palette (dataviz skill): categorical slots 1-3, recessive ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABELS = {"none": "No augmentation", "codec": "+ codec", "noise": "+ noise", "channel": "+ channel",
          "mix": "+ mix (1 family/clip)", "all": "+ all 3 families"}


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def roc_points(y, p):
    order = np.argsort(-p)
    ys = y[order]
    tpr = np.r_[0, np.cumsum(ys) / ys.sum()]
    fpr = np.r_[0, np.cumsum(1 - ys) / (1 - ys).sum()]
    return fpr, tpr


def build_report(run_dir: Path, cfg) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    record = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
    shortcut = json.loads((run_dir / "shortcut_diagnostic.json").read_text(encoding="utf-8"))
    conds = summary["conditions"]
    names = [c["condition"] for c in conds]
    control = summary["primary_comparison"]["control"]
    treatment = summary["primary_comparison"]["treatment"]
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Figure 1: target EER by condition with speaker-cluster CI
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(conds))
    means = np.array([c["target_eer_mean"] for c in conds]) * 100
    lo = np.array([c["target_eer_ci_speaker"][0] for c in conds]) * 100
    hi = np.array([c["target_eer_ci_speaker"][1] for c in conds]) * 100
    colors = [INK2 if n == control else (ORANGE if n == treatment else BLUE) for n in names]
    ax.bar(x, means, width=0.62, color=colors, edgecolor=SURFACE, linewidth=2, zorder=2)
    ax.errorbar(x, means, yerr=[means - lo, hi - means], fmt="none", ecolor=INK, elinewidth=1.2, capsize=4, zorder=3)
    for xi, m, h in zip(x, means, hi):
        ax.text(xi, h + 0.6, f"{m:.1f}%", ha="center", va="bottom", fontsize=10, color=INK)
    ax.axhline(means[names.index(control)], color=INK2, linewidth=1, linestyle="--", zorder=1)
    ax.set_xticks(x, [LABELS.get(n, n) for n in names], rotation=18, ha="right")
    ax.set_ylabel("Target EER (%)  - lower is better")
    ax.set_title("ASVspoof 2019 LA -> In-the-Wild: target EER by training condition\n"
                 "bars = mean of 3 seeds; whiskers = 95% speaker-cluster bootstrap CI", fontsize=11, loc="left")
    ax.set_ylim(0, max(hi) * 1.18)
    fig.savefig(fig_dir / "fig1_target_eer_by_condition.png")
    plt.close(fig)

    # Figure 2: ROC on target (seed-averaged probabilities)
    preds = read_csv(run_dir / "predictions_target.csv")
    y = np.array([LABEL_TO_INT[r["label"]] for r in preds])
    seeds = record["seeds"]

    def avg_prob(cond):  # mean classifier logit over seeds
        return np.mean([[float(r[f"score_{cond}_seed{s}"]) for r in preds] for s in seeds], axis=0)

    per_seed = read_csv(run_dir / "metrics_per_seed.csv")

    def mean_threshold(cond):
        return float(np.mean([float(r["threshold_logit"]) for r in per_seed
                              if r["condition"] == cond and r["split"] == "target"]))

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for cond, color in ((control, INK2), (treatment, ORANGE), ("all", BLUE)):
        if cond not in names:
            continue
        p = avg_prob(cond)
        fpr, tpr = roc_points(y, p)
        eer = eer_and_threshold(y, p)[0]
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{LABELS[cond]} (EER {eer * 100:.1f}%)")
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    ax.plot([0, 1], [1, 0], color=GRID, linewidth=1, linestyle=":")
    ax.set_xlabel("False positive rate (real flagged as fake)")
    ax.set_ylabel("True positive rate (fake detected)")
    ax.set_title("Target ROC (In-the-Wild)", loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(fig_dir / "fig2_target_roc.png")
    plt.close(fig)

    # Figure 3: score distributions on target, control vs treatment (logit scale, source threshold marked)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, cond in zip(axes, (control, treatment)):
        p = avg_prob(cond)
        thr = mean_threshold(cond)
        lo_b, hi_b = np.percentile(p, [0.5, 99.5])
        lo_b, hi_b = min(lo_b, thr - 0.5), max(hi_b, thr + 0.5)
        bins = np.linspace(lo_b, hi_b, 45)
        ax.hist(np.clip(p[y == 0], lo_b, hi_b), bins=bins, color=BLUE, alpha=0.75, label="real (bona fide)", density=True)
        ax.hist(np.clip(p[y == 1], lo_b, hi_b), bins=bins, color=ORANGE, alpha=0.75, label="fake (spoof)", density=True)
        ax.axvline(thr, color=INK, linewidth=1.2, linestyle="--", label="threshold fixed on source val")
        ax.set_title(LABELS[cond], loc="left", fontsize=11)
        ax.set_xlabel("Classifier score (logit; higher = more likely fake)")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)
    fig.suptitle("Target (In-the-Wild) score distributions, mean of 3 seeds", x=0.01, y=1.04, ha="left", fontsize=11)
    fig.savefig(fig_dir / "fig3_target_score_distributions.png")
    plt.close(fig)

    # Figure 4: exploratory layer profile
    profile = read_csv(run_dir / "layer_profile_EXPLORATORY.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for cond, color in ((control, INK2), (treatment, ORANGE)):
        rows = [r for r in profile if r["condition"] == cond]
        layers = [int(r["layer"]) for r in rows]
        ax.plot(layers, [float(r["target_eer"]) * 100 for r in rows], color=color, linewidth=2, marker="o",
                markersize=5, label=f"{LABELS[cond]} - target")
        ax.plot(layers, [float(r["val_eer"]) * 100 for r in rows], color=color, linewidth=1.2, linestyle="--",
                label=f"{LABELS[cond]} - source validation")
    ax.set_xlabel("XLS-R hidden-state index (0 = before transformer)")
    ax.set_ylabel("EER (%)")
    ax.set_title("EXPLORATORY layer profile (seed 42) - not used for model selection", loc="left", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(fig_dir / "fig4_layer_profile_exploratory.png")
    plt.close(fig)

    # ---------------- RESULTS.md
    def pct(v):
        return f"{v * 100:.2f}"

    def ci(pair):
        return f"[{pair[0] * 100:+.2f}, {pair[1] * 100:+.2f}]"

    prim = next(c for c in conds if c["condition"] == treatment)
    verdict_text = {"improvement_supported": "improvement supported (CI entirely below 0)",
                    "harm_supported": "augmentation made the target EER worse (CI entirely above 0)",
                    "inconclusive": "inconclusive (CI includes 0)"}[summary["primary_verdict"]]
    lines = [
        "# Pilot results: ASVspoof 2019 LA -> In-the-Wild, controlled augmentation",
        "",
        f"- Run: `{run_dir.name}`; created {record['created_at_utc']} (UTC)",
        f"- Encoder: `{record['encoder'].get('model_id')}` commit `{record['encoder'].get('resolved_commit')}` on {record['encoder'].get('gpu') or record['encoder'].get('device')}",
        f"- Target scope: **{summary['target_scope']}** (n = {record['n_target']}); source pool n = {record['n_source_pool']}, source validation n = {record['n_source_val']}",
        f"- Config SHA-256 `{record['config_sha256']}`; manifests frozen and verified before training.",
        f"- Codec encoders available for augmentation: {', '.join(record.get('codec_encoders_available', [])) or 'none recorded'} (pre-registered: libmp3lame, aac, libopus)",
    ]
    if record.get("fake_model_self_test"):
        lines += ["", "> **SOFTWARE SELF-TEST WITH A FAKE ENCODER — THESE NUMBERS ARE NOT RESEARCH RESULTS.**"]
    lines += [
        "",
        "> Pilot status: substitute source domain (ASVspoof 2019 LA instead of DeepSpeak, whose access is not yet evidenced). "
        "It tests the method and pipeline; it is not the final thesis experiment.",
        "",
        "## Primary pre-registered comparison",
        "",
        f"**{LABELS[treatment]} vs {LABELS[control]}**: target EER {pct(prim['target_eer_mean'] - prim['delta_eer_vs_none'])}% -> "
        f"{pct(prim['target_eer_mean'])}% (Δ = {prim['delta_eer_vs_none'] * 100:+.2f} percentage points; "
        f"95% speaker-cluster CI {ci(prim['delta_eer_ci_speaker'])}; clip-level CI {ci(prim['delta_eer_ci_clip'])}).",
        "",
        f"Decision rule: {summary['decision_rule']}",
        "",
        f"**Verdict: {verdict_text}.**",
        "",
        "## All conditions (target = In-the-Wild, mean ± SD over 3 seeds)",
        "",
        "| Condition | Layer | C | Train n | Target EER % [95% CI] | Δ EER vs none (pp) [95% CI] | Target AUC | Bal. acc. @src thr | Macro F1 @src thr | Src-val EER % |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for c in conds:
        lines.append(
            f"| {LABELS.get(c['condition'], c['condition'])} | {c['layer']} | {c['C']} | {c['n_train']} | "
            f"{pct(c['target_eer_mean'])} ± {pct(c['target_eer_sd'])} [{pct(c['target_eer_ci_speaker'][0])}, {pct(c['target_eer_ci_speaker'][1])}] | "
            f"{c['delta_eer_vs_none'] * 100:+.2f} {ci(c['delta_eer_ci_speaker']) if c['condition'] != control else ''} | "
            f"{c['target_roc_auc_mean']:.3f} | {c['target_balanced_accuracy_mean']:.3f} | {c['target_macro_f1_mean']:.3f} | "
            f"{pct(c['source_val_eer_mean'])} |")
    sc = shortcut
    lines += [
        "",
        "Threshold-dependent metrics use the threshold fixed at the source-validation EER point (no target labels used).",
        "",
        "## Shortcut diagnostic (duration / silence only, logistic regression)",
        "",
        "| Features | Source-val EER % | Source-val AUC | Target EER % | Target AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ("raw_duration_and_silence", "model_input_duration_only"):
        d = sc[key]
        lines.append(f"| {', '.join(d['features'])} | {pct(d['source_val_eer'])} | {d['source_val_auc']:.3f} | {pct(d['target_eer'])} | {d['target_auc']:.3f} |")
    lines += [
        "",
        sc["note"],
        "",
        "## Figures",
        "",
        "- `figures/fig1_target_eer_by_condition.png`",
        "- `figures/fig2_target_roc.png`",
        "- `figures/fig3_target_score_distributions.png`",
        "- `figures/fig4_layer_profile_exploratory.png` (exploratory; computed after the primary results were saved)",
        "",
        "## Files",
        "",
        "`selection_grid_source_validation.csv`, `selected_configuration.json`, `metrics_per_seed.csv`, "
        "`predictions_target.csv`, `predictions_source_validation.csv`, `bootstrap_target.json`, `summary.json`, "
        "`shortcut_diagnostic.json`, `layer_profile_EXPLORATORY.csv`, `run_record.json`.",
    ]
    (run_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--run", help="run directory (default: outputs/LATEST_RUN.txt)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run = Path(args.run) if args.run else cfg["_paths"]["output_dir"] / (cfg["_paths"]["output_dir"] / "LATEST_RUN.txt").read_text().strip()
    build_report(run, cfg)
    print(f"Report written to {run / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
