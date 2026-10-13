"""Figures for the EXPLORATORY fixed-layer analysis (reads <run>/exploratory_fixed_layer/*)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pilot_common import load_config
from step4_report import AQUA, BLUE, GRID, INK, INK2, ORANGE, SURFACE, _style

RED, NEUTRAL = "#e34948", "#f0efec"
NAMES = {"codec": "+ codec", "noise": "+ noise", "channel": "+ channel", "mix": "+ mix (primary)", "all": "+ all 3"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run")
    args = parser.parse_args()
    cfg = load_config()
    out_root = cfg["_paths"]["output_dir"]
    run = Path(args.run) if args.run else out_root / (out_root / "LATEST_RUN.txt").read_text().strip()
    ex = run / "exploratory_fixed_layer"
    rows = list(csv.DictReader((ex / "per_layer_summary.csv").open(encoding="utf-8")))
    summary = json.loads((ex / "layer_averaged_summary.json").read_text(encoding="utf-8"))
    layers = sorted({int(r["layer"]) for r in rows})
    conds = ["codec", "noise", "channel", "mix", "all"]
    get = {(r["condition"], int(r["layer"])): r for r in rows}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    _style(plt)
    figs = ex / "figures"
    figs.mkdir(exist_ok=True)

    # A: heatmap of delta EER (pp) vs no augmentation at the same layer
    delta = np.array([[100 * float(get[(c, L)]["delta_vs_none_same_layer"]) for L in layers] for c in conds])
    lim = max(10.0, float(np.ceil(np.abs(delta).max())))
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, NEUTRAL, RED])
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.grid(False)
    im = ax.imshow(delta, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(conds)):
        for j in range(len(layers)):
            v = delta[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if abs(v) > 0.6 * lim else INK)
    ax.set_xticks(range(len(layers)), [str(L) for L in layers])
    ax.set_yticks(range(len(conds)), [NAMES[c] for c in conds])
    ax.set_xlabel("XLS-R hidden-state index (same layer for both models)")
    ax.set_title("EXPLORATORY: change in target EER vs no augmentation at the same layer (percentage points; blue = better)",
                 loc="left", fontsize=10.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Δ EER (pp)")
    fig.savefig(figs / "figA_fixed_layer_delta_heatmap.png")
    plt.close(fig)

    # B: layer-averaged delta with speaker-cluster CI
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ys = np.arange(len(conds))[::-1]
    for y, c in zip(ys, conds):
        info = summary["conditions"][c]
        d = 100 * info["layer_averaged_delta_vs_none"]
        lo, hi = (100 * v for v in info["delta_ci_speaker_cluster"])
        color = BLUE if hi < 0 else INK2
        ax.plot([lo, hi], [y, y], color=color, linewidth=2)
        ax.plot([d], [y], "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(hi + 0.4, y, f"{d:+.1f} pp  [{lo:+.1f}, {hi:+.1f}]  better at {info['layers_better_than_none']}/{len(layers)} layers",
                va="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(ys, [NAMES[c] for c in conds])
    ax.set_xlabel("Layer-averaged Δ target EER vs no augmentation (pp), 95% speaker-cluster CI")
    ax.set_xlim(-8, 12)
    ax.set_title("EXPLORATORY: augmentation effect averaged over 13 fixed layers", loc="left", fontsize=10.5)
    fig.savefig(figs / "figB_layer_averaged_delta.png")
    plt.close(fig)

    # C: target EER by layer for none / noise / mix, with source-validation EER
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for c, color, label in (("none", INK2, "No augmentation"), ("noise", AQUA, "+ noise"), ("mix", ORANGE, "+ mix (primary)")):
        t = [100 * float(get[(c, L)]["target_eer_mean"]) for L in layers]
        v = [100 * float(get[(c, L)]["val_eer_mean"]) for L in layers]
        ax.plot(layers, t, color=color, linewidth=2, marker="o", markersize=5, label=f"{label} — target")
        ax.plot(layers, v, color=color, linewidth=1.2, linestyle="--", label=f"{label} — source validation")
    ax.set_xlabel("XLS-R hidden-state index")
    ax.set_ylabel("EER (%), mean of 3 seeds")
    ax.set_title("Source validation is near 0% at every layer; target EER varies 16–48%", loc="left", fontsize=10.5)
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 52)
    fig.savefig(figs / "figC_layer_profile_none_noise_mix.png")
    plt.close(fig)
    print(f"figures written to {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
