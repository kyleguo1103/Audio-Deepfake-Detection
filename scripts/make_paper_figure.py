"""Redraw the paper's layer-profile figure (Fig. 1) from the committed exploratory results.

    python scripts/make_paper_figure.py            # writes paper_figures/fig_layer_profile.png

The figure is post hoc: target EER of the no-augmentation, mix and noise models with the XLS-R
hidden-state index held fixed, plus the control's source-validation EER.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "pilot_la_to_itw"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--out", default=str(ROOT / "paper_figures" / "fig_layer_profile.png"))
    args = parser.parse_args()
    run = args.run or (PILOT / "outputs" / "LATEST_RUN.txt").read_text(encoding="utf-8").strip()
    rows = list(csv.DictReader((PILOT / "outputs" / run / "exploratory_fixed_layer" / "per_layer_summary.csv")
                               .open(encoding="utf-8", newline="")))
    get = {(r["condition"], int(r["layer"])): r for r in rows}
    layers = sorted({int(r["layer"]) for r in rows})

    def series(cond, key):
        return [100 * float(get[(cond, L)][key]) for L in layers]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 8, "axes.linewidth": 0.6, "savefig.dpi": 600,
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    fig, ax = plt.subplots(figsize=(3.0, 1.45))
    ax.plot(layers, series("none", "target_eer_mean"), "k-o", ms=2.8, lw=1.0, label="No augmentation (target)")
    ax.plot(layers, series("mix", "target_eer_mean"), "k--s", ms=2.6, lw=0.9, mfc="white", label="Mix (target)")
    ax.plot(layers, series("noise", "target_eer_mean"), "k:^", ms=2.8, lw=1.0, mfc="0.6", label="Noise (target)")
    ax.plot(layers, series("none", "val_eer_mean"), "-", color="0.55", lw=0.9,
            label="No augmentation (source val.)")
    ax.annotate("selected:\nnone (2)", xy=(2, 100 * float(get[("none", 2)]["target_eer_mean"])), xytext=(5.0, 41.0),
                fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.5))
    ax.annotate("selected:\nmix (20)", xy=(20, 100 * float(get[("mix", 20)]["target_eer_mean"])), xytext=(14.4, 39.0),
                fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.5))
    ax.set_xlabel("XLS-R hidden-state index (layer)", fontsize=7)
    ax.set_ylabel("EER (%)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.set_xticks(layers)
    ax.set_ylim(0, 52)
    ax.set_xlim(-0.8, 24.8)
    ax.grid(axis="y", color="0.88", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=5.9, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.56, 0.04),
              columnspacing=0.6, handlelength=1.7, handletextpad=0.4)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
