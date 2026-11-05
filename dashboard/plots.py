"""Matplotlib figures for the dashboard (object API only: no pyplot state, safe across requests)."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from detector import CONDITION_NAMES, CONDITIONS, PROFILE_CONDITIONS, Analysis

COLORS = {
    "none": "#5B6573",
    "codec": "#8C6BB1",
    "noise": "#1F8A70",
    "channel": "#C7832A",
    "mix": "#3566D6",
    "all": "#B5527A",
}
FAKE = "#C2412D"
REAL = "#1E8A4C"
INK = "#1F2430"
MUTED = "#6B7280"
GRID = "#E5E7EB"
SR = 16000


def _style(ax) -> None:
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _regions(a: Analysis) -> list[tuple[float, float, str]]:
    """(start_s, end_s, kind) for the parts of the clip the model did not see."""
    n = len(a.waveform)
    trail = int(round(a.info.get("trail_silence_s", 0) * SR)) if not a.info.get("trim_fallback") else 0
    trim_end = max(a.used_end, n - trail)
    out = []
    if a.used_start > 0:
        out.append((0.0, a.used_start / SR, "silence"))
    if trim_end > a.used_end:
        out.append((a.used_end / SR, trim_end / SR, "crop"))
    if n > trim_end:
        out.append((trim_end / SR, n / SR, "silence"))
    return out


def waveform_figure(a: Analysis) -> Figure:
    fig = Figure(figsize=(9.2, 4.4), dpi=110, facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.35], hspace=0.12)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    wav = a.waveform
    t_end = max(len(wav) / SR, 1e-3)

    # min/max envelope keeps long files fast to draw
    bins = min(len(wav), 2400)
    if bins > 0:
        edges = np.linspace(0, len(wav), bins + 1).astype(int)
        mins = np.array([wav[s:e].min() if e > s else 0 for s, e in zip(edges[:-1], edges[1:])])
        maxs = np.array([wav[s:e].max() if e > s else 0 for s, e in zip(edges[:-1], edges[1:])])
        tt = edges[:-1] / SR
        ax1.fill_between(tt, mins, maxs, color=INK, linewidth=0, alpha=0.85)
    peak = float(np.max(np.abs(wav))) if len(wav) else 1.0
    ax1.set_ylim(-peak * 1.1 - 1e-6, peak * 1.1 + 1e-6)
    ax1.set_yticks([])
    _style(ax1)
    ax1.tick_params(labelbottom=False)
    ax1.grid(False)

    if len(wav) >= 512:
        dither = np.random.default_rng(0).normal(scale=1e-5, size=len(wav)).astype(np.float32)  # avoids log(0) in digital silence
        loud = float(np.max(np.abs(wav))) > 1e-3
        with np.errstate(divide="ignore", invalid="ignore"):
            ax2.specgram(wav + dither, NFFT=512, Fs=SR, noverlap=352, cmap="magma", scale="dB",
                         vmin=-120 if loud else None)
    ax2.set_ylim(0, 8000)
    ax2.set_yticks([0, 2000, 4000, 6000, 8000])
    ax2.set_yticklabels(["0", "2k", "4k", "6k", "8k Hz"])
    ax2.set_xlabel("Time (s)", color=MUTED, fontsize=9)
    _style(ax2)
    ax2.grid(False)
    ax1.set_xlim(0, t_end)

    for s, e, kind in _regions(a):
        for ax in (ax1, ax2):
            ax.axvspan(s, e, facecolor="#9CA3AF" if kind == "silence" else "#F3C969",
                       edgecolor="#B7862A" if kind == "crop" else "none",
                       alpha=0.45 if ax is ax1 else 0.55, linewidth=0,
                       hatch=None if kind == "silence" else "//")
    ax1.axvspan(a.used_start / SR, a.used_end / SR, color=COLORS["mix"], alpha=0.06, linewidth=0)
    handles = [Patch(color=COLORS["mix"], alpha=0.25, label=f"Analysed ({(a.used_end - a.used_start) / SR:.2f} s)"),
               Patch(color="#9CA3AF", alpha=0.6, label="Trimmed silence")]
    if any(k == "crop" for _, _, k in _regions(a)):
        handles.append(Patch(facecolor="#F3C969", alpha=0.7, hatch="//", label="Beyond the 6 s limit"))
    ax1.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=7.5, frameon=False,
               ncol=len(handles), borderaxespad=0.2)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.12)
    return fig


def comparison_figure(a: Analysis) -> Figure:
    conds = [c for c in CONDITIONS if c in a.scores]
    fig = Figure(figsize=(6.4, 0.42 * len(conds) + 1.1), dpi=110, facecolor="white")
    ax = fig.add_subplot(111)
    scores = [a.scores[c].score for c in conds]
    y = np.arange(len(conds))[::-1]
    for yi, c, score in zip(y, conds, scores):
        fake = score >= 0.5
        ax.barh(yi, score, height=0.56, color=FAKE if fake else REAL, alpha=0.88)
        ax.text(min(score + 0.02, 0.98), yi, f"{score:.2f}  {'FAKE' if fake else 'REAL'}", va="center",
                ha="left" if score < 0.86 else "right", fontsize=8.5, color=INK if score < 0.86 else "white")
    ax.axvline(0.5, color=INK, linewidth=1.2)
    ax.text(0.5, len(conds) - 0.25, "decision line", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{CONDITION_NAMES[c]}  · stage {a.scores[c].layer}" for c in conds], fontsize=8.5, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Decision score   real side  ←  0.5  →  fake side   (not a probability)", color=MUTED, fontsize=8.5)
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.95, bottom=0.2 if len(conds) > 3 else 0.3)
    return fig


def layer_figure(a: Analysis, selected_layers: dict[str, int]) -> Figure:
    fig = Figure(figsize=(9.2, 3.4), dpi=110, facecolor="white")
    ax = fig.add_subplot(111)
    ax.axvspan(-0.6, 8.9, color=COLORS["mix"], alpha=0.035, linewidth=0)
    ax.axvspan(8.9, 16.9, color=COLORS["channel"], alpha=0.045, linewidth=0)
    ax.axvspan(16.9, 24.6, color=COLORS["noise"], alpha=0.035, linewidth=0)
    ax.axhspan(0.5, 1.0, color=FAKE, alpha=0.05, linewidth=0)
    ax.axhspan(0.0, 0.5, color=REAL, alpha=0.05, linewidth=0)
    ax.axhline(0.5, color=INK, linewidth=1)
    ax.text(0.005, 0.97, "fake side", transform=ax.transAxes, va="top", fontsize=8, color=FAKE)
    ax.text(0.005, 0.03, "real side", transform=ax.transAxes, va="bottom", fontsize=8, color=REAL)
    ax.text(4, 1.035, "early", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.text(13, 1.035, "middle", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.text(21, 1.035, "later", ha="center", va="bottom", fontsize=8, color=MUTED)
    for c in PROFILE_CONDITIONS:
        pts = sorted(a.layer_margins.get(c, {}).items())
        if not pts:
            continue
        xs, margins = zip(*pts)
        ys = [float(1.0 / (1.0 + np.exp(-np.clip(m, -30, 30)))) for m in margins]
        ax.plot(xs, ys, color=COLORS[c], linewidth=2, marker="o", markersize=3.5, label=CONDITION_NAMES[c],
                solid_capstyle="round", solid_joinstyle="round")
        sel = selected_layers.get(c)
        score_by_layer = dict(zip(xs, ys))
        if sel in score_by_layer:
            ax.plot([sel], [score_by_layer[sel]], marker="o", markersize=11, markerfacecolor="none",
                    markeredgecolor=COLORS[c], markeredgewidth=2)
    ax.set_xlim(-0.6, 24.6)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xticks(range(0, 25, 2))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("XLS-R processing stage   (rings = the stage selected before target evaluation)", color=MUTED, fontsize=8.5)
    ax.set_ylabel("Decision score\n(not probability)", color=MUTED, fontsize=8.5)
    _style(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=8, frameon=False, ncol=3)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.17)
    return fig
