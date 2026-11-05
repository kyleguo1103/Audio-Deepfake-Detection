"""XLS-R Audio Deepfake Detector - dashboard demo (REIT7841, Jingkai Guo).

Runs the detector from the ASVspoof 2019 LA -> In-the-Wild pilot (run_20261011_113206) behind a local
Gradio dashboard.  The audience-facing view is deliberately small: try one clip, understand the
decision, then inspect the research evidence.  Raw logits and tuning values remain available inside
closed technical-detail panels.

Usage (Anaconda Prompt, conda env "xlsr"):
    python app.py                 # opens http://127.0.0.1:7860
    python app.py --share         # also prints a temporary public link
    python app.py --retrain       # refit the probes instead of using the cache
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


def _set_hf_home_early() -> None:
    """Point Hugging Face at the pilot's model cache before gradio imports huggingface_hub."""
    if "HF_HOME" in os.environ:
        return
    root = os.environ.get("PILOT_DATA_ROOT")
    if "--data-root" in sys.argv[:-1]:
        root = sys.argv[sys.argv.index("--data-root") + 1]
    if not root:
        app_dir = Path(__file__).resolve().parent
        pilot_override = os.environ.get("PILOT_DIR")
        if "--pilot-dir" in sys.argv[:-1]:
            pilot_override = sys.argv[sys.argv.index("--pilot-dir") + 1]
        candidates = ([Path(pilot_override)] if pilot_override else []) + [
            app_dir.parent / "pilot_la_to_itw",
            app_dir.parents[1] / "06_Implementation" / "pilot_la_to_itw",
        ]
        for pilot_dir in candidates:
            cfg_path = pilot_dir / "config" / "pilot_config.json"
            if not cfg_path.exists():
                continue
            try:
                root = json.loads(cfg_path.read_text(encoding="utf-8"))["paths"]["data_root"]
                break
            except Exception:  # noqa: BLE001
                continue
        if not root:
            return
    if (Path(root) / "hf_cache").is_dir():
        os.environ["HF_HOME"] = str(Path(root) / "hf_cache")


_set_hf_home_early()

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

import plots  # noqa: E402
from detector import CONDITION_NAMES, CONDITIONS, Analysis, Engine  # noqa: E402

ENGINE: Engine | None = None
DEFAULT_MODEL = "mix"

CSS = """
.hdr {display:flex; flex-wrap:wrap; align-items:flex-end; justify-content:space-between; gap:12px; padding:4px 2px 2px}
.hdr h1 {margin:0; font-size:1.55rem; font-weight:650; letter-spacing:-0.01em}
.hdr p {margin:2px 0 0; opacity:.75; font-size:.92rem}
.chips {display:flex; flex-wrap:wrap; gap:6px}
.chip {font-size:.78rem; padding:3px 9px; border-radius:999px; border:1px solid var(--border-color-primary); white-space:nowrap}
.chip.ok {border-color:#1E8A4C; color:#1E8A4C}
.chip.warn {border-color:#B7791F; color:#B7791F}
.journey {display:grid; grid-template-columns:repeat(4, minmax(130px, 1fr)); gap:8px; margin:8px 0 14px}
.journey div {border:1px solid var(--border-color-primary); border-radius:10px; padding:10px 12px; background:var(--background-fill-secondary)}
.journey b {display:block; color:#3566D6; font-size:.76rem; text-transform:uppercase; letter-spacing:.04em; margin-bottom:3px}
.journey span {font-size:.9rem; line-height:1.35}
.vd {border-radius:12px; padding:16px 18px; border:1px solid var(--border-color-primary)}
.vd.fake {background:rgba(194,65,45,.07); border-color:rgba(194,65,45,.45)}
.vd.real {background:rgba(30,138,76,.07); border-color:rgba(30,138,76,.45)}
.vd.empty {opacity:.7}
.vd-top {display:flex; flex-wrap:wrap; align-items:baseline; gap:10px 14px}
.vd-label {font-size:2.1rem; font-weight:750; letter-spacing:.02em; line-height:1}
.vd.fake .vd-label {color:#C2412D}
.vd.real .vd-label {color:#1E8A4C}
.vd-model {font-size:.9rem; opacity:.8}
.vd-gauge {position:relative; height:10px; border-radius:999px; margin:14px 0 4px;
  background:linear-gradient(90deg, #1E8A4C 0%, rgba(140,140,140,.35) 50%, #C2412D 100%)}
.vd-pin {position:absolute; top:-5px; width:4px; height:20px; border-radius:2px; background:var(--body-text-color); transform:translateX(-50%)}
.vd-mid {position:absolute; left:50%; top:-3px; width:1px; height:16px; background:var(--body-text-color); opacity:.5}
.vd-scale {display:flex; justify-content:space-between; font-size:.75rem; opacity:.7}
.vd-stats {display:grid; grid-template-columns:repeat(auto-fit, minmax(140px, 1fr)); gap:8px 16px; margin-top:12px}
.vd-stats div {font-size:.85rem}
.vd-stats b {display:block; font-size:.72rem; font-weight:600; opacity:.7; text-transform:uppercase; letter-spacing:.04em}
.vd-stats span {font-size:1.05rem; font-variant-numeric:tabular-nums}
.vd-note {margin-top:12px; font-size:.84rem; line-height:1.5}
.vd-truth {margin-top:12px; padding:8px 10px; border-radius:8px; font-size:.88rem; border:1px dashed var(--border-color-primary)}
.vd-truth.hit {border-color:#1E8A4C} .vd-truth.miss {border-color:#C2412D}
.explain-grid {display:grid; grid-template-columns:repeat(3, minmax(190px, 1fr)); gap:10px; margin:8px 0 14px}
.explain-card {border:1px solid var(--border-color-primary); border-radius:11px; padding:12px 13px; background:var(--background-fill-secondary)}
.explain-card h4 {margin:0 0 4px; font-size:.98rem}
.explain-card p {margin:0; font-size:.84rem; line-height:1.42; opacity:.82}
.layer-stack {display:grid; grid-template-columns:1fr 1fr 1fr; border:1px solid var(--border-color-primary); border-radius:12px; overflow:hidden; margin:8px 0 10px}
.layer-stack div {padding:14px 16px; min-height:92px}
.layer-stack div+div {border-left:1px solid var(--border-color-primary)}
.layer-stack b {display:block; margin-bottom:4px}
.layer-stack span {font-size:.84rem; line-height:1.4; opacity:.8}
.layer-early {background:rgba(53,102,214,.07)} .layer-middle {background:rgba(199,131,42,.08)} .layer-late {background:rgba(31,138,112,.08)}
.plain-note {border-left:4px solid #3566D6; padding:8px 12px; background:rgba(53,102,214,.06); border-radius:4px; font-size:.88rem; line-height:1.45}
.metric-key {display:grid; grid-template-columns:repeat(3, minmax(180px, 1fr)); gap:10px; margin:8px 0}
.metric-key div {border-top:3px solid #3566D6; background:var(--background-fill-secondary); border-radius:7px; padding:10px 12px}
.metric-key b {display:block; margin-bottom:3px}.metric-key span {font-size:.83rem; line-height:1.4; opacity:.82}
.footnote {font-size:.8rem; opacity:.7}
@media(max-width:800px){.journey,.explain-grid,.metric-key{grid-template-columns:1fr}.layer-stack{grid-template-columns:1fr}.layer-stack div+div{border-left:0;border-top:1px solid var(--border-color-primary)}}
"""

CONDITION_EXPLANATIONS = {
    "none": ("Original audio only", "The control. The detector sees the source recordings without a simulated change."),
    "codec": ("Compression", "Adds one MP3, AAC or Opus copy. This imitates audio sent through apps or streaming."),
    "noise": ("Background noise", "Adds white, pink or brown noise at a controlled level. This imitates a less clean recording."),
    "channel": ("Device or room", "Changes bandwidth, equalisation or reverberation. This imitates a phone, microphone or room."),
    "mix": ("One random family", "Adds one changed copy per clip, using either codec, noise or channel. Training size stays comparable."),
    "all": ("All three copies", "Adds three separate copies per clip: one codec, one noise and one channel version."),
}


# ----------------------------------------------------------------------------- formatting

def esc(x) -> str:
    return html.escape(str(x))


def header_html() -> str:
    s = ENGINE.status()
    chips = [
        ("ok", "Ready"),
        ("ok", "Runs locally"),
        ("ok" if s["itw_available"] else "warn",
         f"Examples available: {s['itw_clips']:,}" if s["itw_available"] else "Examples unavailable"),
    ]
    chip_html = "".join(f'<span class="chip {c}">{esc(t)}</span>' for c, t in chips)
    return (
        '<div class="hdr"><div><h1>Audio Deepfake Detector</h1>'
        "<p>A research demo that shows a verdict, the evidence behind it, and where the result can be trusted.</p></div>"
        f'<div class="chips">{chip_html}</div></div>'
    )


def journey_html() -> str:
    return (
        '<div class="journey">'
        '<div><b>1 · Add audio</b><span>Upload, record, or choose a labelled example.</span></div>'
        '<div><b>2 · Prepare</b><span>Convert to mono 16 kHz, trim silence, and keep at most 6 seconds.</span></div>'
        '<div><b>3 · Read features</b><span>XLS-R turns the sound into learned feature stages.</span></div>'
        '<div><b>4 · Decide</b><span>A small classifier compares the result with a fixed decision line.</span></div>'
        '</div>'
    )


def condition_cards_html() -> str:
    cards = []
    for cond in CONDITIONS:
        title, description = CONDITION_EXPLANATIONS[cond]
        layer = ENGINE.probes[cond].layer if cond in ENGINE.probes else "?"
        cards.append(
            f'<div class="explain-card"><h4>{esc(CONDITION_NAMES[cond])}</h4>'
            f'<p><b>{esc(title)}</b><br>{esc(description)}<br><br>Reads XLS-R stage <b>{layer}</b> of 24.</p></div>'
        )
    return '<div class="explain-grid">' + "".join(cards) + '</div>'


def layer_explainer_html() -> str:
    return (
        '<div class="layer-stack">'
        '<div class="layer-early"><b>Early stages · 0–8</b><span>Tend to keep local sound details such as energy, pitch and short acoustic patterns.</span></div>'
        '<div class="layer-middle"><b>Middle stages · 10–16</b><span>Combine local details into broader speech patterns. This pilot found the clearest exploratory noise gain here.</span></div>'
        '<div class="layer-late"><b>Later stages · 18–24</b><span>Combine information over more context and can represent more abstract speech patterns.</span></div>'
        '</div>'
        '<div class="plain-note"><b>A layer is a processing stage, not a piece of the audio.</b> '
        'Higher does not automatically mean better. The dashboard tests several stages because the same clip can cross the decision line at different stages.</div>'
    )


def metric_key_html() -> str:
    return (
        '<div class="metric-key">'
        '<div><b>Decision score</b><span>Below 0.5 is the real side. At or above 0.5 is the fake side. It is not a probability.</span></div>'
        '<div><b>Versions agreeing</b><span>How many of the six training versions make the same decision. Agreement is not proof of correctness.</span></div>'
        '<div><b>Ground truth</b><span>Shown only for the labelled In-the-Wild examples. It checks the output and was not used for training.</span></div>'
        '</div>'
    )


def empty_verdict(message: str = "Upload or record a clip, or draw a random In-the-Wild clip, then press Detect.") -> str:
    return f'<div class="vd empty"><div class="vd-note">{esc(message)}</div></div>'


def verdict_html(a: Analysis, cond: str) -> str:
    if cond not in a.scores:
        cond = next(iter(a.scores))
    s = a.scores[cond]
    kind = "fake" if s.fake else "real"
    n_fake = sum(1 for x in a.scores.values() if x.fake)
    n = len(a.scores)
    agree = n_fake if s.fake else n - n_fake
    pin = min(98.0, max(2.0, s.score * 100))
    analysed = (a.used_end - a.used_start) / 16000
    parts = [
        f'<div class="vd {kind}">',
        f'<div class="vd-top"><span class="vd-label">LIKELY {kind.upper()}</span>'
        f'<span class="vd-model">Decision from {esc(s.name)} · reads XLS-R stage {s.layer} of 24</span></div>',
        f'<div class="vd-gauge"><div class="vd-mid"></div><div class="vd-pin" style="left:{pin:.1f}%"></div></div>',
        '<div class="vd-scale"><span>more like real</span><span>decision line · 0.5</span><span>more like fake</span></div>',
        '<div class="vd-stats">',
        f"<div><b>Decision score</b><span>{s.score:.2f}</span></div>",
        f"<div><b>Training versions agreeing</b><span>{agree} of {n}</span></div>",
        f"<div><b>Audio used</b><span>{analysed:.2f} s</span></div>",
        "</div>",
    ]
    gt = a.ground_truth
    if gt:
        truth = gt["label"]
        hit = (truth == "fake") == s.fake
        parts.append(
            f'<div class="vd-truth {"hit" if hit else "miss"}"><b>Known label: {esc(truth.upper())}</b> '
            f"→ this decision is <b>{'correct' if hit else 'wrong'}</b>. "
            "The label is shown only after analysis and was not used to train the detector.</div>")
    parts.append(
        '<div class="vd-note footnote"><b>How to read it:</b> 0.5 is the fixed decision line. The score shows which side '
        'the model is on and how far it moved after a mathematical rescaling. It is not “50% fake” and it is not calibrated confidence.</div>')
    parts.append("</div>")
    return "".join(parts)


def comparison_df(a: Analysis) -> pd.DataFrame:
    rows = []
    has_rec = any(s.recorded_logit is not None for s in a.scores.values())
    for cond in CONDITIONS:
        if cond not in a.scores:
            continue
        s = a.scores[cond]
        row = {
            "Training version": s.name,
            "XLS-R stage": s.layer,
            "C": s.C,
            "Logit": round(s.logit, 3),
            "Threshold": round(s.threshold, 3),
            "Margin": round(s.margin, 3),
            "Verdict": "FAKE" if s.fake else "REAL",
            "Decision score": round(s.score, 3),
            "Pilot ITW EER %": round(s.pilot_target_eer * 100, 1) if s.pilot_target_eer is not None else None,
        }
        if has_rec:
            row["Pilot run logit"] = round(s.recorded_logit, 3) if s.recorded_logit is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


def clip_md(a: Analysis) -> str:
    info = a.info
    used = (a.used_end - a.used_start) / 16000
    trim = ("trimming skipped (less than 0.5 s would remain)" if info.get("trim_fallback")
            else f"{info.get('lead_silence_s', 0):.2f} s leading and {info.get('trail_silence_s', 0):.2f} s trailing silence trimmed")
    decoded = ("decoded by ffmpeg to 16 kHz mono" if info.get("decoder") == "ffmpeg"
               else f"original {info.get('orig_sr', '?')} Hz, resampled to 16 kHz")
    return (f"**What the model received:** {esc(a.source)} · {a.duration_s:.2f} s · {decoded} · "
            f"{trim} · **{used:.2f} s analysed**. The plot below checks the input and analysed region; it does not by itself prove that the voice is fake.")


def outputs_for(a: Analysis, cond: str):
    selected_layers = {c: p.layer for c, p in ENGINE.probes.items()}
    return (verdict_html(a, cond), plots.waveform_figure(a), plots.comparison_figure(a), comparison_df(a),
            plots.layer_figure(a, selected_layers), clip_md(a))


def blank_outputs(message: str):
    return (empty_verdict(message), None, None, None, None, "")


# ----------------------------------------------------------------------------- handlers

def on_detect(audio_path, cond):
    if not audio_path:
        return (*blank_outputs("Upload or record a clip first, or draw a random In-the-Wild clip."), None)
    try:
        a = ENGINE.analyse_path(audio_path)
    except Exception as exc:  # noqa: BLE001
        return (*blank_outputs(f"Could not analyse this file: {exc}"), None)
    return (*outputs_for(a, cond), a)


def on_draw(label, speaker, cond):
    try:
        path, _ = ENGINE.draw_itw(label or "any", None if speaker in (None, "", "__any__") else speaker)
        a = ENGINE.analyse_path(path)
    except Exception as exc:  # noqa: BLE001
        return (gr.update(), *blank_outputs(f"Could not draw an In-the-Wild clip: {exc}"), None)
    return (path, *outputs_for(a, cond), a)


def on_model_change(a, cond):
    if a is None:
        return gr.update()
    return verdict_html(a, cond)


def on_batch(files, cond, progress=gr.Progress()):
    paths = []
    for f in files or []:
        p = f if isinstance(f, str) else getattr(f, "name", None) or getattr(f, "path", None)
        if p:
            paths.append(p)
    if not paths:
        return '<div class="vd empty"><div class="vd-note">Add one or more audio files first.</div></div>', None, None
    rows = []
    for i, p in enumerate(paths):
        progress((i, len(paths)), desc=f"Scoring {Path(p).name}")
        row = {"File": Path(p).name}
        try:
            a = ENGINE.analyse_path(p)
            s = a.scores[cond]
            row.update({"Duration s": round(a.duration_s, 2), "Analysed s": round((a.used_end - a.used_start) / 16000, 2),
                        "Verdict": "FAKE" if s.fake else "REAL", "Fake score": round(s.score, 3),
                        "Margin": round(s.margin, 3),
                        "Models saying fake": f"{sum(x.fake for x in a.scores.values())}/{len(a.scores)}"})
            for c in CONDITIONS:
                if c in a.scores:
                    x = a.scores[c]
                    row[f"{c} L{x.layer}"] = f"{'FAKE' if x.fake else 'REAL'} {x.margin:+.2f}"
        except Exception as exc:  # noqa: BLE001
            row.update({"Verdict": "ERROR", "Models saying fake": str(exc)[:120]})
        rows.append(row)
    df = pd.DataFrame(rows)
    out = Path(tempfile.gettempdir()) / "xlsr_dashboard_batch"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"batch_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    n_fake = int((df["Verdict"] == "FAKE").sum())
    n_real = int((df["Verdict"] == "REAL").sum())
    n_err = int((df["Verdict"] == "ERROR").sum())
    probe = ENGINE.probes[cond]
    summary = (f'<div class="vd"><div class="vd-top"><span class="vd-label" style="font-size:1.5rem">{n_fake} fake · {n_real} real'
               f'{f" · {n_err} error" + ("s" if n_err > 1 else "") if n_err else ""}</span>'
               f'<span class="vd-model">{esc(CONDITION_NAMES[cond])} · layer {probe.layer} · {len(paths)} files</span></div></div>')
    return summary, df, str(csv_path)


# ----------------------------------------------------------------------------- pilot results tab

def results_markdown() -> str:
    s = ENGINE.summary
    lines = [
        f"### Pre-registered run `{ENGINE.run_id}`",
        f"ASVspoof 2019 LA (source) → In-the-Wild (all {ENGINE.run_record.get('n_target', '?'):,} clips, target). "
        f"Mean of {len(ENGINE.cfg['seeds'])} seeds; 95% CIs from 1,000 speaker-cluster bootstrap resamples.",
        "",
        f"**Decision rule:** {s['decision_rule']}  ",
        f"**Verdict for {s['primary_comparison']['treatment']} vs {s['primary_comparison']['control']}:** "
        f"{s['primary_verdict'].replace('_', ' ')}.",
        "",
        "| Condition | Layer | C | Target EER % | 95% CI | Δ vs none (pp) | Δ 95% CI | AUC | Bal. acc. |",
        "|---|---:|---:|---:|---|---:|---|---:|---:|",
    ]
    for c in s["conditions"]:
        lo, hi = c["target_eer_ci_speaker"]
        dlo, dhi = c["delta_eer_ci_speaker"]
        delta = "—" if c["condition"] == "none" else f"{c['delta_eer_vs_none'] * 100:+.1f}"
        dci = "—" if c["condition"] == "none" else f"[{dlo * 100:+.1f}, {dhi * 100:+.1f}]"
        lines.append(f"| {CONDITION_NAMES.get(c['condition'], c['condition'])} | {c['layer']} | {c['C']:g} | "
                     f"{c['target_eer_mean'] * 100:.2f} | {lo * 100:.1f}–{hi * 100:.1f} | {delta} | {dci} | "
                     f"{c['target_roc_auc_mean']:.3f} | {c['target_balanced_accuracy_mean']:.3f} |")
    la_path = ENGINE.run_dir / "exploratory_fixed_layer" / "layer_averaged_summary.json"
    if la_path.exists():
        la = json.loads(la_path.read_text(encoding="utf-8"))
        lines += [
            "",
            "### Matched layers (post hoc, exploratory)",
            f"Each condition refit at the same {len(la['layers'])} layers; change vs no augmentation averaged over layers.",
            "",
            "| Condition | Layer-averaged EER % | Δ vs none (pp) | 95% CI | Better at |",
            "|---|---:|---:|---|---:|",
        ]
        for cond, v in la["conditions"].items():
            ci = v.get("delta_ci_speaker_cluster")
            lines.append(
                f"| {CONDITION_NAMES.get(cond, cond)} | {v['layer_averaged_target_eer'] * 100:.1f} | "
                f"{'—' if cond == 'none' else format(v['layer_averaged_delta_vs_none'] * 100, '+.1f')} | "
                f"{'—' if not ci else f'[{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}]'} | "
                f"{'—' if v.get('layers_better_than_none') is None else str(v['layers_better_than_none']) + '/' + str(len(la['layers']))} |")
        lines.append(f"\nReading: *{la.get('reading', '')}*.")
    return "\n".join(lines)


def results_gallery() -> list[tuple[str, str]]:
    captions = {
        "fig1_target_eer_by_condition.png": "Target EER by condition at the selected layer",
        "fig2_target_roc.png": "ROC curves on In-the-Wild",
        "fig3_target_score_distributions.png": "Score distributions on In-the-Wild",
        "fig4_layer_profile_exploratory.png": "Layer profile (exploratory)",
        "figA_fixed_layer_delta_heatmap.png": "Change vs none by layer (exploratory)",
        "figB_layer_averaged_delta.png": "Layer-averaged change with 95% CI (exploratory)",
        "figC_layer_profile_none_noise_mix.png": "Target vs source-validation EER by layer (exploratory)",
    }
    found = list((ENGINE.run_dir / "figures").glob("*.png")) + list(
        (ENGINE.run_dir / "exploratory_fixed_layer" / "figures").glob("*.png"))
    out_dir = Path(tempfile.gettempdir()) / "xlsr_dashboard_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(found, key=lambda x: x.name):
        dst = out_dir / p.name
        shutil.copyfile(p, dst)  # Gradio serves files from its temp folder without extra permissions
        items.append((str(dst), captions.get(p.name, p.stem)))
    return items


ABOUT_MD = """
### What is fixed in this demo
- **Audio preparation:** mono 16 kHz, trim leading and trailing silence below −40 dB, keep the first 6 seconds, then normalise.
- **Feature extractor:** frozen `facebook/wav2vec2-large-xlsr-53`. The dashboard reads one of 25 feature stages, indexed 0 to 24.
- **Small classifier:** class-balanced logistic regression. It turns the chosen feature stage into a real-side or fake-side decision.
- **Decision line:** fixed using ASVspoof 2019 LA development data. In-the-Wild labels were not used to train, choose a layer, or set the line.

### What this demo cannot tell you
- The decision score is not a calibrated probability and the six training versions are not six independent detectors.
- Source attacks are 2019-era TTS and voice conversion. Modern voice clones may behave differently.
- The exploratory noise pattern still needs an independent, pre-registered confirmation.
- This is a research prototype, not a forensic tool and not evidence about a real recording.

Code and raw results: github.com/kyleguo1103/Audio-Deepfake-Detection
"""


def research_summary_html() -> str:
    return (
        '<div class="explain-grid">'
        '<div class="explain-card"><h4>Confirmed pilot comparison</h4><p>Mix showed 13.1 percentage points lower EER than no augmentation in the pre-registered run. '
        'However, Mix used layer 20 and the control used layer 2, so this is not a pure augmentation effect.</p></div>'
        '<div class="explain-card"><h4>Exploratory fixed-layer pattern</h4><p>When conditions were compared at the same layer, Mix averaged 0.0 points of gain. Noise averaged 4.0 points lower EER and was better at 12 of 13 layers.</p></div>'
        '<div class="explain-card"><h4>What must happen next</h4><p>Pre-declare a target-independent layer policy and test the noise condition on independent data before making a confirmatory claim.</p></div>'
        '</div>'
    )


# ----------------------------------------------------------------------------- layout

def build_ui() -> gr.Blocks:
    choices = ENGINE.model_choices()
    default = DEFAULT_MODEL if any(v == DEFAULT_MODEL for _, v in choices) else choices[0][1]
    speakers = [("Any speaker", "__any__")] + [(s, s) for s in ENGINE.itw.speakers()]
    itw_ok = ENGINE.itw.available

    with gr.Blocks(title="XLS-R Deepfake Detector · REIT7841", theme=gr.themes.Soft(primary_hue="indigo"), css=CSS) as demo:
        gr.HTML(header_html())
        state = gr.State(None)

        with gr.Tab("Try the detector"):
            gr.HTML(journey_html())
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=320):
                    audio_in = gr.Audio(label="1. Add an audio clip", type="filepath", sources=["upload", "microphone"])
                    model_dd = gr.Dropdown(choices=choices, value=default, label="2. Choose the training version for the main verdict",
                                           info="The detector design stays the same. Only the audio used during training changes.")
                    detect_btn = gr.Button("3. Analyse audio", variant="primary", size="lg")
                    with gr.Accordion("Or choose a labelled research example", open=True):
                        if itw_ok:
                            gr.Markdown(f"Choose from {len(ENGINE.itw.rows):,} In-the-Wild clips. The known label appears only after the detector has run.",
                                        elem_classes="footnote")
                        else:
                            gr.Markdown(f"Unavailable: `{ENGINE.itw.reason}`", elem_classes="footnote")
                        label_radio = gr.Radio(choices=[("Any", "any"), ("Known real", "real"), ("Known fake", "fake")],
                                               value="any", label="Example type", interactive=itw_ok)
                        speaker_dd = gr.Dropdown(choices=speakers, value="__any__", label="Optional speaker filter", interactive=itw_ok)
                        draw_btn = gr.Button("Choose and analyse example", interactive=itw_ok)
                with gr.Column(scale=6, min_width=420):
                    verdict = gr.HTML(empty_verdict())
                    gr.HTML(metric_key_html())
            gr.Markdown("### What part of the recording was used")
            clip_info = gr.Markdown("")
            wave_plot = gr.Plot(label="Waveform and spectrogram · grey and hatched regions are not analysed")
            gr.Markdown("### Does the decision change with the training audio?")
            cmp_plot = gr.Plot(label="The same clip through six training versions")
            gr.Markdown("The six bars use the same detector design. Only the clean or augmented training copies differ. "
                        "A score on the other side of 0.5 changes the verdict; the score is not a probability.", elem_classes="footnote")
            gr.Markdown("### Does the decision change with the XLS-R stage?")
            gr.HTML(layer_explainer_html())
            layer_plot = gr.Plot(label="Decision score across XLS-R stages · no augmentation, noise and mix")
            with gr.Accordion("Technical details for reproducibility", open=False):
                gr.Markdown("These values reproduce the frozen research run. They are not needed to understand the main verdict.")
                cmp_df = gr.Dataframe(label="Raw scores and fixed thresholds", interactive=False, wrap=True)

            outs = [verdict, wave_plot, cmp_plot, cmp_df, layer_plot, clip_info, state]
            detect_btn.click(on_detect, inputs=[audio_in, model_dd], outputs=outs)
            draw_btn.click(on_draw, inputs=[label_radio, speaker_dd, model_dd], outputs=[audio_in] + outs)
            model_dd.change(on_model_change, inputs=[state, model_dd], outputs=[verdict])

        with gr.Tab("Understand the elements"):
            gr.Markdown("## One detector design, six training versions")
            gr.Markdown("Augmentation changes the **training copy**, not the audio you upload for detection. It teaches the classifier to see the same real or fake label under simulated listening conditions.")
            gr.HTML(condition_cards_html())
            gr.Markdown("## What an XLS-R layer means")
            gr.HTML(layer_explainer_html())
            gr.Markdown("## What every displayed number means")
            gr.HTML(metric_key_html())
            with gr.Accordion("Exact implementation", open=False):
                gr.Markdown(ABOUT_MD)
                s = ENGINE.status()
                gr.Markdown(f"Run `{s['run_id']}` · encoder `{s['encoder']}` @ `{s['encoder_commit'] or '?'}` · "
                            f"device `{s['device']}` · config SHA-256 `{ENGINE.cfg['_config_sha256'][:12]}…`.", elem_classes="footnote")

        with gr.Tab("Research evidence"):
            gr.Markdown("## What the pilot supports, and what it does not")
            gr.HTML(research_summary_html())
            gr.Markdown("**EER in plain language:** the error rate at the operating point where false accepts and false rejects are equal. Lower is better. "
                        "A percentage-point change is the direct difference between two EER percentages.")
            with gr.Accordion("Full result tables", open=False):
                gr.Markdown(results_markdown())
            with gr.Accordion("All pilot figures", open=False):
                gallery = results_gallery()
                if gallery:
                    gr.Gallery(value=gallery, label="Figures from the frozen pilot run", columns=2, height="auto",
                               object_fit="contain")
            gr.Markdown(f"Probes: `{ENGINE.probe_source}`. Config SHA-256 `{ENGINE.cfg['_config_sha256'][:12]}…`.",
                        elem_classes="footnote")
    return demo


def main() -> int:
    global ENGINE
    parser = argparse.ArgumentParser(description="XLS-R deepfake detector dashboard")
    parser.add_argument("--pilot-dir", help="path to 06_Implementation/pilot_la_to_itw (default: found from this file)")
    parser.add_argument("--data-root", help="override the data_root recorded in the pilot config")
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    parser.add_argument("--retrain", action="store_true", help="refit the probes instead of loading the cache")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach it from other devices on the network")
    parser.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    args = parser.parse_args()
    ENGINE = Engine(args.pilot_dir, args.data_root, args.device, retrain=args.retrain)
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=True,
                allowed_paths=[str(ENGINE.run_dir), tempfile.gettempdir()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
