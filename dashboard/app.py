"""XLS-R Audio Deepfake Detector - dashboard demo (REIT7841, Jingkai Guo).

Runs the detector from the ASVspoof 2019 LA -> In-the-Wild pilot (run_20261011_113206) behind a local
Gradio dashboard:

  Detect         upload or record a clip, or draw a random In-the-Wild clip with its true label;
                 verdict, waveform + spectrogram, all six models side by side, and the score at every layer
  Batch          score many files at once and download the table as CSV
  Pilot results  the pre-registered and matched-layer results with the run's figures
  About          what the detector is, how it was trained, and its limits

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
.footnote {font-size:.8rem; opacity:.7}
"""


# ----------------------------------------------------------------------------- formatting

def esc(x) -> str:
    return html.escape(str(x))


def header_html() -> str:
    s = ENGINE.status()
    chips = [
        ("ok", f"Run {s['run_id']}"),
        ("ok" if s["manifests_ok"] else "warn", "Manifests verified" if s["manifests_ok"] else "Manifest check failed"),
        ("ok", f"{s['encoder']} @ {s['encoder_commit'] or '?'}"),
        ("ok" if "cpu" not in str(s["device"]).lower() else "warn", f"Device: {s['device']}"),
        ("ok" if s["itw_available"] else "warn",
         f"In-the-Wild: {s['itw_clips']:,} clips" if s["itw_available"] else "In-the-Wild unavailable"),
    ]
    chip_html = "".join(f'<span class="chip {c}">{esc(t)}</span>' for c, t in chips)
    return (
        '<div class="hdr"><div><h1>XLS-R Audio Deepfake Detector</h1>'
        "<p>REIT7841 thesis demo · Jingkai Guo · frozen XLS-R-53 features + logistic-regression probes "
        "trained on ASVspoof 2019 LA</p></div>"
        f'<div class="chips">{chip_html}</div></div>'
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
    eer = f"{s.pilot_target_eer * 100:.1f}%" if s.pilot_target_eer is not None else "—"
    bal = f"{s.pilot_bal_acc:.2f}" if s.pilot_bal_acc is not None else "—"
    pin = min(98.0, max(2.0, s.score * 100))
    parts = [
        f'<div class="vd {kind}">',
        f'<div class="vd-top"><span class="vd-label">{kind.upper()}</span>'
        f'<span class="vd-model">{esc(s.name)} · XLS-R layer {s.layer} · C = {s.C:g}</span></div>',
        f'<div class="vd-gauge"><div class="vd-mid"></div><div class="vd-pin" style="left:{pin:.1f}%"></div></div>',
        '<div class="vd-scale"><span>real</span><span>threshold</span><span>fake</span></div>',
        '<div class="vd-stats">',
        f"<div><b>Fake score</b><span>{s.score:.2f}</span></div>",
        f"<div><b>Logit vs threshold</b><span>{s.logit:+.2f} / {s.threshold:+.2f}</span></div>",
        f"<div><b>Models agreeing</b><span>{agree} of {n}</span></div>",
        f"<div><b>Pilot EER on In-the-Wild</b><span>{eer}</span></div>",
        f"<div><b>Pilot balanced acc.</b><span>{bal}</span></div>",
        "</div>",
    ]
    gt = a.ground_truth
    if gt:
        truth = gt["label"]
        hit = (truth == "fake") == s.fake
        rec = s.recorded_logit
        rec_txt = ""
        if rec is not None:
            diff = abs(rec - s.logit)
            rec_txt = (f" Pilot run logit for this clip: {rec:+.3f} (live {s.logit:+.3f}, "
                       f"{'matches' if diff < 0.25 else 'differs by ' + format(diff, '.3f')}).")
        parts.append(
            f'<div class="vd-truth {"hit" if hit else "miss"}"><b>In-the-Wild ground truth: {esc(truth.upper())}</b> '
            f"· speaker {esc(gt['speaker_id'])} · clip {esc(gt['sample_id'])} → "
            f"<b>{'correct' if hit else 'wrong'}</b>.{esc(rec_txt)}</div>")
    parts.append(
        '<div class="vd-note footnote">Fake score = sigmoid(logit − threshold): 0.5 sits on the threshold fixed at the '
        "source-validation EER point. It is not a calibrated probability. Models were trained only on ASVspoof 2019 LA.</div>")
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
            "Model": s.name,
            "Layer": s.layer,
            "C": s.C,
            "Logit": round(s.logit, 3),
            "Threshold": round(s.threshold, 3),
            "Margin": round(s.margin, 3),
            "Verdict": "FAKE" if s.fake else "REAL",
            "Fake score": round(s.score, 3),
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
    dev = ENGINE.status()["device"]
    decoded = ("decoded by ffmpeg to 16 kHz mono" if info.get("decoder") == "ffmpeg"
               else f"original {info.get('orig_sr', '?')} Hz, resampled to 16 kHz")
    return (f"**{esc(a.source)}** · {a.duration_s:.2f} s ({decoded}) · "
            f"{trim} · **{used:.2f} s analysed** (first 6 s after trimming, as in the pilot) · "
            f"XLS-R forward pass {a.seconds_encoder * 1000:.0f} ms on {esc(dev)}")


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
### What this detector is
- **Features:** `facebook/wav2vec2-large-xlsr-53`, frozen. Each clip gives 25 hidden states (index 0 = input to the first
  Transformer block, *k* = output of block *k*), mean-pooled over time to 1,024 numbers each.
- **Classifier:** StandardScaler + class-balanced logistic regression on **one** hidden state. Each of the six models
  (no augmentation, codec, noise, channel, mix, all) uses the layer and regularisation C that the pilot chose on
  ASVspoof 2019 LA development data only.
- **Training data:** 2,400 ASVspoof 2019 LA training clips (the seed-42 80% subset of the 3,000-clip pool; attacks A01–A06),
  plus one augmented copy per family for the augmented models.
- **Threshold:** the score where false-accept and false-reject rates are equal on LA development data (EER point).
  No In-the-Wild label is used to train, tune or set thresholds.
- **Preprocessing (identical to the pilot):** mono 16 kHz → trim silence below −40 dB → keep the first 6 s →
  per-utterance zero-mean, unit-variance.

### How to read the screen
- **Verdict:** the chosen model's decision. *Fake score* is sigmoid(logit − threshold), so 0.5 is the threshold.
- **Model comparison:** the same clip through all six models. Disagreement is common on out-of-domain audio.
- **Score by layer:** probes trained at every even layer for no augmentation, noise and mix. Whether a clip looks fake
  can flip with the layer, which is the paper's main point: layer choice matters as much as augmentation.
- **In-the-Wild clips:** drawn from the pilot's frozen target manifest. The ground truth and the logit recorded in the
  pilot run are shown, so you can see the live system reproduces the published numbers.

### Limits
- Source attacks are 2019-era TTS/VC; modern voice clones can look very different. On In-the-Wild the best of these
  models still has about 17% EER, and the no-augmentation model flags almost every clip as fake.
- The noise result is exploratory (post hoc) and still needs a confirmatory run.
- A research prototype for discussion. Not a forensic tool, and not evidence about any real recording.

Code and raw results: github.com/kyleguo1103/Audio-Deepfake-Detection
(pilot release: tag v0.1-pilot; dashboard: main branch).
"""


# ----------------------------------------------------------------------------- layout

def build_ui() -> gr.Blocks:
    choices = ENGINE.model_choices()
    default = DEFAULT_MODEL if any(v == DEFAULT_MODEL for _, v in choices) else choices[0][1]
    speakers = [("Any speaker", "__any__")] + [(s, s) for s in ENGINE.itw.speakers()]
    itw_ok = ENGINE.itw.available

    with gr.Blocks(title="XLS-R Deepfake Detector · REIT7841", theme=gr.themes.Soft(primary_hue="indigo"), css=CSS) as demo:
        gr.HTML(header_html())
        state = gr.State(None)

        with gr.Tab("Detect"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=320):
                    audio_in = gr.Audio(label="Audio clip", type="filepath", sources=["upload", "microphone"])
                    model_dd = gr.Dropdown(choices=choices, value=default, label="Model for the verdict",
                                           info="All six models are always scored; this one drives the verdict.")
                    detect_btn = gr.Button("Detect", variant="primary", size="lg")
                    with gr.Accordion("Random In-the-Wild clip (with ground truth)", open=True):
                        if itw_ok:
                            gr.Markdown(f"Draws from the {len(ENGINE.itw.rows):,} clips the pilot scored. "
                                        "The clip loads into the player above and is analysed straight away.",
                                        elem_classes="footnote")
                        else:
                            gr.Markdown(f"Unavailable: `{ENGINE.itw.reason}`", elem_classes="footnote")
                        label_radio = gr.Radio(choices=[("Any", "any"), ("Real", "real"), ("Fake", "fake")],
                                               value="any", label="True label", interactive=itw_ok)
                        speaker_dd = gr.Dropdown(choices=speakers, value="__any__", label="Speaker", interactive=itw_ok)
                        draw_btn = gr.Button("Draw and analyse", interactive=itw_ok)
                with gr.Column(scale=6, min_width=420):
                    verdict = gr.HTML(empty_verdict())
                    clip_info = gr.Markdown("")
                    wave_plot = gr.Plot(label="Waveform and spectrogram (shaded parts are not seen by the model)")
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=360):
                    cmp_plot = gr.Plot(label="All six models on this clip")
                with gr.Column(scale=5, min_width=360):
                    cmp_df = gr.Dataframe(label="Scores", interactive=False, wrap=True)
            layer_plot = gr.Plot(label="Score at every XLS-R layer")
            gr.Markdown("Each point is a separate probe trained at that layer on ASVspoof 2019 LA. "
                        "Above zero means that probe calls the clip fake.", elem_classes="footnote")

            outs = [verdict, wave_plot, cmp_plot, cmp_df, layer_plot, clip_info, state]
            detect_btn.click(on_detect, inputs=[audio_in, model_dd], outputs=outs)
            draw_btn.click(on_draw, inputs=[label_radio, speaker_dd, model_dd], outputs=[audio_in] + outs)
            model_dd.change(on_model_change, inputs=[state, model_dd], outputs=[verdict])

        with gr.Tab("Batch"):
            gr.Markdown("Score several files at once. Every file goes through all six models; the table and CSV "
                        "show the chosen model's verdict plus each model's decision and margin.")
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=320):
                    batch_files = gr.File(label="Audio files", file_count="multiple", file_types=["audio"])
                    batch_model = gr.Dropdown(choices=choices, value=default, label="Model for the verdict")
                    batch_btn = gr.Button("Score all files", variant="primary")
                    batch_csv = gr.File(label="Download results (CSV)", interactive=False)
                with gr.Column(scale=6, min_width=420):
                    batch_summary = gr.HTML(empty_verdict("Results appear here."))
                    batch_df = gr.Dataframe(label="Results", interactive=False, wrap=True)
            batch_btn.click(on_batch, inputs=[batch_files, batch_model], outputs=[batch_summary, batch_df, batch_csv])

        with gr.Tab("Pilot results"):
            gr.Markdown(results_markdown())
            gallery = results_gallery()
            if gallery:
                gr.Gallery(value=gallery, label="Figures from the pilot run", columns=2, height="auto",
                           object_fit="contain")

        with gr.Tab("About"):
            gr.Markdown(ABOUT_MD)
            gr.Markdown(f"Probes: `{ENGINE.probe_source}`. Config SHA-256 `{ENGINE.cfg['_config_sha256'][:12]}…`.",
                        elem_classes="footnote")
    return demo


def main() -> int:
    global ENGINE
    parser = argparse.ArgumentParser(description="XLS-R deepfake detector dashboard")
    parser.add_argument("--pilot-dir", help="path to pilot_la_to_itw (default: detected from this file)")
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
