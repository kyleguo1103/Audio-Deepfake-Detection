# XLS-R Layer Selection and Controlled Augmentation in Cross-Domain Audio Deepfake Detection

Code, frozen configurations, pre-registrations and results for a thesis study by **Jingkai Guo**
(Master of Information Technology, The University of Queensland, thesis course REIT7841).

The study asks whether controlled waveform augmentation of the *source* training audio (codec, additive
noise, channel effects) lowers the cross-domain error of a frozen XLS-R-53 + logistic-regression audio
deepfake detector, and whether XLS-R layer selection changes the apparent gain.

- **Source:** ASVspoof 2019 LA (training and speaker-disjoint development partitions)
- **Targets:** In-the-Wild for the pilot; restricted DeepSpeak v2 for the independent fixed-layer follow-up
- **Features:** frozen `facebook/wav2vec2-large-xlsr-53`, 25 hidden states mean-pooled to 1,024 dimensions
- **Classifier:** standardisation + class-balanced logistic regression; layer and C chosen on source validation only

## Evidence status

| Result | Status | Where |
|---|---|---|
| Mix vs no augmentation at each condition's selected layer: target EER 36.58% → 23.48%, Δ −13.11 pp, 95% speaker-cluster CI [−18.58, −6.53] | **Pre-registered** (design frozen 2026-10-09 before any audio was processed). Meets the decision rule, but the two pipelines selected different layers (2 and 20), so it is **not** a clean augmentation effect. | `pilot_la_to_itw/outputs/run_20261011_113206/RESULTS.md`, `summary.json` |
| Layer-selection decomposition: control at layer 20 reaches 21.0%; mix at layer 20 is 2.5 pp worse | **Post hoc** (computed after the primary result was saved) | `exploratory_fixed_layer/per_layer_summary.csv` |
| Matched-layer averages: mix +0.0 pp [−2.8, +2.4], 7/13 layers better; additive noise −4.0 pp [−6.3, −2.1], 12/13 layers better | **Exploratory / post hoc.** Hypothesis-generating only; needs a new pre-registered study and independent validation. | `exploratory_fixed_layer/EXPLORATORY_FIXED_LAYER.md`, `layer_averaged_summary.json` |
| DeepSpeak fixed-layer confirmation: none 48.82% EER, noise 47.65%, Δ −1.17 pp, 95% identity-cluster CI [−3.16, +0.80] | **Confirmatory and inconclusive.** The interval crosses zero, so the pre-specified confirmation rule was not met. | `confirm_la_to_deepspeak/outputs/confirmatory/RESULTS.md` |
| DeepSpeak six-condition follow-up: mix 46.35% EER, Δ −2.47 pp, interval [−5.16, −0.25] | **Exploratory / post-confirmation.** DeepSpeak results were already visible; this is hypothesis-generating only. | `confirm_la_to_deepspeak/outputs/exploratory_followup/RESULTS.md` |

The pre-registration is internal (a time-stamped document plus the SHA-256 of the configuration file
recorded by every run), not a public registry entry. See `pilot_la_to_itw/PREREGISTRATION.md`.

## Repository layout

```
README.md               this file
REPRODUCE.md            how to verify the reported numbers, and how to re-run everything from audio
DATASETS.md             where to obtain each dataset, licences, and which derived files are included
ATTRIBUTION.md          authorship, relationship to prior work, third-party code, models and data
CODE_AVAILABILITY.md    statement for the paper
LICENSE                 MIT licence for the code
scripts/
  verify_results.py     recomputes the reported numbers from the committed raw results (no data needed)
  make_paper_figure.py  redraws the paper's layer-profile figure
dashboard/
  app.py                audience-facing local Gradio detector and explanation interface
  detector.py           pilot-backed detection engine and integration self-check
  plots.py              waveform, model-comparison and layer-profile figures
  tests/                data-free core tests
confirm_la_to_deepspeak/
  PREREGISTRATION.md    fixed-layer none-vs-noise confirmation protocol
  config/               frozen confirmatory and labelled exploratory configurations
  src/                  acquisition, feature, evaluation and verification code
  outputs/              public-safe aggregate results, intervals and figures
pilot_la_to_itw/
  PREREGISTRATION.md    frozen design (with addenda A and B, both written before any result)
  config/pilot_config.json   frozen configuration, SHA-256 a26f7873...39782
  requirements_pilot.txt
  src/                  step0-step4 pipeline, augmentation, exploratory fixed-layer analysis
  tests/run_selftest.py end-to-end software self-test on synthetic audio (no data, no GPU)
  outputs/manifests/    frozen sample manifests (file IDs, labels, speaker IDs, SHA-256 of each clip)
  outputs/run_20261011_113206/   the reported run: metrics, sample-level logits, bootstrap CIs, figures
```

## Quick start: verify the reported numbers (no audio needed)

```bash
pip install numpy==1.26.4 scipy==1.13.1 matplotlib==3.9.2
python scripts/verify_results.py
```

This checks the configuration and manifest hashes, recomputes every target and source-validation EER
from the saved logits, re-runs the pre-registered paired speaker-cluster bootstrap with the frozen seed,
and recomputes the matched-layer averages. It prints `ALL CHECKS PASSED` on the committed results.
A full re-run from audio is described in [REPRODUCE.md](REPRODUCE.md).

## Interactive dashboard

The [dashboard](dashboard/) provides a local demonstration of the same frozen pilot pipeline. It can
analyse an uploaded or recorded clip, compare all six augmentation conditions, visualise how the
decision changes across XLS-R layers, and explain the result and research evidence in audience-facing language.

```bash
cd dashboard
pip install -r requirements_dashboard.txt
python -m unittest discover -s tests -v
python detector.py --check   # requires the local model/data artefacts described in dashboard/README.md
python app.py
```

No model weights, fitted caches, embeddings or audio are committed. The dashboard is a reproducibility
and presentation layer for the pilot, not an independent validation result or a forensic tool. See
[dashboard/README.md](dashboard/README.md) for setup, evidence status and remote-sharing precautions.

## Where each reported number comes from

| Number | File and field |
|---|---|
| Target EER per condition, Δ vs control, CIs, AUC, balanced accuracy | `outputs/run_20261011_113206/summary.json` (`conditions[*]`) |
| Selected layer and C per condition | `selected_configuration.json`; full grid in `selection_grid_source_validation.csv` |
| Per-seed metrics | `metrics_per_seed.csv` |
| Sample-level logits (target, source validation) | `predictions_target.csv`, `predictions_source_validation.csv` |
| Bootstrap intervals (speaker-cluster and clip-level) | `bootstrap_target.json` |
| Duration/silence shortcut diagnostic | `shortcut_diagnostic.json` |
| Target EER at every fixed layer (post hoc) | `exploratory_fixed_layer/per_layer_summary.csv`, `per_layer_per_seed.csv` |
| Layer-averaged Δ, CIs and layer counts (post hoc) | `exploratory_fixed_layer/layer_averaged_summary.json` |
| Software versions, GPU, model commit, run time | `run_record.json` |

## What is not included

No audio is redistributed. Audio, XLS-R embeddings (about 2.5 GB) and the model cache must be
obtained or regenerated locally (see [DATASETS.md](DATASETS.md)). No credentials or access tokens are
stored in this repository. DeepSpeak is gated and is never redistributed here. Its media, raw
annotations, target manifest, embeddings and sample-level predictions remain local; only implementation
code and public-safe aggregate results are included. See `confirm_la_to_deepspeak/DATA_RESTRICTIONS.md`.

## Status and next steps

The In-the-Wild pilot exposed layer-selection confounding. A later fixed-layer, pre-specified
none-vs-noise test on DeepSpeak did not confirm a reliable additive-noise benefit. The repository now
preserves that negative result, the clearly labelled exploratory follow-up and the explanatory dashboard.
Future work should validate a target-independent layer policy on another untouched target domain and
address the large source-to-target score shift before any deployment claim.

## Licence and citation

Code: MIT ([LICENSE](LICENSE)). Files derived from the datasets keep the datasets' licences
(see [DATASETS.md](DATASETS.md)). If you use this repository, please cite it via
[CITATION.cff](CITATION.cff) and cite the datasets and model listed in [ATTRIBUTION.md](ATTRIBUTION.md).
