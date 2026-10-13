# XLS-R Layer Selection and Controlled Augmentation in Cross-Domain Audio Deepfake Detection

Code, frozen configuration, pre-registration and raw results for a pilot study by **Jingkai Guo**
(Master of Information Technology, The University of Queensland, thesis course REIT7841).

The study asks whether controlled waveform augmentation of the *source* training audio (codec, additive
noise, channel effects) lowers the cross-domain error of a frozen XLS-R-53 + logistic-regression audio
deepfake detector, and whether XLS-R layer selection changes the apparent gain.

- **Source:** ASVspoof 2019 LA (training and speaker-disjoint development partitions)
- **Target:** In-the-Wild, all 31,779 clips, used only for final scoring
- **Features:** frozen `facebook/wav2vec2-large-xlsr-53`, 25 hidden states mean-pooled to 1,024 dimensions
- **Classifier:** standardisation + class-balanced logistic regression; layer and C chosen on source validation only

## Evidence status

| Result | Status | Where |
|---|---|---|
| Mix vs no augmentation at each condition's selected layer: target EER 36.58% → 23.48%, Δ −13.11 pp, 95% speaker-cluster CI [−18.58, −6.53] | **Pre-registered** (design frozen 2026-10-09 before any audio was processed). Meets the decision rule, but the two pipelines selected different layers (2 and 20), so it is **not** a clean augmentation effect. | `pilot_la_to_itw/outputs/run_20261011_113206/RESULTS.md`, `summary.json` |
| Layer-selection decomposition: control at layer 20 reaches 21.0%; mix at layer 20 is 2.5 pp worse | **Post hoc** (computed after the primary result was saved) | `exploratory_fixed_layer/per_layer_summary.csv` |
| Matched-layer averages: mix +0.0 pp [−2.8, +2.4], 7/13 layers better; additive noise −4.0 pp [−6.3, −2.1], 12/13 layers better | **Exploratory / post hoc.** Hypothesis-generating only; needs a new pre-registered study and independent validation. | `exploratory_fixed_layer/EXPLORATORY_FIXED_LAYER.md`, `layer_averaged_summary.json` |

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
stored in this repository. Restricted datasets (for example DeepSpeak) are not part of this pilot and
will never be uploaded here.

## Status and next steps

This is a pilot that uses ASVspoof 2019 LA as a substitute source while access to the planned source
dataset is pending. Planned next: a new pre-registered confirmatory study with a target-independent
layer policy that tests additive noise as the hypothesis, and validation on an independent
source-target pair.

## Licence and citation

Code: MIT ([LICENSE](LICENSE)). Files derived from the datasets keep the datasets' licences
(see [DATASETS.md](DATASETS.md)). If you use this repository, please cite it via
[CITATION.cff](CITATION.cff) and cite the datasets and model listed in [ATTRIBUTION.md](ATTRIBUTION.md).
