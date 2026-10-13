# Pilot results: ASVspoof 2019 LA -> In-the-Wild, controlled augmentation

- Run: `run_20261011_113206`; created 2026-10-11T01:37:32+00:00 (UTC)
- Encoder: `facebook/wav2vec2-large-xlsr-53` commit `c3f9d884181a224a6ac87bf8885c84d1cff3384f` on NVIDIA GeForce RTX 3060 Laptop GPU
- Target scope: **all_clips** (n = 31779); source pool n = 3000, source validation n = 2000
- Config SHA-256 `a26f7873f8b54d9224894bae0ab2ccf7b6588e3a1cc42b51e90a44b859f39782`; manifests frozen and verified before training.
- Codec encoders available for augmentation: aac, libmp3lame, libopus (pre-registered: libmp3lame, aac, libopus)

> Pilot status: substitute source domain (ASVspoof 2019 LA instead of DeepSpeak, whose access is not yet evidenced). It tests the method and pipeline; it is not the final thesis experiment.

## Primary pre-registered comparison

**+ mix (1 family/clip) vs No augmentation**: target EER 36.58% -> 23.48% (Δ = -13.11 percentage points; 95% speaker-cluster CI [-18.58, -6.53]; clip-level CI [-13.65, -12.53]).

Decision rule: Improvement supported only if the 95% speaker-cluster bootstrap CI of (EER_mix - EER_none), averaged over seeds, lies entirely below 0; harm if entirely above 0; otherwise inconclusive.

**Verdict: improvement supported (CI entirely below 0).**

## All conditions (target = In-the-Wild, mean ± SD over 3 seeds)

| Condition | Layer | C | Train n | Target EER % [95% CI] | Δ EER vs none (pp) [95% CI] | Target AUC | Bal. acc. @src thr | Macro F1 @src thr | Src-val EER % |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| No augmentation | 2 | 1.0 | 2400 | 36.58 ± 0.25 [26.86, 49.36] | +0.00  | 0.685 | 0.505 | 0.283 | 0.37 |
| + codec | 18 | 0.1 | 4800 | 25.43 ± 0.77 [16.83, 37.96] | -11.15 [-18.10, -5.59] | 0.827 | 0.735 | 0.702 | 0.50 |
| + noise | 10 | 0.1 | 4800 | 17.83 ± 0.30 [13.24, 24.43] | -18.76 [-27.04, -11.71] | 0.892 | 0.792 | 0.747 | 0.40 |
| + channel | 4 | 0.1 | 4800 | 17.21 ± 1.28 [11.69, 24.60] | -19.37 [-26.06, -13.15] | 0.907 | 0.652 | 0.551 | 0.53 |
| + mix (1 family/clip) | 20 | 0.1 | 4800 | 23.48 ± 0.56 [16.79, 33.21] | -13.11 [-18.58, -6.53] | 0.845 | 0.765 | 0.733 | 0.60 |
| + all 3 families | 20 | 0.1 | 9600 | 23.29 ± 0.41 [16.68, 33.37] | -13.29 [-19.00, -6.49] | 0.844 | 0.756 | 0.721 | 0.43 |

Threshold-dependent metrics use the threshold fixed at the source-validation EER point (no target labels used).

## Shortcut diagnostic (duration / silence only, logistic regression)

| Features | Source-val EER % | Source-val AUC | Target EER % | Target AUC |
|---|---:|---:|---:|---:|
| raw_seconds, lead_silence_s, trail_silence_s | 30.10 | 0.747 | 45.23 | 0.561 |
| used_seconds | 43.61 | 0.602 | 42.61 | 0.599 |

Silence is trimmed before XLS-R in every condition. These scores show how much label signal duration/silence metadata alone would carry; high source scores mean the source has a known shortcut.

## Figures

- `figures/fig1_target_eer_by_condition.png`
- `figures/fig2_target_roc.png`
- `figures/fig3_target_score_distributions.png`
- `figures/fig4_layer_profile_exploratory.png` (exploratory; computed after the primary results were saved)

## Files

`selection_grid_source_validation.csv`, `selected_configuration.json`, `metrics_per_seed.csv`, `predictions_target.csv`, `predictions_source_validation.csv`, `bootstrap_target.json`, `summary.json`, `shortcut_diagnostic.json`, `layer_profile_EXPLORATORY.csv`, `run_record.json`.
