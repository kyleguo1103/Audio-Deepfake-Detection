# Exploratory DeepSpeak error analysis

> This analysis explains saved results; it does not retrain models or create a new confirmatory claim.

- Source run: `exploratory_followup_20261104_004109`
- Conditions analysed: none, all, channel, codec, mix, noise
- Target samples: 1800; identities: 56
- Operating point: each condition and seed uses its source-validation EER threshold.

## Main findings

1. The lowest mean target EER in this run is **mix** at 46.35%, compared with 48.82% for none.
2. At the transferred source threshold, mix still misclassifies 16.8% of real clips and 80.6% of fake clips. Its balanced accuracy is 51.3%.
3. Mean none and mix target scores remain strongly related (Pearson r = 0.895); augmentation changes ranking only modestly rather than creating a clean domain-invariant separation.
4. For mix, fake false-negative rates vary from 69.2% (speechify) to 94.5% (elevenlabs) across audio providers.
5. 509 of 1800 clips are wrong for mix under all three source seeds, showing a persistent subset rather than only seed noise.

## Interpretation for the thesis

A lower EER or higher AUC describes ranking quality across possible thresholds. The transferred source threshold is a separate deployment problem. The results show that both representation shift and threshold/calibration shift matter on DeepSpeak. Target-label calibration would change the research protocol, so it is not applied here.

## Files

- `overall_threshold_errors.csv`
- `subgroup_errors.csv`
- `identity_errors.csv`
- `hard_cases_LOCAL_RESTRICTED.csv` (local only; do not upload)
- `summary.json`
- `fig1_class_error_rates.png`
- `fig2_fake_errors_by_provider.png`
- `fig3_score_scatter.png`
- `fig4_identity_error_distribution.png`
