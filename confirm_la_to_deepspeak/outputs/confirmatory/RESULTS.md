# Confirmatory ASVspoof LA to DeepSpeak v2 result

- Frozen XLS-R layer: 10
- Fixed logistic-regression C: 0.1
- Target: 1,800 DeepSpeak v2 test clips from 56 identities
- No augmentation mean EER: 48.82%
- Additive noise mean EER: 47.65%
- Difference, noise minus none: -1.17 percentage points
- 95% identity-cluster bootstrap CI: [-3.16, +0.80] percentage points
- Pre-specified decision: **inconclusive**

The result uses the frozen target manifest once. DeepSpeak labels did not influence training, layer selection, C, preprocessing or thresholds.
