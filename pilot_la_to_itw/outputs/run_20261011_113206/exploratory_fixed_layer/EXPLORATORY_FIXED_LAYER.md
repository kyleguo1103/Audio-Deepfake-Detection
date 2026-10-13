# Exploratory fixed-layer analysis (post hoc, not pre-registered)

> Added on 2026-10-11 after the primary results in `../RESULTS.md` were saved. It does not replace the pre-registered result; it qualifies how that result should be interpreted.
> Script: `src/explore_fixed_layer.py` (figures: `src/plot_exploratory.py`). Computed from the same frozen manifests and embeddings; C is still chosen on source validation only. The primary configuration was reproduced exactly (no augmentation, layer 2, C = 1: target EER 36.58%).

## Why this check was needed

In the primary run, the two compared pipelines selected different XLS-R layers on source validation (no augmentation → layer 2, mix → layer 20). Source-validation EER is below about 2% at almost every layer, so it cannot tell which layer transfers to In-the-Wild. Target EER of the no-augmentation model ranges from about 17% to 40% depending on the layer. The primary Δ therefore mixes two effects: augmentation and layer choice.

## Result: compare conditions at the same layer

Layer-averaged target EER over 13 layers (0, 2, …, 24), mean of 3 seeds; Δ is relative to no augmentation at the same layer; 95% CI from 1,000 speaker-cluster bootstrap resamples (54 speakers).

| Condition | Layer-averaged EER | Δ vs none (pp) [95% CI] | Better than none at |
|---|---:|---|---:|
| No augmentation | 26.0% | — | — |
| + codec | 26.4% | +0.4 [−1.7, +2.4] | 6 / 13 layers |
| **+ noise** | **22.0%** | **−4.0 [−6.3, −2.1]** | **12 / 13 layers** |
| + channel | 24.6% | −1.4 [−4.1, +1.1] | 10 / 13 layers |
| + mix (pre-registered treatment) | 26.0% | +0.0 [−2.8, +2.4] | 7 / 13 layers |
| + all 3 families | 24.5% | −1.4 [−4.6, +1.2] | 9 / 13 layers |

Pre-stated reading rule for this analysis: the mix effect would count as robust to layer choice only if its layer-averaged CI were entirely below 0 and it won at a majority of layers. **It does not meet that rule.**

## Interpretation

1. The pre-registered comparison stands as reported (−13.1 pp, CI [−18.6, −6.5]), but **most of that gain comes from the two pipelines landing on different layers**, not from the mix augmentation itself. At matched layers, mix is on average no better than no augmentation.
2. **Additive noise is the one family with a consistent fixed-layer benefit** (−4.0 pp on average, better at 12 of 13 layers, largest at middle layers 10–14: −8 to −10 pp). This is a post-hoc finding among five augmentation conditions and must be confirmed in a new pre-registered run before it is claimed.
3. Codec augmentation does not help on average; channel effects are mostly positive but uncertain. Mixing families (one per clip) dilutes the noise benefit.
4. The best layers (around 22) reach about 16–17% EER with or without augmentation: augmentation helps most where the representation transfers poorly.
5. **Methodological lesson:** clean in-domain validation is saturated and cannot select robust layers or hyperparameters. The next pre-registered design should fix the layer policy in advance (e.g. report all layers or a fixed layer set) or validate on a shifted proxy (held-out attacks or augmented validation audio), never on target labels.

## Figures

- `figures/figA_fixed_layer_delta_heatmap.png` — Δ EER by layer and augmentation family
- `figures/figB_layer_averaged_delta.png` — layer-averaged Δ with CI
- `figures/figC_layer_profile_none_noise_mix.png` — target vs source-validation EER by layer
