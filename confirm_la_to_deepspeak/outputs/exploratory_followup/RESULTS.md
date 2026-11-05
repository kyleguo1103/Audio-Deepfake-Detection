# Exploratory fixed-layer augmentation follow-up

> **Status: EXPLORATORY_POST_CONFIRMATION.** DeepSpeak had already been evaluated before this comparison was chosen. These results explain the failure and generate hypotheses; they are not a new independent validation.

- Fixed XLS-R layer: 10
- Fixed logistic-regression C: 0.1
- Source seeds: 42, 1337, 2026
- Target: 1,800 DeepSpeak v2 test clips from 56 identities
- Paired uncertainty: 10,000 identity-cluster bootstrap replications

## Target results

| Condition | Mean EER | Delta vs none | 95% paired CI for delta | Mean AUC |
|---|---:|---:|---:|---:|
| none | 48.82% | +0.00 pp | [+0.00, +0.00] pp | 0.492 |
| codec | 46.88% | -1.94 pp | [-3.74, -0.05] pp | 0.518 |
| noise | 47.65% | -1.17 pp | [-3.16, +0.80] pp | 0.510 |
| channel | 48.31% | -0.51 pp | [-3.41, +1.81] pp | 0.524 |
| mix | 46.35% | -2.47 pp | [-5.16, -0.25] pp | 0.538 |
| all | 47.08% | -1.74 pp | [-4.95, +0.32] pp | 0.540 |

## Bounded interpretation

The lowest mean EER was produced by **mix** (46.35%; delta versus none -2.47 pp, paired 95% CI [-5.16, -0.25] pp).

Even if a paired interval excludes zero, the finding remains exploratory because the DeepSpeak target result was already visible. It must not be described as pre-registered or independently confirmed.

## Reusable evidence

- `metrics_per_seed.csv`
- `predictions_target.csv`
- `bootstrap_identity_cluster.json`
- `summary.json`
- `figures/fig1_target_eer_by_condition.png`
- `figures/fig2_delta_eer_vs_none.png`
