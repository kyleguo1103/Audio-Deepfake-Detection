# Code availability

Suggested statement for the paper (replace the tag if a different release is cited):

> Code, the frozen configuration and pre-registration, sample manifests with per-clip SHA-256 hashes,
> sample-level predictions, bootstrap results and verification scripts are available at
> https://github.com/kyleguo1103/Audio-Deepfake-Detection (release `v0.1-pilot`). Audio datasets are not
> redistributed; instructions for obtaining them are given in the repository.

What a reader can check with this repository:

| Claim in the paper | How to check |
|---|---|
| Pre-registered comparison (mix vs none at selected layers): −13.11 pp, 95% CI [−18.58, −6.53] | `python scripts/verify_results.py` (recomputes EERs and re-runs the frozen bootstrap) |
| Selected layers 2 (none) and 20 (mix); source-validation EER ≤ 0.6% | `outputs/run_20261011_113206/selected_configuration.json`, `selection_grid_source_validation.csv`, `summary.json` |
| Layer-selection decomposition (control at layer 20: 21.0%; mix 2.5 pp worse) — post hoc | `exploratory_fixed_layer/per_layer_summary.csv`; printed by `verify_results.py` |
| Matched-layer averages (mix +0.0 pp; noise −4.0 pp, 12/13 layers) — post hoc | `exploratory_fixed_layer/layer_averaged_summary.json`; averages recomputed by `verify_results.py` |
| Figure of target EER by fixed layer | `python scripts/make_paper_figure.py` |
| Design fixed before data processing | `pilot_la_to_itw/PREREGISTRATION.md`, configuration SHA-256 in every `run_record.json` |
