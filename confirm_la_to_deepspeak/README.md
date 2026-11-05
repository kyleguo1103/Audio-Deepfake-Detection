# Fixed-layer ASVspoof LA → DeepSpeak v2 study

This folder is the public-safe implementation and aggregate evidence for the thesis follow-up by Jingkai Guo. It extends the earlier ASVspoof 2019 LA → In-the-Wild pilot with a restricted, independent DeepSpeak v2 target.

## Evidence boundary

Two analyses are deliberately separated.

1. **Confirmatory none vs additive noise** — the protocol, target manifest rules, fixed XLS-R layer 10, `C=0.1`, source seeds and decision rule were frozen before DeepSpeak media acquisition or scoring.
2. **Exploratory six-condition follow-up** — codec, noise, channel, mix and all-family comparisons were run only after the confirmatory DeepSpeak result was visible. They explain the failure and generate hypotheses; they are not a new independent validation.

## Main results

### Confirmatory result

- No augmentation mean DeepSpeak EER: **48.82%**
- Additive-noise mean EER: **47.65%**
- Paired difference: **−1.17 percentage points**
- 95% identity-cluster bootstrap CI: **[−3.16, +0.80] percentage points**
- Pre-specified verdict: **inconclusive**

The experiment neither confirms a reliable benefit nor a reliable harm from additive noise.

### Exploratory follow-up

At the same layer and classifier setting, the lowest mean EER was produced by the size-matched `mix` condition:

- Mix mean EER: **46.35%**
- Difference versus none: **−2.47 percentage points**
- Paired 95% identity-cluster interval: **[−5.16, −0.25] percentage points**

This remains exploratory because DeepSpeak had already been evaluated. It must not be described as pre-registered or independently confirmed.

### Error-analysis result

The source-validation threshold transfers poorly. For `mix`, the mean balanced accuracy on DeepSpeak is about **51.3%**; it misclassifies about **16.8% of real clips** and **80.6% of fake clips** at that operating point. The distinction matters:

- EER/AUC describe ranking across possible thresholds.
- Thresholded deployment metrics also depend on calibration and domain-specific score shift.
- Target-label calibration was not applied because it would change the source-only generalisation protocol.

## Layout

```text
confirm_la_to_deepspeak/
  README.md
  DATA_RESTRICTIONS.md
  config/
    confirm_config.json
    exploratory_followup_2026-11-04.json
  src/
    build_target_manifest.py
    acquire_target_audio.py
    verify_full_acquisition.py
    extract_target_layer10.py
    run_confirmatory_eval.py
    verify_confirmatory_results.py
    run_exploratory_fixed_layer_followup.py
    verify_exploratory_followup.py
    analyse_target_errors.py
  outputs/
    confirmatory/              aggregate metrics, intervals, verification and figures
    exploratory_followup/      aggregate metrics, intervals, verification and figures
```

## Reproduction overview

The code reuses the source manifests and deterministic augmentation embeddings from `pilot_la_to_itw/`. A complete local run requires:

1. authorised access to `faridlab/deepspeak_v2`;
2. the source data described in the repository-level `DATASETS.md`;
3. the Python environment used by the pilot;
4. enough temporary storage for one DeepSpeak archive at a time;
5. a locally cached `facebook/wav2vec2-large-xlsr-53` revision `c3f9d884181a224a6ac87bf8885c84d1cff3384f`.

The local sequence is:

```powershell
$env:PILOT_EMBEDDINGS_DIR = "D:\path\to\pilot_la_to_itw\embeddings"
$env:DEEPSPEAK_LAYER10_DIR = "D:\path\to\deepspeak\embeddings\target\layer10"
python src/build_target_manifest.py
python src/acquire_target_audio.py --help
python src/verify_full_acquisition.py
python src/extract_target_layer10.py
python src/run_confirmatory_eval.py
python src/verify_confirmatory_results.py --run outputs/run_YYYYmmdd_HHMMSS
python src/run_exploratory_fixed_layer_followup.py
python src/verify_exploratory_followup.py --run outputs/exploratory_followup_YYYYmmdd_HHMMSS
python src/analyse_target_errors.py --run outputs/run_YYYYmmdd_HHMMSS
```

The frozen configuration retains the thesis workstation paths as an audit record. When reproducing
elsewhere, update only its `paths` fields and set the two embedding-directory environment variables
shown above. Changing model, data rules, layer, classifier, seed or target rules creates a different experiment.

## Verification

The confirmatory and exploratory result folders contain separate verification reports. The exploratory verifier independently recalculates per-seed EER/AUC, condition means and all 10,000 paired identity-cluster intervals from saved local predictions. The local verification status is **PASS**.

Verification confirms arithmetic and saved-result consistency. It does not upgrade an exploratory analysis into confirmatory evidence.

