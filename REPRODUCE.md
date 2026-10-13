# Reproducing the pilot

There are three levels. Levels 1 and 2 need no datasets.

| Level | What it shows | Needs | Time |
|---|---|---|---|
| 1. Verify reported numbers | Every reported EER, Δ, pre-registered CI and matched-layer average follows from the committed raw results | Python + numpy/scipy | 1–3 min |
| 2. Software self-test | The full pipeline (steps 1–4) runs end to end on synthetic audio and its safety checks work | Python + `requirements_pilot.txt` | 1–2 min |
| 3. Full re-run from audio | Independent regeneration of manifests, embeddings and results | Datasets (≈10 GB), NVIDIA GPU recommended | ≈1 h on an RTX 3060 Laptop |

## Level 1: verify the reported numbers

```bash
pip install numpy==1.26.4 scipy==1.13.1 matplotlib==3.9.2
python scripts/verify_results.py
```

Checks the configuration and manifest SHA-256 values against `run_record.json`, recomputes target and
source-validation EER and AUC from the saved logits, re-runs the paired speaker-cluster bootstrap
(1,000 resamples, frozen seed 7841) and recomputes the matched-layer averages and layer counts.
Expected last line: `ALL CHECKS PASSED`.

The matched-layer confidence intervals cannot be recomputed at this level because per-layer logits are
not saved; they require the embeddings (Level 3, `src/explore_fixed_layer.py`).

To redraw the paper's layer-profile figure: `python scripts/make_paper_figure.py`.

## Level 2: software self-test

```bash
cd pilot_la_to_itw
pip install -r requirements_pilot.txt        # torch is only needed for Level 3
python tests/run_selftest.py
```

Builds tiny synthetic archives with the real ASVspoof/In-the-Wild layouts, runs steps 1–4 with a fake
spectral encoder (no XLS-R download), and checks that step 3 refuses fake embeddings, that editing a
frozen manifest is detected and that `LA.zip` and an extracted LA folder give identical manifests.
Expected: `SELF-TEST PASSED`. Its scores are meaningless by design.

## Level 3: full re-run from audio

### Environment

Tested on Windows 11, Python 3.11.17, torch 2.4.1+cu118, transformers 4.44.2, scikit-learn 1.5.2,
numpy 1.26.4 (see `outputs/run_20261011_113206/run_record.json`).

```bash
conda create -n xlsr python=3.11 -y
conda activate xlsr
# choose the CUDA wheel your driver supports (cu121, or cu118 for older drivers)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
cd pilot_la_to_itw
pip install -r requirements_pilot.txt
```

Set where embeddings and the Hugging Face model cache go (about 4 GB in total):

```bash
export PILOT_DATA_ROOT=/path/to/pilot_data      # Windows: set PILOT_DATA_ROOT=D:\pilot_data
export HF_HOME=/path/to/pilot_data/hf_cache     # Windows: set HF_HOME=D:\pilot_data\hf_cache
```

### Data

Obtain ASVspoof 2019 LA and In-the-Wild as described in [DATASETS.md](DATASETS.md). Keep
`release_in_the_wild.zip` zipped; LA can be `LA.zip` or the extracted `LA` folder.

### Steps (run inside `pilot_la_to_itw/`)

```bash
# 0. environment and data check
python src/step0_check_env.py --la-source /path/to/LA --itw-source /path/to/release_in_the_wild.zip

# 1. rebuild the manifests with the frozen sampling seed and compare them with the committed ones
python src/step1_build_manifests.py --la-source /path/to/LA --itw-source /path/to/release_in_the_wild.zip --force
git diff --stat -- outputs/manifests        # the three CSV files should show no change

# 2. XLS-R feature extraction (resumable; add --workers 0 if multiprocessing fails)
python src/step2_extract_xlsr.py

# 3. selection on source validation, single target evaluation, bootstrap, report and figures
python src/step3_train_eval.py

# exploratory, post hoc analyses (never used for selection)
python src/explore_fixed_layer.py
python src/plot_exploratory.py

# compare the new run with the reported one
cd ..
python scripts/compare_runs.py run_YYYYMMDD_HHMMSS
python scripts/verify_results.py --run run_YYYYMMDD_HHMMSS
```

Notes:

- `--force` in step 1 is needed only because this repository ships the frozen manifests. Step 1 writes
  your local data paths into `outputs/manifests/manifest_record.json`; the CSV files themselves contain
  no local paths, so identical CSVs mean an identical sample selection and identical audio (each row
  stores the clip's SHA-256).
- Step 3 refuses to run if a manifest no longer matches its recorded hash, and every run records the
  configuration hash. Do not edit `config/pilot_config.json`; a different hash marks a deviation from
  the pre-registration.
- `src/check_model_weights.py` optionally confirms that the XLS-R checkpoint loaded without missing or
  unexpected weights (set `HF_HOME` first).

### Expected differences in a re-run

Manifests and the sample selection should be identical. Target EERs may differ slightly because codec
augmentation depends on the local ffmpeg build, and because of BLAS and GPU fp16 arithmetic. A
different *selected layer* for a condition can change its target EER by several points; that
sensitivity is the main finding of the study, so report it rather than tuning it away.
