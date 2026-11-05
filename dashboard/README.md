# XLS-R Audio Deepfake Detector Dashboard

This local Gradio dashboard turns the frozen `run_20261011_113206` pilot into an interactive detector.
It reuses the repository's preprocessing, XLS-R encoder and logistic-regression probe code rather than
implementing a separate model.

## Three audience-facing pages

- **Try the detector:** upload, record or draw a labelled In-the-Wild example; view the decision,
  analysed audio region, six training-version comparison and layer profile.
- **Understand the elements:** plain-language cards explain each augmentation family, what an XLS-R
  layer means and how to read every displayed number.
- **Research evidence:** separates the pre-registered pilot comparison from the fixed-layer post hoc
  analysis and states the limitations before exposing the full tables and figures.

The default verdict uses the pre-registered `mix` condition, but all six models are always scored.
The displayed **decision score** is `sigmoid(logit - source-validation threshold)`: 0.5 is the decision
boundary. It is deliberately not labelled as a probability or confidence value.

## Required local artefacts

The repository intentionally does not contain audio, embeddings, model weights or fitted pickle caches.
Before running the full dashboard, follow [../REPRODUCE.md](../REPRODUCE.md) to obtain the datasets and
extract source embeddings. The pilot config's `data_root` must contain:

```text
hf_cache/                         cached XLS-R-53 checkpoint, or allow a first download
pilot_la_to_itw/embeddings/       cached source-train and source-validation embeddings
raw/release_in_the_wild.zip       optional; enables random labelled target clips
demo_cache/                       created locally for fitted probes
```

You may override the data root with `--data-root PATH`. Restricted datasets are not used by this demo.

## Install and verify

Use the same Python environment as the pilot, then add Gradio:

```bash
cd dashboard
pip install -r requirements_dashboard.txt
python -m unittest discover -s tests -v
python detector.py --check
```

If the model is already cached and the machine is offline:

```bat
set HF_HUB_OFFLINE=1
python detector.py --check
```

The integration check draws five deterministic In-the-Wild clips and compares live logits with the
committed pilot predictions. It prints `CHECK PASSED` when the largest absolute difference is below 0.25.
On the author's verified 2026-10-18 setup, the difference was `0.0000`.

## Launch

```bash
python app.py
```

Open <http://127.0.0.1:7860>. On Windows, `run_dashboard.bat` activates the `xlsr` Conda environment and
starts the same command. Useful options:

```text
--device cpu       run without CUDA
--port 7861        use another port
--retrain          ignore the local probe cache
--data-root PATH   override the pilot data root
--pilot-dir PATH   use another pilot_la_to_itw directory
```

`--share` and `--host 0.0.0.0` expose the interface beyond the local machine. The prototype has no
application-level authentication, so do not use those modes with sensitive recordings or on an
untrusted network.

## Verification status and limits

- Four data-free unit tests cover threshold-centred scoring, hidden-layer selection, core analysis and
  empty-audio rejection.
- The author's offline integration check loaded the frozen model, verified the manifests, found all
  31,779 target rows and exactly reproduced 30 stored logits from five deterministic clips.
- This is a presentation and reproducibility layer for the existing pilot. It is **not** independent
  validation and does not create a new research result.
- The apparent selected-layer gain is confounded by layer selection. The additive-noise pattern is post
  hoc and remains exploratory.
- The source attacks are from ASVspoof 2019 LA. The detector is not a forensic tool and its output must
  not be treated as evidence about a real recording.

Chinese instructions are available in [README_zh.md](README_zh.md). Maintainer notes are in
[docs/codebase/](docs/codebase/).
