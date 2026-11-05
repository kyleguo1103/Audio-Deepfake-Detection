# Technology Stack

## 1) Runtime Summary

| Area | Value | Evidence |
|---|---|---|
| Primary language | Python | `app.py`, `detector.py`, `plots.py` |
| Runtime + version | Python 3.11.17 in the local `xlsr` Conda environment; the repository itself does not pin Python | 2026-10-18 terminal version check; `run_dashboard.bat` |
| Package manager | Conda environment plus `pip` for dashboard extras | `README_zh.md`, `requirements_dashboard.txt`, `run_dashboard.bat` |
| Module/build system | Plain Python modules; no package build or container configuration | `.codebase-scan.txt`, root file layout |

## 2) Production Frameworks and Dependencies

The following versions are the installed versions verified in the local `xlsr` environment on 2026-10-18. Only Gradio is constrained by this folder; the remaining packages come from the pilot environment.

| Dependency | Installed version | Role in system | Evidence |
|---|---:|---|---|
| Gradio | 5.50.0 | Local dashboard, uploads, microphone input, tables and downloads | `app.py`, `requirements_dashboard.txt` |
| PyTorch | 2.4.1+cu118 | XLS-R inference on CPU or CUDA | `detector.py`, `README_zh.md` |
| Transformers | 4.44.2 | Loads `facebook/wav2vec2-large-xlsr-53` | `detector.py`, `README_zh.md` |
| scikit-learn | 1.5.2 | StandardScaler and logistic-regression probes; cache compatibility check | `detector.py`, `requirements_dashboard.txt` |
| NumPy | 1.26.4 | Audio arrays, embeddings and score calculations | `detector.py`, `plots.py` |
| pandas | 2.3.3 | Comparison and batch-result tables/CSV | `app.py` |
| Matplotlib | 3.9.2 | Waveform, spectrogram, model and layer plots | `plots.py` |
| SoundFile | 0.12.1 | Primary decoder for common audio suffixes | `detector.py` |
| SciPy | 1.13.1 | Pilot preprocessing/augmentation dependency | `requirements_dashboard.txt`; imported pilot modules |
| imageio-ffmpeg | 0.5.1 | Supplies ffmpeg for fallback decoding | `detector.py`; imported `pilot_common.ffmpeg_exe()` |

## 3) Development Toolchain

| Tool | Purpose | Evidence |
|---|---|---|
| `detector.py --check` | Integration/reproducibility self-check against recorded pilot logits | `detector.py`, `README_zh.md` |
| Conda environment `xlsr` | Reuses the pilot's model and scientific-Python environment | `run_dashboard.bat`, `README_zh.md` |
| Linter/formatter | [TODO] None configured in this folder | `.codebase-scan.txt` |
| Python `unittest` | Four data-free core tests | `tests/test_detector_core.py`, `TESTING.md` |
| Coverage runner | [TODO] None configured in this folder | `TESTING.md` |

## 4) Key Commands

```bat
conda activate xlsr
pip install -r requirements_dashboard.txt
set PILOT_DATA_ROOT=D:\path\to\pilot_data
set HF_HOME=%PILOT_DATA_ROOT%\hf_cache
set HF_HUB_OFFLINE=1
python detector.py --check
python app.py
```

Optional runtime flags include `--share`, `--host`, `--port`, `--device`, `--data-root`, `--pilot-dir` and `--retrain`.

## 5) Environment and Config

- Config sources: `../pilot_la_to_itw/config/pilot_config.json`, the pilot run's `selected_configuration.json`, `summary.json`, `run_record.json`, and CLI arguments.
- Environment variables read directly: `HF_HOME`, `PILOT_DATA_ROOT`, `PILOT_DIR`. `HF_HUB_OFFLINE=1` is documented for offline startup.
- Runtime constraints: the pilot code and run outputs must remain available; cached source embeddings are required to refit probes; the In-the-Wild archive is optional for uploads but required for random target sampling; the XLS-R checkpoint must be cached or reachable from Hugging Face.
- [TODO] Add a complete reproducible environment file that pins Python and every dependency used by both the dashboard and pilot.

## 6) Evidence

- `README_zh.md`
- `requirements_dashboard.txt`
- `run_dashboard.bat`
- `app.py`
- `detector.py`
- `docs/codebase/.codebase-scan.txt`
