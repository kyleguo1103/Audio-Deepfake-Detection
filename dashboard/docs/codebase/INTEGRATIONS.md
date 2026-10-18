# External Integrations

## 1) Integration Inventory

| System | Type | Purpose | Auth model | Criticality | Evidence |
|---|---|---|---|---|---|
| Pilot code and outputs | Local filesystem | Canonical preprocessing, probe training, selected configuration, metrics and figures | Local file access | High | `detector.py`, `app.py` |
| `facebook/wav2vec2-large-xlsr-53` | Hugging Face model/cache | Frozen 25-hidden-state audio representation | Public model; no token used in code | High | `detector.py`, `README_zh.md` |
| ASVspoof source embeddings | Local data store | Refit logistic-regression probes when cache is absent/stale | Local file access | High for first run/retrain | `detector.py` |
| In-the-Wild archive and manifest | Local dataset/archive | Optional random labelled demo clips and reproducibility comparison | Local file access | Medium | `ItwSampler` in `detector.py` |
| ffmpeg via pilot helper | Local executable | Fallback decoding for M4A/AAC/MP4/WebM and failed common-format reads | None | Medium | `_ffmpeg_decode()` |
| Gradio share service | Optional remote tunnel | Temporary public dashboard link when `--share` is selected | No application authentication configured | Optional/high exposure | `app.py`, `README_zh.md` |

DeepSpeak is not used by this dashboard.

## 2) Data Stores

| Store | Role | Access layer | Key risk | Evidence |
|---|---|---|---|---|
| `{data_root}/demo_cache/*.pkl` | Fitted probe cache | `Engine._load_or_train_probes()` | Pickle must remain trusted and compatible | `detector.py` |
| Pilot output run directory | Frozen JSON/CSV/PNG evidence | `Engine`, results page | Tight coupling to a particular run structure | `app.py`, `detector.py` |
| Temporary directories | Copied target clips, figures and batch CSV downloads | `draw_itw()`, `on_batch()`, `results_gallery()` | Files persist according to OS temp cleanup | `app.py`, `detector.py` |
| In-the-Wild ZIP/source reader | Random-access target audio | `ItwSampler` | Restricted/licensed data must not be published with code | `detector.py`, `README_zh.md` |

There is no database, queue or event bus.

## 3) Secrets and Credentials Handling

- Credential sources: none in the dashboard code.
- Hardcoding check: no API keys or passwords found. Several local Windows paths are intentionally documented or hardcoded in the launcher.
- Hugging Face uses a public model and local cache; no token access is implemented.
- Rotation/lifecycle: not applicable to current local-only operation.

## 4) Reliability and Failure Behavior

- Retry/backoff: none around Hugging Face or ffmpeg calls.
- Timeout policy: none explicitly configured.
- Fallbacks: SoundFile falls back to ffmpeg; missing/unreadable probe cache triggers refitting; unavailable In-the-Wild disables that UI section without disabling uploaded-file detection.
- Offline operation requires the cached model and `HF_HUB_OFFLINE=1` when the library would otherwise make a network metadata request.

## 5) Observability for Integrations

- Startup/probe training emits `[dashboard]` console messages.
- Header status reports manifest verification, encoder commit/device and target availability.
- The CLI self-check reports live-versus-recorded logits and a pass/warning threshold.
- No structured logs, metrics, tracing or persistent audit log are implemented.

## 6) Evidence

- `app.py`
- `detector.py`
- `run_dashboard.bat`
- `README_zh.md`
- `../pilot_la_to_itw/config/pilot_config.json`
