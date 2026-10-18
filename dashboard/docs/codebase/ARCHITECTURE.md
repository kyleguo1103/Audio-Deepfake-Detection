# Architecture

## 1) Architectural Style

- Primary style: a local, single-process layered dashboard: Gradio presentation → detection engine → reused pilot/model and local artefacts.
- Classification basis: UI event handlers call one global `Engine`; the engine owns model/probe state and delegates canonical preprocessing/training functions to the pilot modules.
- Primary constraints:
  1. The dashboard must reproduce the frozen pilot rather than silently retrain or retune on target labels.
  2. XLS-R is expensive to load, so one long-lived engine/encoder instance is shared and requests are queued with concurrency one.
  3. The system depends on local pilot results, embeddings, archives and model cache outside this folder.

## 2) System Flow

```text
upload / microphone / random In-the-Wild clip
  -> decode to mono audio
  -> pilot resampling, silence trim, first-6-second crop and normalisation
  -> frozen XLS-R-53 produces hidden-state embeddings
  -> selected probes and per-layer probes calculate logits minus source thresholds
  -> Gradio renders verdict, plots, tables and optional CSV
```

1. At startup, `app.py` sets the Hugging Face cache early, constructs `Engine`, builds the UI and launches a queued local server.
2. `Engine` loads the pilot config, run ID, selected layer/C values, summary, run record and manifests.
3. Probes are loaded from a versioned pickle cache when compatible; otherwise they are refit from cached source embeddings using source validation only.
4. Audio is decoded by SoundFile for supported suffixes or by ffmpeg as fallback, then the pilot's preprocessing is applied.
5. The frozen XLS-R encoder returns all hidden states; the six selected-condition probes and 39 profile probes (three conditions × 13 layers) score the embedding.
6. The UI reports the chosen condition's verdict while also showing all six models, three layer profiles, preprocessing coverage and, for sampled target clips, ground truth and the recorded pilot logit.

## 3) Layer/Module Responsibilities

| Layer or module | Owns | Must not own | Evidence |
|---|---|---|---|
| `app.py` | Interaction, presentation, batch orchestration, export | Changing model parameters or target-aware tuning | `on_detect()`, `on_batch()`, `build_ui()` |
| `detector.Engine` | Model lifecycle, probes, audio-to-score pipeline, status | UI layout | `Engine.__init__()`, `analyse()` |
| `detector.ItwSampler` | Read-only random access to the frozen target manifest/archive and recorded logits | Training or selection | `ItwSampler` |
| `plots.py` | Stateless figures from `Analysis` | Filesystem/model access | Module functions |
| Pilot `src` modules | Canonical preprocessing, encoder and probe utilities | Dashboard-specific output | Imports inside `Engine` |

## 4) Reused Patterns

| Pattern | Where found | Why it exists |
|---|---|---|
| Data-transfer objects | `Probe`, `ModelScore`, `Analysis` dataclasses | Keeps model output structured across engine, UI and plots |
| Adapter/facade | `Engine` around pilot modules | Presents one dashboard-friendly API without duplicating the pilot pipeline |
| Cache-aside | `_load_or_train_probes()` | Avoids refitting 45 probe jobs at every startup while invalidating incompatible caches |
| Strategy by condition | `CONDITIONS`, `self.probes`, `self.layer_probes` | Runs the same scoring interface for six augmentation conditions and three layer profiles |
| Singleton application state | global `ENGINE` | Loads one large encoder and shares it across events |

## 5) Known Architectural Risks

- The dashboard is coupled to the pilot folder layout and files outside this codebase. Moving the project, outputs or data without matching CLI/config changes breaks startup.
- `app.py` and `detector.py` each combine several responsibilities; changes to model state, results presentation and storage require careful end-to-end regression checks.
- One global encoder with `default_concurrency_limit=1` is safe for the current laptop demo but limits throughput.
- The pickle probe cache is trusted local state; loading an untrusted cache would be unsafe.

## 6) Evidence

- `app.py`
- `detector.py`
- `plots.py`
- `README_zh.md`
- `../pilot_la_to_itw/config/pilot_config.json`
