# `trident/` — package layout

The pipeline is: **segment tissue vs. background → extract patch coordinates → run a model over those patches** (feature extraction, cell segmentation, or VLM Q&A). The `Processor` orchestrates it; the `*_models/` packages supply the models and `wsi_objects/` reads the slides.

## Orchestration & pipeline

| Module | What it does |
|---|---|
| `Processor.py` | The orchestrator. Runs each stage over a directory of slides (`run_segmentation_job`, `run_patching_job`, `run_patch_feature_extraction_job`, `run_slide_feature_extraction_job`, `run_patch_segmentation_job`, `run_vlm_query_job`), handling resume, locking, and per-slide error capture. |
| `State.py` | Per-slide state tracking (`wsi_states/<slide>.json`): which tasks ran, their status/reason, attempts, and output paths. This is what makes runs resumable. |
| `Summary.py` | Run-level reporting: writes the human-readable `summary.md` and the machine-readable `runs/<id>.json` manifest. |
| `Concurrency.py` | The WSI cache pipeline — producer/consumer staging of slides onto local disk for slow/network storage (`--wsi_cache`). |

## Models (one subpackage per model family)

Each follows the same shape: a `*_factory(name)` + a `*_registry` of names, a base class, and thin per-model wrappers. To add a model, subclass the base and register it.

| Package | What it does |
|---|---|
| `segmentation_models/` | **Tissue** vs. background segmentation (HEST, GrandQC, Otsu). Produces the tissue contours. |
| `patch_encoder_models/` | **Patch** encoders (UNI, CONCH, Virchow, …): one embedding per patch. |
| `slide_encoder_models/` | **Slide** encoders (Titan, GigaPath, PRISM, …): one embedding per slide, from patch features. |
| `patch_segmentation_models/` | **Cell/nuclei** instance segmentation (HistoPlus, CellViT++): per-cell polygons + types. |
| `vlm_models/` | **Vision-language** models (Patho-R1): free-text answer to a prompt about an image/ROI. |

## Slide I/O

| Module | What it does |
|---|---|
| `wsi_objects/WSI.py` | The core `WSI` object: lazy slide access plus the per-slide methods each stage calls (`segment_tissue`, `extract_tissue_coords`, `extract_patch_features`, `segment_patches`, `query_patches`, `query_region`, `overlay`). |
| `wsi_objects/WSIFactory.py` | `load_wsi(...)` — picks the right reader from the file/format. |
| `wsi_objects/{OpenSlide,CuCIM,Image,OMEZarr,CZI,SDPC}WSI.py` | Format-specific readers, all behind the common `WSI` interface. |
| `wsi_objects/WSIPatcher.py`, `WSIPatcherDataset.py` | Turn tissue contours into a grid of patch coordinates and serve patch crops to a DataLoader. |
| `Converter.py` | Convert awkward formats to pyramidal TIFF (`trident convert`). |

## Utilities & entry points

| Module | What it does |
|---|---|
| `IO.py` | Shared I/O helpers: slide discovery, HDF5/GeoJSON read-write, coords ↔ instance helpers, weight-path resolution, lock files. |
| `Visualization.py` | **All slide rendering.** Score/attention heatmaps (`visualize_heatmap`) and polygon overlays — tissue/cell contours, outline or translucent fill, whole-slide or ROI — via the shared `render_overlay` core (exposed to users as `WSI.overlay`). |
| `cli.py` | The `trident` console entry point (`trident batch`/`single`/`convert`). |
| `cli_doctor.py` | `trident-doctor` — preflight checks for installs, GPU, and gated-model access. |
| `Maintenance.py` | The `@deprecated` decorator. |
| `__init__.py` | Public exports (`Processor`, `load_wsi`, …). |
