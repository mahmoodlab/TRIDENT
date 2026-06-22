# TRIDENT reference

Complete CLI flags, encoder tables, output layout, the Python API, and install profiles.
Read this when you need an exact flag, an encoder's required resolution, or the library API.
For the workflow and decisions, see [SKILL.md](SKILL.md).

## Contents
- Entry points
- `run_batch_of_slides.py` flags
- `run_single_slide.py` flags
- Patch encoders (24) — embedding dim + required patch_size/mag
- Slide encoders — required patch encoder + patch_size/mag
- Segmenters & artifact removal
- WSI readers & formats
- Output artifacts — everything TRIDENT writes
- Python API (custom pipelines)
- Install profiles & preflight
- Slide conversion

## Entry points

| Invocation | Use |
|---|---|
| `python run_batch_of_slides.py ...` | A directory of WSIs (primary tool). |
| `python run_single_slide.py ...` | One slide, end-to-end. Good for a first validation run. |
| `trident batch -- ...` / `trident single -- ...` | Same as above via the installed CLI (reproducible, from any project). |
| `trident convert ...` | Convert images/WSIs to pyramidal TIFF. |

`--task` ∈ {`seg`, `coords`, `feat`, `all`}. **Each value runs exactly one stage; only `all`
chains them.** `coords` reads the segmentation from `--job_dir` (skips with `geojson_not_found`
if absent); `feat` reads coords from `--job_dir` (or from `--coords_dir`). So `coords`/`feat`
on a fresh `--job_dir` do nothing — run `all`, or run the stages in order. For seg+coords
without features, run `--task seg` then `--task coords`.

## `run_batch_of_slides.py` flags

**Core**
- `--task {seg,coords,feat,all}` (default `seg`)
- `--job_dir PATH` (required) — output dir; also the resume key.
- `--wsi_dir PATH` (required) — directory of WSIs.
- `--gpus INT [INT ...]` — GPU indices; multiple shards pending slides; `-1` = CPU.
  (`--gpu` is the deprecated single-GPU form.)
- `--skip_errors` — continue past slides that error (recommended for big batches).
- `--max_workers INT` — dataloader/concurrency workers; `0` = main process.

**Slide selection / IO**
- `--wsi_ext .svs .tiff ...` — restrict extensions.
- `--custom_list_of_wsis list.csv` — a real on-disk CSV file (not a stream/process-substitution —
  pandas reads the path directly) with column `wsi` (filenames/relative paths, with extension) and
  optional column `mpp`. Only listed slides run.
- `--custom_mpp_keys KEY ...` — metadata keys to read micron-per-pixel from.
- `--reader_type {openslide,image,cucim,sdpc,omezarr,czi}` — force a reader (default: auto).
- `--search_nested` — recurse into subdirectories of `--wsi_dir`.

**Segmentation**
- `--segmenter {hest,grandqc,otsu}` (default `hest`).
- `--seg_conf_thresh FLOAT` (default 0.5) — lower keeps more tissue (try 0.4).
- `--remove_holes` — drop patches over tissue holes (default keeps them).
- `--remove_artifacts` — extra GrandQC pass removing folds/blur/stains/penmarks/OOF (aggressive).
- `--remove_penmarks` — extra pass removing penmarks only (gentle).
- `--seg_batch_size INT`.

**Patching**
- `--mag FLOAT` (default 20.0) — target magnification; floats allowed (e.g. `1.25`).
- `--patch_size INT` (default 512).
- `--overlap INT` (default 0) — absolute pixels (e.g. 128 = 50% overlap on 256px). **Must be `< --patch_size`**; `>=` makes the step ≤ 0 and hangs the patching loop forever.
- `--min_tissue_proportion FLOAT` (default 0.0) — min tissue fraction to keep a patch.
- `--coords_dir PATH` — reuse externally generated coords (e.g. legacy CLAM `*_fp/`).
- `--dump_patches` — during the `coords` task, also write the patch **images** (not just coordinates) to `<mag>x_<ps>px_<ov>px_overlap/patch_images/<slide>/`.
- `--dump_patches_max INT` (default `0` = no limit) — cap the number of patch images dumped per slide.
- `--dump_patches_format {png,jpg}` (default `png`).
- `--dump_patches_jpeg_quality INT` (default 90, 1–100) — only used when format is `jpg`.

**Feature extraction**
- `--patch_encoder NAME` (default `conch_v15`) — see patch encoder table.
- `--patch_encoder_ckpt_path PATH` — local checkpoint (`.pt/.pth/.bin/.safetensors`) for the
  patch encoder, for offline/air-gapped clusters (otherwise weights download from HF). Ignored
  when `--slide_encoder` is set.
- `--patch_encoder_img_size INT` — optional custom input resolution for ViT encoders
  (interpolates positional embeddings via timm `dynamic_img_size`; must be a multiple of
  the model patch size; embedding dim is unchanged).
- `--slide_encoder NAME` — produce slide embeddings (auto-extracts the right patch features first).
- `--feat_batch_size INT`, `--batch_size INT`.

**Cache / locks (slow storage, resume)**
- `--wsi_cache /local/ssd --cache_batch_size 32` — stage slides locally (producer/consumer).
- `--clear_dead_locks` — remove stale `.lock` files from a killed run before starting.
- `--dead_lock_max_age_hours FLOAT` (default 24) — with `--clear_dead_locks`, a `.lock` whose
  target output is missing is treated as stale only once it is older than this many hours.

## `run_single_slide.py` flags

Same model/segmenter/patching flags as batch, but for one slide:
- `--slide_path PATH` (required), `--job_dir PATH` (required), `--gpu INT`.
- `--mag {5,10,20,40}` (default 20 — note: restricted choices here, unlike batch's float).
- `--patch_size INT` (default 256), `--overlap INT`, `--batch_size INT` (default 32).
- `--patch_encoder` (default `conch_v15`), `--patch_encoder_img_size`, `--segmenter`,
  `--seg_conf_thresh`, `--remove_holes`, `--remove_artifacts`, `--remove_penmarks`,
  `--reader_type`, `--custom_mpp_keys`.

## Patch encoders

Loaded via `trident.patch_encoder_models.encoder_factory(name)`. The `patch_size`/`mag`
column is **required** for correct features — copy it verbatim.

| Encoder (`--patch_encoder`) | Dim | Required args |
|---|---:|---|
| `uni_v1` | 1024 | `--patch_size 256 --mag 20` |
| `uni_v2` | 1536 | `--patch_size 256 --mag 20` |
| `conch_v1` | 512 | `--patch_size 512 --mag 20` |
| `conch_v15` (default) | 768 | `--patch_size 512 --mag 20` |
| `virchow` | 2560 | `--patch_size 224 --mag 20` |
| `virchow2` | 2560 | `--patch_size 224 --mag 20` |
| `phikon` | 768 | `--patch_size 224 --mag 20` |
| `phikon_v2` | 1024 | `--patch_size 224 --mag 20` |
| `keep` | 768 | `--patch_size 256 --mag 20` |
| `gigapath` | 1536 | `--patch_size 256 --mag 20` |
| `hoptimus0` | 1536 | `--patch_size 224 --mag 20` |
| `hoptimus1` | 1536 | `--patch_size 224 --mag 20` |
| `h0-mini` | 768/1536 | `--patch_size 224 --mag 20` |
| `musk` | 1024 | `--patch_size 384 --mag 20` |
| `midnight12k` | 3072 | `--patch_size 224 --mag 20` |
| `openmidnight` | 1536 | `--patch_size 224 --mag 20` |
| `gpfm` | 1024 | `--patch_size 224 --mag 20` |
| `genbio-pathfm` | 4608 | `--patch_size 224 --mag 20` |
| `gemma4-e4b` / `gemma4-26b` | 768/1152 | `--patch_size 224 --mag 20` |
| `kaiko-vits8/vits16/vitb8/vitb16/vitl14` | 384/768/1024 | `--patch_size 256 --mag 20` |
| `lunit-vits8` | 384 | `--patch_size 224 --mag 20` |
| `hibou_l` | 1024 | `--patch_size 224 --mag 20` |
| `ctranspath` | 768 | `--patch_size 256 --mag 10` |
| `resnet50` | 1024 | `--patch_size 256 --mag 20` |

## Slide encoders

Loaded via `trident.slide_encoder_models.encoder_factory(name)`. Each pins a patch encoder;
pass its required patch_size/mag.

| Encoder (`--slide_encoder`) | Patch encoder | Required args |
|---|---|---|
| `titan` | conch_v15 | `--patch_size 512 --mag 20` |
| `prism` | virchow | `--patch_size 224 --mag 20` |
| `chief` | ctranspath | `--patch_size 256 --mag 10` |
| `gigapath` | gigapath | `--patch_size 256 --mag 20` |
| `madeleine` | conch_v1 | `--patch_size 256 --mag 10` |
| `feather` | conch_v15 | `--patch_size 512 --mag 20` |
| `feather_uni_v2` | uni_v2 | `--patch_size 256 --mag 20` |
| `care` | conch_v15 | `--patch_size 512 --mag 20` |
| `threads` | conch_v15 | `--patch_size 512 --mag 20` *(coming soon)* |

## Segmenters & artifact removal

- `hest` (default): general tissue-vs-background, GPU.
- `grandqc`: fast H&E (non-commercial license; cite the GrandQC paper).
- `otsu`: classical thresholding, **CPU-only**, no model download — the fallback when no GPU.
- `--remove_penmarks`: removes GrandQC class PenMarking (+Background). Gentle.
- `--remove_artifacts`: keeps **only** GrandQC class "Normal Tissue", removing Fold,
  Darkspot, PenMarking, Edge/Air Bubble, **OOF (out-of-focus)**, Background. Aggressive — a
  slide whose pyramid reads soft (e.g. some MIRAX `.mrxs`) can be classified OOF and fully
  erased. The pipeline then warns and records `reason="artifact_removal_emptied_tissue"` in
  `wsi_states/<slide>.json`; re-run with `--remove_penmarks`.

Segmentation is stored as GeoJSON (`contours_geojson/`), editable in
[QuPath](https://qupath.github.io/) and reloaded on rerun.

The CLI (and `GrandQCArtifactSegmenter.forward()`) only expose **binary keep/remove**. To inspect
the raw **7-class** artifact map yourself: `m = segmentation_model_factory('grandqc_artifact')`
exposes `input_size=512`, `target_mag=10` (MPP≈1.0) and ImageNet-normalized `eval_transforms`; tile
the **tissue region** (e.g. `wsi.create_patcher(patch_size=512, dst_pixel_size=10/m.target_mag,
mask=tissue_gdf)` — masking matters, or empty borders dominate) and take
`argmax(softmax(m.model.predict(tile)))` → class ids 1–7 (Normal Tissue … Background).

## WSI readers & formats

OpenSlide (`.svs`, `.tiff`, `.ndpi`, `.mrxs`, …), CuCIM, plain images (`.png`, `.jpeg`),
SDPC, OME-Zarr (`.zarr`, needs `.[omezarr]`), Zeiss CZI (`.czi`, needs `.[czi]`).
Multi-file formats like MIRAX (`.mrxs`) sit in `--wsi_dir` exactly like single-file slides:
the `.mrxs` index file next to its same-named data folder; the custom list references the
`.mrxs` by basename.

## Output artifacts — everything TRIDENT writes (under `--job_dir`)

Complete list. `<slide>` is the slide basename (no extension); `<cdir>` =
`<mag>x_<patch>px_<overlap>px_overlap` (e.g. `20x_256px_0px_overlap`); `<penc>`/`<senc>` =
patch/slide encoder name.

**Segmentation stage**
```
thumbnails/<slide>.jpg              downscaled slide thumbnail
contours/<slide>.jpg                thumbnail with the tissue contour drawn on top
contours_geojson/<slide>.geojson    tissue polygons (GeoJSON; open/edit in QuPath, reloaded on rerun)
_config_segmentation.json           the exact args used for the seg run
_logs_segmentation.txt              per-slide seg status line (key: "<slide><ext>")
```

**Coords (patching) stage — under `<cdir>/`**
```
<cdir>/patches/<slide>_patches.h5   patch coordinates (see h5 layout below)
<cdir>/visualization/<slide>.jpg    thumbnail with the patch grid drawn on top
<cdir>/patch_images/<slide>/*.png   patch image crops — ONLY when --dump_patches is set
<cdir>/_config_coords.json          args used for the coords run
<cdir>/_logs_coords.txt             per-slide coords status
```

**Patch feature stage — under `<cdir>/`**
```
<cdir>/features_<penc>/<slide>.h5   patch embeddings (see h5 layout below)
<cdir>/_config_feats_<penc>.json    args used for the feature run
<cdir>/_logs_feats_<penc>.txt       per-slide feature status
```

**Slide feature stage — under `<cdir>/`** (only with `--slide_encoder`)
```
<cdir>/slide_features_<senc>/<slide>.h5      one slide embedding, dataset "features" shape (dim,)
<cdir>/_config_slide_features_<senc>.json    args used
<cdir>/_logs_slide_features_<senc>.txt       per-slide status
```

**Run-level bookkeeping (job_dir root)**
```
summary.md                          human-readable report, one section appended per run
runs/<run_id>.json                  machine-readable manifest for that run (args, counts, timing)
wsi_states/<slide>__<hash>.json     per-slide state: task status/reason/message, attempts, outputs, resume info
<output>.lock                       transient lock guarding an in-flight output; auto-removed on completion.
                                    Stale ones (from a killed run) are cleared by --clear_dead_locks.
```

### `.h5` internal layout
- **`patches/<slide>_patches.h5`** — dataset `coords` `(n_patches, 2)` int64 (level-0 x,y). Attrs:
  `patch_size`, `overlap`, `target_magnification`, `patch_size_level0`, `level0_width`,
  `level0_height`, `level0_magnification`, `name`.
  - **`patch_size_level0` is the patch side in level-0 pixels** and **varies per slide**:
    `patch_size_level0 = patch_size × (level0_magnification / target_magnification)` (e.g. a
    `256px@20×` request → 256 on a 20×-native slide but 512 on a 40×-native one). Always read it
    from the `.h5` for level-0 cropping / overlays — never assume it equals `--patch_size`.
- **`features_<penc>/<slide>.h5`** — **self-contained**: holds `coords` (same dataset+attrs as
  above) *and* `features` `(n_patches, dim)` float32 with attrs `encoder`, `name`. So a feature
  file alone carries both embeddings and their patch coordinates.
- **`slide_features_<senc>/<slide>.h5`** — dataset `features` shape `(dim,)`.

## Python API (custom pipelines)

Use the public API when you need to embed TRIDENT in your own code.

Single slide, stage by stage. This mirrors `run_single_slide.py` exactly — note the path
conventions, which are easy to get wrong:
- segmentation runs at the **model's own** `target_mag` (`seg.target_mag`), not the patch mag.
- `save_coords` is the **per-config subdir** `"{job_dir}/{mag}x_{patch}px_{overlap}px_overlap"`,
  NOT the job_dir root. TRIDENT writes `patches/`, `visualization/`, `features_<enc>/` under it.
- `extract_tissue_coords` returns the coords `.h5` path; pass that to feature extraction.
- `device` must be consistent: move the encoder to the same device you pass to `extract_patch_features`.

```python
import os
from trident import load_wsi
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory

job_dir, device, mag, patch_size, overlap = "out", "cuda:0", 20, 256, 0
save_coords = os.path.join(job_dir, f"{mag}x_{patch_size}px_{overlap}px_overlap")

with load_wsi(slide_path="x.svs", lazy_init=False) as slide:
    seg = segmentation_model_factory("hest")          # use "otsu" + device="cpu" if no GPU
    slide.segment_tissue(segmentation_model=seg, target_mag=seg.target_mag,
                         job_dir=job_dir, device=device)

    coords_path = slide.extract_tissue_coords(target_mag=mag, patch_size=patch_size,
                                              save_coords=save_coords, overlap=overlap)
    slide.visualize_coords(coords_path=coords_path,
                           save_patch_viz=os.path.join(save_coords, "visualization"))

    enc = encoder_factory("uni_v1").to(device).eval()  # 256px/20x — match the encoder table
    slide.extract_patch_features(patch_encoder=enc, coords_path=coords_path,
                                 save_features=os.path.join(save_coords, "features_uni_v1"),
                                 device=device)
```

Batch via the orchestrator:

```python
from trident import Processor
proc = Processor(job_dir="out", wsi_source="./wsis", skip_errors=True)
# proc.run_segmentation_job(...) / run_patching_job(...) / run_patch_feature_extraction_job(...)
```

Registries: `from trident.patch_encoder_models import encoder_registry` (and the
`slide_encoder_models` equivalent) list valid names.

For overlays/visualizations, get a downsampled thumbnail with
`load_wsi(path, lazy_init=False).get_thumbnail((max_w, max_h))` → PIL image; then map level-0
`coords` onto it by dividing by the downsample `level0_width / thumb_width` and draw squares of
side `patch_size_level0 / downsample`.

## Install profiles & preflight

```bash
pip install -e .                     # core (transformers, timm, safetensors)
pip install -e ".[patch-encoders]"   # CONCH, MUSK, CTransPath/CHIEF, …
pip install -e ".[slide-encoders]"   # PRISM, GigaPath, Madeleine, …
pip install -e ".[omezarr]" / ".[czi]" / ".[convert]" / ".[full]"
trident-doctor --profile base
trident-doctor --profile patch-encoders --check-gated
```

Pin `timm==0.9.16`. Gated HF encoders need access approval + `huggingface-cli login`.
Some models need manual setup (e.g. local CHIEF path in
`trident/slide_encoder_models/local_ckpts.json`).

Compatibility notes (observed):
- Python 3.10/3.11 is recommended, but runs succeed on newer (e.g. 3.13) as long as `timm==0.9.16`.
- Some **slide encoders load HF remote code that breaks on `transformers` 5.x** — e.g. TITAN fails
  with `AttributeError: 'Titan' object has no attribute 'all_tied_weights_keys'`. If a slide
  encoder errors on load (not a gating/timm error), pin an older `transformers` (4.x), or — in a
  read-only/shared env — monkeypatch before load:
  `from transformers.modeling_utils import PreTrainedModel; PreTrainedModel.all_tied_weights_keys = {}`
  (the batch CLI spawns workers, so put it in a `sitecustomize.py` on `PYTHONPATH`). Note PRISM also
  pulls a heavy, version-pinned dependency set (`transformers==4.42.4`, `environs`, `sacremoses`).
- If the `trident-doctor` console script isn't on PATH (depends on the install), preflight with
  `python -c "import trident; from trident.patch_encoder_models import encoder_factory; encoder_factory('uni_v1')"`
  to confirm imports + gated-model access.

## Slide conversion

```bash
trident convert --input_dir ./wsis --mpp_csv ./to_process.csv \
  --job_dir ./pyramidal_tiff --downscale_by 1 --num_workers 1
```

`--mpp_csv` is required with columns `wsi,mpp` (`wsi` = filename relative to `--input_dir`);
only listed files convert, and an `mpp` value is required per row. Embedded MPP, if present,
is compared to the CSV and mismatches are logged. `--downscale_by 1` keeps full resolution;
raise `--num_workers` for parallelism. After converting, point the normal pipeline
(`run_batch_of_slides.py --wsi_dir`) at the conversion's `--job_dir`.
