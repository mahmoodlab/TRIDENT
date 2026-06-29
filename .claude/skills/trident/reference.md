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
- Cell segmenters (`--task patch_seg`) — HistoPlus, CellViT++
- Vision-language models (`--task vlm` / `run_query_roi.py`) — Patho-R1
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
| `python run_query_roi.py ...` | Interrogate **one** ROI with a VLM (interactive; no coords needed). |

`--task` ∈ {`seg`, `coords`, `feat`, `patch_seg`, `vlm`, `all`}. **Each value runs exactly one
stage; only `all` chains them (seg → coords → feat — not `patch_seg`/`vlm`).** `coords` reads the
segmentation from `--job_dir` (skips with `geojson_not_found` if absent); `feat`, `patch_seg`, and
`vlm` read coords from `--job_dir` (or `feat` from `--coords_dir`). So `coords`/`feat`/`patch_seg`/
`vlm` on a fresh `--job_dir` do nothing — run `all`, or run the stages in order. For seg+coords
without features, run `--task seg` then `--task coords`. `patch_seg` (cell/nuclei segmentation) and
`vlm` (VLM question answering) are alternative consumers of coords — see their sections.

## `run_batch_of_slides.py` flags

**Core**
- `--task {seg,coords,feat,patch_seg,vlm,all}` (default `seg`)
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

**Cell segmentation (`--task patch_seg`)**
- `--patch_segmenter {histoplus,cellvit_plus_plus}` (default `histoplus`) — see Cell segmenters table.
- `--patch_seg_batch_size INT` (default `4`) — patches per batch for cell/nuclei segmentation. Lower
  it if you run out of GPU memory (CellViT++ at 1024px is ~3.8 GB/patch → `8`≈31 GB, `32` OOMs a 95 GB GPU).
- `--patch_segmenter_ckpt_path PATH` — local model weights (offline/air-gapped); otherwise the model
  package downloads them (HistoPlus from gated HF; CellViT++ from Zenodo).
- `--seg_viz` — also write debug overlays (slide overview + full-res sample patches) with a
  color→cell-type legend.

**VLM question answering (`--task vlm`)**
- `--vlm {patho_r1_7b,patho_r1_3b}` (default `patho_r1_7b`) — see Vision-language models table.
- `--vlm_prompt STR` (default `"Describe the tissue in this region."`) — the question asked of
  every tissue patch.
- `--vlm_batch_size INT` (default `4`) — patches per generation batch. Lower it if you OOM.
- `--vlm_max_new_tokens INT` (default `512`) — max tokens generated per patch answer.
- `--vlm_ckpt_path PATH` — local weights / HF repo (offline); otherwise auto-downloaded from HF.

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

## Cell segmenters (`--task patch_seg`)

Cell/nuclei instance segmentation + classification over the tissue patches. Each model lives in
a **separate package** (TRIDENT only wraps it — no model code is vendored). Both install **into the
TRIDENT env** (a separate env is not required — verified). Loaded via
`trident.patch_segmentation_models.patch_segmenter_factory(name)`.

| Encoder (`--patch_segmenter`) | Cell types | Required args | Install (into the TRIDENT env) |
|---|---|---|---|
| `histoplus` | 14 (pan-cancer) | `--patch_size 784 --mag 20` (mpp 0.5) or `--mag 40` (mpp 0.25) | `pip install --no-deps git+https://github.com/owkin/histoplus.git` then `pip install "timm==1.0.8"` — **not on PyPI**; weights **gated** on [HF](https://huggingface.co/Owkin-Bioptimus/histoplus) (CC-BY-NC-ND, needs `HF_TOKEN`) |
| `cellvit_plus_plus` | 5 (PanNuke) | `--patch_size 1024 --mag 40` | `pip install cellvit` — Python 3.10/3.11; on 3.13 its pinned Shapely fails to build → `--no-deps` + `colorama colour geojson natsort opt-einsum pyaml`; checkpoint auto-downloads from Zenodo |

Attribution: HistoPlus — Adjadj/Bannier/Horent et al., arXiv:2508.09926 ([owkin/histoplus](https://github.com/owkin/histoplus)).
CellViT++ — Hörst et al., arXiv:2501.05269 ([TIO-IKIM/CellViT-Plus-Plus](https://github.com/TIO-IKIM/CellViT-Plus-Plus), Apache-2.0 + Commons Clause).

HistoPlus installs into the TRIDENT env — `--no-deps` keeps it from disturbing TRIDENT's/CellViT++'s
pins, and `timm==1.0.8` (which HistoPlus needs and is verified bit-identical for all TRIDENT encoders)
is the only bump. TRIDENT sets the `XFORMERS_IGNORE_FLASH_VERSION_CHECK` env var itself when loading
HistoPlus, so no manual step is needed. (If you later re-run `pip install -e .` it reverts timm to
0.9.16 — just `pip install timm==1.0.8` again.)

Notes:
- **Batch size:** set with `--patch_seg_batch_size` (default `4`). Both models batch fine; lower it if
  you OOM (CellViT++ at 1024px is ~3.8 GB/patch — `8`≈31 GB, `16`≈59 GB, `32` OOMs a 95 GB GPU; HistoPlus
  is lighter, ~1.5 GB/patch).
- `--mag 40` requires a 40×-native slide (mpp ≈ 0.25); a 20×-native slide cannot be upsampled to 40×.
- Output goes to `<cdir>/seg_<model>/` (per model). See Output artifacts.
- `--task patch_seg` runs only this stage; it needs `seg` + `coords` already in `--job_dir`.

## Vision-language models (`--task vlm` / `run_query_roi.py`)

Generative **image+prompt → text** question answering over tissue. Patho-R1 is a Qwen2.5-VL-based
pathology reasoner. The wrapper only drives `transformers` (no model code vendored). Loaded via
`trident.vlm_models.vlm_factory(name)`.

| Model (`--vlm`) | Backbone / memory | Required args | Install (into the TRIDENT env) |
|---|---|---|---|
| `patho_r1_7b` (default) | Qwen2.5-VL, ~16 GB bf16 | `--mag 20 --patch_size 512` | `pip install "transformers>=4.49" accelerate qwen-vl-utils`; weights auto-download from [HF](https://huggingface.co/WenchuanZhang/Patho-R1-7B) (**CC-BY-NC-ND-4.0**, non-commercial) |
| `patho_r1_3b` | Qwen2.5-VL, ~8 GB bf16 | `--mag 20 --patch_size 512` | same; [WenchuanZhang/Patho-R1-3B](https://huggingface.co/WenchuanZhang/Patho-R1-3B) |

Attribution: Zhang et al., *"Patho-R1: A Multimodal Reinforcement Learning-Based Pathology Expert
Reasoner"*, arXiv:2505.11404.

Two modes:
- **Batch** (`--task vlm`): asks `--vlm_prompt` of *every* tissue patch; needs `seg` + `coords`
  already in `--job_dir`. Output goes to `<cdir>/vlm_<model>/` (per model). See Output artifacts.
- **Interactive** (`run_query_roi.py`): asks one prompt of one ROI; no coords needed.

`run_query_roi.py` flags: `--slide_path PATH` (required), `--prompt STR` (required),
`--location X Y` (level-0 px, required), `--size INT` (square edge in px at `--mag`, default 512),
`--mag FLOAT` (default: slide-native), `--vlm NAME` (default `patho_r1_7b`), `--vlm_ckpt_path PATH`,
`--max_new_tokens INT` (default 512), `--gpu INT` (`-1` for CPU), `--reader_type`, `--custom_mpp_keys`.

Notes:
- **Slow / autoregressive.** The batch task sweeps every patch and generation is token-by-token —
  far slower than the feed-forward encoders. Prefer a tight coords set, a higher `--mag` / larger
  `--patch_size` (fewer patches), or the interactive ROI path. `vlm` is **not** part of `--task all`.
- **Batch size** with `--vlm_batch_size` (default `4`); lower it if you OOM. `--vlm_max_new_tokens`
  caps answer length.
- Like any LLM, answers can be confidently wrong — not for clinical use.

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

**Cell segmentation stage — under `<cdir>/`** (only with `--task patch_seg`)
```
<cdir>/seg_<model>/<slide>.geojson         per-cell polygons (level-0 coords) + class/class_name/confidence; open in QuPath
<cdir>/seg_<model>/<slide>.h5              compact cells (see h5 layout below)
<cdir>/seg_<model>/visualization/<slide>_overview.jpg   slide overview with cells + legend — only with --seg_viz
<cdir>/seg_<model>/visualization/<slide>/<x>_<y>.jpg    full-res sample-patch overlays — only with --seg_viz
<cdir>/_config_seg_<model>.json            args used
<cdir>/_logs_seg_<model>.txt               per-slide status
```
`<model>` is the segmenter name (`histoplus`, `cellvit_plus_plus`).

**VLM question-answering stage — under `<cdir>/`** (only with `--task vlm`)
```
<cdir>/vlm_<model>/<slide>.json            {model, prompt, answers:[{x,y,prompt,answer}]} — one entry per patch (level-0 x,y)
<cdir>/vlm_<model>/<slide>.geojson         one patch box per answer (level-0 coords) with prompt/answer properties; open in QuPath
<cdir>/_config_vlm_<model>.json            args used
<cdir>/_logs_vlm_<model>.txt               per-slide status
```
`<model>` is the VLM name (`patho_r1_7b`, `patho_r1_3b`).

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
- **`seg_<model>/<slide>.h5`** (cell segmentation) — group `cells` with ragged polygons:
  `contours` `(M,2)` float32 (all vertices, level-0) + `contour_offsets` `(N+1,)` (cell `i` =
  `contours[offsets[i]:offsets[i+1]]`), plus `centroids` `(N,2)`, `class_ids` `(N,)`,
  `confidences` `(N,)`. Group attrs: `model`, `class_names` (JSON), `mpp`, source coords meta.

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
`slide_encoder_models` equivalent) list valid names. For cell segmentation:
`from trident.patch_segmentation_models import patch_segmenter_factory, patch_segmenter_registry`
then `proc.run_patch_segmentation_job(coords_dir=..., patch_segmenter=patch_segmenter_factory("histoplus"), device="cuda:0", batch_limit=1, visualize=True)` (or, single-slide, `slide.segment_patches(...)`).

For VLM question answering: `from trident.vlm_models import vlm_factory, vlm_registry`, then
`proc.run_vlm_query_job(coords_dir=..., vlm=vlm_factory("patho_r1_7b"), prompt="Is tumor present?", device="cuda:0", batch_limit=4)` for the batch sweep. For a single ROI, skip coords entirely:
`slide.query_region(vlm, "Describe this region.", location=(x, y), size=512, mag=20)` returns the
answer string (`location` is level-0 px, `size` is the edge in px at `mag`). `vlm.generate(images, prompts)` is the low-level call (one prompt per image, or one prompt broadcast to all).

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

Pin `timm==0.9.16` (default) **or `timm==1.0.8`** — both verified bit-identical across all encoders;
1.0.8 is required if HistoPlus shares the env. Gated HF encoders need access approval + `huggingface-cli login`.
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
