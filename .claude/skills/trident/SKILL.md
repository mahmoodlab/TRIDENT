---
name: trident
description: >-
  Process whole-slide pathology images (WSIs) with TRIDENT: tissue segmentation,
  patch coordinate extraction, patch/slide feature (embedding) extraction with
  foundation models (UNI, CONCH, Virchow, Gemma, Titan, GigaPath, etc.), and
  cell/nuclei/structure segmentation (HistoPlus, CellViT++, and the promptable weave/SAM3), and
  VLM question answering over ROIs (Patho-R1). Use when the user works with WSIs
  (.svs/.tiff/.ndpi/.mrxs/.czi/.dcm), mentions TRIDENT, run_batch_of_slides /
  run_single_slide, tissue segmentation, patching, extracting pathology embeddings,
  cell/nuclei/prompted segmentation, or interrogating ROIs with a vision-language model, for downstream ML.
---

# TRIDENT — whole-slide image processing

TRIDENT runs a 3-stage pipeline over whole-slide images (WSIs):

```
tissue segmentation  →  patch coordinates  →  patch / slide embeddings   (--task feat)
   (--task seg)          (--task coords)    ├→  cell / nuclei segmentation (--task patch_seg)
                                            └→  VLM question answering     (--task vlm)
```

`feat`, `patch_seg`, and `vlm` all consume the same patch coordinates; pick whichever output you
need (or run several). Outputs are written under a single `--job_dir` and are **resumable** —
re-running the same command skips finished work.

**Tasks run one stage only — they do NOT auto-run prerequisites.** Pick `--task`:
- `--task all` — runs seg → coords → feat in one go (the usual choice). Requires a `--patch_encoder` (or `--slide_encoder`); it always produces features. (It does **not** run `patch_seg` or `vlm`.)
- `--task seg` / `--task coords` / `--task feat` — run *only* that stage. `coords` needs segmentation already done in `--job_dir`; `feat` needs coords already done (or supply `--coords_dir`). Running `--task coords` on a fresh `--job_dir` produces nothing (it skips with `geojson_not_found`).
- `--task patch_seg` — run an **instance segmentation model** over the tissue patches: a fixed-taxonomy cell/nuclei model (HistoPlus / CellViT++) or the **promptable** weave/SAM3 (segments what `--patch_seg_prompt` names). Like `feat`, it needs `seg` + `coords` done first. See the "Cell / nuclei segmentation" section below.
- `--task vlm` — interrogate the tissue patches with a **vision-language model** (Patho-R1): ask one free-text prompt of every patch, get a free-text answer. Like `feat`/`patch_seg`, it needs `seg` + `coords` done first. See the "VLM question answering" section below. (For a single ROI, use the `WSI.query_region` Python API — no coords needed.)
- **Seg + coords but no features?** There is no single flag — run two commands: `--task seg` then `--task coords` on the same `--job_dir`.

For the full CLI flag list, the complete encoder tables, output layout, and the Python
API, read **[reference.md](reference.md)**. Keep this file for the workflow and decisions.

## Setup (do this once, verify before big jobs)

```bash
pip install -e .                 # core; add ".[patch-encoders]" ".[slide-encoders]" ".[full]" as needed
trident-doctor --profile base    # preflight; use --profile <profile> --check-gated for model access
```

- **`timm==0.9.16`** is the default pin, but **`timm==1.0.8` also works for every encoder**
  (verified — `gigapath`/`hoptimus` give bit-identical features; needed for HistoPlus). Avoid other
  timm 1.x releases (untested → cryptic model-build errors). Python 3.10/3.11 is recommended but
  newer (3.13) works.
- If `trident-doctor` isn't on PATH (install-dependent), preflight instead with
  `python -c "import trident; from trident.patch_encoder_models import encoder_factory; encoder_factory('uni_v1')"`.
- Most encoders download from HuggingFace; gated models (UNI, CONCH, Virchow, …) need an
  approved HF account and `huggingface-cli login`. A load failure usually means missing
  access or a missing optional install — read the error, it names the fix.
- A **slide encoder** that errors on load with something like `all_tied_weights_keys` (not a
  gating/timm error) is a `transformers` 5.x incompatibility (e.g. TITAN) — pin `transformers` 4.x,
  or if you can't change the env, set
  `transformers.modeling_utils.PreTrainedModel.all_tied_weights_keys = {}` before loading (for the
  batch CLI's spawned workers, put that in a `sitecustomize.py` on `PYTHONPATH`).

## The one command users want first

```bash
python run_batch_of_slides.py --task all \
  --wsi_dir ./wsis --job_dir ./trident_processed \
  --patch_encoder uni_v1 --mag 20 --patch_size 256 --gpus 0
```

Segments tissue, extracts patches at 20× / 256px, and writes UNI embeddings. For a single
slide (cautious first run), use `run_single_slide.py --slide_path ./wsis/x.svs ...` with the
same flags. From another project, the CLI equivalents are `trident batch -- ...` /
`trident single -- ...`.

## Decisions you must get right

**1. The encoder dictates `--patch_size` and `--mag` — do not pick them freely.**
Each model was trained at a specific resolution; mismatched values give garbage features.
Always copy the pair from the encoder table in [reference.md](reference.md). Common ones:

| Encoder | use |
|---|---|
| `uni_v1` (1024-d) | `--patch_size 256 --mag 20` |
| `uni_v2` (1536-d) | `--patch_size 256 --mag 20` |
| `conch_v15` (768-d, default) | `--patch_size 512 --mag 20` |
| `virchow` / `virchow2` (2560-d) | `--patch_size 224 --mag 20` |
| `ctranspath` (768-d) | `--patch_size 256 --mag 10` |

**2. Patch vs slide embeddings.** `--patch_encoder X` → one embedding per patch
(`features_X/`, shape `(n_patches, dim)`). `--slide_encoder Y` → one embedding per slide
(`slide_features_Y/`, shape `(dim,)`); it auto-runs the correct patch encoder first. Pass
the slide encoder's required patch_size/mag (from the slide-encoder table). **Still pass
`--task all`** (or `feat`) with `--slide_encoder` — on its own, the default `--task seg`
only segments and you get no embeddings. A slide-encoder run also writes the intermediate
patch features (`features_<patch_encoder>/`) alongside `slide_features_<Y>/`.
("UNI" = `uni_v1`; "UNI2"/"UNI2-h" = `uni_v2`.) The named slide encoders are pretrained
and frozen; to **train your own** aggregator on the extracted patch features, TRIDENT also
ships a randomly-initialized `abmil` slide encoder (set `input_feature_dim` to the patch
encoder's dim) — see the "Trainable ABMIL aggregator" section in [reference.md](reference.md).

**3. Segmenter.** Default `--segmenter hest` (a model — runs on GPU). `grandqc` = fast H&E.
`otsu` = classical, **CPU-only** — on a machine with no GPU you must pass `--segmenter otsu`
explicitly (the default `hest` expects a GPU). For segmentation, `--gpus -1` and `otsu`
go together.
- If segmentation misses tissue, lower `--seg_conf_thresh` (default 0.5 → try 0.4) to retain more.
- Optional clean-up: `--remove_penmarks` (gentle) or `--remove_artifacts` (aggressive:
  folds, blur, stains, OOF…).
- ⚠️ `--remove_artifacts` keeps only "normal tissue" and can erase an entire slide whose
  pyramid reads soft (e.g. some MIRAX/`.mrxs` slides → flagged out-of-focus). If a slide
  ends up with **no patches/features**, check the slide's `wsi_states/` JSON for reason
  `artifact_removal_emptied_tissue` and re-run that slide with `--remove_penmarks` (it trades
  away fold/blur/OOF removal to keep the tissue). The two flags are not combinable.

**4. Compute.** `--gpus 0 1 2` shards pending slides across GPUs; `--gpus -1` forces CPU.
Tune throughput with `--batch_size` / `--feat_batch_size` / `--seg_batch_size` (raise on big
GPUs). For slow/network storage, stage slides locally with `--wsi_cache /local/ssd
--cache_batch_size 32`.

**5. mpp / custom inputs.** To process a **subset**, or to supply **micron-per-pixel** for
slides that lack embedded metadata, use `--custom_list_of_wsis list.csv`. The CSV has a
`wsi` column (filename incl. extension, relative to `--wsi_dir`) and, when slides have no
embedded mpp, a required `mpp` column with the value for each row:

```csv
wsi,mpp
slide_001.tiff,0.5
slide_002.tiff,0.25
```

Use the CSV `mpp` column to *supply* a value; use `--custom_mpp_keys` only when the value
*is* in the slide metadata under a non-standard key. `--mag` in batch mode accepts floats
(e.g. `1.25`).

**6. Reuse existing patch coordinates** (e.g. legacy CLAM `*_fp/`): skip seg+patching and
point feat at them — `--task feat --coords_dir ./extracted_mag20x_patch256_fp ...` with the
encoder's required patch_size/mag (must match the coords' resolution). `--wsi_dir` is still
required (features read pixels from the WSIs).

## Cell / nuclei segmentation (`--task patch_seg`)

Instance segmentation across the tissue patches — either individual cells/nuclei with a fixed
taxonomy (HistoPlus / CellViT++), or an arbitrary prompted structure (weave/SAM3). Another
consumer of the coords from `--task coords`; run `seg` + `coords` first (or a prior `all`), then:

```bash
python run_batch_of_slides.py --task patch_seg \
  --wsi_dir ./wsis --job_dir ./trident_processed \
  --patch_segmenter histoplus --mag 20 --patch_size 784 \
  --seg_viz --gpus 0
```

**Batch size:** `--patch_seg_batch_size` (default `4`); lower it if you OOM. CellViT++ at 1024px is
memory-heavy (~3.8 GB/patch → `8`≈31 GB, `32` OOMs a 95 GB GPU); HistoPlus is lighter.

Three models. HistoPlus/CellViT++ have a **fixed cell taxonomy and required resolution** (copy
verbatim); weave is **promptable** — it segments whatever a text prompt names, at any resolution:

| `--patch_segmenter` | Classes | Required args | Install |
|---|---|---|---|
| `histoplus` | 14 cell types | `--patch_size 784 --mag 20` (or `--mag 40`) | `pip install --no-deps git+https://github.com/owkin/histoplus.git` + `pip install timm==1.0.8` (not on PyPI); gated HF weights |
| `cellvit_plus_plus` | 5 (PanNuke) | `--patch_size 1024 --mag 40` | `pip install cellvit` |
| `weave` | promptable (prompt = the class) | **`--patch_seg_prompt "…"`**; any `--mag`/`--patch_size`; **single GPU** | `pip install 'git+https://github.com/JaumeLab/sam3.git'` + `pip install pycocotools` (not on PyPI); gated HF weights |

`--mag 40` needs a **40×-native** slide (mpp ≈ 0.25); on a 20×-native slide (`objective-power 20`)
40× is unreachable — use HistoPlus@`--mag 20` or a 40× slide for CellViT++.

**weave (promptable SAM3).** JaumeLab's SAM3 finetuned for histopathology — segments whatever
`--patch_seg_prompt` names (e.g. `"tumor"`, `"glomeruli"`) in every patch, instead of a fixed cell
taxonomy. **Any** `--mag`/`--patch_size` works (SAM3 resizes internally, like a VLM). `--patch_seg_conf_thresh`
(default 0.5) sets the score threshold. Only text-prompt whole-slide mode is exposed (no box prompt).
Install needs `pycocotools` too. **Single GPU only** (`--gpus 0`); for another physical GPU use
`CUDA_VISIBLE_DEVICES=<n>` + `--gpus 0`. Its deps (timm 1.x, numpy 1.x, transformers 5.x) conflict
with TRIDENT's feature-encoder / HistoPlus pins — use a dedicated env if mixing.

**Output = semantic region map by default.** SAM3 produces per-instance masks, but weave segments
*regions*, so TRIDENT dissolves the per-tile polygons into contiguous same-class regions by default —
unioning across patch seams (with a morphological-close bridge, so seams vanish even at `--overlap 0`)
and dropping within-tile duplicate masks. `class_name` = the prompt, `confidence` = area-weighted mean.
Pass `--patch_seg_no_dissolve` to keep the **raw per-instance** polygons (instance-level, own scores),
like HistoPlus/CellViT++. So: `--overlap` is **optional** (dissolve already removes seams); adding it
only helps the model see edge-crossing structures whole (slightly better boundaries, more compute).

Output is keyed **per prompt**: `seg_weave_<prompt>/` (e.g. `seg_weave_tumor/`, `seg_weave_necrosis/`),
with matching `_config_seg_weave_<prompt>.json` / `_logs_...`. So you can run several prompts on the
same `--job_dir`/coords and they coexist (and resume independently) instead of overwriting each other.

```bash
python run_batch_of_slides.py --task patch_seg \
  --wsi_dir ./wsis --job_dir ./trident_processed \
  --patch_segmenter weave --patch_seg_prompt "tumor" \
  --mag 20 --patch_size 1024 --seg_viz --gpus 0
```

Install — **both run in the TRIDENT env** (no separate env needed):
- **CellViT++:** `pip install cellvit` (Python 3.10/3.11; on 3.13 use `--no-deps` then add
  `colorama colour geojson natsort opt-einsum pyaml`). Weights auto-download from Zenodo.
- **HistoPlus:** `pip install --no-deps git+https://github.com/owkin/histoplus.git && pip install timm==1.0.8`.
  Not on PyPI; weights gated on [HF](https://huggingface.co/Owkin-Bioptimus/histoplus) (accept the
  license + `huggingface-cli login`). `timm 1.0.8` is required by HistoPlus and verified safe for all
  TRIDENT encoders. (TRIDENT handles HistoPlus's xformers quirk automatically — no env vars needed.)
- **Batch size:** `--patch_seg_batch_size` (default `4`). Both models batch fine; lower it if you OOM
  (CellViT++ at 1024px is ~3.8 GB/patch).
- `--seg_viz` (optional) also writes debug overlays with a color→cell-type legend.
- Output dir is keyed per model: `<cdir>/seg_<model>/` — for weave `<model>` is `weave_<prompt>` (so prompts coexist). Outputs: a QuPath-ready
  GeoJSON of per-cell polygons + a compact HDF5 + (with `--seg_viz`) visualizations.

## VLM question answering (`--task vlm`)

Interrogate tissue with a **vision-language model** (Patho-R1, a Qwen2.5-VL pathology
reasoner): give a free-text prompt, get a free-text answer. Ask the same prompt of *every*
patch (another consumer of the coords from `--task coords`; run `seg` + `coords` first, or a
prior `all`):

```bash
python run_batch_of_slides.py --task vlm \
  --wsi_dir ./wsis --job_dir ./trident_processed \
  --vlm patho_r1_7b --vlm_prompt "Is tumor present? Describe the tissue." \
  --mag 20 --patch_size 512 --gpus 0
```

To query a **single ROI** programmatically (no coords needed), use the Python API
`WSI.query_region(vlm, prompt, location, size, mag)` — see reference.md.

| `--vlm` | Backbone | Args | Install |
|---|---|---|---|
| `patho_r1_7b` (default) | Qwen2.5-VL, ~16 GB bf16 | any `--mag` / `--patch_size` (e.g. `--mag 20 --patch_size 512`) | `pip install "transformers>=4.49" accelerate qwen-vl-utils` |
| `patho_r1_3b` | Qwen2.5-VL, ~8 GB bf16 | any `--mag` / `--patch_size` | same |

- **Any magnification works** — unlike the patch/slide encoders (whose `--mag`/`--patch_size` are
  fixed by training; a mismatch gives garbage features), a VLM accepts arbitrary input sizes. Pick
  `--mag`/`--patch_size` to frame the field of view you want; there is no required pair. A lower
  `--mag` or larger `--patch_size` gives more context and fewer (so faster) patches.
- **Runs in the TRIDENT env** (no separate env). Weights auto-download from HF on first use;
  **CC-BY-NC-ND-4.0 (non-commercial)**.
- **Slow:** generation is autoregressive and the batch task sweeps every patch — prefer a tight
  coords set, a coarser field of view (a **lower** `--mag` or **larger** `--patch_size` covers more
  tissue per patch → fewer patches), or a single-ROI `WSI.query_region` call. It is **not** part of `--task all`.
- **Batch size:** `--vlm_batch_size` (default `4`); lower it if you OOM. `--vlm_max_new_tokens`
  (default `512`) caps answer length. Other flags: `--vlm_ckpt_path` (local weights/repo).
- Output dir is keyed per model: `<cdir>/vlm_<model>/` (see Outputs) — a `<slide>.json` of
  per-patch answers + a QuPath-ready `<slide>.geojson` (patch boxes carrying the answer).
- ⚠️ Answers can be **confidently wrong** (like any LLM) — never use them for clinical decisions.

## Overlays & visualization (Python API)

To **overlay** tissue or cell/nuclei segmentation on a slide, use the native `WSI.overlay` —
it reads the GeoJSON the pipeline already writes (`contours_geojson/<slide>.geojson` for
tissue, `seg_<model>/<slide>.geojson` for cells) and renders on a whole-slide thumbnail or a
cropped ROI:

```python
from trident import load_wsi
with load_wsi("./wsis/x.svs", lazy_init=False) as slide:
    slide.overlay("./out/contours_geojson/x.geojson", mode="fill", saveto="tissue.jpg")
    slide.overlay("./out/20x_784px_0px_overlap/seg_histoplus/x.geojson",
                  region=(10000, 8000, 4096, 4096), color_by="class", saveto="roi_cells.jpg")
```

`mode="outline"` (boundaries) or `"fill"` (translucent, holes preserved); `region=(x,y,w,h)`
in level-0 px crops an ROI (omit → whole-slide thumbnail); `color_by="class"` colors per cell
type with a legend. Returns a PIL image. For per-patch **score/attention heatmaps** use
`from trident import visualize_heatmap`. During `--task patch_seg`, pass `--seg_viz` to auto-write
overlays. Full signature + the shared `render_overlay` core: see [reference.md](reference.md).

## Outputs (under `--job_dir`)

```
thumbnails/<slide>.jpg                slide thumbnail
contours/<slide>.jpg                  thumbnail + tissue contour overlay
contours_geojson/<slide>.geojson      tissue polygons (editable in QuPath)
_config_segmentation.json             args used for the seg run
_logs_segmentation.txt                per-slide seg status
<mag>x_<ps>px_<ov>px_overlap/         e.g. 20x_256px_0px_overlap/
    patches/<slide>_patches.h5        patch coords (dataset `coords`, (n,2), + patch attrs)
    visualization/<slide>.jpg         patch-grid overlay
    patch_images/<slide>/*.png        patch image crops — only with --dump_patches
    features_<enc>/<slide>.h5         patch embeddings: datasets `features` (n,dim) + `coords`
    slide_features_<enc>/<slide>.h5   slide embedding `features` (dim,) — only with --slide_encoder
    seg_<model>/<slide>.geojson       per-cell polygons + class/class_name/confidence — only with --task patch_seg
    seg_<model>/<slide>.h5            compact cells: contours+contour_offsets, centroids, class_ids, confidences
    seg_<model>/visualization/        <slide>_overview.jpg + <slide>/ patch overlays — only with --seg_viz
    vlm_<model>/<slide>.json          per-patch VLM answers {model,prompt,answers:[{x,y,prompt,answer}]} — only with --task vlm
    vlm_<model>/<slide>.geojson       one patch box per answer, carrying prompt/answer (QuPath)
    _config_coords.json / _config_feats_<enc>.json / _config_slide_features_<enc>.json / _config_vlm_<model>.json
    _logs_coords.txt / _logs_feats_<enc>.txt / _logs_slide_features_<enc>.txt / _logs_vlm_<model>.txt
summary.md                            human-readable run report (one section per run)
runs/<id>.json                        machine-readable run manifest
wsi_states/<slide>__<hash>.json       per-slide tasks, attempts, errors, resume reason
<output>.lock                         transient in-flight guard (cleared by --clear_dead_locks)
```

A feature `.h5` is **self-contained** (it stores `coords` alongside `features`). The
patch/feature subdir is named only from `mag/patch/overlap`, so changing those creates a
*separate* folder rather than reusing one. Full per-artifact detail + h5 attrs: see
[reference.md](reference.md).

To inspect results: open an `.h5` with `h5py` and read `coords` / `features`; check
`summary.md` or `wsi_states/` to see what succeeded, was skipped, or errored, and why.

To also export the actual **patch images** (not just coordinates), add `--dump_patches`
to a `coords`/`all` run — writes PNGs (or `--dump_patches_format jpg`) to
`<coords_dir>/patch_images/<slide>/`, capped by `--dump_patches_max` (see reference.md).

## Workflow for the agent

1. Confirm the goal: which **encoder** (drives patch_size/mag), patch vs slide embeddings,
   GPU availability.
2. Verify env (`trident-doctor`, or the `encoder_factory('uni_v1')` import fallback if it's not
   on PATH), and HF access for gated encoders.
3. Build the command from the recipe above; pull the exact patch_size/mag from
   [reference.md](reference.md). Prefer `run_single_slide.py` to validate on one slide first.
4. Run; then **verify**: confirm the expected `features_<enc>/` / `slide_features_<enc>/`
   files exist with sensible shapes, and scan `wsi_states/` for skips/errors. Report any
   slide that produced no output and the reason.
5. Re-run on the **same `--job_dir` with identical `--mag`/`--patch_size`/`--overlap`/encoder**
   to resume (changing them starts a new output dir instead of resuming). Finished slides are
   skipped; errored/unfinished ones are retried. Add `--clear_dead_locks` only if a previous
   run was killed and left stale `.lock` files.

## Common pitfalls

- Mismatched `--patch_size`/`--mag` for the chosen encoder → meaningless features.
- **`--overlap` must be `< --patch_size`.** `--overlap >= --patch_size` makes the patch step ≤ 0 and **hangs forever** (no error). Overlap is absolute pixels (e.g. `128` = 50% of a 256px patch).
- Re-patching with a changed `--min_tissue_proportion` (or seg settings) on an existing `--job_dir` has **no effect** — the coords folder is keyed only by `mag/patch/overlap`, so the old coords are reused. Use a fresh `--job_dir` to re-patch.
- `--patch_encoder_img_size` and `--patch_encoder_ckpt_path` apply only to `--patch_encoder`; they are silently ignored when `--slide_encoder` is set.
- `--task coords`/`feat` on a fresh `--job_dir` → silently skips (no prior stage); use `--task all`, or run the stages in order.
- `--slide_encoder` without `--task all`/`feat` → only segmentation runs, no embeddings.
- No-GPU machine without `--segmenter otsu` → default `hest` tries to use a GPU.
- `timm` not pinned to `0.9.16` (Python 3.10/3.11) → cryptic model-build errors; a bad timm can also look like a model *load* failure.
- Gated HF model without access → load failure (request access + `huggingface-cli login`).
- Empty output after `--remove_artifacts` → see Decision 3.
- Changing `--mag`/`--patch_size`/`--overlap` on a rerun → new output folder instead of a resume.
- Wrong reader auto-detected → force it with `--reader_type {openslide,image,cucim,sdpc,omezarr,czi}`.
- `--task patch_seg` "not installed" → install the model **into the TRIDENT env** (CellViT++ from PyPI; HistoPlus `--no-deps` from git + `timm==1.0.8`, gated weights; weave from `git+https://github.com/JaumeLab/sam3.git` **plus `pip install pycocotools`**, gated weights). Out of GPU memory → lower `--patch_seg_batch_size`. See the cell-segmentation section.
- `--patch_segmenter weave` errors immediately with "requires --patch_seg_prompt" → weave is promptable; pass e.g. `--patch_seg_prompt "tumor"`. (The prompt is ignored by the fixed-taxonomy cell models.)
- `weave` "only runs on the default CUDA device (cuda:0)" → SAM3 pins internal tensors to cuda:0; run with `--gpus 0` (no multi-GPU sharding). For another physical GPU: `CUDA_VISIBLE_DEVICES=<n>` + `--gpus 0`.
- `--task vlm` ImportError → `pip install "transformers>=4.49" accelerate qwen-vl-utils`. Out of GPU memory → lower `--vlm_batch_size`. Whole-slide `vlm` is slow (autoregressive, every patch) — for a single region use the `WSI.query_region` Python API, not the batch sweep. `vlm` is **not** part of `--task all`. See the VLM section.
