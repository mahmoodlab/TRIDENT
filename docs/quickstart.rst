Quickstart
==========

This page is a working reference for the CLI: what to run, what knobs matter, and
where outputs land.

If you are new: start with one slide, then scale up.

.. code-block:: bash

   # 1. validate settings on one slide
   python run_single_slide.py --slide_path ./wsis/example.svs --job_dir ./out \
       --patch_encoder uni_v1 --mag 20 --patch_size 256

   # 2. run the full batch when happy
   python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./out \
       --patch_encoder uni_v1 --mag 20 --patch_size 256 --skip_errors

End-to-end pipeline
-------------------

``--task all`` runs the three stages in order. You can also run them individually
on the same ``--job_dir`` — TRIDENT will pick up where it left off.

.. code-block:: bash

   python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./out \
       --patch_encoder uni_v1 --mag 20 --patch_size 256

This produces:

1. **Tissue segmentation** (``contours/``, ``contours_geojson/``, ``thumbnails/``).
2. **Patch coordinates** (``<mag>x_<patch>px_<overlap>px_overlap/patches/<slide>_patches.h5``).
3. **Patch features** (``<mag>x_<patch>px_<overlap>px_overlap/features_<encoder>/<slide>.h5``).

Equivalent unified CLI:

.. code-block:: bash

   trident batch  -- --task all --wsi_dir ./wsis --job_dir ./out --patch_encoder uni_v1 --mag 20 --patch_size 256
   trident single -- --slide_path ./wsis/example.svs --job_dir ./out --patch_encoder uni_v1 --mag 20 --patch_size 256
   trident doctor -- --profile base

Outputs and run tracking
------------------------

In your ``--job_dir``, TRIDENT writes:

- ``summary.md``: appended once per run; counts (completed / skipped / errored), per-encoder breakdown, and a short error list.
- ``runs/<run_id>.json``: one manifest per CLI invocation (args, timestamps, status).
- ``wsi_states/<slide>__<hash>.json``: per-slide machine-readable state (tasks, attempts, outputs, last error, resume info).
- ``contours/`` + ``contours_geojson/``: tissue masks (open ``.geojson`` in `QuPath <https://qupath.github.io/>`_ to QC/edit).
- ``<mag>x_<patch>px_<overlap>px_overlap/``: per-config coords and feature dirs.

Resume and skip behavior
------------------------

Re-running on the same ``--job_dir`` is the recommended way to retry / extend a job:

- If the expected output for a (slide, task) already exists and is **not locked**, TRIDENT
  marks the task **skipped**. No recomputation.
- ``.lock`` files mark tasks that are currently being written. If a worker crashes mid-task,
  the lock can become stale (an "orphan"). Clean those safely with:

  .. code-block:: bash

     python run_batch_of_slides.py --clear_dead_locks --dead_lock_max_age_hours 24 \
         --task all --wsi_dir ./wsis --job_dir ./out ...

  This removes only locks where (a) the target output already exists, or (b) the writer
  PID is dead on this host, or (c) the lock is unreadable / legacy and older than
  ``--dead_lock_max_age_hours`` (default 24). **Active locks from running jobs are never removed.**

Multi-GPU and multi-worker
--------------------------

Use ``--gpus`` to shard pending slides across devices:

.. code-block:: bash

   # Two GPUs (production)
   python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./out \
       --patch_encoder uni_v1 --mag 20 --patch_size 256 --gpus 0 1

   # Two CPU workers (no GPU)
   python run_batch_of_slides.py --task seg --wsi_dir ./wsis --job_dir ./out \
       --segmenter otsu --gpus -1 -1

Notes:

- Pending slides are sharded round-robin across the listed GPU IDs.
- Duplicate **positive** GPU IDs are deduplicated (running two workers on the same CUDA
  device wastes memory). Duplicate ``-1`` entries are kept (each is an independent CPU worker).
- ``--gpu`` (singular) is the legacy form and still works, but prefer ``--gpus``.

Caching for slow / network storage
----------------------------------

If WSIs sit on a slow network drive, copy them in batches to a local SSD via the
producer/consumer cache pipeline:

.. code-block:: bash

   python run_batch_of_slides.py --task all --wsi_dir /mnt/nfs/wsis --job_dir ./out \
       --patch_encoder uni_v1 --mag 20 --patch_size 256 \
       --gpus 0 1 \
       --wsi_cache /local/ssd/cache --cache_batch_size 32

The cache directory is wiped and recreated at the start of each run (this is **separate**
from lock cleanup, which is opt-in via ``--clear_dead_locks``).

High-signal knobs
-----------------

- ``--segmenter``:

  - ``grandqc`` — fast, accurate on clean H&E.
  - ``hest`` — better on IHC / dirtier slides.
  - ``otsu`` — CPU-only fallback, no model weights needed.

- ``--mag`` / ``--patch_size`` / ``--overlap`` define the patch grid; the same values must be
  used across ``coords`` and ``feat`` runs.
- ``--min_tissue_proportion`` (0.0 to 1.0) raises the bar for keeping a patch; 0.3–0.7
  removes many weak edge patches.
- ``--remove_artifacts`` / ``--remove_penmarks``: extra artifact-cleaning segmentation pass.
- ``--search_nested``: discover slides in nested subfolders.
- ``--custom_list_of_wsis my.csv``: process a CSV subset (column ``wsi`` with paths
  relative to ``--wsi_dir``; optional ``mpp`` column).
- ``--reader_type {openslide,cucim,image,sdpc,omezarr,czi}``: force a backend, mostly
  for debugging.
- ``--max_workers 0``: force single-process data loading (use this if your environment has
  DataLoader multiprocessing issues).

Stage-only examples
-------------------

**Segmentation only**

.. code-block:: bash

   python run_batch_of_slides.py --task seg --wsi_dir ./wsis --job_dir ./out --segmenter grandqc

**Patching only** (with patch images for QC)

.. code-block:: bash

   python run_batch_of_slides.py --task coords --wsi_dir ./wsis --job_dir ./out \
       --mag 20 --patch_size 256 \
       --dump_patches --dump_patches_max 100 --dump_patches_format jpg --dump_patches_jpeg_quality 90

**Feature extraction only** (reusing existing coords)

.. code-block:: bash

   python run_batch_of_slides.py --task feat --wsi_dir ./wsis --job_dir ./out \
       --patch_encoder uni_v1 --mag 20 --patch_size 256

**Slide-level embeddings**

.. code-block:: bash

   python run_batch_of_slides.py --task feat --wsi_dir ./wsis --job_dir ./out \
       --slide_encoder titan --mag 20 --patch_size 512

If patch features for the required underlying encoder don't exist, TRIDENT extracts them automatically.

**Cell / structure segmentation** (per-instance polygons; reuses existing coords)

.. code-block:: bash

   python run_batch_of_slides.py --task patch_seg --wsi_dir ./wsis --job_dir ./out \
       --patch_segmenter histoplus --mag 20 --patch_size 784 --feat_batch_size 1 --seg_viz

Runs a fixed-taxonomy cell model (``histoplus`` or ``cellvit_plus_plus``) or the promptable
``weave`` over tissue patches and writes, under ``<mag>x_<patch>px_<overlap>px_overlap/seg_<model>/``:

- ``<slide>.geojson``: one polygon per instance with ``class`` / ``class_name`` / ``confidence`` (open in QuPath).
- ``<slide>.h5``: compact per-instance storage (``contours`` + ``contour_offsets``, ``centroids``, ``class_ids``, ``confidences``).
- ``visualization/`` (with ``--seg_viz``): a slide overview plus full-resolution sample-patch overlays, both with a color→class legend.

Each model lives in its own package and pulls conflicting deps, so install it in a **separate environment**:
HistoPlus is **not on PyPI** — ``pip install git+https://github.com/owkin/histoplus.git`` (gated weights on
HuggingFace; needs ``HF_TOKEN``); CellViT++ is ``pip install cellvit`` (use Python 3.10/3.11; on 3.13 its pinned
Shapely fails to build — install ``--no-deps`` and add ``colorama colour geojson natsort opt-einsum pyaml``).
Use the ``--mag`` / ``--patch_size`` the model expects (HistoPlus: ``784 @ 20x``; CellViT++: ``1024 @ 40x``).
On recent PyTorch, run HistoPlus with ``--feat_batch_size 1`` (its batched attention kernel can crash).

**weave** (promptable SAM3) segments whatever ``--patch_seg_prompt`` names instead of a fixed taxonomy:

.. code-block:: bash

   python run_batch_of_slides.py --task patch_seg --wsi_dir ./wsis --job_dir ./out \
       --patch_segmenter weave --patch_seg_prompt "tumor" --mag 20 --patch_size 1008 --seg_viz

Any ``--mag`` / ``--patch_size`` works (SAM3 resizes internally; its native input is **1008×1008**, so
``--patch_size 1008`` avoids an extra resize). Output is a **semantic region map** by
default (per-tile detections dissolved into contiguous same-class regions; ``--patch_seg_no_dissolve`` keeps
raw per-instance masks) and is keyed **per prompt** — ``seg_weave_<prompt>/`` — so prompts coexist on one
``--job_dir``. Install ``pip install git+https://github.com/JaumeLab/sam3.git pycocotools`` (gated weights,
`JaumeLab/sam3-finetuned <https://huggingface.co/JaumeLab/sam3-finetuned>`__). **Single GPU only** (``--gpus 0``;
for another GPU use ``CUDA_VISIBLE_DEVICES=<n>``).

**VLM question answering** (free-text Q&A over ROIs; reuses existing coords)

.. code-block:: bash

   # Ask the same question of every tissue patch
   python run_batch_of_slides.py --task vlm --wsi_dir ./wsis --job_dir ./out \
       --vlm patho_r1_7b --vlm_prompt "Is tumor present? Describe the tissue." \
       --mag 20 --patch_size 512

To interrogate a single ROI instead, use the Python API
``slide.query_region(vlm, prompt, location, size, mag)`` (no coords needed).

Runs a pathology vision-language model (`Patho-R1 <https://huggingface.co/WenchuanZhang/Patho-R1-7B>`__,
Qwen2.5-VL) and writes, under ``<mag>x_<patch>px_<overlap>px_overlap/vlm_<model>/``:

- ``<slide>.json``: per-patch ``{x, y, prompt, answer}``.
- ``<slide>.geojson``: one patch box per answer carrying ``prompt`` / ``answer`` (open in QuPath).

**Any magnification works.** Unlike the patch/slide encoders (whose ``--mag`` / ``--patch_size`` are
fixed by training), a VLM accepts arbitrary input sizes — pick ``--mag`` / ``--patch_size`` to frame
the field of view you want; ``--mag 20 --patch_size 512`` above is just an example.

Install into the TRIDENT env: ``pip install "transformers>=4.49" accelerate qwen-vl-utils`` (weights
auto-download from HuggingFace; **CC-BY-NC-ND-4.0**, non-commercial). Generation is autoregressive and
the batch task sweeps every patch, so it is slow and **not** part of ``--task all`` — prefer a tight
coords set, a coarser field of view (a lower ``--mag`` or larger ``--patch_size`` → fewer patches), or
the single-ROI ``query_region`` API. Lower ``--vlm_batch_size`` (default 4) if you OOM. Answers can be
confidently wrong — not for clinical use.

**Convert awkward formats to pyramidal TIFF**

.. code-block:: bash

   trident convert --input_dir ./wsis --mpp_csv ./wsis/to_process.csv --job_dir ./pyramidal_tiff --downscale_by 1 --num_workers 1

Common failure modes
--------------------

- **"Patch features not found" during slide embeddings**: each slide encoder requires a specific
  patch encoder (mapping in ``trident.slide_encoder_models.load.slide_to_patch_encoder_name``).
  Run patch features with the right encoder, or let TRIDENT auto-extract them by passing
  ``--slide_encoder``.
- **OOM during feature extraction**: lower ``--feat_batch_size`` (or ``--batch_size``), or pick a smaller patch encoder / patch size.
- **No slides discovered**: add ``--search_nested`` for nested layouts; or check that your
  CSV uses the column name ``wsi`` and **relative** paths under ``--wsi_dir``.
- **Pipeline looks stuck**: check for stale ``.lock`` files. After confirming no TRIDENT
  process is running, re-run with ``--clear_dead_locks``.
- **Offline / no internet**: set ``HF_TOKEN`` only when needed; otherwise put weights into
  ``trident/*/local_ckpts.json`` or pass ``--patch_encoder_ckpt_path``.

Argument cheat sheet
--------------------

The list below is not exhaustive — for full defaults and choices, scroll to "Raw parser help".

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Use
   * - ``--task {seg,coords,feat,patch_seg,vlm,all}``
     - Pipeline stage. ``all`` runs ``seg → coords → feat``. ``patch_seg`` runs a cell segmentation model, and ``vlm`` runs a VLM Q&A, both over the coords from ``coords``.
   * - ``--gpus 0 1`` / ``--gpus -1 -1``
     - Multi-GPU sharding (positive IDs) or multi-CPU workers (``-1`` entries).
   * - ``--max_workers``
     - DataLoader workers. ``0`` forces single-process loading.
   * - ``--clear_dead_locks``, ``--dead_lock_max_age_hours``
     - Safe cleanup of stale ``.lock`` files; active locks are never touched.
   * - ``--skip_errors``
     - Continue when a slide fails. Errors are recorded in ``summary.md`` and ``wsi_states/``.
   * - ``--segmenter`` / ``--seg_conf_thresh``
     - Tissue segmenter and its threshold.
   * - ``--remove_holes`` / ``--remove_artifacts`` / ``--remove_penmarks``
     - Mask post-processing.
   * - ``--mag`` / ``--patch_size`` / ``--overlap``
     - Patch grid definition (must match between ``coords`` and ``feat``).
   * - ``--min_tissue_proportion``
     - 0..1 floor on tissue overlap to keep a patch.
   * - ``--coords_dir``
     - Custom coords directory (e.g. to feed legacy CLAM coordinates into ``--task feat``).
   * - ``--dump_patches`` / ``--dump_patches_max`` / ``--dump_patches_format`` / ``--dump_patches_jpeg_quality``
     - Save patch images to disk during ``coords`` (debug / QC).
   * - ``--patch_encoder`` / ``--patch_encoder_ckpt_path`` / ``--slide_encoder``
     - Encoders. See API page for full list.
   * - ``--patch_segmenter {histoplus,cellvit_plus_plus,weave}`` / ``--patch_segmenter_ckpt_path`` / ``--seg_viz``
     - Segmenter for ``--task patch_seg``, optional local weights, and debug overlays with a class legend.
   * - ``--patch_seg_prompt`` / ``--patch_seg_conf_thresh`` / ``--patch_seg_no_dissolve``
     - For ``weave``: the text prompt to segment (required), the score threshold (default 0.5), and an opt-out of the default seam dissolve (keeps raw per-instance masks).
   * - ``--patch_seg_simplify_tol``
     - Polygon simplification tolerance (level-0 px) for ``--task patch_seg``; shrinks GeoJSON/HDF5 10-70x with <0.1%% area change. Default auto-scales per model; 0 disables.
   * - ``--vlm {patho_r1_7b,patho_r1_3b}`` / ``--vlm_prompt`` / ``--vlm_batch_size`` / ``--vlm_max_new_tokens`` / ``--vlm_ckpt_path``
     - VLM for ``--task vlm``: model, the question asked of every patch, generation batch size, answer-length cap, and optional local weights.
   * - ``--batch_size`` / ``--seg_batch_size`` / ``--feat_batch_size``
     - Stage-specific batch overrides.
   * - ``--wsi_dir`` / ``--wsi_ext`` / ``--search_nested`` / ``--custom_list_of_wsis`` / ``--custom_mpp_keys`` / ``--reader_type``
     - Slide discovery and reader controls.
   * - ``--wsi_cache`` / ``--cache_batch_size``
     - Local cache pipeline for slow source storage.

Raw parser help
---------------

For exact defaults, choices, and the complete flag list:

.. literalinclude:: generated/run_batch_of_slides_help.txt
   :language: text
