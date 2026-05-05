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
   * - ``--task {seg,coords,feat,all}``
     - Pipeline stage. ``all`` runs ``seg → coords → feat``.
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
