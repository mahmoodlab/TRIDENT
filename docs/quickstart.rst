Quickstart
==================

Trident provides user-facing command-line scripts for processing large batches of whole-slide images (WSIs).

This page explains how to quickly get started, and provides detailed help for available options.

Use this rule of thumb:

- Use ``run_single_slide.py`` (or ``trident single``) to test settings on one slide.
- Use ``run_batch_of_slides.py`` (or ``trident batch``) once settings are validated.

---

Processing a batch of slides
----------

To process a batch of WSIs through segmentation, patch extraction, and feature extraction in one go,  
run the following command:

.. code-block:: bash

    python run_batch_of_slides.py --task all --wsi_dir output/wsis --job_dir output --patch_encoder uni_v1 --mag 20 --patch_size 256

Equivalent wrapper CLI:

.. code-block:: bash

    trident batch -- --task all --wsi_dir output/wsis --job_dir output --patch_encoder uni_v1 --mag 20 --patch_size 256

Other wrapper entrypoints:

.. code-block:: bash

    trident single -- --slide_path ./wsis/example.svs --job_dir ./trident_processed --patch_encoder uni_v1 --mag 20 --patch_size 256
    trident doctor -- --profile base

This command runs the full pipeline:

1. Segment tissue areas in the slides.
2. Extract patch coordinates over tissue.
3. Extract patch-level features using the selected encoder.

Typical Examples
-----------------

**Segmentation Only**  
(Segment tissue regions and save binary masks.)

.. code-block:: bash

    python run_batch_of_slides.py --task seg --wsi_dir input_wsis --job_dir output

Available segmenters:

- ``hest`` (default, learned model)
- ``grandqc`` (learned model, fast for H&E)
- ``otsu`` (image-processing-only fallback, runs at 1.25x on CPU)

Recommended choices:

- Clean H&E: prefer ``grandqc`` (fast and reliable).
- IHC or dirtier slides: prefer ``hest``.
- Limited resources or fallback mode: use ``otsu``.

**Patch Extraction Only**  
(Extract patch coordinates from tissue regions.)

.. code-block:: bash

    python run_batch_of_slides.py --task coords --wsi_dir input_wsis --job_dir output --mag 20 --patch_size 256

To additionally dump the patch *images* (useful for debugging), enable:

- ``--dump_patches``: write patch images under ``<job_dir>/<coords_dir>/patch_images/<slide_name>/``
- ``--dump_patches_max``: cap number of patches written per slide (0 = no limit)
- ``--dump_patches_format``: ``png`` (default) or ``jpg``
- ``--dump_patches_jpeg_quality``: JPEG quality (1-100) when using ``jpg``

Example (dump first 100 patches as JPEGs):

.. code-block:: bash

    python run_batch_of_slides.py --task coords --wsi_dir input_wsis --job_dir output --mag 20 --patch_size 256 --dump_patches --dump_patches_max 100 --dump_patches_format jpg --dump_patches_jpeg_quality 90

To control how strict tissue overlap should be during patch selection, use:

- ``--min_tissue_proportion`` (0.0 to 1.0)
- Default is permissive (keeps patches if they overlap tissue even minimally).

**Feature Extraction Only**  
(Extract features from patches using a patch encoder.)

.. code-block:: bash

    python run_batch_of_slides.py --task feat --wsi_dir input_wsis --job_dir output --patch_encoder uni_v1 --mag 20 --patch_size 256

**Converter (pyramidal TIFF)**  
(Convert listed files from CSV to pyramidal TIFF.)

.. code-block:: bash

    trident convert --input_dir ./wsis --mpp_csv ./wsis/to_process.csv --job_dir ./pyramidal_tiff --downscale_by 1 --num_workers 1

Argument guide (run_batch_of_slides.py)
---------------------------------------

This section explains every parser argument in plain language.

Generic arguments
^^^^^^^^^^^^^^^^^

- ``--task``: chooses pipeline stage.
  - ``seg`` = segmentation only
  - ``coords`` = patch coordinate extraction only
  - ``feat`` = feature extraction only
  - ``all`` = run ``seg -> coords -> feat`` in order
- ``--job_dir``: output root directory for all generated files.
- ``--gpu``: GPU index used by learned segmentation and feature extraction.
- ``--skip_errors``: continue processing other slides if one slide fails.
- ``--max_workers``: worker count for slide collection/processing internals.
- ``--batch_size``: shared batch size baseline for segmentation and feature extraction.
- ``--seg_batch_size``: segmentation-specific override for ``--batch_size``.
- ``--feat_batch_size``: feature-specific override for ``--batch_size``.

When to change:

- Use ``--skip_errors`` for long production runs.
- Start with default ``--batch_size`` and increase only if memory allows.

WSI discovery and reading
^^^^^^^^^^^^^^^^^^^^^^^^^

- ``--wsi_dir``: directory containing input slides.
- ``--wsi_ext``: optional allowed extension filter (for mixed folders).
- ``--search_nested``: recursively discover slides in nested subfolders.
- ``--custom_list_of_wsis``: CSV subset list to process selected slides only.
- ``--custom_mpp_keys``: metadata keys to read MPP from non-standard slide headers.
- ``--reader_type``: force backend reader (``openslide``, ``cucim``, ``image``, ``sdpc``, ``omezarr``).

When to change:

- Use ``--search_nested`` when slides are organized by subfolders.
- Use ``--custom_list_of_wsis`` to run curated/retry batches.
- Set ``--reader_type`` only for debugging backend-specific read issues.

Segmentation arguments
^^^^^^^^^^^^^^^^^^^^^^

- ``--segmenter``: segmentation model choice (``hest``, ``grandqc``, ``otsu``).
- ``--seg_conf_thresh``: tissue confidence threshold for learned segmenters.
- ``--remove_holes``: exclude hole regions from final tissue mask.
- ``--remove_artifacts``: run extra artifact-cleaning segmentation pass.
- ``--remove_penmarks``: lighter artifact mode focused on penmark removal.

Operational behavior:

- ``hest`` and ``grandqc`` run on GPU.
- ``grandqc`` is typically fast on clean H&E.
- ``hest`` is often preferred for IHC/dirtier slides.
- ``otsu`` is model-free fallback and runs on CPU at 1.25x.
- ``--remove_artifacts`` / ``--remove_penmarks`` add extra segmentation time.

Patching arguments
^^^^^^^^^^^^^^^^^^

- ``--mag``: target magnification for patch coordinates/features.
- ``--patch_size``: patch size in pixels at target magnification.
- ``--overlap``: absolute patch overlap in pixels.
- ``--min_tissue_proportion``: minimum tissue overlap ratio (0.0 to 1.0) to keep a patch.
- ``--coords_dir``: custom coordinates directory to read/write.
- ``--dump_patches``: dump patch images to disk during the coords task.
- ``--dump_patches_max``: max patch images to dump per slide (0 = unlimited).
- ``--dump_patches_format``: patch image format (``png`` or ``jpg``).
- ``--dump_patches_jpeg_quality``: JPEG quality (1-100) when dumping ``jpg``.

When to change:

- Increase ``--min_tissue_proportion`` to remove weak edge patches.
- Use ``--coords_dir`` to reuse precomputed coordinates.

Feature extraction arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``--patch_encoder``: patch encoder name for patch-level features.
- ``--patch_encoder_ckpt_path``: local patch-encoder checkpoint path (offline/custom).
- ``--slide_encoder``: optional slide encoder name for slide-level embeddings.

How this works:

- If ``--slide_encoder`` is not provided, TRIDENT extracts patch features only.
- If ``--slide_encoder`` is provided, TRIDENT computes patch features as needed, then slide embeddings.
- Patch and slide feature extraction are GPU workflows.

Caching arguments
^^^^^^^^^^^^^^^^^

- ``--wsi_cache``: local cache directory used when source storage is slow.
- ``--cache_batch_size``: max slides staged in cache per batch.

When to use:

- Use caching when source slides are on slow network disks and you run large jobs.
- Keep cache on fast local SSD storage for best effect.

Raw parser help
---------------

For exact defaults and full flag list, see:

.. literalinclude:: generated/run_batch_of_slides_help.txt
   :language: text

