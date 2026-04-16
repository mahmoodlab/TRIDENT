Frequently Asked Questions
==========================

This page groups common questions by theme.

Troubleshooting by symptom
--------------------------

- **Found 0 slides / missing files** → :ref:`why no slides <faq-found-0-slides>`
- **Pipeline looks stuck** (locks) → :ref:`locks <faq-locks>`
- **Re-run / resume on same job_dir** → :ref:`resume <faq-resume-job-dir>`
- **Job is slow** → :ref:`performance <faq-slow>`
- **Slide embeddings complain about missing patch features** → :ref:`missing patch features <faq-patch-features-not-found>`
- **Offline cluster / no internet** → :ref:`offline <faq-offline>`
- **One slide keeps failing** → :ref:`debug one slide <faq-one-slide-failing>`
- **Where are outputs / what happened** → :ref:`where results are <faq-where-results>`

Getting started and discovery
-----------------------------

.. _faq-legacy-clam:

.. dropdown:: **How do I extract embeddings from legacy CLAM coordinates?**

   Use the `--coords_dir` flag to pass CLAM-style patch coordinates:

   .. code-block:: bash

      python run_batch_of_slides.py --task feat --wsi_dir wsis --job_dir legacy_dir --coords_dir extracted_coords --patch_encoder uni_v1

.. _faq-found-0-slides:

.. dropdown:: **TRIDENT says “Found 0 valid slides”. Why?**

   Common causes:

   - Your folder is nested: add ``--search_nested``.
   - Your extension filter is too strict: remove ``--wsi_ext`` or include the right extensions.
   - You used ``--custom_list_of_wsis`` but the CSV is wrong:
     - CSV must contain a ``wsi`` column
     - values must be **relative paths under** ``--wsi_dir`` (e.g., ``patientA/slide.svs``)

.. dropdown:: **My WSIs are in multiple subfolders. How can I process them all?**

   By default, only the top-level directory is scanned. Use `--search_nested` to recursively search for WSIs in all nested folders and include them in processing.

.. dropdown:: **What does `--reader_type` do? Which one should I use?**

   TRIDENT can force a reader backend. Use this mostly for debugging:

   - ``openslide``: default for many WSI formats (``.svs``, ``.tif/.tiff``, ``.ndpi``, ``.mrxs``, …; also ``.dcm`` if your OpenSlide build supports it)
   - ``cucim``: GPU-friendly WSI reading (when available)
   - ``image``: standard images via PIL (``.png``, ``.jpg/.jpeg``)
   - ``sdpc``: SDPC files
   - ``omezarr``: OME-Zarr / NGFF Zarr
   - ``czi``: Zeiss CZI (requires the optional CZI dependency)

Metadata and patching semantics
-------------------------------

.. dropdown:: **My WSIs have no micron-per-pixel (MPP) or magnification metadata. What should I do?**

   PNGs and JPEGs do not store MPP metadata in the file itself. If you're working with such formats, passing a CSV via `--custom_list_of_wsis` is **required**. This CSV should include at least two columns: `wsi` and `mpp`.

   Example:

   .. code-block:: text

      wsi,mpp
      TCGA-AJ-A8CV-01Z-00-DX1_1.png,0.25
      TCGA-AJ-A8CV-01Z-00-DX1_2.png,0.25
      TCGA-AJ-A8CV-01Z-00-DX1_3.png,0.25

   If you're using OpenSlide-readable formats (e.g., `.svs`, `.tiff`), this CSV is optional—but you can still use it to:

   - Restrict processing to a specific subset of slides
   - Override incorrect MPP metadata

.. dropdown:: **I want to skip patches on holes.**

   By default, TRIDENT includes all tissue patches (including holes). Use `--remove_holes` to exclude them. Not recommended, as "holes" often help define the tissue microenvironment.

Models and compute (GPU/CPU)
----------------------------

.. dropdown:: **Which tissue vs. background segmenter should I use?**

   TRIDENT supports three segmenters:

   - ``hest``: preferred for IHC and dirtier slides.
   - ``grandqc``: often preferred for clean H&E workflows (fast and reliable).
   - ``otsu``: image-processing-only fallback (no segmentation weights required), runs at 1.25x on CPU.

.. dropdown:: **Which tasks need GPU and which are fine on CPU?**

   - Segmentation:
     - ``hest`` and ``grandqc`` use GPU.
     - Optional artifact cleanup (``--remove_artifacts`` / ``--remove_penmarks``) adds additional segmentation cost.
     - ``otsu`` runs on CPU.
   - Patching:
     - CPU-only; usually fast, but can be CPU-intensive on very large slides or heavy overlap settings.
     - Use ``--min_tissue_proportion`` to require more tissue overlap and reduce weak/edge patches.
     - For debugging, you can also dump patch images during the coords task using: ``--dump_patches --dump_patches_format {png,jpg} --dump_patches_jpeg_quality 90 --dump_patches_max 100``.
   - Feature extraction:
     - Patch-level and slide-level feature extraction require GPU in practice.

Performance
-----------

.. _faq-slow:

.. dropdown:: **My job is slow. What are the usual bottlenecks?**

   - **I/O bound** (common on network drives): enable ``--wsi_cache`` on a local SSD.
   - **GPU bound** (feature extraction): reduce ``--feat_batch_size`` / ``--batch_size`` if you see OOM.
   - **Too many patches**: increase ``--min_tissue_proportion`` or decrease overlap.

.. dropdown:: **I don’t have enough local SSD storage and my WSIs are on a slow remote disk. How can I accelerate processing?**

   When WSIs are stored on slow network or external drives, processing can be very slow. Use `--wsi_cache ./cache --cache_batch_size 32` to enable local caching. WSIs will be copied in batches to a local SSD, processed in parallel, and automatically cleaned up after use. This significantly reduces I/O bottlenecks.

.. dropdown:: **Why does `trident convert` exist if TRIDENT already reads many formats?**

   The converter is mainly for uncommon formats that OpenSlide does not handle well. It uses BioFormats-backed readers when possible, then writes pyramidal TIFF outputs for downstream workflows.

Reliability, resume, and debugging
----------------------------------

.. _faq-where-results:

.. dropdown:: **Where can I see what TRIDENT has done (and what failed)?**

   In your ``--job_dir``:

   - ``summary.md``: appended once per run; compact counts and per-model breakdown, plus a short error list.
   - ``runs/<run_id>.json``: per-run JSON manifest (args, timestamps, status).
   - ``wsi_states/<slide>__<hash>.json``: per-slide state (task attempts, outputs, and resume info).

.. _faq-resume-job-dir:

.. dropdown:: **How do I safely re-run or resume a job on an existing `--job_dir`?**

   TRIDENT is designed so that **re-running on the same ``--job_dir`` is usually safe**:

   - If an output already exists and is not locked, the corresponding task is marked **skipped**.
   - State is persisted under ``wsi_states/``; you can inspect it to see what has already run.
   - When in doubt, start by re-running **only one stage** (e.g., ``--task feat``) instead of ``--task all``.

.. dropdown:: **How should I read the `wsi_states/*.json` files? What do the fields mean?**

   At a high level:

   - ``slide``: identity (name, extension, absolute path, reader type).
   - ``meta``: one-shot WSI metadata snapshot (e.g., dimensions, mpp, level_count) when available.
   - ``tasks``: one entry per logical task (``segmentation``, ``coords``, ``patch_features:<encoder>``, ``slide_features:<encoder>``) with:
     - ``status`` (``not_started``, ``running``, ``completed``, ``skipped``, ``error``),
     - ``reason`` (why it was skipped/errored, if known),
     - ``attempts`` (merged start/finish records with timestamps and durations),
     - ``outputs`` (paths + existence/bytes).
   - ``summary``: a compact view of task statuses, grouped by patch/slide encoder.
   - ``resume``: last task + status + last error, useful when debugging failed runs.

.. _faq-locks:

.. dropdown:: **How do locks (`.lock` files) work and when is it safe to remove them?**

   TRIDENT uses simple filesystem locks (``<output>.lock``) to avoid two workers writing the same file:

   - A task creates a ``.lock`` file when it starts, and removes it on success or handled error.
   - If you see a stale ``.lock`` file but no corresponding running process, it usually means a crash or interruption.
   - It is generally safe to **delete stale locks** once you've ensured no TRIDENT process is still running for that job, then re-run the affected task.

.. _faq-patch-features-not-found:

.. dropdown:: **I’m running slide embeddings and it says “Patch features not found”.**

   Slide encoders require a specific patch encoder (internal mapping).

   Fix:

   - run patch features for the required encoder under the same ``coords_dir``, or
   - run slide features and let TRIDENT auto-extract missing patch features.

.. _faq-one-slide-failing:

.. dropdown:: **One slide keeps failing while `--skip_errors` is on. How do I debug it?**

   - Check ``summary.md`` and the slide’s entry in ``wsi_states/`` to see which task and error are reported.
   - Re-run a **small test** focusing only on that slide:

     - via API: use ``load_wsi`` + a minimal pipeline around the failing step, or
     - via CLI: create a CSV for that slide only (``--custom_list_of_wsis``) and re-run the relevant task.

   - Once you understand/fix the cause, you can safely re-run the full batch with ``--skip_errors`` again.

Environment and advanced usage
------------------------------

.. dropdown:: **Which Python versions are supported? What about 3.12+?**

   TRIDENT is tested and packaged for **Python 3.10 and 3.11** (see ``pyproject.toml``).
   Python 3.12+ may work at the pure-Python level, but binary dependencies (PyTorch, OpenSlide, etc.) and some pinned versions are **not guaranteed** to be compatible.
   For production use, stick to 3.10/3.11 until explicit 3.12+ support is advertised.

.. dropdown:: **How can I control where TRIDENT stores downloaded weights and caches?**

   TRIDENT follows a simple hierarchy:

   - If ``TRIDENT_HOME`` is set, weights and related files go under that directory.
   - Else, it falls back to ``$XDG_CACHE_HOME/trident`` (defaulting to ``~/.cache/trident`` if unset).

   On clusters with small home directories, point ``TRIDENT_HOME`` or ``XDG_CACHE_HOME`` to a larger scratch or project disk.

.. dropdown:: **Can I plug in my own custom patch or slide encoder?**

   Yes. The recommended approach is:

   - Wrap your patch encoder in ``CustomInferenceEncoder`` (see ``trident/patch_encoder_models/load.py``).
   - Wrap your slide encoder in ``CustomSlideEncoder`` (see ``trident/slide_encoder_models/load.py``).
   - Use the API (``Processor`` + your custom encoder) rather than the CLI for these advanced cases.

   This way, you still benefit from TRIDENT’s I/O and patching pipeline while controlling the model.

.. _faq-offline:

.. dropdown:: **I work on a cluster without Internet access. How can I use models offline?**

   You can use local checkpoint files by editing the model registry files in Trident. This allows you to cache or pre-download all necessary models for both segmentation and patch encoding.

   **1. Segmentation Models**

   Update the segmentation model registry at:
   `trident/segmentation_models/local_ckpts.json`

   Example:

   .. code-block:: json

      {
        "hest": "./ckpts/trident/deeplabv3_seg_v4.ckpt",
        "grandqc": "./ckpts/trident/Tissue_Detection_MPP10.pth",
        "grandqc_artifact": "./ckpts/trident/GrandQC_MPP1_state_dict.pth"
      }

   **2. Patch Encoder Models**

   Update the patch encoder model registry at:
   `trident/patch_encoder_models/local_ckpts.json`

   Example:

   .. code-block:: json

      {
        "conch_v1": "./ckpts/conch_patch_encoder/pytorch_model.bin",
        "uni_v1": "./ckpts/uni_patch_encoder/pytorch_model.bin",
        "uni_v2": "./ckpts/uni2_patch_encoder/pytorch_model.bin",
        "ctranspath": "./ckpts/ctranspath_patch_encoder/CHIEF_CTransPath.pth",
        "phikon": "./ckpts/phikon_patch_encoder/pytorch_model.bin",
        "resnet50": "./ckpts/resnet_patch_encoder/pytorch_model.bin",
        "gigapath": "./ckpts/gigapath_patch_encoder/pytorch_model.bin",
        "virchow": "./ckpts/virchow_patch_encoder/pytorch_model.bin",
        "virchow2": "./ckpts/virchow2_patch_encoder/pytorch_model.bin",
        "hoptimus0": "./ckpts/hoptimus0_patch_encoder/pytorch_model.bin",
        "hoptimus1": "./ckpts/hoptimus1_patch_encoder/pytorch_model.bin",
        "phikon_v2": "./ckpts/phikon-v2_patch_encoder/model.safetensors",
        "kaiko-vitb8": "./ckpts/kaiko_vitb8_patch_encoder/model.safetensors",
        "kaiko-vitb16": "./ckpts/kaiko_vitb16_patch_encoder/model.safetensors",
        "kaiko-vits8": "./ckpts/kaiko_vits8_patch_encoder/model.safetensors",
        "kaiko-vits16": "./ckpts/kaiko_vits16_patch_encoder/model.safetensors",
        "kaiko-vitl14": "./ckpts/kaiko_vitl14_patch_encoder/model.safetensors",
        "lunit-vits8": "./ckpts/lunit_patch_encoder/model.safetensors",
        "conch_v15": "./ckpts/conchv1_5_patch_encoder/pytorch_model_vision.bin"
      }

   **3. Alternative Option**

   You can also directly pass a local checkpoint path at runtime using the `--patch_encoder_ckpt_path` argument in `run_batch_of_slides.py`.

   **4. Optional: Pre-download All Models in Advance**

   Full credit to @haydenych. If you'd like to automatically download all model weights in advance (e.g., from a connected machine), use the following:

   .. code-block:: bash

      XDG_CACHE_HOME="<YOUR_CACHE_DIR>" HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>" python run_predownload_weights.py

   This will fetch all segmentation, patch encoder, and slide encoder weights supported in Trident.

   To run downstream tasks using the cached models:

   .. code-block:: bash

      XDG_CACHE_HOME="<YOUR_CACHE_DIR>" python run_single_slide.py ...
      XDG_CACHE_HOME="<YOUR_CACHE_DIR>" python run_batch_of_slides.py ...

   Example `run_predownload_weights.py` script (can be adapted based on needs):

   .. code-block:: python

      from trident.segmentation_models import segmentation_model_factory
      from trident.patch_encoder_models.load import encoder_factory as patch_encoder_model_factory
      from trident.slide_encoder_models.load import encoder_factory as slide_encoder_model_factory

      segmentation_models = ["hest", "grandqc", "grandqc_artifact", "otsu"]
      for model in segmentation_models:
          try:
              segmentation_model_factory(model)
          except Exception as e:
              print(f"Failed to download weights for {model}: {e}")

      patch_encoder_models = [
          "conch_v1", "uni_v1", "uni_v2", "ctranspath", "phikon", "resnet50", "gigapath",
          "virchow", "virchow2", "hoptimus0", "hoptimus1", "phikon_v2", "conch_v15",
          "musk", "hibou_l", "kaiko-vits8", "kaiko-vits16", "kaiko-vitb8", "kaiko-vitb16",
          "kaiko-vitl14", "lunit-vits8"
      ]
      for model in patch_encoder_models:
          try:
              patch_encoder_model_factory(model)
          except Exception as e:
              print(f"Failed to download weights for {model}: {e}")

      slide_encoder_models = [
          "threads", "titan", "prism", "gigapath", "chief", "madeleine", "mean-virchow",
          "mean-virchow2", "mean-conch_v1", "mean-conch_v15", "mean-ctranspath", "mean-gigapath",
          "mean-resnet50", "mean-hoptimus0", "mean-phikon", "mean-phikon_v2", "mean-musk",
          "mean-uni_v1", "mean-uni_v2"
      ]
      for model in slide_encoder_models:
          try:
              slide_encoder_model_factory(model)
          except Exception as e:
              print(f"Failed to download weights for {model}: {e}")