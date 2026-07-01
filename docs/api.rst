API Reference
=============

This section documents the **public API** of TRIDENT. 

When to use the API vs CLI:

- Use the CLI (``run_batch_of_slides.py`` / ``trident batch``) for standard reproducible runs.
- Use the API when embedding Trident in your own Python pipeline, custom loops, or experiments.
- Start with the CLI first, then move to API once the workflow is validated.

Minimal API usage
----------------------------------------

Load a slide and read regions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``load_wsi`` as a context manager so file handles are released:

.. code-block:: python

   from trident import load_wsi

   with load_wsi("./wsis/example.svs", lazy_init=False) as wsi:
       print(wsi.dimensions, wsi.mpp)
       patch = wsi.read_region((0, 0), level=0, size=(512, 512))

Run the pipeline with ``Processor``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CLI entrypoints are thin wrappers around ``Processor``.

.. code-block:: python

   from trident import Processor
   from trident.segmentation_models.load import segmentation_model_factory
   from trident.patch_encoder_models.load import encoder_factory as patch_encoder_factory

   processor = Processor(job_dir="./job", wsi_source="./wsis", search_nested=True, skip_errors=True)

   seg = segmentation_model_factory("grandqc", confidence_thresh=0.5)
   processor.run_segmentation_job(seg, device="cuda:0", batch_size=16)

   processor.run_patching_job(target_magnification=20, patch_size=256, overlap=0, min_tissue_proportion=0.0)

   enc = patch_encoder_factory("uni_v1")
   processor.run_patch_feature_extraction_job(coords_dir="20x_256px_0px_overlap", patch_encoder=enc, device="cuda:0", batch_limit=64)

   # Cell segmentation (per-cell polygons + types) over the same coords:
   from trident.patch_segmentation_models import patch_segmenter_factory
   cell_seg = patch_segmenter_factory("histoplus")
   processor.run_patch_segmentation_job(coords_dir="20x_784px_0px_overlap", patch_segmenter=cell_seg, device="cuda:0", batch_limit=1, visualize=True)

   # VLM question answering (free-text answer per patch) over the same coords:
   from trident.vlm_models import vlm_factory
   vlm = vlm_factory("patho_r1_7b")
   processor.run_vlm_query_job(coords_dir="20x_512px_0px_overlap", vlm=vlm, prompt="Is tumor present?", device="cuda:0", batch_limit=4)

Overlay segmentations on a slide or ROI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``WSI.overlay`` is the single, native entrypoint for drawing polygon geometries — tissue
contours or cell/nuclei instances — on a whole-slide thumbnail or on a cropped region of
interest. It reads the GeoJSON the pipeline already writes (``contours_geojson/<slide>.geojson``
for tissue, ``seg_<model>/<slide>.geojson`` for cells) and shares its renderer with the
built-in segmentation visualizations, so styling is consistent everywhere.

.. code-block:: python

   from trident import load_wsi

   with load_wsi("./wsis/example.svs", lazy_init=False) as wsi:
       # Tissue vs background, translucent fill, on a whole-slide thumbnail
       wsi.overlay("./job/contours_geojson/example.geojson", mode="fill", saveto="tissue.jpg")

       # Nuclear segmentation colored by cell type, on a 4096x4096 ROI at (x=10000, y=8000)
       wsi.overlay(
           "./job/20x_784px_0px_overlap/seg_histoplus/example.geojson",
           region=(10000, 8000, 4096, 4096),
           mode="outline", color_by="class", saveto="roi_cells.jpg",
       )

Use ``mode="outline"`` for boundaries or ``mode="fill"`` for translucent regions (holes
preserved), ``region=(x, y, w, h)`` (level-0 pixels) to crop an ROI or omit it for a
whole-slide thumbnail, and ``color_by="class"`` to color per cell type with a legend.
For per-patch **score** heatmaps (e.g. attention) use :func:`trident.visualize_heatmap`.

Outputs and run tracking
^^^^^^^^^^^^^^^^^^^^^^^^

In ``job_dir`` (same as the CLI):

- ``summary.md``: appended once per run; compact counts + per-model breakdown + errors
- ``runs/<run_id>.json``: per-run manifest (args, timestamps, status)
- ``wsi_states/<slide>__<hash>.json``: per-slide state (attempts, outputs, resume info)

Notes for power users
^^^^^^^^^^^^^^^^^^^^^

- **Nested datasets**: ``search_nested=True`` uses relative paths under ``wsi_source`` (mirrors CLI ``--search_nested``).
- **Subset runs**: pass ``custom_list_of_wsis="subset.csv"``; the CSV must have a ``wsi`` column.
- **Reader selection**: force a backend with ``reader_type="openslide" | "cucim" | "image" | "sdpc" | "omezarr" | "czi"``.
- **Slide encoders**: slide embeddings require a specific underlying patch encoder. The mapping lives in ``trident.slide_encoder_models.load.slide_to_patch_encoder_name``. If patch features are missing for that encoder, ``run_slide_feature_extraction_job`` extracts them on the fly.
- **Resume / idempotency**: every job uses self-describing ``.lock`` files (PID, host, timestamp). If an output exists and is not actively locked, the job is skipped on re-run. Use ``trident.IO.clear_dead_locks(job_dir)`` (or pass ``--clear_dead_locks`` to the CLI) to remove orphaned locks safely.
- **Multi-GPU**: the CLI handles GPU sharding via ``--gpus``. From Python, run separate ``Processor`` instances per shard with disjoint ``selected_wsi_paths`` and distinct ``device="cuda:N"`` arguments to the run-* methods.

.. contents::
   :local:
   :depth: 2


Trident
-------

Core of TRIDENT with `Processor` and WSI building.

.. automodule:: trident
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:


Segmentation Models
-------------------

Semantic segmentation models for tissue vs. background detection and filtering.

.. automodule:: trident.segmentation_models
   :members:
   :undoc-members:


Patch Encoders
--------------

Factory for loading patch-level encoder models.

.. list-table:: 
   :header-rows: 1
   :widths: 18 10 40 32

   * - Patch Encoder
     - Dim
     - Args
     - Link
   * - **UNI**
     - 1024
     - ``--patch_encoder uni_v1 --patch_size 256 --mag 20``
     - `MahmoodLab/UNI <https://huggingface.co/MahmoodLab/UNI>`__
   * - **UNI2-h**
     - 1536
     - ``--patch_encoder uni_v2 --patch_size 256 --mag 20``
     - `MahmoodLab/UNI2-h <https://huggingface.co/MahmoodLab/UNI2-h>`__
   * - **CONCH**
     - 512
     - ``--patch_encoder conch_v1 --patch_size 512 --mag 20``
     - `MahmoodLab/CONCH <https://huggingface.co/MahmoodLab/CONCH>`__
   * - **CONCHv1.5**
     - 768
     - ``--patch_encoder conch_v15 --patch_size 512 --mag 20``
     - `MahmoodLab/conchv1_5 <https://huggingface.co/MahmoodLab/conchv1_5>`__
   * - **Virchow**
     - 2560
     - ``--patch_encoder virchow --patch_size 224 --mag 20``
     - `paige-ai/Virchow <https://huggingface.co/paige-ai/Virchow>`__
   * - **Virchow2**
     - 2560
     - ``--patch_encoder virchow2 --patch_size 224 --mag 20``
     - `paige-ai/Virchow2 <https://huggingface.co/paige-ai/Virchow2>`__
   * - **Phikon**
     - 768
     - ``--patch_encoder phikon --patch_size 224 --mag 20``
     - `owkin/phikon <https://huggingface.co/owkin/phikon>`__
   * - **Phikon-v2**
     - 1024
     - ``--patch_encoder phikon_v2 --patch_size 224 --mag 20``
     - `owkin/phikon-v2 <https://huggingface.co/owkin/phikon-v2/>`__
   * - **KEEP**
     - 768
     - ``--patch_encoder keep --patch_size 256 --mag 20``
     - `Astaxanthin/KEEP <https://huggingface.co/Astaxanthin/KEEP>`__
   * - **Prov-Gigapath**
     - 1536
     - ``--patch_encoder gigapath --patch_size 256 --mag 20``
     - `prov-gigapath <https://huggingface.co/prov-gigapath/prov-gigapath>`__
   * - **H-Optimus-0**
     - 1536
     - ``--patch_encoder hoptimus0 --patch_size 224 --mag 20``
     - `bioptimus/H-optimus-0 <https://huggingface.co/bioptimus/H-optimus-0>`__
   * - **H-Optimus-1**
     - 1536
     - ``--patch_encoder hoptimus1 --patch_size 224 --mag 20``
     - `bioptimus/H-optimus-1 <https://huggingface.co/bioptimus/H-optimus-1>`__
   * - **H0-mini**
     - 768/1536
     - ``--patch_encoder h0-mini --patch_size 224 --mag 20``
     - `bioptimus/H0-mini <https://huggingface.co/bioptimus/H0-mini>`__
   * - **MUSK**
     - 1024
     - ``--patch_encoder musk --patch_size 384 --mag 20``
     - `xiangjx/musk <https://huggingface.co/xiangjx/musk>`__
   * - **Midnight-12k**
     - 3072
     - ``--patch_encoder midnight12k --patch_size 224 --mag 20``
     - `kaiko-ai/midnight <https://huggingface.co/kaiko-ai/midnight>`__
   * - **OpenMidnight**
     - 1536
     - ``--patch_encoder openmidnight --patch_size 224 --mag 20``
     - `SophontAI/OpenMidnight <https://huggingface.co/SophontAI/OpenMidnight>`__
   * - **GPFM**
     - 1024
     - ``--patch_encoder gpfm --patch_size 224 --mag 20``
     - `majiabo/GPFM <https://huggingface.co/majiabo/GPFM>`__
   * - **GenBio-PathFM**
     - 4608
     - ``--patch_encoder genbio-pathfm --patch_size 224 --mag 20``
     - `genbio-ai/genbio-pathfm <https://huggingface.co/genbio-ai/genbio-pathfm>`__
   * - **Gemma 4**
     - 768/1152
     - ``--patch_encoder {gemma4-e4b, gemma4-26b} --patch_size 224 --mag 20``
     - `google/gemma-4-E4B <https://huggingface.co/google/gemma-4-E4B>`__ / `google/gemma-4-26B-A4B <https://huggingface.co/google/gemma-4-26B-A4B>`__
   * - **Kaiko**
     - 384/768/1024
     - ``--patch_encoder kaiko-vit* --patch_size 256 --mag 20``
     - `Kaiko Collection <https://huggingface.co/collections/1aurent/kaikoai-models-66636c99d8e1e34bc6dcf795>`__
   * - **Lunit**
     - 384
     - ``--patch_encoder lunit-vits8 --patch_size 224 --mag 20``
     - `1aurent/lunit <https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino>`__
   * - **Hibou**
     - 1024
     - ``--patch_encoder hibou_l --patch_size 224 --mag 20``
     - `histai/hibou-L <https://huggingface.co/histai/hibou-L>`__
   * - **CTransPath-CHIEF**
     - 768
     - ``--patch_encoder ctranspath --patch_size 256 --mag 10``
     - —
   * - **ResNet50**
     - 1024
     - ``--patch_encoder resnet50 --patch_size 256 --mag 20``
     - —

.. automodule:: trident.patch_encoder_models
   :members:
   :undoc-members:


Cell Segmenters
---------------

Factory for cell/structure segmentation models used by ``run_patch_segmentation_job`` (``--task patch_seg``).
HistoPlus and CellViT++ return per-cell polygons + cell types; **weave** (SAM3) returns polygons for a
free-text prompt. Models ship in their own packages.

.. list-table::
   :header-rows: 1
   :widths: 20 12 36 32

   * - Segmenter
     - Classes
     - Args
     - Install / Link
   * - **HistoPlus**
     - 14 cell types
     - ``--patch_segmenter histoplus --patch_size 784 --mag 20``
     - ``pip install git+https://github.com/owkin/histoplus.git`` · gated `Owkin-Bioptimus/histoplus <https://huggingface.co/Owkin-Bioptimus/histoplus>`__
   * - **CellViT++**
     - 5 (PanNuke)
     - ``--patch_segmenter cellvit_plus_plus --patch_size 1024 --mag 40``
     - ``pip install cellvit`` · `TIO-IKIM/CellViT-Plus-Plus <https://github.com/TIO-IKIM/CellViT-Plus-Plus>`__
   * - **weave** (SAM3)
     - promptable
     - ``--patch_segmenter weave --patch_seg_prompt "tumor"`` (any ``--mag``/``--patch_size``)
     - ``pip install git+https://github.com/JaumeLab/sam3.git pycocotools`` · gated `JaumeLab/sam3-finetuned <https://huggingface.co/JaumeLab/sam3-finetuned>`__

.. note::
   Install these in a **separate environment** (they pull conflicting deps, e.g. HistoPlus needs ``timm==1.0.8``;
   weave/SAM3 pulls ``timm`` 1.x, ``numpy`` 1.x, ``transformers`` 5.x).
   HistoPlus is **not on PyPI** (install from the git URL above) and its weights are gated on HuggingFace
   (accept the license, set ``HF_TOKEN``); on recent PyTorch run it with ``--feat_batch_size 1`` (batched attention
   can crash). CellViT++ is on PyPI — use Python 3.10/3.11; on 3.13 its pinned Shapely fails to build, so install
   ``--no-deps`` and add ``colorama colour geojson natsort opt-einsum pyaml``.

.. note::
   **weave** is a promptable SAM3 finetuned for histopathology: it segments whatever ``--patch_seg_prompt``
   names (e.g. ``"tumor"``, ``"glomeruli"``), at any resolution (native input **1008×1008**, so
   ``--patch_size 1008`` avoids an extra resize). It outputs a **semantic region map** by
   default — per-tile detections are dissolved into contiguous same-class regions (``--patch_seg_no_dissolve``
   keeps raw per-instance masks). Output is keyed **per prompt** (``seg_weave_<prompt>/``), so prompts coexist
   on one ``--job_dir``. Runs on a **single GPU** (``--gpus 0``; for another GPU use ``CUDA_VISIBLE_DEVICES=<n>``).
   Also install ``pycocotools`` (imported by the fork but not declared).

.. automodule:: trident.patch_segmentation_models
   :members:
   :undoc-members:


Vision-Language Models
----------------------

Factory for generative vision-language models used by ``run_vlm_query_job`` (``--task vlm``) and
``WSI.query_region`` (single-ROI Python API). Given an image + a free-text prompt
they return a free-text answer.

.. list-table::
   :header-rows: 1
   :widths: 20 18 30 32

   * - VLM
     - Backbone
     - Args
     - Install / Link
   * - **Patho-R1 7B**
     - Qwen2.5-VL
     - ``--vlm patho_r1_7b`` (any ``--mag`` / ``--patch_size``)
     - ``pip install "transformers>=4.49" accelerate qwen-vl-utils`` · `WenchuanZhang/Patho-R1-7B <https://huggingface.co/WenchuanZhang/Patho-R1-7B>`__
   * - **Patho-R1 3B**
     - Qwen2.5-VL
     - ``--vlm patho_r1_3b`` (any ``--mag`` / ``--patch_size``)
     - same · `WenchuanZhang/Patho-R1-3B <https://huggingface.co/WenchuanZhang/Patho-R1-3B>`__

.. note::
   **Any magnification works** — unlike the patch/slide encoders (whose ``--mag`` / ``--patch_size`` are
   fixed by training), a VLM accepts arbitrary input sizes; pick them to frame the field of view you want.
   Runs in the TRIDENT env; weights auto-download from HuggingFace (**CC-BY-NC-ND-4.0**, non-commercial).
   Generation is autoregressive and ``--task vlm`` sweeps every patch, so it is slow and **not** part of
   ``--task all`` — prefer a tight coords set, a coarser field of view (lower ``--mag`` / larger
   ``--patch_size`` → fewer patches), or the single-ROI ``query_region`` API. Lower ``--vlm_batch_size``
   (default 4) if you OOM. Answers can be confidently wrong — not for clinical use.

.. automodule:: trident.vlm_models
   :members:
   :undoc-members:


Slide Encoders
--------------

Factory for slide-level encoder models.

.. list-table:: 
   :header-rows: 1
   :widths: 20 20 40 32

   * - Slide Encoder
     - Patch Encoder
     - Args
     - Link
   * - **Threads**
     - conch_v15
     - ``--slide_encoder threads --patch_size 512 --mag 20``
     - *(Coming Soon!)*
   * - **Titan**
     - conch_v15
     - ``--slide_encoder titan --patch_size 512 --mag 20``
     - `MahmoodLab/TITAN <https://huggingface.co/MahmoodLab/TITAN>`__
   * - **PRISM**
     - virchow
     - ``--slide_encoder prism --patch_size 224 --mag 20``
     - `paige-ai/Prism <https://huggingface.co/paige-ai/Prism>`__
   * - **CHIEF**
     - ctranspath
     - ``--slide_encoder chief --patch_size 256 --mag 10``
     - `CHIEF <https://github.com/hms-dbmi/CHIEF>`__
   * - **GigaPath**
     - gigapath
     - ``--slide_encoder gigapath --patch_size 256 --mag 20``
     - `prov-gigapath <https://huggingface.co/prov-gigapath/prov-gigapath>`__
   * - **Madeleine**
     - conch_v1
     - ``--slide_encoder madeleine --patch_size 256 --mag 10``
     - `MahmoodLab/madeleine <https://huggingface.co/MahmoodLab/madeleine>`__
   * - **Feather**
     - conch_v15
     - ``--slide_encoder feather --patch_size 512 --mag 20``
     - `MahmoodLab/feather <https://huggingface.co/MahmoodLab/abmil.base.conch_v15.pc108-24k>`__

.. automodule:: trident.slide_encoder_models
   :members:
   :undoc-members:
