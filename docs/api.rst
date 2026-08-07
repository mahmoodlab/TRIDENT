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

In the model tables on this page: 🔒 gated on HuggingFace (accept the terms while logged in; some
need manual approval) · 🌐 open download. Licenses are those declared by the model host — check
them before any commercial use.

.. list-table::
   :header-rows: 1
   :widths: 20 24 34 22

   * - Segmenter
     - Args
     - Link
     - License
   * - **HEST** (default)
     - ``--segmenter hest``
     - `MahmoodLab/hest-tissue-seg <https://huggingface.co/MahmoodLab/hest-tissue-seg>`__
     - 🌐 `CC-BY-NC-SA-4.0 <https://creativecommons.org/licenses/by-nc-sa/4.0/>`__
   * - **GrandQC**
     - ``--segmenter grandqc``
     - `cpath-ukk/grandqc <https://github.com/cpath-ukk/grandqc>`__
     - 🌐 `CC-BY-NC-SA-4.0 <https://creativecommons.org/licenses/by-nc-sa/4.0/>`__
   * - **Otsu**
     - ``--segmenter otsu``
     - —
     - — (classical, no model)

.. automodule:: trident.segmentation_models
   :members:
   :undoc-members:


Patch Encoders
--------------

Factory for loading patch-level encoder models.

.. list-table:: 
   :header-rows: 1
   :widths: 16 8 32 22 22

   * - Patch Encoder
     - Dim
     - Args
     - Link
     - License
   * - **UNI**
     - 1024
     - ``--patch_encoder uni_v1 --patch_size 256 --mag 20``
     - `MahmoodLab/UNI <https://huggingface.co/MahmoodLab/UNI>`__
     - 🔒 `CC-BY-NC-ND-4.0 <https://creativecommons.org/licenses/by-nc-nd/4.0/>`__
   * - **UNI2-h**
     - 1536
     - ``--patch_encoder uni_v2 --patch_size 256 --mag 20``
     - `MahmoodLab/UNI2-h <https://huggingface.co/MahmoodLab/UNI2-h>`__
     - 🔒 `CC-BY-NC-ND-4.0 <https://creativecommons.org/licenses/by-nc-nd/4.0/>`__
   * - **CONCH**
     - 512
     - ``--patch_encoder conch_v1 --patch_size 512 --mag 20``
     - `MahmoodLab/CONCH <https://huggingface.co/MahmoodLab/CONCH>`__
     - 🔒 `CC-BY-NC-ND-4.0 <https://creativecommons.org/licenses/by-nc-nd/4.0/>`__
   * - **CONCHv1.5**
     - 768
     - ``--patch_encoder conch_v15 --patch_size 512 --mag 20``
     - `MahmoodLab/conchv1_5 <https://huggingface.co/MahmoodLab/conchv1_5>`__
     - 🔒 `CC-BY-NC-ND-4.0 <https://creativecommons.org/licenses/by-nc-nd/4.0/>`__
   * - **Virchow**
     - 2560
     - ``--patch_encoder virchow --patch_size 224 --mag 20``
     - `paige-ai/Virchow <https://huggingface.co/paige-ai/Virchow>`__
     - 🔒 `Apache-2.0 <https://www.apache.org/licenses/LICENSE-2.0>`__
   * - **Virchow2**
     - 2560
     - ``--patch_encoder virchow2 --patch_size 224 --mag 20``
     - `paige-ai/Virchow2 <https://huggingface.co/paige-ai/Virchow2>`__
     - 🔒 `CC-BY-NC-ND-4.0 <https://creativecommons.org/licenses/by-nc-nd/4.0/>`__
   * - **Virchow2 (CLS only)**
     - 1280
     - ``--patch_encoder virchow2-cls --patch_size 224 --mag 20``
     - `paige-ai/Virchow2 <https://huggingface.co/paige-ai/Virchow2>`__
     - 🔒 `CC-BY-NC-ND-4.0 <https://creativecommons.org/licenses/by-nc-nd/4.0/>`__
   * - **Phikon**
     - 768
     - ``--patch_encoder phikon --patch_size 224 --mag 20``
     - `owkin/phikon <https://huggingface.co/owkin/phikon>`__
     - 🌐 `Owkin non-commercial <https://github.com/owkin/HistoSSLscaling/blob/main/LICENSE.txt>`__
   * - **Phikon-v2**
     - 1024
     - ``--patch_encoder phikon_v2 --patch_size 224 --mag 20``
     - `owkin/phikon-v2 <https://huggingface.co/owkin/phikon-v2/>`__
     - 🌐 `Owkin non-commercial <https://huggingface.co/owkin/phikon-v2/blob/main/LICENSE.pdf>`__
   * - **KEEP**
     - 768
     - ``--patch_encoder keep --patch_size 256 --mag 20``
     - `Astaxanthin/KEEP <https://huggingface.co/Astaxanthin/KEEP>`__
     - 🌐 `MIT <https://opensource.org/license/mit>`__
   * - **Prov-Gigapath**
     - 1536
     - ``--patch_encoder gigapath --patch_size 256 --mag 20``
     - `prov-gigapath <https://huggingface.co/prov-gigapath/prov-gigapath>`__
     - 🔒 `Apache-2.0 <https://www.apache.org/licenses/LICENSE-2.0>`__
   * - **Prov-Gigapath-Flash**
     - 384
     - ``--patch_encoder gigapath-flash --patch_size 256 --mag 20``
     - `prov-gigapath-flash <https://huggingface.co/prov-gigapath/prov-gigapath-flash>`__
     - 🔒 `Apache-2.0 <https://www.apache.org/licenses/LICENSE-2.0>`__
   * - **H-Optimus-0**
     - 1536
     - ``--patch_encoder hoptimus0 --patch_size 224 --mag 20``
     - `bioptimus/H-optimus-0 <https://huggingface.co/bioptimus/H-optimus-0>`__
     - 🔒 `Apache-2.0 <https://www.apache.org/licenses/LICENSE-2.0>`__
   * - **H-Optimus-1**
     - 1536
     - ``--patch_encoder hoptimus1 --patch_size 224 --mag 20``
     - `bioptimus/H-optimus-1 <https://huggingface.co/bioptimus/H-optimus-1>`__
     - 🔒 `CC-BY-NC-ND-4.0 <https://creativecommons.org/licenses/by-nc-nd/4.0/>`__
   * - **H0-mini**
     - 768/1536
     - ``--patch_encoder h0-mini --patch_size 224 --mag 20``
     - `bioptimus/H0-mini <https://huggingface.co/bioptimus/H0-mini>`__
     - 🔒 `CC-BY-NC-ND-4.0 <https://creativecommons.org/licenses/by-nc-nd/4.0/>`__
   * - **MUSK**
     - 1024
     - ``--patch_encoder musk --patch_size 384 --mag 20``
     - `xiangjx/musk <https://huggingface.co/xiangjx/musk>`__
     - 🔒 `CC-BY-NC-ND-4.0 <https://creativecommons.org/licenses/by-nc-nd/4.0/>`__
   * - **Midnight-12k**
     - 3072
     - ``--patch_encoder midnight12k --patch_size 224 --mag 20``
     - `kaiko-ai/midnight <https://huggingface.co/kaiko-ai/midnight>`__
     - 🌐 `MIT <https://opensource.org/license/mit>`__
   * - **Phaet**
     - 1024
     - ``--patch_encoder phaet --patch_size 224 --mag 20``
     - `wearewaiv/phaet <https://huggingface.co/wearewaiv/phaet>`__
     - 🔒 `Waiv non-commercial <https://huggingface.co/wearewaiv/phaet/blob/main/LICENSE.pdf>`__
   * - **Mascaret**
     - 1536/3072
     - ``--patch_encoder mascaret --patch_size 224 --mag 20``
     - `wearewaiv/mascaret <https://huggingface.co/wearewaiv/mascaret>`__
     - 🔒 `Waiv non-commercial <https://huggingface.co/wearewaiv/mascaret/blob/main/LICENSE.pdf>`__
   * - **OpenMidnight**
     - 1536
     - ``--patch_encoder openmidnight --patch_size 224 --mag 20``
     - `SophontAI/OpenMidnight <https://huggingface.co/SophontAI/OpenMidnight>`__
     - 🔒 `Apache-2.0 <https://www.apache.org/licenses/LICENSE-2.0>`__
   * - **GPFM**
     - 1024
     - ``--patch_encoder gpfm --patch_size 224 --mag 20``
     - `majiabo/GPFM <https://huggingface.co/majiabo/GPFM>`__
     - 🌐 `MIT <https://opensource.org/license/mit>`__
   * - **GenBio-PathFM**
     - 4608
     - ``--patch_encoder genbio-pathfm --patch_size 224 --mag 20``
     - `genbio-ai/genbio-pathfm <https://huggingface.co/genbio-ai/genbio-pathfm>`__
     - 🌐 `GenBio AI Community <https://huggingface.co/genbio-ai/genbio-pathfm/blob/main/LICENSE.txt>`__
   * - **Gemma 4**
     - 768/1152
     - ``--patch_encoder {gemma4-e4b, gemma4-26b} --patch_size 224 --mag 20``
     - `google/gemma-4-E4B <https://huggingface.co/google/gemma-4-E4B>`__ / `google/gemma-4-26B-A4B <https://huggingface.co/google/gemma-4-26B-A4B>`__
     - 🌐 `Gemma Terms <https://ai.google.dev/gemma/docs/gemma_4_license>`__
   * - **Kaiko**
     - 384/768/1024
     - ``--patch_encoder kaiko-vit* --patch_size 256 --mag 20``
     - `Kaiko Collection <https://huggingface.co/collections/1aurent/kaikoai-models-66636c99d8e1e34bc6dcf795>`__
     - 🌐 `Kaiko non-commercial <https://github.com/kaiko-ai/towards_large_pathology_fms/blob/main/LICENSE>`__
   * - **Lunit**
     - 384
     - ``--patch_encoder lunit-vits8 --patch_size 224 --mag 20``
     - `1aurent/lunit <https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino>`__
     - 🌐 `Lunit non-commercial <https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino>`__
   * - **Hibou**
     - 1024
     - ``--patch_encoder hibou_l --patch_size 224 --mag 20``
     - `histai/hibou-L <https://huggingface.co/histai/hibou-L>`__
     - 🔒 `Apache-2.0 <https://www.apache.org/licenses/LICENSE-2.0>`__
   * - **CTransPath-CHIEF**
     - 768
     - ``--patch_encoder ctranspath --patch_size 256 --mag 10``
     - —
     - 🌐 `GPL-3.0 <https://www.gnu.org/licenses/gpl-3.0.html>`__
   * - **ResNet50**
     - 1024
     - ``--patch_encoder resnet50 --patch_size 256 --mag 20``
     - —
     - 🌐 `BSD-3-Clause <https://opensource.org/license/bsd-3-clause>`__

.. automodule:: trident.patch_encoder_models
   :members:
   :undoc-members:


Slide Encoders
--------------

Factory for slide-level encoder models.

.. list-table:: 
   :header-rows: 1
   :widths: 16 14 32 20 22

   * - Slide Encoder
     - Patch Encoder
     - Args
     - Link
     - License
   * - **Threads**
     - conch_v15
     - ``--slide_encoder threads --patch_size 512 --mag 20``
     - *(Coming Soon!)*
     - —
   * - **Titan**
     - conch_v15
     - ``--slide_encoder titan --patch_size 512 --mag 20``
     - `MahmoodLab/TITAN <https://huggingface.co/MahmoodLab/TITAN>`__
     - 🔒 `CC-BY-NC-ND-4.0 <https://creativecommons.org/licenses/by-nc-nd/4.0/>`__
   * - **PRISM**
     - virchow
     - ``--slide_encoder prism --patch_size 224 --mag 20``
     - `paige-ai/Prism <https://huggingface.co/paige-ai/Prism>`__
     - 🔒 `CC-BY-NC-ND-4.0 <https://creativecommons.org/licenses/by-nc-nd/4.0/>`__
   * - **PRISM2**
     - virchow2-cls
     - ``--slide_encoder prism2 --patch_size 224 --mag 20``
     - `paige-ai/Prism2 <https://huggingface.co/paige-ai/Prism2>`__
     - 🔒 `CC-BY-NC-ND-4.0 <https://creativecommons.org/licenses/by-nc-nd/4.0/>`__
   * - **CHIEF**
     - ctranspath
     - ``--slide_encoder chief --patch_size 256 --mag 10``
     - `CHIEF <https://github.com/hms-dbmi/CHIEF>`__
     - 🌐 `AGPL-3.0 <https://www.gnu.org/licenses/agpl-3.0.html>`__
   * - **GigaPath**
     - gigapath
     - ``--slide_encoder gigapath --patch_size 256 --mag 20``
     - `prov-gigapath <https://huggingface.co/prov-gigapath/prov-gigapath>`__
     - 🔒 `Apache-2.0 <https://www.apache.org/licenses/LICENSE-2.0>`__
   * - **GigaPath-Flash**
     - gigapath-flash
     - ``--slide_encoder gigapath-flash --patch_size 256 --mag 20``
     - `prov-gigapath-flash <https://huggingface.co/prov-gigapath/prov-gigapath-flash>`__
     - 🔒 `Apache-2.0 <https://www.apache.org/licenses/LICENSE-2.0>`__
   * - **Madeleine**
     - conch_v1
     - ``--slide_encoder madeleine --patch_size 256 --mag 10``
     - `MahmoodLab/madeleine <https://huggingface.co/MahmoodLab/madeleine>`__
     - 🔒 `MIT <https://opensource.org/license/mit>`__
   * - **Feather**
     - conch_v15
     - ``--slide_encoder feather --patch_size 512 --mag 20``
     - `MahmoodLab/feather <https://huggingface.co/MahmoodLab/abmil.base.conch_v15.pc108-24k>`__
     - 🔒 `CC-BY-NC-ND-4.0 <https://creativecommons.org/licenses/by-nc-nd/4.0/>`__
   * - **Feather-UNI2**
     - uni_v2
     - ``--slide_encoder feather_uni_v2 --patch_size 256 --mag 20``
     - `MahmoodLab/feather <https://huggingface.co/MahmoodLab/abmil.base.uni_v2.pc108-24k>`__
     - 🔒 `CC-BY-NC-ND-4.0 <https://creativecommons.org/licenses/by-nc-nd/4.0/>`__
   * - **CARE**
     - conch_v15
     - ``--slide_encoder care --patch_size 512 --mag 20``
     - `Zipper-1/CARE <https://huggingface.co/Zipper-1/CARE>`__
     - 🔒 `CC-BY-NC-4.0 <https://creativecommons.org/licenses/by-nc/4.0/>`__
   * - **ABMIL**
     - any
     - Python API only — untrained aggregator
     - —
     - —

.. automodule:: trident.slide_encoder_models
   :members:
   :undoc-members:
