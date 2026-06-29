.. image:: _static/trident_crop.jpg
   :align: right
   :width: 220px

Welcome to **TRIDENT**!
======================================

**TRIDENT** is a scalable toolkit for **large-scale whole-slide image (WSI) processing**, developed at the `Mahmood Lab <https://mahmoodlab.org>`_ at **Harvard Medical School** and **Brigham and Women's Hospital**.

What TRIDENT does, end to end
-----------------------------

Take a folder of slides, get back tissue masks, patch coordinates, and patch / slide
embeddings with one command scalable to thousands of slides.

.. code-block:: bash

   python run_batch_of_slides.py \
       --task all \
       --wsi_dir ./wsis --job_dir ./out \
       --segmenter grandqc \
       --patch_encoder uni_v1 --mag 20 --patch_size 256 \
       --gpus 0 1

Highlights
----------

**Pipeline**

- **One command for three stages**: segmentation → patch coordinates → patch / slide features.
  Run them together (``--task all``) or independently (``seg`` / ``coords`` / ``feat``).
- **More consumers of the same coords**: cell / nuclei segmentation (``--task patch_seg``;
  HistoPlus, CellViT++) and VLM question answering over ROIs (``--task vlm``; Patho-R1).
- **Smart skip & resume**: if an output already exists and is not actively being written,
  TRIDENT skips that task. Re-running on the same ``--job_dir`` is safe by design.
- **Per-run reports**: every run produces ``summary.md`` (human-readable) plus
  ``runs/<id>.json`` and ``wsi_states/<slide>.json`` (machine-readable) so you always know
  what happened, what failed, and what to retry.

**Scale**

- **Multi-GPU sharding** out of the box: ``--gpus 0 1 2 3`` splits pending slides evenly
  across devices. Works in both standard and cache modes.
- **Multi-CPU fallback**: ``--gpus -1 -1`` runs two CPU workers in parallel. Useful for
  segmentation-only or otsu pipelines on machines without a GPU.
- **WSI cache pipeline** for slow / network storage: ``--wsi_cache /local/ssd`` stages
  slides locally in batches via a producer / consumer pipeline, processes them, then
  evicts — no manual copying.

**Reliability**

- **Self-describing locks**: each ``.lock`` file carries the writer's PID, host, and
  timestamp. Stale locks from crashes are cleaned safely with ``--clear_dead_locks``,
  active locks from running jobs are never touched.
- **``--skip_errors``** for production: failed slides are recorded and the batch keeps
  going. The summary tells you exactly which slides need attention.

**Models**

- **22+ patch encoders**: UNI / UNI2-h, CONCH / CONCHv1.5, Virchow / Virchow2, Phikon /
  Phikon-v2, KEEP, Prov-GigaPath, H-Optimus 0/1, H0-mini, MUSK, Midnight-12k,
  OpenMidnight, GPFM, GenBio-PathFM, Kaiko (5 variants), Lunit, Hibou-L, CTransPath,
  ResNet50.
- **6+ slide encoders**: Threads (coming soon), Titan, PRISM, CHIEF, GigaPath, Madeleine,
  Feather. Auto-runs the correct underlying patch encoder for you.
- **Bring your own encoder**: ``CustomInferenceEncoder`` / ``CustomSlideEncoder`` plug
  into the same pipeline.

**File formats**

- OpenSlide (``.svs``, ``.tiff``, ``.ndpi``, ``.mrxs``, …), CuCIM, plain images
  (``.png``, ``.jpeg``), SDPC (``.sdpc``), OME-Zarr / NGFF, Zeiss CZI (``.czi``), DICOM
  (``.dcm``, via OpenSlide). Use ``trident convert`` to make awkward formats friendly
  (pyramidal TIFF).

**Operability**

- ``trident-doctor --profile {base,patch-encoders,slide-encoders,convert,full}`` is a
  preflight check that catches missing dependencies, missing weights, or HF gating
  before you launch a long job.
- Offline-friendly: drop weights in
  ``trident/{segmentation,patch_encoder,slide_encoder}_models/local_ckpts.json`` or pass
  ``--patch_encoder_ckpt_path`` directly.

Quick command guide
-------------------

Use these calls based on your goal:

- ``python run_single_slide.py ...``: run one slide end-to-end. Use this first to validate settings.
- ``python run_batch_of_slides.py ...``: run many slides. Use this for production jobs.
- ``trident batch -- ...`` / ``trident single -- ...``: same workflows through one unified CLI.
- ``trident convert ...``: convert listed files to pyramidal TIFF before downstream processing.
- ``trident doctor -- --profile ...``: check environment readiness before long runs.

If you are new, start with ``single`` on one known-good slide, then scale to ``batch``.

.. toctree::
   :maxdepth: 2
   :caption: 📚 Contents

   installation
   quickstart
   tutorials
   api
   faq
   citation

.. note::
   🧪 This project is supported by **NIH NIGMS R35GM138216** and is under active development by the Mahmood Lab.
