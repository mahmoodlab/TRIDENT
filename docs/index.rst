.. image:: _static/trident_crop.jpg
   :align: right
   :width: 220px

Welcome to **TRIDENT**!
======================================

**TRIDENT** is a scalable toolkit for **large-scale whole-slide image (WSI) processing**, developed at the `Mahmood Lab <https://mahmoodlab.org>`_ at **Harvard Medical School** and **Brigham and Women's Hospital**.

🚀 **What TRIDENT offers:**

- **Tissue vs. background segmentation** for H&E, IHC, special stains, and artifact removal
- **Patch-level feature extraction** using 20+ foundation models
- **Slide-level feature extraction** via 5+ pretrained model backbones
- Native support for **OpenSlide**, **CuCIM**, and **PIL-compatible** formats

Explore the **end-to-end pipeline**, from segmentation to slide-level representation — all powered by the latest **foundation models** for computational pathology.

---

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
