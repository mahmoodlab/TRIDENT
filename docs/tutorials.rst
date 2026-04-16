Tutorials
=========

Browse our interactive guides:

- `1-Step-by-Step-Patch-Feature-Extraction-with-Trident.ipynb <https://github.com/mahmoodlab/TRIDENT/blob/main/tutorials/1-Step-by-Step-Patch-Feature-Extraction-with-Trident.ipynb>`_: Guided whole-slide image processing.
- `2-Using-Trident-With-Your-Custom-Patch-Encoder.ipynb <https://github.com/mahmoodlab/TRIDENT/blob/main/tutorials/2-Using-Trident-With-Your-Custom-Patch-Encoder.ipynb>`_: Using Trident with a custom patch encoder. 
- `3-Training-a-WSI-Classification-Model-with-ABMIL-and-Heatmaps.ipynb <https://github.com/mahmoodlab/TRIDENT/blob/main/tutorials/3-Training-a-WSI-Classification-Model-with-ABMIL-and-Heatmaps.ipynb>`_: Training an ABMIL model with attention heatmaps.

When to use each tutorial:

- Tutorial 1: start here if you are new and want a complete end-to-end example.
- Tutorial 2: use this when integrating your own patch encoder into Trident workflows.
- Tutorial 3: use this after feature extraction, when moving to downstream training and interpretation.

Practical recipes (what most users actually do)
-----------------------------------------------

Validate on one slide (fast QA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before launching a large run, validate:

- the reader backend can open your files
- MPP/magnification behaves as expected
- segmentation quality is acceptable
- patch count is in the right ballpark

.. code-block:: bash

   python run_single_slide.py --slide_path ./wsis/example.svs --job_dir ./job --segmenter grandqc --mag 20 --patch_size 256

Then inspect:

- ``job/summary.md`` (what happened)
- ``job/wsi_states/`` (per-slide details and errors)
- ``job/contours_geojson/`` (open in QuPath if you want to QC/edit)

Scale to a dataset (batch)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./job --segmenter grandqc --patch_encoder uni_v1 --mag 20 --patch_size 256 --skip_errors

If WSIs are on slow storage, add caching:

.. code-block:: bash

   python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./job --wsi_cache /localssd/cache --cache_batch_size 32 --skip_errors

Retry only a subset (curated CSV)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a CSV with a ``wsi`` column containing **relative paths** under ``--wsi_dir``:

.. code-block:: text

   wsi
   patientA/slide1.svs
   patientB/slide7.svs

Then run:

.. code-block:: bash

   python run_batch_of_slides.py --task feat --wsi_dir ./wsis --job_dir ./job --custom_list_of_wsis retry.csv --patch_encoder uni_v1 --mag 20 --patch_size 256

Offline clusters (no internet)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Put weights into the local registries:

  - ``trident/segmentation_models/local_ckpts.json``
  - ``trident/patch_encoder_models/local_ckpts.json``
  - ``trident/slide_encoder_models/local_ckpts.json``

- Or pass a patch checkpoint directly:

.. code-block:: bash

   python run_batch_of_slides.py --task feat --wsi_dir ./wsis --job_dir ./job --patch_encoder uni_v1 --patch_encoder_ckpt_path /path/to/weights.bin --mag 20 --patch_size 256
