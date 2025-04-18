Quickstart
==========

🚀 Process a full directory of WSIs:

.. code-block:: bash

   python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./trident_processed --patch_encoder uni_v1 --mag 20 --patch_size 256

🧪 Test a single WSI:

.. code-block:: bash

   python run_single_slide.py --slide_path ./wsis/sample.svs --job_dir ./trident_processed --patch_encoder uni_v1 --mag 20 --patch_size 256

👣 Or go step-by-step:

- Tissue segmentation
- Patch extraction
- Feature extraction

Tissue Segmentation
-------------------
Segment WSIs into tissue vs. background with:

.. code-block:: bash

   python run_batch_of_slides.py --task seg --wsi_dir ./wsis --job_dir ./trident_processed --segmenter hest --remove_artifacts

Outputs: GeoJSONs, contours, thumbnails.

Patch Extraction
----------------
Extract tissue patches at desired magnification:

.. code-block:: bash

   python run_batch_of_slides.py --task coords --wsi_dir ./wsis --job_dir ./trident_processed --mag 20 --patch_size 256

Outputs: Patch coordinates and visualizations.

Patch Feature Extraction
------------------------
Embed patches using any supported foundation model:

.. code-block:: bash

   python run_batch_of_slides.py --task feat --wsi_dir ./wsis --job_dir ./trident_processed --patch_encoder uni_v1

Slide Feature Extraction
------------------------
Embed entire slides via models like TITAN or GigaPath:

.. code-block:: bash

   python run_batch_of_slides.py --task feat --slide_encoder titan --patch_size 512 --mag 20

