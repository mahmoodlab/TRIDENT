Quickstart
==================

Trident provides user-facing command-line scripts for processing large batches of whole-slide images (WSIs).

This page explains how to quickly get started, and provides detailed help for available options.

---

Processing a batch of slides
----------

To process a batch of WSIs through segmentation, patch extraction, and feature extraction in one go,  
run the following command:

.. code-block:: bash

    python run_batch_of_slides.py --task all --wsi_dir output/wsis --job_dir output --patch_encoder uni_v1 --mag 20 --patch_size 256

- `--wsi_dir`: Folder containing your whole-slide images (.svs, .tiff, etc.)
- `--job_dir`: Folder where all outputs (masks, coordinates, features) will be stored
- `--patch_encoder`: Pre-trained encoder to use (e.g., `uni_v1`, `conch_v15`, etc.)
- `--mag`: Target magnification level for patches (e.g., 20x)
- `--patch_size`: Size of patches to extract (e.g., 256px)

This will:
1. Segment tissue areas in the slides.
2. Extract patch coordinates over the tissue.
3. Extract patch-level features using the specified encoder.

Typical Examples
-----------------

**Segmentation Only**  
(Segment tissue regions and save binary masks.)

.. code-block:: bash

    python run_batch_of_slides.py --task seg --wsi_dir input_wsis --job_dir output

**Patch Extraction Only**  
(Extract patch coordinates from tissue regions.)

.. code-block:: bash

    python run_batch_of_slides.py --task coords --wsi_dir input_wsis --job_dir output --mag 20 --patch_size 256

**Feature Extraction Only**  
(Extract features from patches using a patch encoder.)

.. code-block:: bash

    python run_batch_of_slides.py --task feat --wsi_dir input_wsis --job_dir output --patch_encoder uni_v1 --mag 20 --patch_size 256

---

Help Text
---------

The full list of available arguments and options is shown below.

.. note::

   .. raw:: html

      <div style="padding: 12px; background-color: #f9fafb; border: 1px solid #e0e0e0; border-radius: 8px; box-shadow: 2px 2px 8px rgba(0,0,0,0.05);">

      <h3 style="margin-top: 0px; margin-bottom: 10px;">
         🛠️ <b>Command-line Argument Reference</b>
      </h3>

   .. literalinclude:: generated/run_batch_of_slides_help.txt
      :language: text

   .. raw:: html

      </div>

Notes
-----

- If you use `--task all`, it will run segmentation, patch extraction, and feature extraction sequentially.
- For feature extraction, you can choose either a **patch encoder** (e.g., `uni_v1`) or a **slide encoder** (e.g., `threads`).
- You can cache WSIs locally using `--wsi_cache` for faster processing on networked filesystems.
- You can control the number of parallel workers with `--max_workers`.

For more advanced settings (artifact removal, segmentation confidence thresholds, slide readers),  
refer to the full help text above.

