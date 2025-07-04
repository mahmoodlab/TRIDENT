Frequently Asked Questions
==========================

.. dropdown:: **How do I extract embeddings from legacy CLAM coordinates?**

   Use the `--coords_dir` flag to pass CLAM-style patch coordinates:

   .. code-block:: bash

      python run_batch_of_slides.py --task feat --wsi_dir wsis --job_dir legacy_dir --coords_dir extracted_coords --patch_encoder uni_v1


.. dropdown:: **My WSIs have no micron-per-pixel (MPP) or magnification metadata. What should I do?**

   PNGs and JPEGs do not store MPP metadata in the file itself. If you're working with such formats, passing a CSV via `--custom_list_of_wsis` is **required**. This CSV should include at least two columns: `wsi` and `mpp`.

   Example:

   .. code-block:: csv

      wsi,mpp
      TCGA-AJ-A8CV-01Z-00-DX1_1.png,0.25
      TCGA-AJ-A8CV-01Z-00-DX1_2.png,0.25
      TCGA-AJ-A8CV-01Z-00-DX1_3.png,0.25

   If you're using OpenSlide-readable formats (e.g., `.svs`, `.tiff`), this CSV is optional—but you can still use it to:

   - Restrict processing to a specific subset of slides
   - Override incorrect or missing MPP metadata


.. dropdown:: **I want to skip patches on holes.**

   By default, TRIDENT includes all tissue patches (including holes). Use `--remove_holes` to exclude them. Not recommended, as "holes" often help define the tissue microenvironment.

.. dropdown:: **I don’t have enough local SSD storage and my WSIs are on a slow remote disk. How can I accelerate processing?**

   When WSIs are stored on slow network or external drives, processing can be very slow. Use `--wsi_cache ./cache --cache_batch_size 32` to enable local caching. WSIs will be copied in batches to a local SSD, processed in parallel, and automatically cleaned up after use. This significantly reduces I/O bottlenecks.

.. dropdown:: **My WSIs are in multiple subfolders. How can I process them all?**

   By default, only the top-level directory is scanned. Use `--search_nested` to recursively search for WSIs in all nested folders and include them in processing.

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

      segmentation_models = ["hest", "grandqc", "grandqc_artifact"]
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