### 🔨 **Running Trident for TITAN Inference**:

**Using TITAN for a single slide**

Run this command to perform segmentation, patching, and TITAN slide-level feature extraction for:

1. A single slide
```bash
SLIDE_PATH=./wsis/xxxx.svs
OUTPUT_DIR=./trident_single_processed

python run_single_slide.py \
 --slide_path $SLIDE_PATH \
 --job_dir $OUTPUT_DIR \
 --slide_encoder titan \
 --mag 20 --patch_size 512
```

2. A directory of slides
```bash
SLIDE_DIR=./wsis
OUTPUT_DIR=./trident_batch_processed

python run_batch_of_slides.py \
 --task all \
 --wsi_dir $SLIDE_DIR \
 --job_dir $OUTPUT_DIR \
 --slide_encoder titan \
 --mag 20 --patch_size 512
```

**Further Step-by-Step Instructions:**

**Step 1: Tissue Segmentation:** Segments tissue vs. background from a dir of WSIs
 - **Command**:
   ```bash
   python run_batch_of_slides.py --task seg --wsi_dir ./wsis --job_dir ./trident_processed --gpu 0 --segmenter grandqc
   ```
   - `--task seg`: Specifies that you want to do tissue segmentation.
   - `--wsi_dir ./wsis`: Path to dir with your WSIs.
   - `--job_dir ./trident_processed`: Output dir for processed results.
   - `--gpu 0`: Uses GPU with index 0.
   - `--segmenter`: Segmentation model. Defaults to `grandqc` for fast H&E segmentation. Add the option `--remove_artifacts` for additional artifact clean up.
 - **Outputs**:
   - WSI thumbnails in `./trident_processed/thumbnails`.
   - WSI thumbnails with tissue contours in `./trident_processed/contours`.
   - GeoJSON files containing tissue contours in `./trident_processed/contours_geojson`. These can be opened in [QuPath](https://qupath.github.io/) for editing/quality control, if necessary.

 **Step 2: Tissue Patching:** Extracts patches from segmented tissue regions at a specific magnification.
 - **Command**:
   ```bash
   python run_batch_of_slides.py --task coords --wsi_dir ./wsis --job_dir ./trident_processed --mag 20 --patch_size 512 --overlap 0
   ```
   - `--task coords`: Specifies that you want to do patching.
   - `--wsi_dir wsis`: Path to the dir with your WSIs.
   - `--job_dir ./trident_processed`: Output dir for processed results.
   - `--mag 20`: Extracts patches at 20x magnification.
   - `--patch_size 512`: Each patch is 512x512 pixels.
   - `--overlap 0`: Patches overlap by 0 pixels (**always** an absolute number in pixels, e.g., `--overlap 128` for 50% overlap for 256x256 patches.
 - **Outputs**:
   - Patch coordinates as h5 files in `./trident_processed/20x_512px/patches`.
   - WSI thumbnails annotated with patch borders in `./trident_processed/20x_512px/visualization`.

 **Step 3a: Patch Feature Extraction:** Extracts features from tissue patches using a specified encoder
 - **Command**:
   ```bash
   python run_batch_of_slides.py --task feat --wsi_dir ./wsis --job_dir ./trident_processed --patch_encoder conch_v15 --mag 20 --patch_size 512
   ```
   - `--task feat`: Specifies that you want to do feature extraction.
   - `--wsi_dir wsis`: Path to the dir with your WSIs.
   - `--job_dir ./trident_processed`: Output dir for processed results.
   - `--patch_encoder conch_v15`: Uses the  **CONCHv1.5** patch encoder. See below for list of supported models. 
   - `--mag 20`: Features are extracted from patches at 20x magnification.
   - `--patch_size 512`: Patches are 512x512 pixels in size.
 - **Outputs**: 
   - Features are saved as h5 files in `./trident_processed/20x_256px/features_conch_v15`. (Shape: `(n_patches, 768)`)

This deployment of TRIDENT (specialized for **TITAN** inference) supports the following patch encoders, loaded via `./trident/patch_encoder_models/load.py`. The **CONCHv1.5** model checkpoint is made available in the local path: `./trident/patch_encoder_models/model_zoo/conchv1_5/pytorch_model_vision.bin`. Other models can be made available upon request.

- **UNI**: [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI)  (`--patch_encoder uni_v1 --patch_size 256 --mag 20`)
- **UNIv2**: [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h)  (`--patch_encoder uni_v2 --patch_size 256 --mag 20`)
- **CONCH**: [MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH)  (`--patch_encoder conch_v1 --patch_size 512 --mag 20`)
- **CONCHv1.5**: [MahmoodLab/conchv1_5](https://huggingface.co/MahmoodLab/conchv1_5)  (`--patch_encoder conch_v15 --patch_size 512 --mag 20`)

**Step 3b: Slide Feature Extraction:** Extracts slide embeddings using  **TITAN** as the slide encoder. Choosing **TITAN** will automatically select the **CONCHv1.5** as the patch encoder to extract features (if not already extracted). The **TITAN** model checkpoint is made available in the local path: `./trident/slide_encoder_models/model_zoo/titan/pytorch_model_vision.bin`

 - **Command**:
   ```bash
   python run_batch_of_slides.py --task feat --wsi_dir ./wsis --job_dir ./trident_processed --slide_encoder titan --mag 20 --patch_size 512 
   ```
   - `--task feat`: Specifies that you want to do feature extraction.
   - `--wsi_dir wsis`: Path to the dir containing WSIs.
   - `--job_dir ./trident_processed`: Output dir for processed results.
   - `--slide_encoder titan`: Uses the `Titan` slide encoder. See below for supported models.
   - `--mag 20`: Features are extracted from patches at 20x magnification.
   - `--patch_size 512`: Patches are 512x512 pixels in size.
 - **Outputs**: 
   - Features are saved as h5 files in `./trident_processed/20x_512px/slide_features_titan`. (Shape: `(768)`)

Please see the tutorial `./tutorial/TITAN-walkthrough.ipynb` for more support, the `./docs/detailed_README.md` for additional features and usage, and the `./docs/FAQ.md` for additional questions and answers.

### 🙋 FAQ
- **Q**: How do I extract patch embeddings from legacy patch coordinates extracted with [CLAM](https://github.com/mahmoodlab/CLAM)?
   - **A**:
      ```bash
      python run_batch_of_slides.py --task feat --wsi_dir ..wsis --job_dir legacy_dir --patch_encoder uni_v1 --mag 20 --patch_size 256 --coords_dir extracted_mag20x_patch256_fp/
      ```
- **Q**: How do I keep patches corresponding to holes in the tissue?
   - **A**: In `run_batch_of_slides`, this behavior is default. Set `--remove_holes` to exclude patches on top of holes.

- **Q**: I see weird messages when building models using timm. What is happening?
   - **A**: Make sure `timm==0.9.16` is installed. `timm==1.X.X` creates issues with most models. 

- **Q**: How can I use `run_single_slide.py` and `run_batch_of_slides.py` in other repos with minimal work?
  - **A**: Make sure `trident` is installed using `pip install -e .`. Then, both scripts are exposed and can be integrated into any Python code, e.g., as

```python
import sys 
from run_single_slide import main

sys.argv = [
    "run_single_slide",
    '--slide_path', "output/wsis/394140.svs",
    "--job_dir", 'output/',
    "--mag", "20",
    "--patch_size", '256'
]

main()
```

- **Q**: I am not satisfied with the tissue vs background segmentation. What can I do?
   - **A**: Trident uses GeoJSON to store and load segmentations. This format is natively supported by [QuPath](https://qupath.github.io/). You can load the Trident segmentation into QuPath, modify it using QuPath's annotation tools, and save the updated segmentation back to GeoJSON.
   - **A**: You can try another segmentation model by specifying `segmenter --grandqc`.

- **Q**: I want to process a custom list of WSIs. Can I do it? Also, most of my WSIs don't have the micron per pixel (mpp) stored. Can I pass it?
   - **A**: Yes using the `--custom_list_of_wsis` argument. Provide a list of WSI names in a CSV (with slide extension, `wsi`). Optionally, provide the mpp (field `mpp`)
