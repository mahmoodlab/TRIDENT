# 🔱   Trident

 [arXiv](https://arxiv.org/pdf/2502.06750) | [Blog](https://www.linkedin.com/pulse/announcing-new-open-source-tools-accelerate-ai-pathology-andrew-zhang-loape/?trackingId=pDkifo54SRuJ2QeGiGcXpQ%3D%3D) | [Cite](https://github.com/mahmoodlab/trident?tab=readme-ov-file#reference)
 | [License](https://github.com/mahmoodlab/trident?tab=License-1-ov-file)
 
Trident is a toolkit for large-scale whole-slide image processing.
This project was developed by the [Mahmood Lab](https://faisal.ai/) at Harvard Medical School and Brigham and Women's Hospital. This work was funded by NIH NIGMS R35GM138216.

> [!NOTE]
> Contributions are welcome! Please report any issues. You may also contribute by opening a pull request.

### Key Features:

<img align="right" src="_readme/trident_crop.jpg" width="250px" />

- **Tissue Segmentation**: Extract tissue from background (H&E, IHC, etc.).
- **Patch Extraction**: Extract tissue patches of any size and magnification.
- **Patch Feature Extraction**: Extract patch embeddings from 20+ foundation models, including [UNI](https://www.nature.com/articles/s41591-024-02857-3), [Virchow](https://www.nature.com/articles/s41591-024-03141-0), [H-Optimus-0](https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0) and more...
- **Slide Feature Extraction**: Extract slide embeddings from 5+ slide foundation models, including [Threads](https://arxiv.org/abs/2501.16652) (coming soon!), [Titan](https://arxiv.org/abs/2411.19666), and [GigaPath](https://www.nature.com/articles/s41586-024-07441-w). 

### Updates:
- 04.25: Remove artifacts from the tissue segmentation with `--remove_artifacts`. Works well for H&E.  
- 02.25: New image converter from `czi`, `png`, etc to `tiff`.
- 02.25: Support for [GrandQC](https://www.nature.com/articles/s41467-024-54769-y) tissue vs. background segmentation.
- 02.25: Support for [Madeleine](https://github.com/mahmoodlab/MADELEINE/tree/main), [Hibou](https://github.com/HistAI/hibou), [Lunit](https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino), [Kaiko](https://huggingface.co/histai/hibou-L), and [H-Optimus-1](https://huggingface.co/bioptimus/H-optimus-1) models.

### 🔨 1. **Installation**:
- Create an environment: `conda create -n "trident" python=3.10`, and activate it `conda activate trident`.
- Cloning: `git clone https://github.com/mahmoodlab/trident.git && cd trident`.
- Local installation: `pip install -e .`.

Additional packages may be required to load some pretrained models. Follow error messages for instructions.

### 🔨 2. **Running Trident for TITAN Inference**:

**Already familiar with WSI processing?** Perform segmentation, patching, and TITAN slide feature extraction from a directory of WSIs with:

```
python run_batch_of_slides.py --task all --wsi_dir ./wsis --job_dir ./trident_processed --patch_encoder titan --mag 20 --patch_size 512
```

**Feeling cautious?**

Run this command to perform all processing steps for a **single** slide:
```
python run_single_slide.py --slide_path ./wsis/xxxx.svs --job_dir ./trident_processed --patch_encoder utitan --mag 20 --patch_size 512
```

**Or follow step-by-step instructions:**

**Step 1: Tissue Segmentation:** Segments tissue vs. background from a dir of WSIs
 - **Command**:
   ```bash
   python run_batch_of_slides.py --task seg --wsi_dir ./wsis --job_dir ./trident_processed --gpu 0 --segmenter hest
   ```
   - `--task seg`: Specifies that you want to do tissue segmentation.
   - `--wsi_dir ./wsis`: Path to dir with your WSIs.
   - `--job_dir ./trident_processed`: Output dir for processed results.
   - `--gpu 0`: Uses GPU with index 0.
   - `--segmenter`: Segmentation model. Defaults to `hest`. Switch to `grandqc` for fast H&E segmentation. Add the option `--remove_artifacts` for additional artifact clean up.
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
   - Features are saved as h5 files in `./trident_processed/20x_256px/features_conch_v15`. (Shape: `(n_patches, feature_dim)`)

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
   - Features are saved as h5 files in `./trident_processed/20x_256px/slide_features_titan`. (Shape: `(feature_dim)`)

Please see the tutorials for more support, the `./docs/detailed_README.md` for additional features and usage, and the `./docs/FAQ.md` for additional questions and answers.

## License and Terms of Use

ⓒ Mahmood Lab. This repository is released under the [CC-BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en) license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of this repository is prohibited and requires prior approval. By downloading any pretrained encoder, you agree to follow the model's respective license.

## Acknowledgements

This repository is a fork of the original [TRIDENT GitHub repository](https://github.com/mahmoodlab/TRIDENT), which has been simplifed for TITAN inference. This repisotry is also built on top of amazing repositories such as [Timm](https://github.com/huggingface/pytorch-image-models/), [HuggingFace](https://huggingface.co/docs/datasets/en/index), and open-source contributions from the community. We thank the authors and developers for their contribution. 

## Funding
This work was funded by NIH NIGMS [R35GM138216](https://reporter.nih.gov/search/sWDcU5IfAUCabqoThQ26GQ/project-details/10029418).

<img src="_readme/joint_logo.png"> 
