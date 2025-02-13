# 🔱   Trident

 [arXiv](https://arxiv.org/pdf/2502.06750) | [Blog](https://www.linkedin.com/pulse/announcing-new-open-source-tools-accelerate-ai-pathology-andrew-zhang-loape/?trackingId=pDkifo54SRuJ2QeGiGcXpQ%3D%3D) | [Cite](https://github.com/mahmoodlab/trident?tab=readme-ov-file#reference)
 | [License](https://github.com/mahmoodlab/trident?tab=License-1-ov-file)
 
Trident is a toolkit for large-scale whole-slide image processing.
This project was developed by the [Mahmood Lab](https://faisal.ai/) at Harvard Medical School and Brigham and Women's Hospital. This work was funded by NIH NIGMS R35GM138216.

> [!NOTE]
> Contributions are welcome! Please report any issues. You may also contribute by opening a pull request.

## Key Features:

<img align="right" src="_readme/trident_crop.jpg" width="250px" />

- **Tissue Segmentation**: Extract tissue from background using a DeepLabv3 model (supports H&E, IHC, penmark and artifact removal, etc.).
- **Patch Extraction**: Extract tissue patches of any size and magnification.
- **Patch Feature Extraction**: Extract patch embeddings from tissue patches using 20 popular foundation models, including [UNI](https://www.nature.com/articles/s41591-024-02857-3), [CONCH](https://www.nature.com/articles/s41591-024-02856-4), [Virchow](https://www.nature.com/articles/s41591-024-03141-0), [H-Optimus-0](https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0) and many more...
- **Slide Feature Extraction**: Extract slide embeddings from pre-extracted patch embeddings using 5 whole-slide foundation models, including [Threads](https://arxiv.org/abs/2501.16652) (coming soon!), [Titan](https://arxiv.org/abs/2411.19666), 
[PRISM](https://arxiv.org/abs/2405.10254), [GigaPath](https://www.nature.com/articles/s41586-024-07441-w) and [CHIEF](https://www.nature.com/articles/s41586-024-07894-z). 

## Getting Started:

### 🔨 1. **Installation**:
- Create an environment: `conda create -n "trident" python=3.10`, and activate it `conda activate trident`.
- **Install from local clone**:
    - `git clone https://github.com/mahmoodlab/trident.git && cd trident`.
    - Local install with running `pip install -e .`.
- **Install with pip**:
    - `pip install git+https://github.com/mahmoodlab/trident.git `
- Additional packages may be required if you are loading specific pretrained models. Follow error messages for additional instructions.

### 🔨 2. **Running Trident**:

**Already familiar with WSI processing?**

Want patch embeddings? Perform segmentation, patching, and UNI feature extraction for a directory of WSIs in a single command:

```
python run_batch_of_slides.py --task all --wsi_dir wsis --job_dir ./trident_processed --patch_encoder uni_v1 --mag 20 --patch_size 256
```

**Feeling cautious?**

Run this script to perform all processing steps for just a single slide:
```
python run_single_slide.py --slide_path wsis/xxxx.svs --job_dir ./trident_processed --patch_encoder uni_v1 --mag 20 --patch_size 256
```

**Or follow step-by-step instructions:**

**Step 1: Tissue Segmentation:** Segments tissue vs. background from a list of WSIs
 - **Command**:
   ```bash
   python run_batch_of_slides.py --task seg --wsi_dir ./wsis --job_dir ./trident_processed --gpu 0
   ```
   - `--task seg`: Specifies that you want to do tissue segmentation.
   - `--wsi_dir ./wsis`: Path to the directory containing WSIs.
   - `--job_dir ./trident_processed`: Output directory for processed results.
   - `--gpu 0`: Uses GPU with index 0 for computation.
 - **Outputs**:
   - WSI thumbnails are saved in `./trident_processed/thumbnails`.
   - WSI thumbnails annotated with tissue contours are saved in `./trident_processed/contours`.
   - GeoJSON files containing tissue contours are saved in `./trident_processed/contours_geojson`. These can be opened in [QuPath](https://qupath.github.io/) for editing/quality control, if necessary.

 **Step 2: Tissue Patching:** Extracts patches from segmented tissue regions at a specific magnification.
 - **Command**:
   ```bash
   python run_batch_of_slides.py --task coords --wsi_dir wsis --job_dir ./trident_processed --mag 20 --patch_size 256 --overlap 0
   ```
   - `--task coords`: Specifies that you want to do patching.
   - `--wsi_dir wsis`: Path to the directory containing WSIs.
   - `--job_dir ./trident_processed`: Output directory for processed results.
   - `--mag 20`: Extracts patches at 20x magnification.
   - `--patch_size 256`: Each patch is 256x256 pixels.
   - `--overlap 0`: Patches overlap by 0 pixels. Note that this is the absolute overlap in pixels, i.e. use `--overlap 128` for 50% overlap on patches of size 256.
 - **Outputs**:
   - Patch coordinates are saved as h5 files in `./trident_processed/20x_256px/patches`.
   - WSI thumbnails annotated with patch borders are saved in `./trident_processed/20x_256px/visualization`.

 **Step 3a: Patch Feature Extraction:** Extracts features from tissue patches using a specified encoder
 - **Command**:
   ```bash
   python run_batch_of_slides.py --task feat --wsi_dir wsis --job_dir ./trident_processed --patch_encoder uni_v1 --mag 20 --patch_size 256 
   ```
   - `--task feat`: Specifies that you want to do feature extraction.
   - `--wsi_dir wsis`: Path to the directory containing WSIs.
   - `--job_dir ./trident_processed`: Output directory for processed results.
   - `--patch_encoder uni_v1`: Uses the `UNIv1` patch encoder. Could also be `conch_v1`, `ctranspath`, `gigapath`, `virchow`, `hoptimus0`, etc. See below for list of supported models. 
   - `--mag 20`: Features are extracted from patches at 20x magnification.
   - `--patch_size 256`: Patches are 256x256 pixels in size.
 - **Outputs**: 
   - Features are saved as h5 files in `./trident_processed/20x_256px/features_uni_v1`. (Shape: `(n_patches, feature_dim)`)

Trident supports 20 patch encoders, loaded via a patch-level [`encoder_factory`](https://github.com/mahmoodlab/trident/blob/main/trident/patch_encoder_models/load.py#L14). Models requiring specific installations will return error messages with additional instructions. Gated models on HuggingFace require access requests.

- **UNI**: [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI)  (`--patch_encoder uni_v1`)
- **UNIv2**: [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h)  (`--patch_encoder uni_v2`)
- **CONCH**: [MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH)  (`--patch_encoder conch_v1`)
- **CONCHv1.5**: [MahmoodLab/conchv1_5](https://huggingface.co/MahmoodLab/conchv1_5)  (`--patch_encoder conch_v15`)
- **Virchow**: [paige-ai/Virchow](https://huggingface.co/paige-ai/Virchow)  (`--patch_encoder virchow`)
- **Virchow2**: [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2)  (`--patch_encoder virchow2`)
- **Phikon**: [owkin/phikon](https://huggingface.co/owkin/phikon)  (`--patch_encoder phikon`)
- **Phikon-v2**: [owkin/phikon-v2](https://huggingface.co/owkin/phikon-v2/)  (`--patch_encoder phikon_v2`)
- **Prov-Gigapath**: [prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath)  (`--patch_encoder gigapath`)
- **H-Optimus-0**: [bioptimus/H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0)  (`--patch_encoder hoptimus0`)
- **MUSK**: [xiangjx/musk](https://huggingface.co/xiangjx/musk)  (`--patch_encoder musk`)
- **Kaiko**: Hosted on TorchHub  (`--patch_encoder kaiko-vits8, kaiko-vits16, kaiko-vitb8, kaiko-vitb16, kaiko-vitl14`)
- **Lunit**: [1aurent/vit_small_patch8_224.lunit_dino](https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino)  (`--patch_encoder lunit-vits8`)
- **Hibou**: [histai/hibou-L](https://huggingface.co/histai/hibou-L)  (`--patch_encoder hibou_l`)
- **CTransPath-CHIEF**: Automatic download  (`--patch_encoder ctranspath`)
- **ResNet50**: Hosted on torchvision.  (`--patch_encoder resnet50`)

**Step 3b: Slide Feature Extraction:** Extracts slide embeddings using a slide encoder. Will also automatically extract patch embeddings. 
 - **Command**:
   ```bash
   python run_batch_of_slides.py --task feat --wsi_dir wsis --job_dir ./trident_processed --slide_encoder titan --mag 20 --patch_size 512 
   ```
   - `--task feat`: Specifies that you want to do feature extraction.
   - `--wsi_dir wsis`: Path to the directory containing WSIs.
   - `--job_dir ./trident_processed`: Output directory for processed results.
   - `--slide_encoder titan`: Uses the `Titan` slide encoder. Could also be `prism`, `gigapath`, `chief`, and `threads` (coming soon!). 
   - `--mag 20`: Features are extracted from patches at 20x magnification.
   - `--patch_size 512`: Patches are 512x512 pixels in size.
 - **Outputs**: 
   - Features are saved as h5 files in `./trident_processed/20x_256px/slide_features_titan`. (Shape: `(feature_dim)`)

Trident supports 5 slide encoders, loaded via a slide-level [`encoder_factory`](https://github.com/mahmoodlab/trident/blob/main/trident/slide_encoder_models/load.py#L14). Models requiring specific installations will return error messages with additional instructions. Gated models on HuggingFace require access requests.
- **Threads**: Coming Soon! [MahmoodLab/threads](https://huggingface.co/MahmoodLab/threads) (`--slide_encoder threads`). Based on `conch_v15` with `512x512` @20x.
- **Titan**: [MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN) (`--slide_encoder titan`). Based on `conch_v15` with `512x512` @20x.
- **PRISM**: [paige-ai/Prism](https://huggingface.co/paige-ai/Prism) (`--slide_encoder prism`). Based on `virchow` with `256x256` @20x.
- **CHIEF**: [CHIEF](https://github.com/hms-dbmi/CHIEF) (`--slide_encoder chief`). Based on `ctranspath` with `256x256` @10x.
- **GigaPath**: [prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath)  (`--slide_encoder gigapath`). Based on `gigapath` with `256x256x` @20x.
- **Madeleine**: [MahmoodLab/madeleine](https://huggingface.co/MahmoodLab/madeleine) (`--slide_encoder madeleine`). Based on `conch_v1` with `256x256` @10x.

> [!NOTE]
> If you have a patient containing multiple slides, you have two ways for constructing whole-patient embeddings: processing each slide independently and taking the average of the slide features (late fusion) or pooling all patches together and processing that as a single "pseudo-slide" (early fusion). You can use Trident-generated slide embeddings in your own late fusion pipeline, or use Trident-generated patch embeddings in your own early fusion pipeline. For an implementation of both fusion strategies, please check out our sister repository [Patho-Bench](https://github.com/mahmoodlab/Patho-Bench).

Please see our [tutorials](https://github.com/mahmoodlab/trident/tree/main/tutorials) for more cool things you can do with Trident and a more [detailed readme](https://github.com/mahmoodlab/trident/blob/main/DETAILS.md) for additional features.

## 🙋 FAQ
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
   - **A**: You can try rerunning the segmentation step at a higher magnification (which may be slower, but more accurate).
   - **A**: You can finetune the Trident segmentation model on a few annotated examples to improve performance for your specific use case.

- **Q**: I want to process a custom list of WSIs. Can I do it? Also, most of my WSIs don't have the micron per pixel (mpp) stored. Can I pass it?
   - **A**: Yes using the `--custom_list_of_wsis` argument. Provide a list of WSI names in a CSV (with slide extension, `wsi`). Optionally, provide the mpp (field `mpp`)
 
 - **Q**: Do I need to install any additional packages to use Trident?
   - **A**: Most pretrained models require additional dependencies (e.g., the CTransPath patch encoder requires `pip install timm_ctp`). When you load a model using Trident, it will tell you what dependencies are missing and how to install them. 

## License and Terms of Use

ⓒ Mahmood Lab. This repository is released under the [CC-BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en) license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of this repository is prohibited and requires prior approval. By downloading any pretrained encoder, you agree to follow the model's respective license.

## Acknowledgements

The project was built on top of amazing repositories such as [Timm](https://github.com/huggingface/pytorch-image-models/), [HuggingFace](https://huggingface.co/docs/datasets/en/index), and open-source contributions from the community. We thank the authors and developers for their contribution. 

## Issues

- The preferred mode of communication is via GitHub issues.
- If GitHub issues are inappropriate, email gjaume@bwh.harvard.edu and andrewzh@mit.edu.
- Immediate response to minor issues may not be available.

## Funding
This work was funded by NIH NIGMS [R35GM138216](https://reporter.nih.gov/search/sWDcU5IfAUCabqoThQ26GQ/project-details/10029418).

## How to cite

If you find our work useful in your research or if you use parts of this code, please consider citing our papers:

```
@article{zhang2025standardizing,
  title={Accelerating Data Processing and Benchmarking of AI Models for Pathology},
  author={Zhang, Andrew and Jaume, Guillaume and Vaidya, Anurag and Ding, Tong and Mahmood, Faisal},
  journal={arXiv preprint arXiv:2502.06750},
  year={2025}
}

@article{vaidya2025molecular,
  title={Molecular-driven Foundation Model for Oncologic Pathology},
  author={Vaidya, Anurag and Zhang, Andrew and Jaume, Guillaume and Song, Andrew H and Ding, Tong and Wagner, Sophia J and Lu, Ming Y and Doucet, Paul and Robertson, Harry and Almagro-Perez, Cristina and others},
  journal={arXiv preprint arXiv:2501.16652},
  year={2025}
}

```

<img src="_readme/joint_logo.png"> 
