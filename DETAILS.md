# TRIDENT Details

This document complements `README.md`.

## Preflight checklist

Run these checks before launching long jobs:

```bash
trident-doctor --profile base
trident-doctor --profile patch-encoders --check-gated
trident-doctor --profile slide-encoders
trident-doctor --profile convert
```

Use `--profile full --check-gated` only when you need one consolidated report.

Interpretation:

- `FAIL`: missing required dependency or config. Fix before running.
- `WARN`: optional dependency missing, gated access missing, or partial environment risk.
- `PASS`: check succeeded.

## Input conventions

### WSI batch inputs (`run_batch_of_slides.py`)

- `--wsi_dir` expects a directory of slide files.
- By default, only top-level files are scanned.
- Add `--search_nested` to recursively scan subdirectories.
- Use `--custom_list_of_wsis <csv>` to process a curated subset.

`--custom_list_of_wsis` CSV format:

- Required column: `wsi` (filename with extension).
- Optional column: `mpp` (numeric microns-per-pixel).

### Converter inputs (`trident convert`)

- `--input_dir` points to files that may be converted.
- `--mpp_csv` is required and must include `wsi,mpp`.
- `wsi` is resolved relative to `--input_dir`.
- `mpp` must be numeric for each row you want converted.

Example:

```csv
wsi,mpp
case_001.czi,0.25
case_002.svs,0.50
```

## Output structure and artifacts

Given `--job_dir ./trident_processed`, common artifacts are:

- `thumbnails`: WSI thumbnails.
- `contours`: thumbnails with tissue contours.
- `contours_geojson`: editable contours for QuPath workflows.
- `<mag>x_<patch_size>px/patches`: patch coordinates.
- `<mag>x_<patch_size>px/visualization`: patch overlays for QC.
- `<mag>x_<patch_size>px/features_<patch_encoder>`: patch embeddings.
- `<mag>x_<patch_size>px/slide_features_<slide_encoder>`: slide embeddings.

For conversion runs, output TIFF files are written to the `--job_dir` given to `trident convert`.

## Quality control checklist

Validate these before downstream modeling:

- Segmentation contours align with tissue foreground.
- Artifact-heavy regions are removed when expected (`--remove_artifacts`, `--remove_penmarks`).
- Patch overlays cover tissue at intended density.
- Magnification and patch size match encoder requirements.
- Feature files exist for all expected slides.

If contours are not acceptable:

- Try `--segmenter grandqc` for H&E workflows.
- Adjust `--seg_conf_thresh`.
- Edit `contours_geojson` in QuPath and rerun downstream tasks.

## Performance and scaling patterns

### Multiprocessing with lock-based coordination

TRIDENT supports running multiple instances of the same task in parallel. Instances avoid duplicate work by using slide lockfiles.

Recommended pattern:

1. Start one task instance.
2. Start additional instances with identical arguments.
3. Monitor CPU, RAM, and storage throughput.

Notes:

- More workers are not always faster; I/O often becomes the bottleneck.
- It is safer to scale gradually (1 -> 2 -> 3 processes).

### Cache-assisted throughput

If raw slides are on slower storage, cache-first workflows can reduce end-to-end time for heavy feature extraction.

Pattern:

1. Run segmentation/coords normally.
2. Set `--wsi_cache` to a fast local path for runs that read many patches.
3. Launch feature extraction with the same cache path so repeated reads use local storage.

Safety note:

- Only enable cache clearing if you are sure `--wsi_cache` does not point to raw source data.

## Converter deep dive

### What conversion does

`trident convert` wraps `AnyToTiffConverter` and writes pyramidal `.tiff` outputs.

Core arguments:

- `--input_dir`: directory with source images/WSIs.
- `--mpp_csv`: CSV with `wsi,mpp` rows.
- `--job_dir`: output directory for converted TIFF files.
- `--downscale_by`: integer >= 1.
- `--num_workers`: `1` sequential, `0` all CPU cores.
- `--bigtiff`: enable BigTIFF output.

Examples:

```bash
trident convert --input_dir ./wsis --mpp_csv ./to_process.csv --job_dir ./pyramidal_tiff --downscale_by 1 --num_workers 1
trident convert --input_dir ./wsis --mpp_csv ./to_process.csv --job_dir ./pyramidal_tiff --downscale_by 2 --num_workers 0 --bigtiff
```

Behavior details:

- Only files listed in `mpp_csv` are attempted.
- Unsupported extensions are skipped.
- Missing files are skipped and reported.
- Embedded MPP may be detected and compared against CSV MPP; CSV value is used for writing output resolution metadata.

### Converter dependencies

Install:

```bash
pip install -e ".[convert]"
```

System libraries typically required on Linux:

```bash
sudo apt-get update
sudo apt-get install -y libvips libvips-dev libopenslide0 libopenslide-dev
```

Common optional/format-specific packages:

- `aicsimageio` for BioFormats-backed reads.
- `pylibCZIrw` for CZI reads.

### Converter troubleshooting

`ModuleNotFoundError: pyvips`

- Install `pyvips` and system `libvips`.

`cannot load library 'libvips.so'`

- Install `libvips` system packages and re-open shell/session.

`pylibCZIrw is required for CZI files`

- Install `pylibCZIrw`.

`MPP CSV must contain columns ['mpp', 'wsi']`

- Fix headers to exactly `wsi,mpp`.

`No valid conversion tasks found from CSV entries`

- Ensure filenames exist under `--input_dir`.
- Ensure listed extensions are supported.
- Ensure `mpp` values are numeric and non-empty.

## Using encoders in custom Python pipelines

### Patch encoders

```python
from trident.patch_encoder_models.load import encoder_factory

encoder = encoder_factory("uni_v1")
print(encoder.enc_name)
print(encoder.eval_transforms)
print(encoder.precision)
```

### Slide encoders

```python
from trident.slide_encoder_models.load import encoder_factory

encoder = encoder_factory("titan")
print(encoder.enc_name)
print(encoder.precision)
```

Some encoders accept kwargs:

```python
encoder = encoder_factory("conch_v1", with_proj=True)
```

