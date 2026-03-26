# Tests

This folder contains two test tiers:

- **Fast unit tests**: deterministic, no network, no large model downloads.
- **Heavy integration tests**: real data/model paths, optional network/GPU, slower.

## 1) Default (fast) test run

Runs only fast unit tests by default.

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## 2) Run heavy integration tests

Enable with:

```bash
TRIDENT_RUN_INTEGRATION_TESTS=1 python -m unittest discover -s tests -p "test_*.py" -v
```

### Requirements for integration tests

- `huggingface_hub` installed.
- Internet access to download test assets/models.
- Optional dependencies depending on the test module:
  - `cv2` / `opencv-python`
  - `matplotlib`
  - `geopandas`
  - `shapely`

If these are missing, relevant heavy tests are skipped.

## 3) Run GPU-only integration tests

Some integration tests require CUDA and are additionally gated.

```bash
TRIDENT_RUN_INTEGRATION_TESTS=1 TRIDENT_RUN_GPU_TESTS=1 python -m unittest discover -s tests -p "test_*.py" -v
```

### Requirements for GPU tests

- CUDA-capable GPU.
- PyTorch with CUDA support.
- GPU memory sufficient for selected models.

## 4) Run hardware-dependent benchmarks (opt-in)

Some tests under `tests/benchmarks/` are **benchmarks**:

- They can take minutes to run.
- They are hardware-dependent (disk type, OS cache behavior, filesystem, backend).
- They may download real WSIs for reproducibility.

Enable benchmarks with:

```bash
TRIDENT_RUN_INTEGRATION_TESTS=1 TRIDENT_RUN_BENCHMARKS=1 python -m unittest discover -s tests/benchmarks -p "test_*.py" -v
```

### Patch scan order benchmark (row-major vs col-major)

This benchmark compares patch extraction speed for `WSIPatcher(scan_order="row-major")` vs
`WSIPatcher(scan_order="col-major")` on a real `CMU-1.tiff` slide downloaded from
`MahmoodLab/unit-testing` on Hugging Face.

Example run (recommended defaults):

```bash
TRIDENT_RUN_INTEGRATION_TESTS=1 \
TRIDENT_RUN_BENCHMARKS=1 \
TRIDENT_BENCHMARK_MAX_PATCHES=20000 \
TRIDENT_BENCHMARK_REPEATS=5 \
TRIDENT_BENCHMARK_ALTERNATE=1 \
python -m unittest tests.benchmarks.test_patch_scan_order_benchmark -v
```

Optional knobs:

- `TRIDENT_BENCHMARK_MAG` (default: `20`)
- `TRIDENT_BENCHMARK_PATCH_SIZE` (default: `256`)
- `TRIDENT_BENCHMARK_OVERLAP` (default: `0`)
- `TRIDENT_BENCHMARK_MAX_PATCHES` (default: `20000`)
- `TRIDENT_BENCHMARK_REPEATS` (default: `5`)
- `TRIDENT_BENCHMARK_ALTERNATE` (default: `1`, alternate orders between rounds)
- `TRIDENT_BENCHMARK_READER` (default: auto; e.g. `openslide`)

## Notes

- Heavy tests are intentionally skipped in normal local/CI quick runs.
- Skip messages in unittest output tell you which flag/dependency is missing.
- Keep fast tests green first; run heavy tests before releases or major model/pipeline changes.
