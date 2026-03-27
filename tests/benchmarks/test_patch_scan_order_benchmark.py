from __future__ import annotations

import os
import unittest
import time
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Literal, Optional

import numpy as np

from trident.wsi_objects.WSIFactory import load_wsi
from trident.wsi_objects.WSIPatcher import WSIPatcher

try:
    from huggingface_hub import snapshot_download
    HAS_HF_HUB = True
except Exception:
    snapshot_download = None
    HAS_HF_HUB = False

from tests._test_gating import RUN_INTEGRATION_TESTS


ScanOrder = Literal["row-major", "col-major"]


@dataclass
class RunResult:
    scan_order: ScanOrder
    seconds: float
    patches: int
    patches_per_sec: float


def _iter_patches(
    *,
    slide_path: str,
    mag: int,
    patch_size: int,
    overlap: int,
    scan_order: ScanOrder,
    max_patches: Optional[int],
    reader_type: Optional[str],
) -> RunResult:
    # Opening/closing per run makes runs comparable.
    with load_wsi(slide_path=slide_path, reader_type=reader_type, lazy_init=False) as wsi:
        patcher = WSIPatcher(
            wsi=wsi,
            patch_size=patch_size,
            src_mag=wsi.mag,
            dst_mag=mag,
            overlap=overlap,
            coords_only=False,
            pil=False,
            scan_order=scan_order,
        )

        n = len(patcher)
        if max_patches is not None:
            n = min(n, max_patches)

        # Warmup: touch first patch to trigger any one-time init in backends.
        if n > 0:
            _ = patcher[0]

        t0 = time.perf_counter()

        # Force actual reads by touching pixel data.
        acc = 0
        for i in range(n):
            tile, _x, _y = patcher[i]
            acc += int(np.asarray(tile, dtype=np.uint8).sum())

        t1 = time.perf_counter()
        _ = acc

        seconds = t1 - t0
        pps = float("inf") if seconds == 0 else (n / seconds)
        return RunResult(scan_order=scan_order, seconds=seconds, patches=n, patches_per_sec=pps)


def _summarize(results: list[RunResult], order: ScanOrder) -> str:
    xs = [r.patches_per_sec for r in results if r.scan_order == order]
    ts = [r.seconds for r in results if r.scan_order == order]
    if not xs:
        return f"{order}: no runs"
    mu = mean(xs)
    sd = stdev(xs) if len(xs) >= 2 else 0.0
    tmu = mean(ts)
    return (
        f"{order}: patches/s mean={mu:.2f} stdev={sd:.2f} (n={len(xs)}), "
        f"seconds mean={tmu:.2f}"
    )


@unittest.skipUnless(
    RUN_INTEGRATION_TESTS and HAS_HF_HUB and os.environ.get("TRIDENT_RUN_BENCHMARKS") == "1",
    "Set TRIDENT_RUN_INTEGRATION_TESTS=1, TRIDENT_RUN_BENCHMARKS=1 and install huggingface_hub to run benchmarks.",
)
class TestPatchScanOrderBenchmark(unittest.TestCase):
    """
    Hardware-dependent benchmark on a real WSI downloaded from Hugging Face.

    This uses the same HF dataset repo as `tests/test_processor.py`.
    """

    HF_REPO = "MahmoodLab/unit-testing"
    OUT_DIR = os.path.join("test_benchmark_output", "wsis")
    TEST_SLIDE_NAME = "CMU-1.tiff"

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.OUT_DIR, exist_ok=True)
        cls.local_wsi_dir = snapshot_download(
            repo_id=cls.HF_REPO,
            repo_type="dataset",
            local_dir=cls.OUT_DIR,
            allow_patterns=[cls.TEST_SLIDE_NAME],
        )
        cls.slide_path = os.path.join(cls.local_wsi_dir, cls.TEST_SLIDE_NAME)
        if not os.path.exists(cls.slide_path):
            raise RuntimeError(
                f"Expected benchmark slide '{cls.TEST_SLIDE_NAME}' not found in {cls.local_wsi_dir}"
            )

    def test_benchmark(self):
        slide_path = self.slide_path

        mag = int(os.environ.get("TRIDENT_BENCHMARK_MAG", "20"))
        patch_size = int(os.environ.get("TRIDENT_BENCHMARK_PATCH_SIZE", "256"))
        overlap = int(os.environ.get("TRIDENT_BENCHMARK_OVERLAP", "0"))
        max_patches_env = os.environ.get("TRIDENT_BENCHMARK_MAX_PATCHES")
        max_patches = int(max_patches_env) if max_patches_env else 20000
        repeats = int(os.environ.get("TRIDENT_BENCHMARK_REPEATS", "5"))
        reader = os.environ.get("TRIDENT_BENCHMARK_READER") or None
        alternate = os.environ.get("TRIDENT_BENCHMARK_ALTERNATE", "1") == "1"

        schedule: list[ScanOrder] = []
        if alternate:
            for _ in range(repeats):
                schedule.extend(["row-major", "col-major"])
        else:
            schedule = ["row-major"] * repeats + ["col-major"] * repeats

        results: list[RunResult] = []

        print(f"Slide: {os.path.abspath(slide_path)}")
        print(f"mag={mag} patch_size={patch_size} overlap={overlap} max_patches={max_patches}")
        print(f"repeats={repeats} reader={reader} alternate={alternate}")
        print()

        for idx, order in enumerate(schedule, start=1):
            r = _iter_patches(
                slide_path=slide_path,
                mag=mag,
                patch_size=patch_size,
                overlap=overlap,
                scan_order=order,
                max_patches=max_patches,
                reader_type=reader,
            )
            results.append(r)
            print(
                f"[{idx:02d}/{len(schedule):02d}] {order:9s}  "
                f"{r.patches:7d} patches  {r.seconds:8.2f}s  {r.patches_per_sec:8.2f} patches/s"
            )

        print("\n" + _summarize(results, "row-major"))
        print(_summarize(results, "col-major"))

        row = mean([r.patches_per_sec for r in results if r.scan_order == "row-major"])
        col = mean([r.patches_per_sec for r in results if r.scan_order == "col-major"])
        if col > 0:
            print(f"\nSpeedup (row-major / col-major): {row / col:.2f}x")


