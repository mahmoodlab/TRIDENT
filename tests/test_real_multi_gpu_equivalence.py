"""
Real multi-GPU stress test for `run_batch_of_slides`.

Runs the actual pipeline twice on real WSIs and compares scientific outputs
(coords + features) byte-content:
    - single GPU  : --gpus 0
    - two GPUs    : --gpus 0 1

This exercises the real `multiprocessing` spawn path used in production, not
the in-process inline mock used by `test_multi_gpu_equivalence_patch_encoders.py`.

Gated by:
- TRIDENT_RUN_INTEGRATION_TESTS=1
- TRIDENT_RUN_GPU_TESTS=1
- At least 2 visible CUDA devices
"""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import torch

from tests._test_gating import RUN_INTEGRATION_TESTS

try:
    from huggingface_hub import HfApi, hf_hub_download

    HAS_HF_HUB = True
except Exception:
    HAS_HF_HUB = False


_RUN_GPU = os.environ.get("TRIDENT_RUN_GPU_TESTS") == "1"
_HAS_TWO_GPUS = torch.cuda.is_available() and torch.cuda.device_count() >= 2


@unittest.skipUnless(
    RUN_INTEGRATION_TESTS and HAS_HF_HUB and _RUN_GPU and _HAS_TWO_GPUS,
    "Set TRIDENT_RUN_INTEGRATION_TESTS=1 and TRIDENT_RUN_GPU_TESTS=1 with >=2 CUDA GPUs.",
)
class TestRealMultiGPUEquivalence(unittest.TestCase):
    HF_REPO = "MahmoodLab/unit-testing"

    def _download_two_wsis_and_csv(self, wsi_dir: Path, csv_path: Path) -> list[str]:
        api = HfApi()
        files = api.list_repo_files(repo_id=self.HF_REPO, repo_type="dataset")
        svs_files = sorted(f for f in files if f.lower().endswith(".svs"))
        required = [
            "394140.svs",
            "TCGA-AN-A0XW-01Z-00-DX1.811E11E7-FA67-46BB-9BC6-1FD0106B789D.svs",
        ]
        missing = [x for x in required if x not in svs_files]
        if missing:
            self.skipTest(f"Required integration fixtures missing from {self.HF_REPO}: {missing}")

        for filename in required:
            hf_hub_download(
                repo_id=self.HF_REPO,
                repo_type="dataset",
                filename=filename,
                local_dir=str(wsi_dir),
            )

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["wsi"])
            writer.writeheader()
            for filename in required:
                writer.writerow({"wsi": filename})
        return required

    def _argv(self, *, job_dir: Path, wsi_dir: Path, csv_path: Path, gpus: list[int]) -> list[str]:
        return [
            "run_batch_of_slides.py",
            "--task", "all",
            "--job_dir", str(job_dir),
            "--wsi_dir", str(wsi_dir),
            "--custom_list_of_wsis", str(csv_path),
            "--segmenter", "otsu",
            "--patch_encoder", "uni_v1",
            "--mag", "1.25",
            "--patch_size", "512",
            "--overlap", "0",
            "--batch_size", "2",
            "--seg_batch_size", "2",
            "--feat_batch_size", "2",
            "--max_workers", "1",
            "--skip_errors",
            "--gpus", *[str(x) for x in gpus],
        ]

    def _collect_content_hashes(self, job_dir: Path) -> dict[str, str]:
        import h5py
        import numpy as np

        coords_dir = "1.25x_512px_0px_overlap"
        out: dict[str, str] = {}

        for coords_fp in sorted((job_dir / coords_dir / "patches").glob("*_patches.h5")):
            with h5py.File(coords_fp, "r") as f:
                coords = f["coords"][:]
                patch_size = int(f["coords"].attrs.get("patch_size", -1))
                overlap = int(f["coords"].attrs.get("overlap", -1))
            h = hashlib.sha256()
            h.update(np.ascontiguousarray(coords).tobytes())
            h.update(str(patch_size).encode("utf-8"))
            h.update(str(overlap).encode("utf-8"))
            out[str(coords_fp.relative_to(job_dir))] = h.hexdigest()

        for feats_fp in sorted((job_dir / coords_dir / "features_uni_v1").glob("*.h5")):
            with h5py.File(feats_fp, "r") as f:
                feats = f["features"][:]
                encoder = f["features"].attrs.get("encoder", "")
                name = f["features"].attrs.get("name", "")
            h = hashlib.sha256()
            h.update(np.ascontiguousarray(feats).tobytes())
            h.update(str(encoder).encode("utf-8"))
            h.update(str(name).encode("utf-8"))
            out[str(feats_fp.relative_to(job_dir))] = h.hexdigest()

        self.assertTrue(out, "No key outputs found to compare.")
        return out

    def test_single_gpu_equals_two_gpu_outputs_uni_v1(self):
        """
        Running with `--gpus 0` and `--gpus 0 1` should produce content-identical
        coords and features (the multi-GPU path only shards work, it does not
        change the math).
        """
        import run_batch_of_slides as rbs

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wsi_dir = tmp_path / "wsis"
            wsi_dir.mkdir(parents=True, exist_ok=True)
            csv_path = tmp_path / "wsis.csv"
            self._download_two_wsis_and_csv(wsi_dir, csv_path)

            job_single = tmp_path / "job_single"
            job_single.mkdir(parents=True, exist_ok=True)
            rbs.sys.argv = self._argv(
                job_dir=job_single, wsi_dir=wsi_dir, csv_path=csv_path, gpus=[0]
            )
            rbs.main()
            h_single = self._collect_content_hashes(job_single)

            job_multi = tmp_path / "job_multi"
            job_multi.mkdir(parents=True, exist_ok=True)
            rbs.sys.argv = self._argv(
                job_dir=job_multi, wsi_dir=wsi_dir, csv_path=csv_path, gpus=[0, 1]
            )
            rbs.main()
            h_multi = self._collect_content_hashes(job_multi)

            self.assertEqual(
                h_single,
                h_multi,
                "Single-GPU outputs differ from multi-GPU sharded outputs.",
            )


if __name__ == "__main__":
    unittest.main()
