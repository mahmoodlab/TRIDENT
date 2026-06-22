import csv
import os
import tempfile
import unittest
from pathlib import Path
import hashlib

import torch

from tests._test_gating import RUN_INTEGRATION_TESTS

try:
    from huggingface_hub import hf_hub_download, HfApi

    HAS_HF_HUB = True
except Exception:
    HAS_HF_HUB = False


@unittest.skipUnless(
    RUN_INTEGRATION_TESTS and HAS_HF_HUB,
    "Set TRIDENT_RUN_INTEGRATION_TESTS=1 and install huggingface_hub to run heavy integration tests.",
)
class TestRunBatchOfSlidesIntegrationOutputs(unittest.TestCase):
    HF_REPO = "MahmoodLab/unit-testing"

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _download_two_wsis_and_csv(self, wsi_dir: Path, csv_path: Path) -> list[str]:
        svs_names = self._pick_two_svs()
        for filename in svs_names:
            hf_hub_download(
                repo_id=self.HF_REPO,
                repo_type="dataset",
                filename=filename,
                local_dir=str(wsi_dir),
            )

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["wsi"])
            writer.writeheader()
            for filename in svs_names:
                writer.writerow({"wsi": filename})
        return svs_names

    def _base_all_args(self, *, job_dir: Path, wsi_dir: Path, csv_path: Path, extra: list[str] | None = None) -> list[str]:
        argv = [
            "run_batch_of_slides.py",
            "--task",
            "all",
            "--job_dir",
            str(job_dir),
            "--wsi_dir",
            str(wsi_dir),
            "--custom_list_of_wsis",
            str(csv_path),
            "--segmenter",
            "otsu",
            "--patch_encoder",
            "uni_v1",
            "--mag",
            "1.25",
            "--patch_size",
            "512",
            "--overlap",
            "0",
            "--batch_size",
            "2",
            "--seg_batch_size",
            "2",
            "--feat_batch_size",
            "2",
            "--max_workers",
            "1",
            "--gpus",
            "-1",
            "--skip_errors",
        ]
        if extra:
            argv.extend(extra)
        return argv

    def _base_task_args(
        self,
        *,
        task: str,
        job_dir: Path,
        wsi_dir: Path,
        csv_path: Path,
        extra: list[str] | None = None,
    ) -> list[str]:
        argv = self._base_all_args(job_dir=job_dir, wsi_dir=wsi_dir, csv_path=csv_path, extra=None)
        # Replace --task all with requested task.
        i = argv.index("--task")
        argv[i + 1] = task
        if extra:
            argv.extend(extra)
        return argv

    def _pick_two_svs(self) -> list[str]:
        api = HfApi()
        files = api.list_repo_files(repo_id=self.HF_REPO, repo_type="dataset")
        svs_files = sorted([f for f in files if f.lower().endswith(".svs")])
        if len(svs_files) < 2:
            self.skipTest(f"Need at least 2 .svs in {self.HF_REPO}")

        # Require known fixtures for stable, assertable patch counts.
        required = [
            "394140.svs",
            "TCGA-AN-A0XW-01Z-00-DX1.811E11E7-FA67-46BB-9BC6-1FD0106B789D.svs",
        ]
        missing = [x for x in required if x not in svs_files]
        if missing:
            self.skipTest(f"Required integration fixtures missing from {self.HF_REPO}: {missing}")
        return list(required)

    def test_uni_v1_all_outputs_created_and_h5_content_valid(self):
        # Import here so the module-level argparse uses our argv.
        import run_batch_of_slides as rbs
        import h5py

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wsi_dir = tmp_path / "wsis"
            job_dir = tmp_path / "job"
            wsi_dir.mkdir(parents=True, exist_ok=True)
            job_dir.mkdir(parents=True, exist_ok=True)

            csv_path = tmp_path / "wsis.csv"
            svs_names = self._download_two_wsis_and_csv(wsi_dir, csv_path)

            # Force CPU and keep patch count small for test runtime:
            # - `--segmenter otsu` runs segmentation on CPU.
            # - `--gpus -1` forces worker device="cpu" for feature extraction.
            # - low magnification reduces number of extracted patches.
            # Default coords dir naming used by `run_batch_of_slides.py`.
            mag_str = f"{float(1.25):g}"
            coords_dir = f"{mag_str}x_{512}px_{0}px_overlap"
            expected = {
                "394140": {"n_patches": 8, "embed_dim": 1024},
                "TCGA-AN-A0XW-01Z-00-DX1.811E11E7-FA67-46BB-9BC6-1FD0106B789D": {"n_patches": 7, "embed_dim": 1024},
            }
            expected_first_embedding_394140 = [
                -0.20356396,
                0.63925189,
                2.49347925,
                0.68662268,
                -2.45239115,
                1.82746553,
                -0.27057070,
                -2.36906433,
                -0.77510327,
                -0.12984683,
                1.46824217,
                -3.22552824,
                -1.58553052,
                2.88736224,
                0.83875608,
                -0.92192686,
            ]
            expected_first_embedding_394140_norm = 39.72064468200738

            argv = [
                *self._base_all_args(job_dir=job_dir, wsi_dir=wsi_dir, csv_path=csv_path),
            ]

            # Make the run deterministic regardless of host CUDA presence.
            rbs.sys.argv = argv
            rbs.main()

            # 1) Segmentation outputs.
            contours_dir = job_dir / "contours"
            contours_geojson_dir = job_dir / "contours_geojson"
            self.assertTrue(contours_dir.is_dir(), "Missing `contours/` output dir")
            self.assertTrue(contours_geojson_dir.is_dir(), "Missing `contours_geojson/` output dir")

            # 2) Coords outputs.
            patches_dir = job_dir / coords_dir / "patches"
            self.assertTrue(patches_dir.is_dir(), "Missing patch coords `patches/` output dir")

            # 3) Feature outputs.
            features_dir = job_dir / coords_dir / "features_uni_v1"
            self.assertTrue(features_dir.is_dir(), "Missing patch features `features_uni_v1/` output dir")

            # Validate per-slide artifacts exist and have expected content.
            for filename in svs_names:
                stem = os.path.splitext(filename)[0]

                # Segmentation.
                self.assertTrue((contours_dir / f"{stem}.jpg").exists(), f"Missing contour jpg for {stem}")

                # Coords.
                coords_fp = patches_dir / f"{stem}_patches.h5"
                self.assertTrue(coords_fp.exists(), f"Missing coords h5 for {stem}")
                with h5py.File(coords_fp, "r") as f:
                    self.assertIn("coords", f, f"coords dataset missing for {stem}")
                    self.assertEqual(
                        tuple(f["coords"].shape),
                        (expected[stem]["n_patches"], 2),
                        f"Unexpected coords shape for {stem}",
                    )
                    self.assertEqual(int(f["coords"].attrs["patch_size"]), 512)
                    self.assertEqual(int(f["coords"].attrs.get("overlap", 0)), 0)

                # Features.
                feats_fp = features_dir / f"{stem}.h5"
                self.assertTrue(feats_fp.exists(), f"Missing features h5 for {stem}")
                with h5py.File(feats_fp, "r") as f:
                    self.assertIn("features", f, f"features dataset missing for {stem}")
                    feats = f["features"]
                    # We expect either some features, or empty if segmentation/coords yields none.
                    self.assertEqual(
                        tuple(feats.shape),
                        (expected[stem]["n_patches"], expected[stem]["embed_dim"]),
                        f"Unexpected features shape for {stem}",
                    )
                    self.assertEqual(feats.attrs.get("encoder"), "uni_v1")
                    self.assertEqual(feats.attrs.get("name"), stem)
                    if stem == "394140":
                        import numpy as np

                        row0 = feats[0, :].astype(np.float64)
                        self.assertTrue(
                            np.allclose(
                                row0[:16],
                                np.array(expected_first_embedding_394140, dtype=np.float64),
                                rtol=1e-4,
                                atol=1e-4,
                            ),
                            "UNI v1 first embedding (394140) drifted: first 16 dims mismatch.",
                        )
                        self.assertTrue(
                            np.isclose(
                                float(np.linalg.norm(row0)),
                                float(expected_first_embedding_394140_norm),
                                rtol=1e-4,
                                atol=1e-4,
                            ),
                            "UNI v1 first embedding (394140) drifted: L2 norm mismatch.",
                        )

            # Final sanity: should have created at least one feature file.
            created = [p for p in features_dir.iterdir() if p.suffix == ".h5"]
            self.assertEqual(len(created), 2, "Expected exactly 2 feature files for the 2 WSIs")

    def test_otsu_penmarks_5x_uni_all_outputs(self):
        """
        Single `--task all` invocation that chains, in one run:
          * otsu tissue segmentation        (`--segmenter otsu`)
          * GrandQC penmark detection        (`--remove_penmarks`)
          * 256x256 patching at 5x           (`--mag 5 --patch_size 256`)
          * UNI feature extraction           (`--patch_encoder uni_v1`)

        Runs over a mix of single-file `.svs` slides and a multi-file `.mrxs`
        (MIRAX) slide to exercise both reader paths through the batch pipeline.
        MRXS slides are laid out exactly like every other format: the `.mrxs`
        index file(s) sit directly in `wsi_dir`, each alongside its same-named
        data folder, and the custom list references them by basename. This keeps
        slide discovery identical to single-file formats.

        `--remove_penmarks` is used rather than `--remove_artifacts`: the full
        artifact remover is aggressive enough to discard all of CMU-1's tissue,
        whereas penmark-only removal preserves it.

        Asserts that segmentation, coords, and feature outputs are all produced
        by the single run and are mutually consistent (one feature row per patch,
        UNI's 1024-d embedding).
        """
        import csv as csv_mod
        import shutil
        import run_batch_of_slides as rbs
        import h5py
        from huggingface_hub import snapshot_download

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wsi_dir = tmp_path / "wsis"
            job_dir = tmp_path / "job"
            wsi_dir.mkdir(parents=True, exist_ok=True)
            job_dir.mkdir(parents=True, exist_ok=True)

            # Single-file SVS fixtures: dropped directly into wsi_dir.
            svs_names = self._pick_two_svs()
            for filename in svs_names:
                hf_hub_download(
                    repo_id=self.HF_REPO,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=str(wsi_dir),
                )

            # Multi-file MRXS fixture. The repo stores it nested as
            # `CMU-1/CMU-1.mrxs` + `CMU-1/CMU-1/<data>`; flatten it into the
            # standard MIRAX layout so it sits in wsi_dir like the SVS files:
            #   wsi_dir/CMU-1.mrxs          (index file)
            #   wsi_dir/CMU-1/<data files>  (same-named sibling data folder)
            staging = snapshot_download(
                repo_id=self.HF_REPO,
                repo_type="dataset",
                allow_patterns=["CMU-1/*"],
            )
            shutil.copy(os.path.join(staging, "CMU-1", "CMU-1.mrxs"), wsi_dir / "CMU-1.mrxs")
            shutil.copytree(os.path.join(staging, "CMU-1", "CMU-1"), wsi_dir / "CMU-1")
            self.assertTrue((wsi_dir / "CMU-1.mrxs").is_file(), "MRXS index file not laid out in wsi_dir.")
            self.assertTrue((wsi_dir / "CMU-1").is_dir(), "MRXS data folder not laid out alongside index.")

            # Custom list: 'wsi' column holds basenames relative to wsi_dir,
            # identical handling for single-file and multi-file formats.
            slide_names = [*svs_names, "CMU-1.mrxs"]
            csv_path = tmp_path / "wsis.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv_mod.DictWriter(f, fieldnames=["wsi"])
                writer.writeheader()
                for name in slide_names:
                    writer.writerow({"wsi": name})

            # Output stems are basenames without extension (e.g. "CMU-1").
            expected_stems = [os.path.splitext(os.path.basename(p))[0] for p in slide_names]

            mag, patch_size, overlap = 5, 256, 0
            # Use GPU when available (GrandQC + UNI are deep models); fall back to
            # CPU so the test stays runnable on CPU-only hosts.
            gpu_arg = "0" if torch.cuda.is_available() else "-1"
            argv = [
                "run_batch_of_slides.py",
                "--task", "all",
                "--job_dir", str(job_dir),
                "--wsi_dir", str(wsi_dir),
                "--custom_list_of_wsis", str(csv_path),
                "--wsi_ext", ".svs", ".mrxs",
                "--segmenter", "otsu",
                "--remove_penmarks",
                "--patch_encoder", "uni_v1",
                "--mag", str(mag),
                "--patch_size", str(patch_size),
                "--overlap", str(overlap),
                "--batch_size", "2",
                "--seg_batch_size", "2",
                "--feat_batch_size", "2",
                "--max_workers", "1",
                "--gpus", gpu_arg,
                "--skip_errors",
            ]
            rbs.sys.argv = argv
            rbs.main()

            mag_str = f"{float(mag):g}"
            coords_dir = f"{mag_str}x_{patch_size}px_{overlap}px_overlap"

            # 1) Segmentation outputs (otsu + grandqc penmark pass).
            contours_dir = job_dir / "contours"
            contours_geojson_dir = job_dir / "contours_geojson"
            self.assertTrue(contours_dir.is_dir(), "Missing `contours/` output dir")
            self.assertTrue(contours_geojson_dir.is_dir(), "Missing `contours_geojson/` output dir")

            # 2) Coords outputs (256px @ 5x).
            patches_dir = job_dir / coords_dir / "patches"
            self.assertTrue(patches_dir.is_dir(), "Missing patch coords `patches/` output dir")

            # 3) Feature outputs (UNI v1).
            features_dir = job_dir / coords_dir / "features_uni_v1"
            self.assertTrue(features_dir.is_dir(), "Missing patch features `features_uni_v1/` output dir")

            for stem in expected_stems:
                # Segmentation artifacts.
                self.assertTrue((contours_dir / f"{stem}.jpg").exists(), f"Missing contour jpg for {stem}")
                self.assertTrue(
                    (contours_geojson_dir / f"{stem}.geojson").exists(),
                    f"Missing contour geojson for {stem}",
                )

                # Coords: derive patch count dynamically (depends on tissue after
                # artifact removal) rather than hard-coding a magic number.
                coords_fp = patches_dir / f"{stem}_patches.h5"
                self.assertTrue(coords_fp.exists(), f"Missing coords h5 for {stem}")
                with h5py.File(coords_fp, "r") as f:
                    self.assertIn("coords", f, f"coords dataset missing for {stem}")
                    n_patches = int(f["coords"].shape[0])
                    self.assertEqual(int(f["coords"].shape[1]), 2, f"coords should be (N, 2) for {stem}")
                    self.assertGreater(n_patches, 0, f"Expected >0 patches for {stem}")
                    self.assertEqual(int(f["coords"].attrs["patch_size"]), patch_size)
                    self.assertEqual(int(f["coords"].attrs.get("overlap", 0)), overlap)

                # Features: one 1024-d UNI row per patch coordinate.
                feats_fp = features_dir / f"{stem}.h5"
                self.assertTrue(feats_fp.exists(), f"Missing features h5 for {stem}")
                with h5py.File(feats_fp, "r") as f:
                    self.assertIn("features", f, f"features dataset missing for {stem}")
                    feats = f["features"]
                    self.assertEqual(
                        int(feats.shape[0]),
                        n_patches,
                        f"features/coords count mismatch for {stem}",
                    )
                    self.assertEqual(int(feats.shape[1]), 1024, "UNI v1 embedding dim should be 1024")
                    self.assertEqual(f["features"].attrs.get("encoder"), "uni_v1")
                    self.assertEqual(f["features"].attrs.get("name"), stem)

            created = [p for p in features_dir.iterdir() if p.suffix == ".h5"]
            self.assertEqual(len(created), len(expected_stems), "Expected one feature file per WSI")

    def test_idempotent_rerun_outputs_unchanged(self):
        import run_batch_of_slides as rbs

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wsi_dir = tmp_path / "wsis"
            job_dir = tmp_path / "job"
            wsi_dir.mkdir(parents=True, exist_ok=True)
            job_dir.mkdir(parents=True, exist_ok=True)

            csv_path = tmp_path / "wsis.csv"
            svs_names = self._download_two_wsis_and_csv(wsi_dir, csv_path)

            argv = self._base_all_args(job_dir=job_dir, wsi_dir=wsi_dir, csv_path=csv_path)
            rbs.sys.argv = argv
            rbs.main()

            coords_dir = "1.25x_512px_0px_overlap"
            patches_dir = job_dir / coords_dir / "patches"
            features_dir = job_dir / coords_dir / "features_uni_v1"
            contours_dir = job_dir / "contours"

            tracked: list[Path] = []
            for filename in svs_names:
                stem = os.path.splitext(filename)[0]
                tracked.extend(
                    [
                        contours_dir / f"{stem}.jpg",
                        patches_dir / f"{stem}_patches.h5",
                        features_dir / f"{stem}.h5",
                    ]
                )

            for p in tracked:
                self.assertTrue(p.exists(), f"Missing expected output after first run: {p}")

            hashes_before = {str(p.relative_to(job_dir)): self._sha256_file(p) for p in tracked}

            # Re-run: should skip already-computed work and not mutate outputs.
            rbs.sys.argv = argv
            rbs.main()

            hashes_after = {str(p.relative_to(job_dir)): self._sha256_file(p) for p in tracked}
            self.assertEqual(hashes_before, hashes_after, "Re-run mutated outputs; expected idempotent behavior.")

    def test_all_equals_seg_then_coords_then_feat(self):
        """
        Running seg -> coords -> feat sequentially should yield the same final
        on-disk artifacts as running `--task all` once.
        """
        import run_batch_of_slides as rbs

        def collect_content_hashes(job_dir: Path) -> dict[str, str]:
            """
            Compare stable *content* rather than raw file bytes:
            - HDF5 files embed run-specific absolute paths in attrs, which differ across job dirs.
            """
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

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wsi_dir = tmp_path / "wsis"
            wsi_dir.mkdir(parents=True, exist_ok=True)
            csv_path = tmp_path / "wsis.csv"
            self._download_two_wsis_and_csv(wsi_dir, csv_path)

            # Baseline: --task all
            job_all = tmp_path / "job_all"
            job_all.mkdir(parents=True, exist_ok=True)
            rbs.sys.argv = self._base_all_args(job_dir=job_all, wsi_dir=wsi_dir, csv_path=csv_path)
            rbs.main()
            hashes_all = collect_content_hashes(job_all)

            # Sequential: seg -> coords -> feat
            job_seq = tmp_path / "job_seq"
            job_seq.mkdir(parents=True, exist_ok=True)
            for task in ["seg", "coords", "feat"]:
                rbs.sys.argv = self._base_task_args(task=task, job_dir=job_seq, wsi_dir=wsi_dir, csv_path=csv_path)
                rbs.main()
            hashes_seq = collect_content_hashes(job_seq)

            self.assertEqual(hashes_all, hashes_seq, "Sequential seg/coords/feat differed from --task all outputs.")

    def test_single_worker_equals_two_workers_cpu(self):
        """
        Real-data equivalence: splitting work across two CPU workers should not
        change the final coords/features content.
        """
        import run_batch_of_slides as rbs
        import h5py
        import numpy as np

        def collect_content_hashes(job_dir: Path) -> dict[str, str]:
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

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wsi_dir = tmp_path / "wsis"
            wsi_dir.mkdir(parents=True, exist_ok=True)
            csv_path = tmp_path / "wsis.csv"
            self._download_two_wsis_and_csv(wsi_dir, csv_path)

            # Single worker (CPU).
            job_single = tmp_path / "job_single"
            job_single.mkdir(parents=True, exist_ok=True)
            rbs.sys.argv = self._base_all_args(job_dir=job_single, wsi_dir=wsi_dir, csv_path=csv_path, extra=["--gpus", "-1"])
            rbs.main()
            h_single = collect_content_hashes(job_single)

            # Two workers on CPU (two shards).
            job_two = tmp_path / "job_two"
            job_two.mkdir(parents=True, exist_ok=True)
            rbs.sys.argv = self._base_all_args(job_dir=job_two, wsi_dir=wsi_dir, csv_path=csv_path, extra=["--gpus", "-1", "-1"])
            rbs.main()
            h_two = collect_content_hashes(job_two)

            self.assertEqual(h_single, h_two, "Single-worker outputs differ from two-worker sharded outputs.")

    def test_coords_are_deterministic_across_runs(self):
        """
        Patch coordinate generation should be stable given fixed inputs/config.
        """
        import run_batch_of_slides as rbs
        import h5py
        import numpy as np

        coords_dir = "1.25x_512px_0px_overlap"

        def read_coords(job_dir: Path) -> dict[str, np.ndarray]:
            patches_dir = job_dir / coords_dir / "patches"
            out: dict[str, np.ndarray] = {}
            for fp in sorted(patches_dir.glob("*_patches.h5")):
                stem = fp.name.replace("_patches.h5", "")
                with h5py.File(fp, "r") as f:
                    out[stem] = f["coords"][:]
            self.assertTrue(out, "No coords files found.")
            return out

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wsi_dir = tmp_path / "wsis"
            wsi_dir.mkdir(parents=True, exist_ok=True)
            csv_path = tmp_path / "wsis.csv"
            self._download_two_wsis_and_csv(wsi_dir, csv_path)

            job_a = tmp_path / "job_a"
            job_a.mkdir(parents=True, exist_ok=True)
            rbs.sys.argv = self._base_all_args(job_dir=job_a, wsi_dir=wsi_dir, csv_path=csv_path)
            rbs.main()
            coords_a = read_coords(job_a)

            job_b = tmp_path / "job_b"
            job_b.mkdir(parents=True, exist_ok=True)
            rbs.sys.argv = self._base_all_args(job_dir=job_b, wsi_dir=wsi_dir, csv_path=csv_path)
            rbs.main()
            coords_b = read_coords(job_b)

            self.assertEqual(set(coords_a.keys()), set(coords_b.keys()))
            for k in coords_a:
                self.assertTrue(
                    np.array_equal(coords_a[k], coords_b[k]),
                    f"Coords drifted across runs for slide '{k}'.",
                )

    def test_cache_mode_outputs_match_non_cache_mode(self):
        """
        The producer/consumer cache pipeline (`--wsi_cache`) should not change
        final coords/features content.
        """
        import run_batch_of_slides as rbs
        import h5py
        import numpy as np

        coords_dir = "1.25x_512px_0px_overlap"

        def collect_content_hashes(job_dir: Path) -> dict[str, str]:
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

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wsi_dir = tmp_path / "wsis"
            wsi_dir.mkdir(parents=True, exist_ok=True)
            csv_path = tmp_path / "wsis.csv"
            self._download_two_wsis_and_csv(wsi_dir, csv_path)

            # Non-cache baseline.
            job_plain = tmp_path / "job_plain"
            job_plain.mkdir(parents=True, exist_ok=True)
            rbs.sys.argv = self._base_all_args(job_dir=job_plain, wsi_dir=wsi_dir, csv_path=csv_path)
            rbs.main()
            h_plain = collect_content_hashes(job_plain)

            # Cache mode.
            job_cache = tmp_path / "job_cache"
            job_cache.mkdir(parents=True, exist_ok=True)
            cache_dir = tmp_path / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            rbs.sys.argv = self._base_all_args(
                job_dir=job_cache,
                wsi_dir=wsi_dir,
                csv_path=csv_path,
                extra=["--wsi_cache", str(cache_dir), "--cache_batch_size", "2"],
            )
            rbs.main()
            h_cache = collect_content_hashes(job_cache)

            self.assertEqual(h_plain, h_cache, "Cache-mode outputs differ from non-cache outputs.")

    def test_dump_patches_writes_expected_number_of_images(self):
        """
        `--dump_patches --dump_patches_max N` should materialize exactly N patch images per slide.
        """
        import run_batch_of_slides as rbs

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wsi_dir = tmp_path / "wsis"
            job_dir = tmp_path / "job"
            wsi_dir.mkdir(parents=True, exist_ok=True)
            job_dir.mkdir(parents=True, exist_ok=True)

            csv_path = tmp_path / "wsis.csv"
            svs_names = self._download_two_wsis_and_csv(wsi_dir, csv_path)

            # Coords depends on segmentation outputs (geojson). Run seg first, then coords with dump flags.
            rbs.sys.argv = self._base_task_args(task="seg", job_dir=job_dir, wsi_dir=wsi_dir, csv_path=csv_path)
            rbs.main()

            rbs.sys.argv = self._base_task_args(
                task="coords",
                job_dir=job_dir,
                wsi_dir=wsi_dir,
                csv_path=csv_path,
                extra=[
                    "--dump_patches",
                    "--dump_patches_max",
                    "3",
                    "--dump_patches_format",
                    "png",
                ],
            )
            rbs.main()

            coords_dir = "1.25x_512px_0px_overlap"
            patch_images_root = job_dir / coords_dir / "patch_images"
            self.assertTrue(patch_images_root.is_dir(), "Missing patch_images/ directory after --dump_patches.")

            for filename in svs_names:
                stem = os.path.splitext(filename)[0]
                slide_dir = patch_images_root / stem
                self.assertTrue(slide_dir.is_dir(), f"Missing dumped patch image directory for {stem}")
                images = sorted([p for p in slide_dir.iterdir() if p.suffix.lower() == ".png"])
                self.assertEqual(
                    len(images),
                    3,
                    f"Expected exactly 3 dumped patch images for {stem}, got {len(images)}",
                )
                for p in images:
                    self.assertGreater(p.stat().st_size, 0, f"Dumped patch image is empty: {p}")

