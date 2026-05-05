import os
import tempfile
import unittest

import numpy as np

from tests._test_gating import RUN_INTEGRATION_TESTS


class _TinyPatchEncoder:
    """
    Small CPU-only patch encoder for integration testing.

    Matches the minimal interface expected by `WSI.extract_patch_features`:
    - `.enc_name`
    - `.eval_transforms`
    - `.precision`
    - `.embedding_dim`
    - callable on a tensor batch
    """

    def __init__(self):
        import torch
        from torchvision import transforms as T

        self.enc_name = "tiny_test_encoder"
        self.embedding_dim = 8
        self.precision = torch.float32
        self.eval_transforms = T.Compose([T.ToTensor()])

        self._torch = torch

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, imgs):
        # imgs: (B, C, H, W) float tensor in [0,1]
        torch = self._torch
        if imgs.ndim != 4:
            raise ValueError(f"expected imgs to be 4D, got {imgs.shape}")
        b, c, _, _ = imgs.shape
        mean = imgs.mean(dim=(2, 3))  # (B, C)
        if c < 3:
            mean = torch.nn.functional.pad(mean, (0, 3 - c))
        mean3 = mean[:, :3]
        # Expand deterministically to 8 dims.
        feats = torch.cat(
            [
                mean3,
                mean3.mean(dim=1, keepdim=True),
                mean3.max(dim=1, keepdim=True).values,
                mean3.min(dim=1, keepdim=True).values,
                torch.ones((b, 1), dtype=imgs.dtype, device=imgs.device),
                torch.zeros((b, 1), dtype=imgs.dtype, device=imgs.device),
            ],
            dim=1,
        )
        if feats.shape[1] != self.embedding_dim:
            raise AssertionError(f"unexpected embedding dim {feats.shape[1]}")
        return feats


class TestCZIHFIntegrationFeatureExtraction(unittest.TestCase):
    def test_download_and_extract_patch_features_from_czi(self):
        if not RUN_INTEGRATION_TESTS:
            self.skipTest("Set TRIDENT_RUN_INTEGRATION_TESTS=1 to run heavy integration tests.")

        try:
            import pylibCZIrw  # noqa: F401
        except Exception:
            self.skipTest("pylibCZIrw not installed")

        try:
            from huggingface_hub import hf_hub_download, HfApi
        except Exception:
            self.skipTest("huggingface_hub not installed")

        from trident import load_wsi
        from trident.IO import coords_to_h5

        repo_id = "MahmoodLab/unit-testing"
        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        czi_files = [f for f in files if f.lower().endswith(".czi")]
        if not czi_files:
            self.skipTest(f"No .czi files found in {repo_id}")

        # Prefer the known filename if present.
        preferred = "239_GS_PD_P1.czi"
        filename = preferred if preferred in czi_files else czi_files[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            czi_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=tmpdir,
            )

            with load_wsi(slide_path=czi_path, reader_type="czi", lazy_init=False) as wsi:
                w, h = wsi.get_dimensions()
                self.assertGreater(w, 0)
                self.assertGreater(h, 0)

                # Build a small set of coordinates safely inside bounds.
                patch_size = 256
                coords = [
                    [0, 0],
                    [min(patch_size, w - patch_size), 0],
                    [0, min(patch_size, h - patch_size)],
                    [min(patch_size, w - patch_size), min(patch_size, h - patch_size)],
                    [max(0, w // 2 - patch_size // 2), max(0, h // 2 - patch_size // 2)],
                ]
                coords = [[int(x), int(y)] for x, y in coords]

                coords_dir = os.path.join(tmpdir, "coords")
                os.makedirs(os.path.join(coords_dir, "patches"), exist_ok=True)
                coords_fp = os.path.join(coords_dir, "patches", f"{wsi.name}_patches.h5")

                src_mag = int(round(float(wsi.mag))) if wsi.mag is not None else 20
                target_mag = 20
                coords_to_h5(
                    coords=coords,
                    save_path=coords_fp,
                    patch_size=patch_size,
                    src_mag=src_mag,
                    target_mag=target_mag,
                    save_coords=coords_dir,
                    width=w,
                    height=h,
                    name=wsi.name,
                    overlap=0,
                )

                encoder = _TinyPatchEncoder()
                feats_dir = os.path.join(tmpdir, "features")
                out_fp = wsi.extract_patch_features(
                    patch_encoder=encoder,
                    coords_path=coords_fp,
                    save_features=feats_dir,
                    device="cpu",
                    saveas="h5",
                    batch_limit=4,
                    verbose=False,
                )
                self.assertTrue(os.path.exists(out_fp))

                # Validate output contents.
                import h5py

                with h5py.File(out_fp, "r") as f:
                    features = f["features"][:]
                    coords_out = f["coords"][:]
                self.assertEqual(coords_out.shape[0], len(coords))
                self.assertEqual(features.shape[0], len(coords))
                self.assertEqual(features.shape[1], encoder.embedding_dim)
                self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()

