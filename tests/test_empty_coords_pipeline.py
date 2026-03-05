import os
import tempfile
import unittest

import geopandas as gpd
import h5py
import numpy as np
import torch
from PIL import Image
from shapely.geometry import Polygon

from trident.IO import read_coords
from trident.wsi_objects.ImageWSI import ImageWSI


class DummyPatchEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_name = "dummy_patch"
        self.precision = torch.float32
        self.embedding_dim = 4

    @staticmethod
    def eval_transforms(img):
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def forward(self, x):
        # Deterministic lightweight embedding with shape [B, 4]
        bsz = x.shape[0]
        pooled = x.mean(dim=(2, 3))
        out = torch.zeros((bsz, self.embedding_dim), dtype=pooled.dtype, device=pooled.device)
        out[:, : min(pooled.shape[1], self.embedding_dim)] = pooled[:, : min(pooled.shape[1], self.embedding_dim)]
        return out


class DummySlideEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.precision = torch.float32
        self.called = False

    def forward(self, batch, device="cpu"):
        self.called = True
        # Return one deterministic slide embedding
        return torch.zeros((1, 8), dtype=torch.float32, device=device)


class TestEmptyCoordsPipeline(unittest.TestCase):
    def setUp(self):
        self.tmpdir_ctx = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_ctx.name
        self.slide_path = os.path.join(self.tmpdir, "synthetic_slide.png")

        # Synthetic 20x-equivalent image with sparse signal.
        img = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        img[100:120, 100:120, :] = 0
        Image.fromarray(img).save(self.slide_path)

    def tearDown(self):
        self.tmpdir_ctx.cleanup()

    @staticmethod
    def _tiny_tissue_mask():
        return gpd.GeoDataFrame(
            geometry=[
                Polygon(
                    [
                        (100, 100),
                        (120, 100),
                        (120, 120),
                        (100, 120),
                        (100, 100),
                    ]
                )
            ]
        )

    def _build_wsi_with_mask(self):
        wsi = ImageWSI(slide_path=self.slide_path, mpp=0.5, lazy_init=False)
        wsi.gdf_contours = self._tiny_tissue_mask()
        return wsi

    def test_high_min_tissue_proportion_produces_empty_coords(self):
        wsi = self._build_wsi_with_mask()
        coords_path = wsi.extract_tissue_coords(
            target_mag=1.25,
            patch_size=256,
            save_coords=self.tmpdir,
            overlap=0,
            min_tissue_proportion=0.25,
        )
        attrs, coords = read_coords(coords_path)
        self.assertEqual(coords.shape, (0, 2))
        self.assertEqual(attrs["patch_size"], 256)

    def test_visualize_coords_handles_empty_coords(self):
        wsi = self._build_wsi_with_mask()
        coords_path = wsi.extract_tissue_coords(
            target_mag=1.25,
            patch_size=256,
            save_coords=self.tmpdir,
            min_tissue_proportion=0.25,
        )
        viz_dir = os.path.join(self.tmpdir, "viz")
        viz_path = wsi.visualize_coords(coords_path=coords_path, save_patch_viz=viz_dir)
        self.assertTrue(os.path.exists(viz_path))
        with Image.open(viz_path) as img:
            self.assertGreater(img.size[0], 0)
            self.assertGreater(img.size[1], 0)

    def test_extract_patch_features_handles_empty_coords(self):
        wsi = self._build_wsi_with_mask()
        coords_path = wsi.extract_tissue_coords(
            target_mag=1.25,
            patch_size=256,
            save_coords=self.tmpdir,
            min_tissue_proportion=0.25,
        )

        feats_dir = os.path.join(self.tmpdir, "features")
        out_path = wsi.extract_patch_features(
            patch_encoder=DummyPatchEncoder(),
            coords_path=coords_path,
            save_features=feats_dir,
            device="cpu",
            saveas="h5",
            batch_limit=16,
        )
        self.assertTrue(os.path.exists(out_path))
        with h5py.File(out_path, "r") as f:
            self.assertEqual(f["coords"].shape, (0, 2))
            self.assertEqual(f["features"].shape, (0, 4))

    def test_extract_slide_features_skips_encoder_on_empty_patch_features(self):
        wsi = self._build_wsi_with_mask()
        coords_path = wsi.extract_tissue_coords(
            target_mag=1.25,
            patch_size=256,
            save_coords=self.tmpdir,
            min_tissue_proportion=0.25,
        )
        patch_feats_path = wsi.extract_patch_features(
            patch_encoder=DummyPatchEncoder(),
            coords_path=coords_path,
            save_features=os.path.join(self.tmpdir, "features"),
            device="cpu",
            saveas="h5",
            batch_limit=16,
        )

        slide_encoder = DummySlideEncoder()
        slide_out = wsi.extract_slide_features(
            patch_features_path=patch_feats_path,
            slide_encoder=slide_encoder,
            save_features=os.path.join(self.tmpdir, "slide_features"),
            device="cpu",
        )
        self.assertTrue(os.path.exists(slide_out))
        self.assertFalse(slide_encoder.called)
        with h5py.File(slide_out, "r") as f:
            self.assertEqual(f["features"].shape, (0,))
            self.assertEqual(f["coords"].shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
