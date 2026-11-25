import os
import tempfile
import unittest
import warnings

import h5py
import numpy as np
import torch
from PIL import Image

from trident.wsi_objects.ImageWSI import ImageWSI


class DummyPatchEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_name = "dummy_patch"
        self.precision = torch.float32
        self.embedding_dim = 3

    @staticmethod
    def eval_transforms(img):
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def forward(self, x):
        return x.mean(dim=(2, 3))


class TestLegacyCoordsFeatures(unittest.TestCase):
    """Feature extraction from CLAM / Fishing-Rod coords (`patch_size` + `patch_level` attrs)."""

    def setUp(self):
        self.tmpdir_ctx = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_ctx.name
        self.slide_path = os.path.join(self.tmpdir, "synthetic_slide.png")
        img = np.full((512, 512, 3), 255, dtype=np.uint8)
        img[:256, :256] = 0  # first patch black, the others white
        Image.fromarray(img).save(self.slide_path)

        self.coords = np.array([[0, 0], [256, 0], [0, 256]], dtype=np.int64)
        self.coords_path = os.path.join(self.tmpdir, "legacy_patches.h5")
        with h5py.File(self.coords_path, "w") as f:
            dset = f.create_dataset("coords", data=self.coords)
            dset.attrs["patch_size"] = 256
            dset.attrs["patch_level"] = 0

    def tearDown(self):
        self.tmpdir_ctx.cleanup()

    def test_extract_patch_features_from_legacy_coords(self):
        wsi = ImageWSI(slide_path=self.slide_path, mpp=0.5, lazy_init=False)
        save_dir = os.path.join(self.tmpdir, "features")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = wsi.extract_patch_features(
                patch_encoder=DummyPatchEncoder(),
                coords_path=self.coords_path,
                save_features=save_dir,
                device="cpu",
            )

        with h5py.File(out, "r") as f:
            features = f["features"][:]
            coords = f["coords"][:]
            coords_attrs = dict(f["coords"].attrs)

        np.testing.assert_array_equal(coords, self.coords)
        self.assertEqual(features.shape, (3, 3))
        np.testing.assert_allclose(features[0], 0.0)   # black patch
        np.testing.assert_allclose(features[1:], 1.0)  # white patches
        self.assertEqual(int(coords_attrs["patch_size"]), 256)
        self.assertEqual(int(coords_attrs["patch_level"]), 0)


if __name__ == "__main__":
    unittest.main()
