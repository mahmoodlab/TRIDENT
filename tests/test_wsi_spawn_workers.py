import multiprocessing as mp
import os
import pickle
import tempfile
import unittest
import warnings
from unittest.mock import patch

import h5py
import numpy as np
import torch
from PIL import Image

from trident.wsi_objects.ImageWSI import ImageWSI


class MeanRGBEncoder(torch.nn.Module):
    """Module-level, so spawned DataLoader workers can unpickle the transforms."""

    enc_name = "mean_rgb"
    precision = torch.float32
    embedding_dim = 3

    @staticmethod
    def eval_transforms(img):
        return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

    def forward(self, x):
        return x.mean(dim=(2, 3))


class TestSpawnWorkers(unittest.TestCase):
    """Slides cross into spawned DataLoader workers (the only start method on Windows)."""

    def setUp(self):
        self.tmp_ctx = tempfile.TemporaryDirectory()
        self.tmp = self.tmp_ctx.name
        self.slide_path = os.path.join(self.tmp, "slide.png")
        img = (np.random.default_rng(0).random((512, 512, 3)) * 255).astype(np.uint8)
        Image.fromarray(img).save(self.slide_path)
        self.coords_path = os.path.join(self.tmp, "coords.h5")
        coords = np.array([[x, y] for x in (0, 128, 256) for y in (0, 128, 256)], dtype=np.int64)
        with h5py.File(self.coords_path, "w") as f:
            d = f.create_dataset("coords", data=coords)
            d.attrs.update(patch_size=128, level0_magnification=20, target_magnification=20)

    def tearDown(self):
        self.tmp_ctx.cleanup()

    def _features(self, out_name):
        wsi = ImageWSI(slide_path=self.slide_path, mpp=0.5, lazy_init=False, max_workers=2)
        fp = wsi.extract_patch_features(MeanRGBEncoder(), self.coords_path,
                                        os.path.join(self.tmp, out_name), device="cpu", batch_limit=4)
        with h5py.File(fp, "r") as f:
            return f["features"][:]

    def test_slide_pickles(self):
        wsi = ImageWSI(slide_path=self.slide_path, mpp=0.5, lazy_init=False)
        before = wsi.read_region((0, 0), level=0, size=(64, 64), read_as="numpy")
        clone = pickle.loads(pickle.dumps(wsi))
        np.testing.assert_array_equal(clone.read_region((0, 0), level=0, size=(64, 64), read_as="numpy"), before)

    def test_spawn_workers_match_default(self):
        # Only the outputs are compared: another test module reloads WSIPatcher, which can make
        # patchers unpicklable in the same process and trigger the (correct) fallback.
        default = self._features("default")
        with patch.object(mp, "get_all_start_methods", return_value=["spawn"]), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spawned = self._features("spawn")
        np.testing.assert_array_equal(default, spawned)

    def test_czi_slide_pickles(self):
        try:
            import pylibCZIrw  # noqa: F401
        except Exception:
            self.skipTest("pylibCZIrw not installed")
        from trident import load_wsi

        czi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "czis", "239_GS_PD_P1.czi"))
        if not os.path.exists(czi_path):
            self.skipTest(f"Missing test asset: {czi_path}")
        wsi = load_wsi(slide_path=czi_path, lazy_init=False)
        before = wsi.read_region((0, 0), level=0, size=(64, 64), read_as="numpy")
        clone = pickle.loads(pickle.dumps(wsi))
        np.testing.assert_array_equal(clone.read_region((0, 0), level=0, size=(64, 64), read_as="numpy"), before)
        wsi.release()
        clone.release()


if __name__ == "__main__":
    unittest.main()
