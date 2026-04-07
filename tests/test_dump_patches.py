import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from trident.IO import coords_to_h5
from trident.wsi_objects.WSI import WSI


class DummyWSIForDump(WSI):
    """
    Minimal in-memory WSI that supports create_patcher -> dump_patches.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _lazy_initialize(self) -> None:
        super()._lazy_initialize()
        if not self._initialized:
            self.width, self.height = 512, 512
            self.dimensions = (self.width, self.height)
            self.level_count = 1
            self.level_downsamples = [1.0]
            self.level_dimensions = [self.dimensions]
            if self.mpp is None:
                self.mpp = 0.5
            self.mag = 20
            self._initialized = True

    def get_dimensions(self):
        return self.dimensions

    def get_best_level_and_custom_downsample(self, downsample: float, tolerance: float = 0.01):
        return 0, 1.0

    def read_region(self, location, level, size, read_as="pil"):
        w, h = size
        # simple deterministic pattern
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:, :, 0] = 100
        arr[:, :, 1] = 50
        arr[:, :, 2] = 200
        img = Image.fromarray(arr, mode="RGB")
        if read_as == "pil":
            return img
        return np.array(img)

    def get_thumbnail(self, size):
        return self.read_region((0, 0), 0, size, read_as="pil")


class TestDumpPatches(unittest.TestCase):
    def test_dump_patches_writes_pngs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wsi = DummyWSIForDump(slide_path="dummy.svs", lazy_init=False, mpp=0.5)

            coords_path = os.path.join(tmpdir, "coords.h5")
            coords_to_h5(
                coords=[(0, 0), (128, 0), (0, 128)],
                save_path=coords_path,
                patch_size=64,
                src_mag=wsi.mag,
                target_mag=20,
                save_coords=tmpdir,
                width=wsi.width,
                height=wsi.height,
                name=wsi.name,
                overlap=0,
            )

            out_dir = wsi.dump_patches(coords_path=coords_path, save_patches_dir=tmpdir, max_patches=2)
            self.assertTrue(os.path.isdir(out_dir))
            pngs = [f for f in os.listdir(out_dir) if f.endswith(".png")]
            self.assertEqual(len(pngs), 2)

    def test_dump_patches_writes_jpgs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wsi = DummyWSIForDump(slide_path="dummy.svs", lazy_init=False, mpp=0.5)

            coords_path = os.path.join(tmpdir, "coords.h5")
            coords_to_h5(
                coords=[(0, 0), (128, 0), (0, 128)],
                save_path=coords_path,
                patch_size=64,
                src_mag=wsi.mag,
                target_mag=20,
                save_coords=tmpdir,
                width=wsi.width,
                height=wsi.height,
                name=wsi.name,
                overlap=0,
            )

            out_dir = wsi.dump_patches(
                coords_path=coords_path,
                save_patches_dir=tmpdir,
                max_patches=2,
                image_format="jpg",
                jpeg_quality=85,
            )
            self.assertTrue(os.path.isdir(out_dir))
            jpgs = [f for f in os.listdir(out_dir) if f.endswith(".jpg")]
            self.assertEqual(len(jpgs), 2)


if __name__ == "__main__":
    unittest.main()

