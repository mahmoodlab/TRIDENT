import unittest
import numpy as np

from trident.wsi_objects.WSI import WSI
from trident.wsi_objects.WSIPatcher import WSIPatcher


class _MockPyramidWSI(WSI):
    def __init__(self, level_dims, mpp=0.25):
        super().__init__("mock.svs", lazy_init=True, mpp=mpp)
        self.width, self.height = level_dims[0]
        self.dimensions = level_dims[0]
        self.level_dimensions = level_dims
        self.level_count = len(level_dims)
        self.level_downsamples = tuple(
            ((self.width / w) + (self.height / h)) / 2.0 for w, h in level_dims
        )
        self.mag = 40
        self._initialized = True
        self.read_calls = []

    def _lazy_initialize(self):
        pass

    def get_dimensions(self):
        return self.dimensions

    def read_region(self, location, level, size, read_as="numpy"):
        self.read_calls.append((location, level, size))
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)


class TestWSIPatcherLevelDownsample(unittest.TestCase):
    def test_sub_integer_level_downsample_not_truncated(self):
        # Odd base dimensions with ceil-rounded pyramid levels produce
        # level_downsamples slightly below 2.0 and 4.0 (e.g. 1.9998, 3.9988).
        wsi = _MockPyramidWSI([(10001, 10001), (5001, 5001), (2501, 2501)], mpp=0.25)

        patcher_20x = WSIPatcher(wsi, patch_size=256, src_mag=40, dst_mag=20, overlap=64)
        self.assertEqual(patcher_20x.level, 1)
        self.assertEqual(patcher_20x.patch_size_level, 256)
        self.assertEqual(patcher_20x.overlap_level, 64)

        patcher_20x.get_tile_xy(0, 0)
        self.assertEqual(wsi.read_calls[-1], ((0, 0), 1, (256, 256)))

        patcher_10x = WSIPatcher(wsi, patch_size=512, src_mag=40, dst_mag=10, overlap=128)
        self.assertEqual(patcher_10x.level, 2)
        self.assertEqual(patcher_10x.patch_size_level, 512)
        self.assertEqual(patcher_10x.overlap_level, 128)

    def test_fractional_level_downsample_preserved(self):
        # Pyramid level with non-integer downsample (2.5) used for 10x (downsample=4.0).
        wsi = _MockPyramidWSI([(10000, 10000), (4000, 4000)], mpp=0.25)

        patcher = WSIPatcher(wsi, patch_size=250, src_mag=40, dst_mag=10, overlap=50)
        self.assertEqual(patcher.level, 1)
        self.assertEqual(patcher.patch_size_level, 400)
        self.assertEqual(patcher.overlap_level, 80)


if __name__ == "__main__":
    unittest.main()
