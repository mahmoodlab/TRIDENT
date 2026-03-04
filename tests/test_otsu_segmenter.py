import unittest

import numpy as np
import torch

from trident.segmentation_models import (
    OtsuSegmenter,
    apply_otsu_thresholding,
    mask_rgb,
    segmentation_model_factory,
)


class TestOtsuSegmenter(unittest.TestCase):
    def test_mask_rgb_preserves_shape(self):
        rgb = np.full((32, 32, 3), 200, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        out = mask_rgb(rgb, mask)
        self.assertEqual(out.shape, rgb.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_apply_otsu_thresholding_binary_output(self):
        # White background with a darker "tissue-like" blob and black border.
        tile = np.full((128, 128, 3), 255, dtype=np.uint8)
        tile[30:95, 30:95] = np.array([150, 80, 120], dtype=np.uint8)
        tile[:2, :, :] = 0
        tile[-2:, :, :] = 0
        tile[:, :2, :] = 0
        tile[:, -2:, :] = 0

        mask = apply_otsu_thresholding(tile)
        self.assertEqual(mask.shape, tile.shape[:2])
        self.assertEqual(mask.dtype, np.uint8)
        unique = set(np.unique(mask).tolist())
        self.assertTrue(unique.issubset({0, 1}))

    def test_apply_otsu_thresholding_all_white_tile(self):
        tile = np.full((128, 128, 3), 255, dtype=np.uint8)
        mask = apply_otsu_thresholding(tile)
        self.assertEqual(mask.shape, (128, 128))
        self.assertTrue(np.all(mask == 1))

    def test_apply_otsu_thresholding_invalid_shape_raises(self):
        tile = np.full((128, 128), 255, dtype=np.uint8)  # grayscale instead of RGB
        with self.assertRaises(Exception):
            _ = apply_otsu_thresholding(tile)

    def test_factory_builds_otsu_segmenter(self):
        seg = segmentation_model_factory("otsu")
        self.assertIsInstance(seg, OtsuSegmenter)
        self.assertEqual(seg.input_size, 512)
        self.assertEqual(seg.target_mag, 1.25)

    def test_otsu_forward_output_shape_dtype(self):
        seg = OtsuSegmenter()
        img = np.full((1, 3, 64, 64), 1.0, dtype=np.float32)
        img[:, :, 20:44, 20:44] = 0.4
        x = torch.from_numpy(img)

        y = seg(x)
        self.assertEqual(tuple(y.shape), (1, 64, 64))
        self.assertEqual(y.dtype, torch.uint8)
        unique = set(torch.unique(y).cpu().numpy().tolist())
        self.assertTrue(unique.issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
