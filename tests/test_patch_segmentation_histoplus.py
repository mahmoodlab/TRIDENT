"""
Real-model integration test for HistoPlus through the TRIDENT patch_seg path.

This test runs the actual Owkin HistoPlus CellViT model. It is skipped automatically unless:
    * the `histoplus` package is importable,
    * a CUDA device is available (HistoPlus on CPU is impractically slow), and
    * a bundled test slide is present under `wsis/`.

It validates that the TRIDENT wrapper (`HistoPlusSegmenter`) loads the real model, reuses
its normalization, runs it on real tissue patches via TRIDENT's batching, and returns
well-formed per-cell instances. Because HistoPlus is gated on Hugging Face, weights are
downloaded on first run (an accepted license + HF token are required).

Run it (in an environment with histoplus installed), e.g.:
    pytest tests/test_patch_segmentation_histoplus.py -q -s
"""

import importlib.util
import os
import unittest

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WSI_DIR = os.path.join(REPO_ROOT, "wsis")

_HAS_HISTOPLUS = importlib.util.find_spec("histoplus") is not None
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False


def _find_slide_dir():
    if not os.path.isdir(WSI_DIR):
        return None
    for f in os.listdir(WSI_DIR):
        if f.lower().endswith((".tif", ".tiff", ".svs", ".ndpi")):
            return WSI_DIR
    return None


@unittest.skipUnless(_HAS_HISTOPLUS, "histoplus not installed")
@unittest.skipUnless(_HAS_CUDA, "CUDA not available (HistoPlus on CPU is too slow)")
@unittest.skipUnless(_find_slide_dir(), "No bundled test slide under wsis/")
class TestHistoPlusReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from trident import load_wsi
        from trident.patch_segmentation_models import patch_segmenter_factory

        cls.seg = patch_segmenter_factory("histoplus", mpp=0.5, inference_image_size=784)
        cls.seg.to("cuda:0").eval()
        slide_dir = _find_slide_dir()
        slide_file = [f for f in os.listdir(slide_dir)
                      if f.lower().endswith((".tif", ".tiff", ".svs", ".ndpi"))][0]
        cls.slide = load_wsi(slide_path=os.path.join(slide_dir, slide_file), lazy_init=False)

    def test_loads_expected_taxonomy(self):
        # HistoPlus ships a 15-class pan-cancer taxonomy; index 0 is background.
        self.assertEqual(self.seg.seg_name, "histoplus")
        self.assertEqual(len(self.seg.class_names), 15)
        self.assertEqual(self.seg.class_names[0], "Background")
        self.assertAlmostEqual(self.seg.target_mpp, 0.5, places=3)

    def test_predict_patches_on_real_tissue(self):
        patcher = self.slide.create_patcher(
            patch_size=784, src_mag=20, dst_mag=20, overlap=0, pil=True
        )
        n = len(patcher)
        self.assertGreater(n, 0)
        # NOTE: process one patch at a time. The vendored CellViT attention (xformers)
        # segfaults on batched input under torch>=2.10; batch_size=1 is the safe path
        # (TRIDENT's patch_seg task should likewise use --feat_batch_size 1 there).
        per_image = []
        for i in (n // 2, n // 2 + 1):
            tile, _, _ = patcher[i]
            imgs = self.seg.eval_transforms(tile).unsqueeze(0).to("cuda:0")
            with torch.inference_mode():
                per_image.extend(self.seg.predict_patches(imgs))

        self.assertEqual(len(per_image), 2)
        # Real H&E tissue should yield a non-trivial number of cells on at least one patch.
        self.assertTrue(any(len(p) > 0 for p in per_image))
        for instances in per_image:
            for inst in instances:
                contour = np.asarray(inst["contour"])
                self.assertEqual(contour.ndim, 2)
                self.assertEqual(contour.shape[1], 2)
                self.assertGreaterEqual(contour.shape[0], 3)
                self.assertGreaterEqual(inst["class_id"], 1)             # never background
                self.assertLess(inst["class_id"], len(self.seg.class_names))
                self.assertGreaterEqual(inst["confidence"], 0.0)
                self.assertLessEqual(inst["confidence"], 1.0)
                self.assertEqual(np.asarray(inst["centroid"]).shape, (2,))


if __name__ == "__main__":
    unittest.main()
