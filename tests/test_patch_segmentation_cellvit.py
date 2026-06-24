"""
Real-model integration test for CellViT++ through the TRIDENT patch_seg path.

Runs the actual TIO-IKIM CellViT++ model (CellViT-SAM-H). Skipped automatically unless:
    * the `cellvit` package is importable,
    * a CUDA device is available, and
    * a bundled test slide is present under `wsis/`.

Validates that the TRIDENT wrapper (`CellViTPlusPlusSegmenter`) loads the real checkpoint
(auto-downloaded from Zenodo on first run), runs the model + its `calculate_instance_map`
post-processor via TRIDENT's batching, and returns well-formed per-cell instances.

Run it (in an environment with cellvit installed):
    pytest tests/test_patch_segmentation_cellvit.py -q -s
"""

import importlib.util
import os
import unittest

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WSI_DIR = os.path.join(REPO_ROOT, "wsis")

_HAS_CELLVIT = importlib.util.find_spec("cellvit") is not None
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


@unittest.skipUnless(_HAS_CELLVIT, "cellvit not installed")
@unittest.skipUnless(_HAS_CUDA, "CUDA not available")
@unittest.skipUnless(_find_slide_dir(), "No bundled test slide under wsis/")
class TestCellViTPlusPlusReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from trident import load_wsi
        from trident.patch_segmentation_models import patch_segmenter_factory

        cls.seg = patch_segmenter_factory("cellvit_plus_plus", magnification=40)
        cls.seg.to("cuda:0").eval()
        slide_dir = _find_slide_dir()
        slide_file = [f for f in os.listdir(slide_dir)
                      if f.lower().endswith((".tif", ".tiff", ".svs", ".ndpi"))][0]
        cls.slide = load_wsi(slide_path=os.path.join(slide_dir, slide_file), lazy_init=False)

    def test_loads_pannuke_taxonomy(self):
        self.assertEqual(self.seg.seg_name, "cellvit_plus_plus")
        # Default CellViT++ checkpoints use the 6-class PanNuke taxonomy (incl. background).
        self.assertEqual(len(self.seg.class_names), 6)
        self.assertEqual(self.seg.class_names[0], "Background")
        self.assertIn("Neoplastic", self.seg.class_names)

    def test_predict_patches_on_real_tissue(self):
        patcher = self.slide.create_patcher(
            patch_size=256, src_mag=20, dst_mag=20, overlap=0, pil=True
        )
        n = len(patcher)
        self.assertGreater(n, 0)
        tile, _, _ = patcher[n // 2]
        imgs = self.seg.eval_transforms(tile).unsqueeze(0).to("cuda:0")

        with torch.inference_mode():
            per_image = self.seg.predict_patches(imgs)

        self.assertEqual(len(per_image), 1)
        for inst in per_image[0]:
            contour = np.asarray(inst["contour"])
            self.assertEqual(contour.ndim, 2)
            self.assertEqual(contour.shape[1], 2)
            self.assertGreaterEqual(contour.shape[0], 3)
            self.assertGreaterEqual(inst["class_id"], 1)               # never background
            self.assertLess(inst["class_id"], len(self.seg.class_names))
            self.assertGreaterEqual(inst["confidence"], 0.0)
            self.assertLessEqual(inst["confidence"], 1.0)
            self.assertEqual(np.asarray(inst["centroid"]).shape, (2,))


if __name__ == "__main__":
    unittest.main()
