"""
Integration test for the `patch_seg` task on a real slide using a *fake* instance model.

This exercises the full TRIDENT pipeline end-to-end (tissue seg -> coords -> patch_seg)
and the three output artifacts (GeoJSON, HDF5, visualization) + coordinate stitching,
WITHOUT needing HistoPlus/CellViT++. The fake model emits deterministic per-patch
instances so geometry can be checked exactly.

Skips automatically if the bundled test slide is not present.
"""

import json
import os
import unittest

import h5py
import numpy as np
import torch
from torchvision import transforms

from trident import Processor
from trident.segmentation_models.load import segmentation_model_factory
from trident.patch_segmentation_models import BasePatchSegmenter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WSI_DIR = os.path.join(REPO_ROOT, "wsis")


def _find_slide():
    if not os.path.isdir(WSI_DIR):
        return None
    for f in os.listdir(WSI_DIR):
        if f.lower().endswith((".tif", ".tiff", ".svs", ".ndpi")):
            return WSI_DIR
    return None


class _FakeInstanceSegmenter(BasePatchSegmenter):
    """Emits exactly one centered square instance (class 1) per patch, in patch-px coords."""

    def __init__(self):
        super().__init__()
        self.seg_name = "fake_cells"
        self.class_names = ["background", "cell"]
        self.eval_transforms = transforms.Compose([transforms.ToTensor()])
        self.precision = torch.float32
        self.model = torch.nn.Identity()

    def _build(self):
        return None, None, None

    def predict_patches(self, imgs):
        b, _, h, w = imgs.shape
        out = []
        for _ in range(b):
            contour = np.array(
                [[w // 4, h // 4], [w // 4, 3 * h // 4],
                 [3 * w // 4, 3 * h // 4], [3 * w // 4, h // 4]], dtype=np.float64
            )
            out.append([{
                "contour": contour,
                "class_id": 1,
                "class_name": "cell",
                "confidence": 0.8,
                "centroid": np.array([w / 2.0, h / 2.0]),
            }])
        return out


@unittest.skipUnless(_find_slide(), "No bundled test slide under wsis/")
class TestPatchSegPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmp = tempfile.mkdtemp(prefix="patchseg_")
        cls.wsi_dir = _find_slide()
        p = Processor(job_dir=cls.tmp, wsi_source=cls.wsi_dir, skip_errors=False)
        otsu = segmentation_model_factory("otsu", confidence_thresh=0.5)
        p.run_segmentation_job(otsu, seg_mag=otsu.target_mag, device="cpu")
        cls.coords_dir = os.path.relpath(
            p.run_patching_job(target_magnification=5, patch_size=512, overlap=0, visualize=False),
            cls.tmp,
        )
        p.release()

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_pipeline_produces_aligned_artifacts(self):
        import geopandas as gpd

        p = Processor(job_dir=self.tmp, wsi_source=self.wsi_dir, skip_errors=False)
        seg = _FakeInstanceSegmenter()
        out_dir = p.run_patch_segmentation_job(
            coords_dir=self.coords_dir, patch_segmenter=seg, device="cpu",
            batch_limit=16, visualize=True,
        )
        p.release()

        geojsons = [f for f in os.listdir(out_dir) if f.endswith(".geojson")]
        self.assertEqual(len(geojsons), 1)
        name = geojsons[0][:-len(".geojson")]
        geojson_path = os.path.join(out_dir, geojsons[0])
        h5_path = os.path.join(out_dir, f"{name}.h5")
        viz_overview = os.path.join(out_dir, "visualization", f"{name}_overview.jpg")
        viz_patch_dir = os.path.join(out_dir, "visualization", name)

        # GeoJSON: one polygon per patch, all class 1, confidence 0.8.
        gdf = gpd.read_file(geojson_path)
        n_polys = len(gdf)
        self.assertGreater(n_polys, 0)
        self.assertEqual(sorted(gdf["class"].unique().tolist()), [1])
        self.assertTrue(np.allclose(gdf["confidence"].to_numpy(), 0.8))

        # HDF5: same instance count, ragged contours reconstruct, attrs carry class_names.
        with h5py.File(h5_path, "r") as f:
            g = f["cells"]
            offsets = g["contour_offsets"][:]
            class_ids = g["class_ids"][:]
            self.assertEqual(len(class_ids), n_polys)
            self.assertEqual(len(offsets), n_polys + 1)
            self.assertEqual(set(class_ids.tolist()), {1})
            self.assertEqual(json.loads(g.attrs["class_names"]), ["background", "cell"])

        # Visualization artifacts present: slide overview + full-res sample patch overlays.
        self.assertTrue(os.path.exists(viz_overview))
        self.assertTrue(os.path.isdir(viz_patch_dir))
        patch_jpgs = [f for f in os.listdir(viz_patch_dir) if f.endswith(".jpg")]
        self.assertGreater(len(patch_jpgs), 0)

        # Coordinates are in level-0 space: every polygon lies within slide bounds.
        from trident import load_wsi
        with load_wsi(slide_path=os.path.join(self.wsi_dir, [
            x for x in os.listdir(self.wsi_dir)
            if x.lower().endswith((".tif", ".tiff", ".svs", ".ndpi"))][0]), lazy_init=False) as s:
            W, H = s.width, s.height
        # Coords must be in level-0 space. Edge patches can overhang the slide edge by up
        # to a patch, so allow a 10% margin (a scale bug would be multiples off, not 2%).
        minx, miny, maxx, maxy = gdf.total_bounds
        self.assertGreaterEqual(minx, 0)
        self.assertGreaterEqual(miny, 0)
        self.assertLessEqual(maxx, W * 1.1)
        self.assertLessEqual(maxy, H * 1.1)

    def test_rerun_skips(self):
        seg = _FakeInstanceSegmenter()
        p = Processor(job_dir=self.tmp, wsi_source=self.wsi_dir, skip_errors=False)
        # First run already happened in the prior test for some configs; ensure idempotency.
        p.run_patch_segmentation_job(
            coords_dir=self.coords_dir, patch_segmenter=seg, device="cpu", batch_limit=16,
        )
        out_dir = p.run_patch_segmentation_job(
            coords_dir=self.coords_dir, patch_segmenter=seg, device="cpu", batch_limit=16,
        )
        p.release()
        self.assertTrue(os.path.isdir(out_dir))


if __name__ == "__main__":
    unittest.main()
