"""
Integration test for the `vlm` task on a real slide using a *fake* VLM.

Exercises the full TRIDENT pipeline end-to-end (tissue seg -> coords -> vlm) and the two
output artifacts (JSON answers + QuPath GeoJSON) + coordinate stitching + resume, WITHOUT
downloading Patho-R1. The fake model returns a deterministic answer per patch so the
record/geometry alignment can be checked exactly.

Skips automatically if the bundled test slide is not present.
"""

import json
import os
import unittest

import numpy as np
import torch

from trident import Processor
from trident.segmentation_models.load import segmentation_model_factory
from trident.vlm_models import BaseVLM

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WSI_DIR = os.path.join(REPO_ROOT, "wsis")
_SLIDE_EXTS = (".tif", ".tiff", ".svs", ".ndpi")


def _find_slide_dir():
    if not os.path.isdir(WSI_DIR):
        return None
    for f in os.listdir(WSI_DIR):
        if f.lower().endswith(_SLIDE_EXTS):
            return WSI_DIR
    return None


def _slide_file(slide_dir):
    return [f for f in os.listdir(slide_dir) if f.lower().endswith(_SLIDE_EXTS)][0]


class _FakeVLM(BaseVLM):
    """Returns a deterministic ``FAKE::<prompt>`` answer per image; no weights needed."""

    def __init__(self):
        super().__init__()
        self.vlm_name = "fake_vlm"
        self.model = torch.nn.Identity()
        self.n_generate_calls = 0

    def _build(self):
        return None, None, None

    def generate(self, images, prompts, max_new_tokens=None):
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        self.n_generate_calls += len(images)
        return [f"FAKE::{p}" for p in prompts]


@unittest.skipUnless(_find_slide_dir(), "No bundled test slide under wsis/")
class TestVLMPipeline(unittest.TestCase):
    PROMPT = "Is tumor present?"

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmp = tempfile.mkdtemp(prefix="vlm_")
        cls.wsi_dir = _find_slide_dir()
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
        out_dir = p.run_vlm_query_job(
            coords_dir=self.coords_dir, vlm=_FakeVLM(), prompt=self.PROMPT,
            device="cpu", batch_limit=8,
        )
        p.release()

        # Output dir is keyed per model.
        self.assertTrue(out_dir.endswith("vlm_fake_vlm"))
        jsons = [f for f in os.listdir(out_dir) if f.endswith(".json")]
        self.assertEqual(len(jsons), 1)
        name = jsons[0][:-len(".json")]
        json_path = os.path.join(out_dir, jsons[0])
        geojson_path = os.path.join(out_dir, f"{name}.geojson")

        # JSON: one record per patch, prompt + deterministic answer threaded through.
        with open(json_path) as f:
            payload = json.load(f)
        self.assertEqual(payload["model"], "fake_vlm")
        self.assertEqual(payload["prompt"], self.PROMPT)
        records = payload["answers"]
        n = len(records)
        self.assertGreater(n, 0)
        for rec in records:
            self.assertEqual(set(rec), {"x", "y", "prompt", "answer"})
            self.assertEqual(rec["prompt"], self.PROMPT)
            self.assertEqual(rec["answer"], f"FAKE::{self.PROMPT}")
            self.assertIsInstance(rec["x"], int)
            self.assertIsInstance(rec["y"], int)

        # GeoJSON: one box per patch, carrying prompt + answer; loadable as annotations.
        gdf = gpd.read_file(geojson_path)
        self.assertEqual(len(gdf), n)
        self.assertIn("answer", gdf.columns)
        self.assertIn("prompt", gdf.columns)
        self.assertTrue((gdf["answer"] == f"FAKE::{self.PROMPT}").all())

        # Boxes are in level-0 space: within slide bounds (allow edge overhang margin).
        from trident import load_wsi
        with load_wsi(slide_path=os.path.join(self.wsi_dir, _slide_file(self.wsi_dir)),
                      lazy_init=False) as s:
            W, H = s.width, s.height
        minx, miny, maxx, maxy = gdf.total_bounds
        self.assertGreaterEqual(minx, 0)
        self.assertGreaterEqual(miny, 0)
        self.assertLessEqual(maxx, W * 1.1)
        self.assertLessEqual(maxy, H * 1.1)

        # Config + logs written alongside the coords dir, like the other tasks.
        coords_abs = os.path.join(self.tmp, self.coords_dir)
        self.assertTrue(os.path.exists(os.path.join(coords_abs, "_config_vlm_fake_vlm.json")))
        self.assertTrue(os.path.exists(os.path.join(coords_abs, "_logs_vlm_fake_vlm.txt")))

    def test_rerun_skips(self):
        p = Processor(job_dir=self.tmp, wsi_source=self.wsi_dir, skip_errors=False)
        out1 = p.run_vlm_query_job(
            coords_dir=self.coords_dir, vlm=_FakeVLM(), prompt=self.PROMPT,
            device="cpu", batch_limit=8,
        )
        name = [f for f in os.listdir(out1) if f.endswith(".json")][0]
        mtime_before = os.path.getmtime(os.path.join(out1, name))
        out2 = p.run_vlm_query_job(
            coords_dir=self.coords_dir, vlm=_FakeVLM(), prompt="A DIFFERENT PROMPT",
            device="cpu", batch_limit=8,
        )
        p.release()
        # Idempotent: the existing answers are kept (not overwritten by the new prompt).
        self.assertEqual(out1, out2)
        with open(os.path.join(out2, name)) as f:
            payload = json.load(f)
        self.assertEqual(payload["prompt"], self.PROMPT)  # original, not "A DIFFERENT PROMPT"
        self.assertEqual(os.path.getmtime(os.path.join(out2, name)), mtime_before)

    def test_query_region_interactive(self):
        from trident import load_wsi
        with load_wsi(slide_path=os.path.join(self.wsi_dir, _slide_file(self.wsi_dir)),
                      lazy_init=False) as s:
            answer = s.query_region(
                vlm=_FakeVLM(), prompt="Describe this ROI.",
                location=(0, 0), size=256, mag=5,
            )
        self.assertEqual(answer, "FAKE::Describe this ROI.")


class TestVLMEmptyCoords(unittest.TestCase):
    """`query_patches` on a slide with zero tissue patches writes valid empty artifacts."""

    def setUp(self):
        import tempfile
        from PIL import Image
        self.tmp_ctx = tempfile.TemporaryDirectory()
        self.tmp = self.tmp_ctx.name
        self.slide_path = os.path.join(self.tmp, "synthetic_slide.png")
        img = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        img[100:120, 100:120, :] = 0
        Image.fromarray(img).save(self.slide_path)

    def tearDown(self):
        self.tmp_ctx.cleanup()

    def _empty_coords_wsi(self):
        import geopandas as gpd
        from shapely.geometry import Polygon
        from trident.wsi_objects.ImageWSI import ImageWSI
        wsi = ImageWSI(slide_path=self.slide_path, mpp=0.5, lazy_init=False)
        wsi.gdf_contours = gpd.GeoDataFrame(
            geometry=[Polygon([(100, 100), (120, 100), (120, 120), (100, 120), (100, 100)])]
        )
        coords_path = wsi.extract_tissue_coords(
            target_mag=1.25, patch_size=256, save_coords=self.tmp,
            overlap=0, min_tissue_proportion=0.25,  # forces zero valid patches
        )
        return wsi, coords_path

    def test_empty_coords_writes_valid_artifacts(self):
        import geopandas as gpd
        wsi, coords_path = self._empty_coords_wsi()
        vlm = _FakeVLM()
        save_dir = os.path.join(self.tmp, "vlm_out")
        out = wsi.query_patches(
            vlm=vlm, coords_path=coords_path, prompt="Anything here?",
            save_dir=save_dir, device="cpu", batch_limit=4,
        )
        self.assertTrue(os.path.exists(out))
        # No patches -> the model is never invoked.
        self.assertEqual(vlm.n_generate_calls, 0)
        with open(out) as f:
            payload = json.load(f)
        self.assertEqual(payload["answers"], [])
        self.assertEqual(payload["prompt"], "Anything here?")
        geojson_path = os.path.join(save_dir, f"{wsi.name}.geojson")
        self.assertTrue(os.path.exists(geojson_path))
        gdf = gpd.read_file(geojson_path)
        self.assertEqual(len(gdf), 0)


if __name__ == "__main__":
    unittest.main()
