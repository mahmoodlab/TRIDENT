"""
Unit tests for the unified overlay renderer.

Covers the shared ``render_overlay`` core (used by tissue, cell and patch-grid
visualizations) and the public ``WSI.overlay`` entrypoint. Dependency-light: uses an
``ImageWSI`` built from a synthetic PNG, no segmentation models or real slides required.
"""

import os
import tempfile
import unittest

import numpy as np
from PIL import Image

import geopandas as gpd
from shapely import Polygon

from trident.Visualization import render_overlay
from trident.wsi_objects.ImageWSI import ImageWSI


def _square(x, y, s):
    return Polygon([(x, y), (x + s, y), (x + s, y + s), (x, y + s)])


class TestRenderOverlay(unittest.TestCase):
    def test_outline_draws_boundary_only(self):
        canvas = np.full((100, 100, 3), 255, dtype=np.uint8)
        ext = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.int32)
        out = render_overlay(canvas, [(ext, [], (0, 0, 255))], mode='outline', thickness=2)
        self.assertIs(out, canvas)  # in place
        # boundary pixel changed, interior center untouched (outline, not fill)
        self.assertFalse(np.array_equal(canvas[20, 50], [255, 255, 255]))
        self.assertTrue(np.array_equal(canvas[50, 50], [255, 255, 255]))

    def test_fill_blends_interior(self):
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)  # black
        ext = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.int32)
        render_overlay(canvas, [(ext, [], (0, 0, 200))], mode='fill', alpha=0.5)
        # interior is alpha*color over black = ~100 in R channel (BGR idx 2)
        self.assertAlmostEqual(int(canvas[50, 50][2]), 100, delta=2)
        # outside the polygon stays black
        self.assertTrue(np.array_equal(canvas[5, 5], [0, 0, 0]))

    def test_fill_preserves_holes(self):
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        ext = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.int32)
        hole = np.array([[40, 40], [60, 40], [60, 60], [40, 60]], dtype=np.int32)
        render_overlay(canvas, [(ext, [hole], (0, 0, 200))], mode='fill', alpha=0.5)
        self.assertGreater(int(canvas[20, 20][2]), 0)            # filled ring
        self.assertTrue(np.array_equal(canvas[50, 50], [0, 0, 0]))  # hole untinted

    def test_invalid_mode_raises(self):
        canvas = np.zeros((10, 10, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            render_overlay(canvas, [], mode='nope')


class TestWSIOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.img_path = os.path.join(cls.tmp.name, "synthetic.png")
        # 1200x800 pink-ish image so overlays are visible
        arr = np.full((800, 1200, 3), 200, dtype=np.uint8)
        Image.fromarray(arr).save(cls.img_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _wsi(self):
        return ImageWSI(self.img_path, lazy_init=False, mpp=0.5)

    def _cells_gdf(self):
        return gpd.GeoDataFrame(
            [{"class": 1, "class_name": "tumor", "geometry": _square(100, 100, 60)},
             {"class": 2, "class_name": "lymph", "geometry": _square(400, 300, 60)}],
            columns=["class", "class_name", "geometry"], geometry="geometry",
        )

    def test_thumbnail_overlay_returns_image(self):
        wsi = self._wsi()
        im = wsi.overlay(self._cells_gdf(), mode='outline', color_by='class')
        self.assertIsInstance(im, Image.Image)
        # thumbnail fits within the max_dim box (aspect preserved), never exceeds it
        self.assertLessEqual(max(im.size), 2000)

    def test_max_dim_caps_raster(self):
        wsi = self._wsi()
        im = wsi.overlay(self._cells_gdf(), max_dim=500)
        # get_thumbnail fits *within* the box, so the long side is <= max_dim (and close to it)
        self.assertLessEqual(max(im.size), 500)
        self.assertGreaterEqual(max(im.size), 498)

    def test_region_overlay_size(self):
        wsi = self._wsi()
        im = wsi.overlay(self._cells_gdf(), region=(50, 50, 400, 400), color_by='class')
        self.assertEqual(im.size, (400, 400))  # < max_dim -> 1:1

    def test_saveto_writes_file(self):
        wsi = self._wsi()
        out = os.path.join(self.tmp.name, "ov.jpg")
        wsi.overlay(self._cells_gdf(), mode='fill', color_by='class', saveto=out)
        self.assertTrue(os.path.exists(out) and os.path.getsize(out) > 0)

    def test_path_input(self):
        wsi = self._wsi()
        gj = os.path.join(self.tmp.name, "cells.geojson")
        self._cells_gdf().to_file(gj, driver="GeoJSON")
        im = wsi.overlay(gj, color_by='class')
        self.assertIsInstance(im, Image.Image)

    def test_empty_gdf_does_not_crash(self):
        wsi = self._wsi()
        empty = self._cells_gdf().iloc[0:0]
        im = wsi.overlay(empty, mode='outline')
        self.assertIsInstance(im, Image.Image)

    def test_overlay_actually_marks_pixels(self):
        wsi = self._wsi()
        base = np.array(wsi.overlay(self._cells_gdf().iloc[0:0]))  # no geometries
        drawn = np.array(wsi.overlay(self._cells_gdf(), mode='fill', color_by='class'))
        self.assertFalse(np.array_equal(base, drawn))


if __name__ == "__main__":
    unittest.main()
