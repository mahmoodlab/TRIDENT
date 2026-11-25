import os
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from trident import load_wsi
from trident.wsi_objects.ISyntaxWSI import ISyntaxWSI

"""
Unit tests for the Philips iSyntax reader. The `isyntax` backend (pyisyntax) is replaced by a
fake module, so these tests run in CI without the optional dependency or a real slide.
"""


class _FakeISyntax:
    """Minimal stand-in for `isyntax.ISyntax`: a 3-level pyramid with downsamples 1, 2, 4."""

    opened = []

    def __init__(self, path):
        self.path = path
        self.dimensions = (4000, 2000)
        self.level_count = 3
        self.level_downsamples = (1.0, 2.0, 4.0)
        self.level_dimensions = ((4000, 2000), (2000, 1000), (1000, 500))
        self.mpp_x = 0.25
        self.mpp_y = 0.25
        self.offset_x = 0
        self.offset_y = 0
        self.reads = []
        self.closed = False

    @classmethod
    def open(cls, path):
        handle = cls(path)
        cls.opened.append(handle)
        return handle

    def read_region(self, x, y, width, height, level=0):
        self.reads.append((x, y, width, height, level))
        region = np.zeros((height, width, 4), dtype=np.uint8)
        region[..., 0] = level
        region[..., 3] = 255
        return region

    def close(self):
        self.closed = True


class TestISyntaxReader(unittest.TestCase):
    def setUp(self):
        _FakeISyntax.opened = []
        self._modules = patch.dict(sys.modules, {"isyntax": types.SimpleNamespace(ISyntax=_FakeISyntax)})
        self._modules.start()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.slide_path = os.path.join(self._tmpdir.name, "slide.isyntax")
        open(self.slide_path, "wb").close()

    def tearDown(self):
        self._modules.stop()
        self._tmpdir.cleanup()

    def test_autodetect_by_extension_and_metadata(self):
        wsi = load_wsi(self.slide_path, lazy_init=False)
        try:
            self.assertIsInstance(wsi, ISyntaxWSI)
            self.assertEqual(wsi.get_dimensions(), (4000, 2000))
            self.assertEqual(wsi.level_count, 3)
            self.assertAlmostEqual(wsi.mpp, 0.25)
            self.assertEqual(wsi.mag, 40)
        finally:
            wsi.release()

    def test_read_region_converts_level0_location_to_level_coordinates(self):
        wsi = load_wsi(self.slide_path, reader_type="isyntax", lazy_init=False)
        try:
            tile = wsi.read_region((800, 400), level=2, size=(64, 32), read_as="numpy")
            self.assertEqual(tile.shape, (32, 64, 3))
            self.assertEqual(tile.dtype, np.uint8)
            self.assertTrue((tile[..., 0] == 2).all())
            self.assertEqual(wsi.img.reads[-1], (200, 100, 64, 32, 2))

            pil_tile = wsi.read_region((0, 0), level=0, size=(16, 16))
            self.assertEqual(pil_tile.mode, "RGB")
            self.assertEqual(pil_tile.size, (16, 16))

            with self.assertRaises(ValueError):
                wsi.read_region((0, 0), level=3, size=(16, 16))
        finally:
            wsi.release()

    def test_thumbnail_preserves_aspect_ratio(self):
        wsi = load_wsi(self.slide_path, lazy_init=False)
        try:
            thumb = wsi.get_thumbnail((200, 200))
            self.assertEqual(thumb.mode, "RGB")
            self.assertEqual(thumb.size, (200, 100))
            # Reads the coarsest level that is still at least as large as the thumbnail.
            self.assertEqual(wsi.img.reads[-1][-1], 2)
        finally:
            wsi.release()

    def test_pickle_drops_handle_and_reopens(self):
        wsi = load_wsi(self.slide_path, lazy_init=False)
        clone = pickle.loads(pickle.dumps(wsi))
        try:
            self.assertIsNone(clone.img)
            tile = clone.read_region((0, 0), level=0, size=(8, 8), read_as="numpy")
            self.assertEqual(tile.shape, (8, 8, 3))
            self.assertEqual(len(_FakeISyntax.opened), 2)
        finally:
            wsi.release()
            clone.release()

    def test_release_closes_backend(self):
        wsi = load_wsi(self.slide_path, lazy_init=False)
        handle = wsi.img
        wsi.release()
        self.assertTrue(handle.closed)
        self.assertIsNone(wsi.img)

    def test_missing_backend_raises_install_hint(self):
        with patch.dict(sys.modules, {"isyntax": None}):
            with self.assertRaises(ImportError) as ctx:
                load_wsi(self.slide_path, lazy_init=False)
        self.assertIn("pyisyntax", str(ctx.exception))

    def test_wrong_extension_for_forced_reader(self):
        with self.assertRaises(ValueError):
            load_wsi(os.path.join(self._tmpdir.name, "slide.svs"), reader_type="isyntax")


if __name__ == "__main__":
    unittest.main()
