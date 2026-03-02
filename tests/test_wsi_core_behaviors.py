import unittest
from unittest.mock import patch

from trident.IO import splitext
import trident.wsi_objects.WSIFactory as wsifactory
from trident.wsi_objects.WSI import WSI


class DummyWSI(WSI):
    """Lightweight WSI used to test context-manager lifecycle."""

    def __init__(self, *args, **kwargs):
        self.released = False
        super().__init__(*args, **kwargs)

    def release(self) -> None:
        self.released = True


class TestSplitExt(unittest.TestCase):
    def test_splitext_handles_compound_ome_tif(self):
        stem, ext = splitext("slide.ome.tif")
        self.assertEqual(stem, "slide")
        self.assertEqual(ext, ".ome.tif")

    def test_splitext_handles_compound_ome_tiff(self):
        stem, ext = splitext("slide.ome.tiff")
        self.assertEqual(stem, "slide")
        self.assertEqual(ext, ".ome.tiff")

    def test_splitext_falls_back_to_standard_extensions(self):
        stem, ext = splitext("slide.svs")
        self.assertEqual(stem, "slide")
        self.assertEqual(ext, ".svs")


class TestWSIFactoryRouting(unittest.TestCase):
    def test_auto_reader_routes_ome_tif_to_openslide(self):
        with patch.object(wsifactory, "OpenSlideWSI", return_value="open_reader") as open_mock, \
             patch.object(wsifactory, "ImageWSI", return_value="image_reader") as image_mock, \
             patch.object(wsifactory, "SDPCWSI", return_value="sdpc_reader"):
            reader = wsifactory.load_wsi("/tmp/sample.ome.tif", reader_type=None)
            self.assertEqual(reader, "open_reader")
            open_mock.assert_called_once_with(slide_path="/tmp/sample.ome.tif")
            image_mock.assert_not_called()

    def test_auto_reader_routes_ome_tiff_to_openslide(self):
        with patch.object(wsifactory, "OpenSlideWSI", return_value="open_reader") as open_mock, \
             patch.object(wsifactory, "ImageWSI", return_value="image_reader") as image_mock, \
             patch.object(wsifactory, "SDPCWSI", return_value="sdpc_reader"):
            reader = wsifactory.load_wsi("/tmp/sample.ome.tiff", reader_type=None)
            self.assertEqual(reader, "open_reader")
            open_mock.assert_called_once_with(slide_path="/tmp/sample.ome.tiff")
            image_mock.assert_not_called()


class TestWSIContextManager(unittest.TestCase):
    def test_context_manager_releases_on_normal_exit(self):
        wsi = DummyWSI(slide_path="dummy.ome.tif", lazy_init=True)
        self.assertFalse(wsi.released)
        with wsi as scoped_wsi:
            self.assertIs(scoped_wsi, wsi)
        self.assertTrue(wsi.released)

    def test_context_manager_releases_and_does_not_swallow_exceptions(self):
        wsi = DummyWSI(slide_path="dummy.ome.tif", lazy_init=True)
        with self.assertRaises(RuntimeError):
            with wsi:
                raise RuntimeError("boom")
        self.assertTrue(wsi.released)


if __name__ == "__main__":
    unittest.main()
