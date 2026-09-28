import unittest
import tempfile
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

from trident.IO import splitext, coords_to_h5, read_coords
import trident.wsi_objects.WSIFactory as wsifactory
from trident.wsi_objects.OpenSlideWSI import OpenSlideWSI
from trident.wsi_objects.WSI import WSI
from trident.wsi_objects.WSIPatcher import WSIPatcher


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

    def test_splitext_handles_ome_zarr(self):
        stem, ext = splitext("slide.ome.zarr")
        self.assertEqual(stem, "slide")
        self.assertEqual(ext, ".ome.zarr")


class TestWSIFactoryRouting(unittest.TestCase):
    def test_auto_reader_routes_ome_tif_to_openslide(self):
        with patch.object(wsifactory, "OpenSlideWSI", return_value="open_reader") as open_mock, \
             patch.object(wsifactory, "ImageWSI", return_value="image_reader") as image_mock, \
             patch.object(wsifactory, "SDPCWSI", return_value="sdpc_reader"):
            reader = wsifactory.load_wsi("/tmp/sample.ome.tif", reader_type=None)
            self.assertEqual(reader, "open_reader")
            open_mock.assert_called_once_with(slide_path="/tmp/sample.ome.tif", lazy_init=False)
            image_mock.assert_not_called()

    def test_auto_reader_routes_ome_tiff_to_openslide(self):
        with patch.object(wsifactory, "OpenSlideWSI", return_value="open_reader") as open_mock, \
             patch.object(wsifactory, "ImageWSI", return_value="image_reader") as image_mock, \
             patch.object(wsifactory, "SDPCWSI", return_value="sdpc_reader"):
            reader = wsifactory.load_wsi("/tmp/sample.ome.tiff", reader_type=None)
            self.assertEqual(reader, "open_reader")
            open_mock.assert_called_once_with(slide_path="/tmp/sample.ome.tiff", lazy_init=False)
            image_mock.assert_not_called()

    def test_explicit_lazy_init_true_is_forwarded(self):
        with patch.object(wsifactory, "OpenSlideWSI", return_value="open_reader") as open_mock:
            reader = wsifactory.load_wsi("/tmp/sample.svs", reader_type="openslide", lazy_init=True)
            self.assertEqual(reader, "open_reader")
            open_mock.assert_called_once_with(slide_path="/tmp/sample.svs", lazy_init=True)


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


class DummyPatcherWSI:
    def __init__(self):
        self.level_downsamples = [1]
        self.width = 512
        self.height = 512

    def get_dimensions(self):
        return self.width, self.height

    def get_best_level_and_custom_downsample(self, downsample, tolerance=0.1):
        return 0, 1.0

    def read_region(self, location, level, size, read_as='numpy'):
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)


class TestEmptyCoordsBehavior(unittest.TestCase):
    def test_coords_to_h5_persists_empty_coords_as_nx2(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "coords.h5")
            coords_to_h5(
                coords=[],
                save_path=out_path,
                patch_size=256,
                src_mag=20,
                target_mag=20,
                save_coords=tmpdir,
                width=1000,
                height=1000,
                name="dummy",
                overlap=0,
            )
            _, coords = read_coords(out_path)
            self.assertEqual(coords.shape, (0, 2))

    def test_patcher_accepts_empty_custom_coords(self):
        patcher = WSIPatcher(
            wsi=DummyPatcherWSI(),
            patch_size=256,
            src_mag=20,
            dst_mag=1,
            custom_coords=np.empty((0, 2), dtype=np.int64),
            coords_only=True,
        )
        self.assertEqual(len(patcher), 0)
        self.assertEqual(patcher.valid_coords.shape, (0, 2))


class _FakeVipsImage:
    def __init__(self, array):
        self.array = array
        self.height, self.width = array.shape[:2]
        self.bands = 1 if array.ndim == 2 else array.shape[2]

    def cast(self, _format):
        return self

    def extract_band(self, start, n=None):
        if n is None:
            return _FakeVipsImage(self.array[:, :, start])
        return _FakeVipsImage(self.array[:, :, start:start + n])

    def write_to_memory(self):
        return self.array.tobytes()


class TestOpenSlideThumbnailBehavior(unittest.TestCase):
    @staticmethod
    def _make_wsi(level_count=1, slide_path="single-layer.tiff"):
        wsi = OpenSlideWSI.__new__(OpenSlideWSI)
        wsi.level_count = level_count
        wsi.slide_path = slide_path
        wsi.img = MagicMock()
        return wsi

    def test_single_level_tiff_uses_vips_thumbnail(self):
        source = np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4)
        fake_thumbnail = _FakeVipsImage(source)
        thumbnail_mock = MagicMock(return_value=fake_thumbnail)
        fake_pyvips = SimpleNamespace(Image=SimpleNamespace(thumbnail=thumbnail_mock))
        wsi = self._make_wsi()

        with patch.dict(sys.modules, {"pyvips": fake_pyvips}):
            result = wsi.get_thumbnail((100, 50))

        thumbnail_mock.assert_called_once_with(
            "single-layer.tiff",
            100,
            height=50,
            size="down",
        )
        wsi.img.get_thumbnail.assert_not_called()
        self.assertEqual(result.mode, "RGB")
        np.testing.assert_array_equal(np.asarray(result), source[:, :, :3])

    def test_thumbnail_falls_back_when_vips_is_unavailable(self):
        wsi = self._make_wsi()
        wsi.img.get_thumbnail.return_value = Image.new("L", (3, 2))

        with patch.dict(sys.modules, {"pyvips": None}):
            result = wsi.get_thumbnail((100, 50))

        wsi.img.get_thumbnail.assert_called_once_with((100, 50))
        self.assertEqual(result.mode, "RGB")

    def test_multilevel_tiff_keeps_openslide_thumbnail_path(self):
        wsi = self._make_wsi(level_count=2)
        wsi.img.get_thumbnail.return_value = Image.new("RGB", (3, 2))

        result = wsi.get_thumbnail((100, 50))

        wsi.img.get_thumbnail.assert_called_once_with((100, 50))
        self.assertEqual(result.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
