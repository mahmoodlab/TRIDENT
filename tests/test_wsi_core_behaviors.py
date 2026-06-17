import unittest
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch
import tempfile
import os
import numpy as np

from trident.IO import splitext, coords_to_h5, read_coords, create_lock, remove_lock
import trident.wsi_objects.WSIFactory as wsifactory
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


class TestIOLocks(unittest.TestCase):
    def test_create_lock_does_not_overwrite_existing_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "output.h5")

            self.assertTrue(create_lock(target))
            with open(f"{target}.lock", "r", encoding="utf-8") as f:
                first_payload = json.load(f)

            self.assertFalse(create_lock(target))
            with open(f"{target}.lock", "r", encoding="utf-8") as f:
                second_payload = json.load(f)

            self.assertEqual(second_payload, first_payload)
            remove_lock(target)
            self.assertFalse(os.path.exists(f"{target}.lock"))

    def test_create_lock_allows_only_one_concurrent_acquirer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "output.h5")
            workers = 8
            barrier = Barrier(workers)

            def acquire_once():
                barrier.wait()
                return create_lock(target)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(lambda _: acquire_once(), range(workers)))

            self.assertEqual(sum(1 for result in results if result), 1)
            self.assertTrue(os.path.exists(f"{target}.lock"))
            with open(f"{target}.lock", "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertIn("pid", payload)
            self.assertIn("hostname", payload)
            self.assertIn("created_at", payload)


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


if __name__ == "__main__":
    unittest.main()
