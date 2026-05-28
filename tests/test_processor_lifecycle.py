import unittest
import importlib
import os
import tempfile
from unittest.mock import MagicMock, patch

from trident import Processor
from trident.IO import create_lock

processor_module = importlib.import_module("trident.Processor")


class _Ctx:
    """Simple context manager helper for lifecycle tests."""

    def __init__(self, value, on_exit):
        self.value = value
        self.on_exit = on_exit

    def __enter__(self):
        return self.value

    def __exit__(self, exc_type, exc, tb):
        self.on_exit()
        return False


class _DummyLoop:
    def __init__(self, items, **kwargs):
        self.items = items

    def __iter__(self):
        return iter(self.items)

    def set_postfix_str(self, _message):
        return None


class TestProcessorLifecycle(unittest.TestCase):
    def _processor_with_wsi(self, job_dir, wsi):
        processor = Processor.__new__(Processor)
        processor.job_dir = job_dir
        processor.skip_errors = True
        processor.max_workers = 1
        processor.wsis = [wsi]
        processor.save_config = MagicMock()
        return processor

    def _wsi(self):
        wsi = MagicMock()
        wsi.name = "slide"
        wsi.ext = ".svs"
        wsi.slide_path = "/tmp/slide.svs"
        wsi.dimensions = (100, 100)
        wsi.mpp = 0.5
        wsi.mag = 20
        wsi.level_count = 1
        wsi.release = MagicMock()
        return wsi

    def test_release_closes_exitstack_and_clears_wsis(self):
        exit_calls = {"count": 0}

        def on_exit():
            exit_calls["count"] += 1

        slide1 = MagicMock(name="slide1")
        slide2 = MagicMock(name="slide2")
        ctx1 = _Ctx(slide1, on_exit)
        ctx2 = _Ctx(slide2, on_exit)

        with patch.object(processor_module, "collect_valid_slides", return_value=(["/tmp/a.svs", "/tmp/b.svs"], ["a.svs", "b.svs"])), \
             patch.object(processor_module, "load_wsi", side_effect=[ctx1, ctx2]), \
             patch.object(processor_module.os.path, "exists", return_value=False):
            processor = Processor(
                job_dir="/tmp/job",
                wsi_source="/tmp/wsi",
                wsi_ext=[".svs"],
            )

        self.assertEqual(len(processor.wsis), 2)
        processor.release()
        self.assertEqual(exit_calls["count"], 2)
        self.assertEqual(len(processor.wsis), 0)
        self.assertIsNone(processor._wsi_stack)

    def test_release_is_idempotent(self):
        processor = Processor.__new__(Processor)
        processor.wsis = []
        processor._wsi_stack = None
        processor.release()
        processor.release()
        self.assertEqual(processor.wsis, [])
        self.assertIsNone(processor._wsi_stack)

    def test_init_failure_closes_previously_entered_contexts(self):
        exit_calls = {"count": 0}

        def on_exit():
            exit_calls["count"] += 1

        ctx1 = _Ctx(MagicMock(name="slide1"), on_exit)

        with patch.object(processor_module, "collect_valid_slides", return_value=(["/tmp/a.svs", "/tmp/b.svs"], ["a.svs", "b.svs"])), \
             patch.object(processor_module, "load_wsi", side_effect=[ctx1, RuntimeError("boom")]), \
             patch.object(processor_module.os.path, "exists", return_value=False):
            with self.assertRaises(RuntimeError):
                Processor(
                    job_dir="/tmp/job",
                    wsi_source="/tmp/wsi",
                    wsi_ext=[".svs"],
                )

        self.assertEqual(exit_calls["count"], 1)

    def test_slide_feature_job_skips_patch_extraction_when_already_processed(self):
        processor = Processor.__new__(Processor)
        processor.job_dir = "/tmp/job"
        processor.skip_errors = False
        processor.wsis = [MagicMock(name="wsi")]
        processor.wsis[0].name = "slide"
        processor.wsis[0].ext = ".ome.tif"
        processor.wsis[0].release = MagicMock()
        processor.run_patch_feature_extraction_job = MagicMock()
        processor.save_config = MagicMock()

        slide_encoder = MagicMock()
        slide_encoder.enc_name = "titan"

        def fake_exists(path):
            # Pretend the slide feature exists to avoid entering heavy extraction branch.
            return path.endswith("/slide.h5")

        with patch.object(processor_module.os.path, "isdir", return_value=True), \
             patch.object(processor_module.os, "listdir", return_value=["slide.h5"]), \
             patch.object(processor_module.os.path, "exists", side_effect=fake_exists), \
             patch.object(processor_module.os, "makedirs"), \
             patch.object(processor_module, "is_locked", return_value=False), \
             patch.object(processor_module, "update_log"):
            processor.run_slide_feature_extraction_job(
                coords_dir="20x_256px_0px_overlap",
                slide_encoder=slide_encoder,
                device="cpu",
                saveas="h5",
            )

        processor.run_patch_feature_extraction_job.assert_not_called()

    def test_segmentation_error_removes_lock_when_skip_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wsi = self._wsi()
            contour_path = os.path.join(tmpdir, "contours", "slide.jpg")

            def write_partial_contour_then_fail(**_kwargs):
                with open(contour_path, "w", encoding="utf-8") as f:
                    f.write("partial")
                raise RuntimeError("boom")

            wsi.segment_tissue.side_effect = write_partial_contour_then_fail
            processor = self._processor_with_wsi(tmpdir, wsi)

            with patch.object(processor_module, "tqdm", side_effect=lambda items, **kwargs: _DummyLoop(items, **kwargs)), \
                 patch.object(processor_module, "update_task_state"), \
                 patch.object(processor_module, "update_log"):
                processor.run_segmentation_job(
                    segmentation_model=MagicMock(),
                    seg_mag=10,
                    batch_size=1,
                    device="cpu",
                )

            self.assertFalse(os.path.exists(os.path.join(tmpdir, "contours", "slide.jpg.lock")))
            self.assertFalse(os.path.exists(contour_path))

    def test_segmentation_post_acquire_existing_output_releases_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wsi = self._wsi()
            processor = self._processor_with_wsi(tmpdir, wsi)

            def create_lock_then_finish(path):
                acquired = create_lock(path)
                with open(path, "w", encoding="utf-8") as f:
                    f.write("finished by another worker")
                return acquired

            with patch.object(processor_module, "tqdm", side_effect=lambda items, **kwargs: _DummyLoop(items, **kwargs)), \
                 patch.object(processor_module, "create_lock", side_effect=create_lock_then_finish), \
                 patch.object(processor_module, "update_task_state"), \
                 patch.object(processor_module, "update_log"):
                processor.run_segmentation_job(
                    segmentation_model=MagicMock(),
                    seg_mag=10,
                    batch_size=1,
                    device="cpu",
                )

            wsi.segment_tissue.assert_not_called()
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "contours", "slide.jpg.lock")))

    def test_coords_error_removes_lock_when_skip_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tissue_seg_path = os.path.join(tmpdir, "slide.geojson")
            with open(tissue_seg_path, "w", encoding="utf-8") as f:
                f.write("{}")

            wsi = self._wsi()
            wsi.tissue_seg_path = tissue_seg_path
            coords_path = os.path.join(tmpdir, "20x_256px_0px_overlap", "patches", "slide_patches.h5")

            def write_partial_coords_then_fail(**_kwargs):
                with open(coords_path, "w", encoding="utf-8") as f:
                    f.write("partial")
                raise RuntimeError("boom")

            wsi.extract_tissue_coords.side_effect = write_partial_coords_then_fail
            processor = self._processor_with_wsi(tmpdir, wsi)

            with patch.object(processor_module, "tqdm", side_effect=lambda items, **kwargs: _DummyLoop(items, **kwargs)), \
                 patch.object(processor_module.gpd, "read_file", return_value=MagicMock(empty=False)), \
                 patch.object(processor_module, "update_task_state"), \
                 patch.object(processor_module, "update_log"):
                coords_dir = processor.run_patching_job(
                    target_magnification=20,
                    patch_size=256,
                    overlap=0,
                )

            self.assertFalse(os.path.exists(os.path.join(coords_dir, "patches", "slide_patches.h5.lock")))
            self.assertFalse(os.path.exists(coords_path))

    def test_coords_existing_output_with_lock_is_reported_locked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            coords_dir = "20x_256px_0px_overlap"
            patches_dir = os.path.join(tmpdir, coords_dir, "patches")
            os.makedirs(patches_dir, exist_ok=True)
            coords_path = os.path.join(patches_dir, "slide_patches.h5")
            with open(coords_path, "w", encoding="utf-8") as f:
                f.write("partial")
            self.assertTrue(create_lock(coords_path))

            wsi = self._wsi()
            wsi.tissue_seg_path = os.path.join(tmpdir, "slide.geojson")
            processor = self._processor_with_wsi(tmpdir, wsi)

            with patch.object(processor_module, "tqdm", side_effect=lambda items, **kwargs: _DummyLoop(items, **kwargs)), \
                 patch.object(processor_module, "update_task_state") as update_state, \
                 patch.object(processor_module, "update_log"):
                processor.run_patching_job(
                    target_magnification=20,
                    patch_size=256,
                    overlap=0,
                )

            wsi.extract_tissue_coords.assert_not_called()
            self.assertEqual(update_state.call_args.kwargs["reason"], "locked")
            self.assertTrue(os.path.exists(f"{coords_path}.lock"))

    def test_patch_features_error_removes_lock_when_skip_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            coords_dir = "20x_256px_0px_overlap"
            patches_dir = os.path.join(tmpdir, coords_dir, "patches")
            os.makedirs(patches_dir, exist_ok=True)
            with open(os.path.join(patches_dir, "slide_patches.h5"), "w", encoding="utf-8") as f:
                f.write("coords")

            wsi = self._wsi()
            processor = self._processor_with_wsi(tmpdir, wsi)
            patch_encoder = MagicMock()
            patch_encoder.enc_name = "uni_v1"
            features_path = os.path.join(tmpdir, coords_dir, "features_uni_v1", "slide.h5")

            def write_partial_features_then_fail(**_kwargs):
                with open(features_path, "w", encoding="utf-8") as f:
                    f.write("partial")
                raise RuntimeError("boom")

            wsi.extract_patch_features.side_effect = write_partial_features_then_fail

            with patch.object(processor_module, "tqdm", side_effect=lambda items, **kwargs: _DummyLoop(items, **kwargs)), \
                 patch.object(processor_module, "update_task_state"), \
                 patch.object(processor_module, "update_log"):
                features_dir = processor.run_patch_feature_extraction_job(
                    coords_dir=coords_dir,
                    patch_encoder=patch_encoder,
                    device="cpu",
                    saveas="h5",
                    batch_limit=1,
                )

            self.assertFalse(os.path.exists(os.path.join(features_dir, "slide.h5.lock")))
            self.assertFalse(os.path.exists(features_path))

    def test_slide_features_error_removes_lock_when_skip_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            coords_dir = "20x_256px_0px_overlap"
            patch_features_dir = os.path.join(tmpdir, coords_dir, "features_mockpatch")
            os.makedirs(patch_features_dir, exist_ok=True)
            with open(os.path.join(patch_features_dir, "slide.h5"), "w", encoding="utf-8") as f:
                f.write("features")

            wsi = self._wsi()
            processor = self._processor_with_wsi(tmpdir, wsi)
            processor.run_patch_feature_extraction_job = MagicMock()
            slide_encoder = MagicMock()
            slide_encoder.enc_name = "mean-mockpatch"
            slide_features_path = os.path.join(tmpdir, coords_dir, "slide_features_mean-mockpatch", "slide.h5")

            def write_partial_slide_features_then_fail(**_kwargs):
                with open(slide_features_path, "w", encoding="utf-8") as f:
                    f.write("partial")
                raise RuntimeError("boom")

            wsi.extract_slide_features.side_effect = write_partial_slide_features_then_fail

            with patch.object(processor_module, "tqdm", side_effect=lambda items, **kwargs: _DummyLoop(items, **kwargs)), \
                 patch.object(processor_module, "update_task_state"), \
                 patch.object(processor_module, "update_log"):
                slide_features_dir = processor.run_slide_feature_extraction_job(
                    coords_dir=coords_dir,
                    slide_encoder=slide_encoder,
                    device="cpu",
                    saveas="h5",
                )

            self.assertFalse(os.path.exists(os.path.join(slide_features_dir, "slide.h5.lock")))
            self.assertFalse(os.path.exists(slide_features_path))


if __name__ == "__main__":
    unittest.main()
