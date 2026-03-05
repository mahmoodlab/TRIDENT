import unittest
import importlib
from unittest.mock import MagicMock, patch

from trident import Processor

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


class TestProcessorLifecycle(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
