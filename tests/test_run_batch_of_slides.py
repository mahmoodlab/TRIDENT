import unittest
from unittest.mock import patch

import run_batch_of_slides as batch_mod


class _DummySegmentationModel:
    def __init__(self, target_mag=1.25):
        self.target_mag = target_mag


class _DummyProcessor:
    def __init__(self):
        self.calls = []

    def run_segmentation_job(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class TestRunBatchOfSlides(unittest.TestCase):
    def _base_args(self, segmenter: str):
        class Args:
            pass

        args = Args()
        args.task = "seg"
        args.segmenter = segmenter
        args.seg_conf_thresh = 0.5
        args.remove_holes = False
        args.remove_artifacts = False
        args.remove_penmarks = False
        args.seg_batch_size = None
        args.batch_size = 8
        args.gpu = 2
        return args

    def test_run_task_seg_uses_cpu_for_otsu(self):
        processor = _DummyProcessor()
        args = self._base_args("otsu")

        with patch("trident.segmentation_models.load.segmentation_model_factory", return_value=_DummySegmentationModel()):
            batch_mod.run_task(processor, args)

        self.assertEqual(len(processor.calls), 1)
        kwargs = processor.calls[0][1]
        self.assertEqual(kwargs["device"], "cpu")

    def test_run_task_seg_uses_gpu_for_hest(self):
        processor = _DummyProcessor()
        args = self._base_args("hest")

        with patch("trident.segmentation_models.load.segmentation_model_factory", return_value=_DummySegmentationModel(target_mag=10)):
            batch_mod.run_task(processor, args)

        self.assertEqual(len(processor.calls), 1)
        kwargs = processor.calls[0][1]
        self.assertEqual(kwargs["device"], "cuda:2")


if __name__ == "__main__":
    unittest.main()
