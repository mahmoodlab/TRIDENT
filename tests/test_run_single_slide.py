import unittest
from unittest.mock import MagicMock, patch

import run_single_slide as single_mod


class _SlideContext:
    def __init__(self, slide):
        self.slide = slide

    def __enter__(self):
        return self.slide

    def __exit__(self, exc_type, exc, tb):
        return False


class _SegModel:
    def __init__(self, target_mag=1.25):
        self.target_mag = target_mag


class TestRunSingleSlide(unittest.TestCase):
    def _base_args(self, segmenter: str):
        class Args:
            pass

        args = Args()
        args.gpu = 1
        args.slide_path = "/tmp/fake.svs"
        args.job_dir = "/tmp/job"
        args.patch_encoder = "uni_v1"
        args.mag = 20
        args.patch_size = 256
        args.segmenter = segmenter
        args.seg_conf_thresh = 0.5
        args.remove_holes = False
        args.remove_artifacts = False
        args.remove_penmarks = False
        args.custom_mpp_keys = None
        args.overlap = 0
        args.batch_size = 4
        return args

    def test_process_slide_uses_cpu_for_otsu(self):
        args = self._base_args("otsu")
        slide = MagicMock()
        slide.name = "fake"
        slide.extract_tissue_coords.return_value = "/tmp/job/coords.h5"
        slide.visualize_coords.return_value = "/tmp/job/viz.jpg"

        with patch("run_single_slide.load_wsi", return_value=_SlideContext(slide)), \
             patch("run_single_slide.segmentation_model_factory", return_value=_SegModel(target_mag=1.25)), \
             patch("run_single_slide.encoder_factory", return_value=MagicMock()):
            single_mod.process_slide(args)

        _, seg_kwargs = slide.segment_tissue.call_args
        self.assertEqual(seg_kwargs["device"], "cpu")

    def test_process_slide_uses_gpu_for_hest(self):
        args = self._base_args("hest")
        slide = MagicMock()
        slide.name = "fake"
        slide.extract_tissue_coords.return_value = "/tmp/job/coords.h5"
        slide.visualize_coords.return_value = "/tmp/job/viz.jpg"

        with patch("run_single_slide.load_wsi", return_value=_SlideContext(slide)), \
             patch("run_single_slide.segmentation_model_factory", return_value=_SegModel(target_mag=10)), \
             patch("run_single_slide.encoder_factory", return_value=MagicMock()):
            single_mod.process_slide(args)

        _, seg_kwargs = slide.segment_tissue.call_args
        self.assertEqual(seg_kwargs["device"], "cuda:1")


if __name__ == "__main__":
    unittest.main()
