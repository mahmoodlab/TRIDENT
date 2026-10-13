"""Device routing checks; no GPU or model downloads required."""
import importlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import MagicMock, patch

import numpy as np
import torch

import run_batch_of_slides as batch
import run_single_slide as single

wsi = importlib.import_module("trident.wsi_objects.WSI")


class TestMPSDevice(unittest.TestCase):
    def parse(self, module, extra):
        required = ["--job_dir", "out"]
        required += ["--wsi_dir", "wsis"] if module is batch else ["--slide_path", "slide.svs"]
        with patch.object(sys, "argv", ["trident", *required, *extra]):
            return module.parse_arguments()

    def test_explicit_devices_and_legacy_cuda(self):
        for module in (batch, single):
            for device in ("mps", "cpu"):
                with self.subTest(cli=module.__name__, device=device), \
                     patch("torch.backends.mps.is_available", return_value=True):
                    args = self.parse(module, ["--device", device])
                    self.assertEqual(args.device, device)
                    if module is batch:
                        self.assertEqual(args.gpus, [-1])
            self.assertEqual(self.parse(module, ["--gpu", "2"]).device, "cuda:2")
        self.assertEqual(self.parse(batch, ["--gpus", "0", "1"]).gpus, [0, 1])

    def test_unavailable_mps_fails_before_processing(self):
        for module in (batch, single):
            stderr = io.StringIO()
            with self.subTest(cli=module.__name__), \
                 patch("torch.backends.mps.is_available", return_value=False), \
                 redirect_stderr(stderr), \
                 self.assertRaises(SystemExit) as caught:
                self.parse(module, ["--device", "mps"])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("MPS is not available", stderr.getvalue())

    def test_batch_keeps_explicit_device_without_cuda_or_subprocess(self):
        for device in ("cpu", "mps"):
            with self.subTest(device=device), \
                 patch("torch.backends.mps.is_available", return_value=True):
                args = self.parse(batch, ["--device", device, "--gpus", "0", "1"])
            with patch.object(batch, "parse_arguments", return_value=args), \
                 patch.object(batch, "cleanup_cache"), \
                 patch.object(batch, "start_run"), patch.object(batch, "finalize_run"), \
                 patch.object(batch, "get_pending_slides", return_value=["one.svs", "two.svs"]), \
                 patch.object(batch, "worker_entrypoint") as worker, \
                 patch.object(batch, "_pick_mp_context") as context, \
                 patch("torch.cuda.is_available", return_value=False):
                batch.main()
            self.assertEqual(worker.call_args.args[0].device, device)
            self.assertEqual(worker.call_args.args[0].selected_wsi_paths, ["one.svs", "two.svs"])
            context.assert_not_called()

    def test_single_routes_features_and_artifact_removal(self):
        for device in ("cpu", "mps"):
            with self.subTest(device=device), patch("torch.backends.mps.is_available", return_value=True):
                args = self.parse(single, ["--device", device, "--segmenter", "otsu", "--remove_artifacts"])
                slide = MagicMock()
                slide.name = "slide"
                encoder = MagicMock()
                with patch.object(single, "load_wsi") as loader, \
                     patch.object(single, "segmentation_model_factory"), \
                     patch.object(single, "encoder_factory", return_value=encoder):
                    loader.return_value.__enter__.return_value = slide
                    single.process_slide(args)
                self.assertEqual(slide.segment_tissue.call_args_list[0].kwargs["device"], "cpu")
                self.assertEqual(slide.segment_tissue.call_args_list[1].kwargs["device"], device)
                self.assertEqual(slide.extract_patch_features.call_args.kwargs["device"], device)
                encoder.to.assert_called_once_with(device)

    def test_mps_never_uses_a_multiprocessing_context(self):
        self.assertEqual(wsi._dataloader_context_candidates(8, device="mps"), [None])

    def test_processor_routes_artifact_removal_to_requested_device(self):
        module = importlib.import_module("trident.Processor")
        processor = module.Processor.__new__(module.Processor)
        slide = MagicMock()
        slide.name, slide.ext = "slide", ".svs"
        processor.wsis = [slide]
        processor.skip_errors = False
        processor.save_config = MagicMock()
        processor._record_outcome = MagicMock()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(module, "create_lock"), patch.object(module, "remove_lock"), \
             patch.object(module.gpd, "read_file", return_value=MagicMock(empty=False)):
            processor.job_dir = directory
            processor.run_segmentation_job(MagicMock(), artifact_remover_model=MagicMock(), device="mps")
        self.assertEqual(len(slide.segment_tissue.call_args_list), 2)
        self.assertTrue(all(call.kwargs["device"] == "mps" for call in slide.segment_tissue.call_args_list))

    def test_macos_does_not_try_fork(self):
        with patch.object(wsi.sys, "platform", "darwin"):
            contexts = wsi._dataloader_context_candidates(2)
        self.assertEqual([ctx.get_start_method() if ctx else None for ctx in contexts], ["spawn", None])

    def test_serial_feature_fallback_really_has_zero_workers(self):
        slide = MagicMock()
        slide.max_workers = 8
        encoder = MagicMock(precision=torch.float32)
        attrs = {"patch_size": 256, "level0_magnification": 20, "target_magnification": 20}

        class LoaderReached(Exception):
            pass

        with patch.object(wsi, "read_coords", return_value=(attrs, np.array([[0, 0]]))), \
             patch.object(wsi, "WSIPatcherDataset", return_value=[1]), \
             patch.object(wsi, "_dataloader_context_candidates", return_value=[None]), \
             patch.object(wsi, "DataLoader", side_effect=LoaderReached) as loader:
            with self.assertRaises(LoaderReached):
                wsi.WSI.extract_patch_features(slide, encoder, "coords.h5", "out", device="cpu")
        self.assertEqual(loader.call_args.kwargs["num_workers"], 0)
        self.assertNotIn("multiprocessing_context", loader.call_args.kwargs)

    def test_serial_segmentation_fallback_really_has_zero_workers(self):
        slide = MagicMock()
        slide.mpp = 0.5
        slide.get_dimensions.return_value = (256, 256)
        model = MagicMock(precision=torch.float32)

        class LoaderReached(Exception):
            pass

        with patch.object(wsi, "_dataloader_context_candidates", return_value=[None]), \
             patch.object(wsi, "DataLoader", side_effect=LoaderReached) as loader:
            with self.assertRaises(LoaderReached):
                wsi.WSI._segment_semantic(slide, model, 20, False, "mps", 4, None, 8, None)
        self.assertEqual(loader.call_args.kwargs["num_workers"], 0)

    def test_fallback_does_not_hide_inference_errors(self):
        run = MagicMock(side_effect=RuntimeError("MPS out of memory"))
        with self.assertRaisesRegex(RuntimeError, "MPS out of memory"):
            wsi._run_with_dataloader_ctx_fallback(run, 8, "test", "fallback", "loader", device="mps")
        run.assert_called_once_with(None)

    def test_known_flash_attention_slide_encoders_rejected_on_mps(self):
        for name in ("gigapath", "gigapath-flash", "prism2"):
            args = self.parse(batch, ["--device", "cpu", "--task", "feat", "--slide_encoder", name])
            args.device = "mps"
            with self.subTest(encoder=name), self.assertRaisesRegex(ValueError, "FlashAttention"):
                batch.run_task(MagicMock(), args)


if __name__ == "__main__":
    unittest.main()
