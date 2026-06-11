import unittest
from unittest.mock import patch

import run_batch_of_slides as batch_mod
import run_single_slide as single_mod
from trident.patch_encoder_models import RESIZE_SUPPORTED_PATCH_ENCODERS
from trident.patch_encoder_models.load import (
    _resolve_target_img_size,
    encoder_registry,
)

"""
Unit tests for the configurable patch-encoder input resolution feature
(`--patch_encoder_img_size` / `target_img_size`).

These are deliberately lightweight: they exercise the pure validation helper,
the supported-encoder registry, and the CLI plumbing/gating WITHOUT downloading
weights or instantiating real models, so they run in CI with
`TRIDENT_RUN_INTEGRATION_TESTS=0`. Forward-pass equivalence for the actual
encoders lives in `tests/test_patch_encoders.py` behind the integration gate.
"""


class TestResolveTargetImgSize(unittest.TestCase):
    def test_none_returns_default(self):
        self.assertEqual(_resolve_target_img_size("enc", None, 224, 16), 224)
        self.assertEqual(_resolve_target_img_size("enc", None, 518, 14), 518)

    def test_valid_multiple_returned_unchanged(self):
        self.assertEqual(_resolve_target_img_size("enc", 512, 224, 16), 512)
        self.assertEqual(_resolve_target_img_size("enc", 448, 224, 14), 448)
        # The default itself is always a valid multiple.
        self.assertEqual(_resolve_target_img_size("enc", 224, 224, 16), 224)

    def test_non_multiple_raises(self):
        with self.assertRaises(AssertionError) as ctx:
            _resolve_target_img_size("enc", 500, 224, 16)
        # Error should mention the patch size and the encoder name to be useful.
        msg = str(ctx.exception)
        self.assertIn("enc", msg)
        self.assertIn("16", msg)

    def test_non_positive_raises(self):
        for bad in (0, -16):
            with self.assertRaises(AssertionError):
                _resolve_target_img_size("enc", bad, 224, 16)

    def test_non_integer_raises(self):
        for bad in (512.0, "512"):
            with self.assertRaises(AssertionError):
                _resolve_target_img_size("enc", bad, 224, 16)


class TestResizeSupportedRegistry(unittest.TestCase):
    def test_supported_encoders_exist_in_registry(self):
        """Every name advertised as resize-capable must be a real encoder."""
        missing = sorted(n for n in RESIZE_SUPPORTED_PATCH_ENCODERS if n not in encoder_registry)
        self.assertEqual(missing, [], f"Unknown encoder names in RESIZE_SUPPORTED_PATCH_ENCODERS: {missing}")

    def test_set_is_non_empty(self):
        self.assertGreater(len(RESIZE_SUPPORTED_PATCH_ENCODERS), 0)


class TestParsersExposeFlag(unittest.TestCase):
    def test_batch_parser_defaults_to_none(self):
        parser = batch_mod.build_parser()
        args = parser.parse_args(["--job_dir", "job", "--wsi_dir", "wsis"])
        self.assertIsNone(args.patch_encoder_img_size)

    def test_batch_parser_accepts_value(self):
        parser = batch_mod.build_parser()
        args = parser.parse_args([
            "--job_dir", "job", "--wsi_dir", "wsis",
            "--patch_encoder_img_size", "512",
        ])
        self.assertEqual(args.patch_encoder_img_size, 512)

    def test_single_parser_defaults_to_none(self):
        with patch("sys.argv", ["run_single_slide.py", "--slide_path", "x.svs",
                                "--job_dir", "job"]):
            args = single_mod.parse_arguments()
        self.assertIsNone(args.patch_encoder_img_size)

    def test_single_parser_accepts_value(self):
        with patch("sys.argv", ["run_single_slide.py", "--slide_path", "x.svs",
                                "--job_dir", "job", "--patch_encoder_img_size", "448"]):
            args = single_mod.parse_arguments()
        self.assertEqual(args.patch_encoder_img_size, 448)


class _DummyProcessor:
    def __init__(self):
        self.calls = []

    def run_patch_feature_extraction_job(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class TestRunTaskFeatGating(unittest.TestCase):
    """`run_batch_of_slides.run_task` gating for `--patch_encoder_img_size`."""

    def _feat_args(self, patch_encoder, img_size):
        class Args:
            pass

        args = Args()
        args.task = "feat"
        args.device = "cpu"
        args.slide_encoder = None
        args.patch_encoder = patch_encoder
        args.patch_encoder_ckpt_path = None
        args.patch_encoder_img_size = img_size
        args.mag = 20
        args.patch_size = 256
        args.overlap = 0
        args.coords_dir = None
        args.feat_batch_size = None
        args.batch_size = 8
        return args

    def test_unsupported_encoder_with_img_size_raises(self):
        # `conch_v15` is intentionally NOT in the resize-supported set.
        self.assertNotIn("conch_v15", RESIZE_SUPPORTED_PATCH_ENCODERS)
        processor = _DummyProcessor()
        args = self._feat_args("conch_v15", img_size=512)

        with patch("trident.patch_encoder_models.load.encoder_factory") as mock_factory:
            with self.assertRaises(ValueError) as ctx:
                batch_mod.run_task(processor, args)

        self.assertIn("conch_v15", str(ctx.exception))
        mock_factory.assert_not_called()
        self.assertEqual(processor.calls, [])

    def test_supported_encoder_forwards_target_img_size(self):
        processor = _DummyProcessor()
        args = self._feat_args("uni_v2", img_size=512)

        with patch("trident.patch_encoder_models.load.encoder_factory") as mock_factory:
            mock_factory.return_value = object()
            batch_mod.run_task(processor, args)

        mock_factory.assert_called_once()
        _, kwargs = mock_factory.call_args
        self.assertEqual(kwargs.get("target_img_size"), 512)
        self.assertEqual(len(processor.calls), 1)

    def test_no_img_size_does_not_forward_kwarg(self):
        processor = _DummyProcessor()
        args = self._feat_args("uni_v2", img_size=None)

        with patch("trident.patch_encoder_models.load.encoder_factory") as mock_factory:
            mock_factory.return_value = object()
            batch_mod.run_task(processor, args)

        mock_factory.assert_called_once()
        _, kwargs = mock_factory.call_args
        self.assertNotIn("target_img_size", kwargs)

    def test_missing_attr_is_tolerated(self):
        """Args constructed manually without the attribute must not crash
        (mirrors the `getattr(args, ..., None)` guard added in the PR)."""
        processor = _DummyProcessor()
        args = self._feat_args("uni_v2", img_size=None)
        del args.patch_encoder_img_size

        with patch("trident.patch_encoder_models.load.encoder_factory") as mock_factory:
            mock_factory.return_value = object()
            batch_mod.run_task(processor, args)

        mock_factory.assert_called_once()
        _, kwargs = mock_factory.call_args
        self.assertNotIn("target_img_size", kwargs)


if __name__ == "__main__":
    unittest.main()
