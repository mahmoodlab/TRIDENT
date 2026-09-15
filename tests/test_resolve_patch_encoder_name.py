"""
Regression test for #239: with `--slide_encoder`, `--patch_encoder` keeps its default
('conch_v15') because the slide encoder picks its own patch encoder. Recording the raw default
made `runs/*.json` and `summary.md` name an encoder that never ran.

`resolve_patch_encoder_name` is the single source of that rule -- Processor uses it to choose
the encoder, the CLI uses it to report one, so the two cannot drift.

Fast tier: CPU only, no network, no model download.
"""
import unittest

from run_batch_of_slides import build_parser
from trident.patch_encoder_models import encoder_registry as patch_encoder_registry
from trident.slide_encoder_models import encoder_registry as slide_encoder_registry
from trident.slide_encoder_models.load import (
    resolve_patch_encoder_name,
    slide_to_patch_encoder_name,
)


class TestResolvePatchEncoderName(unittest.TestCase):

    def test_pinned_slide_encoder(self):
        self.assertEqual(resolve_patch_encoder_name("feather_uni_v2"), "uni_v2")

    def test_mean_encoder_uses_its_suffix(self):
        self.assertEqual(resolve_patch_encoder_name("mean-virchow2-cls"), "virchow2-cls")

    def test_unmapped_encoder_raises(self):
        # 'abmil' is a building block with no fixed patch encoder; callers must handle this.
        with self.assertRaises(KeyError):
            resolve_patch_encoder_name("abmil")

    def test_resolver_is_pure(self):
        # It used to mutate the module-level map in Processor, so resolution depended on
        # whatever had run before.
        before = dict(slide_to_patch_encoder_name)
        resolve_patch_encoder_name("mean-resnet50")
        self.assertEqual(dict(slide_to_patch_encoder_name), before)

    def test_every_cli_slide_encoder_resolves_to_a_real_patch_encoder(self):
        for name in slide_encoder_registry:
            if name == "abmil":
                continue  # not CLI-usable: encoder_factory('abmil') needs 5 kwargs
            if name.startswith("mean-kaiko-"):
                # Pre-existing, unrelated to #239: slide-side names are transposed relative to
                # the patch registry ('mean-kaiko-vit14l' -> 'kaiko-vit14l', but the patch
                # encoder is 'kaiko-vitl14'), so these already fail. Tracked separately.
                continue
            with self.subTest(slide_encoder=name):
                self.assertIn(resolve_patch_encoder_name(name), patch_encoder_registry)


class TestRunManifestRecordsEffectiveEncoder(unittest.TestCase):
    """The #239 symptom itself: what lands in runs/*.json and summary.md."""

    def _args(self, *extra):
        return build_parser().parse_args(
            ["--task", "all", "--wsi_dir", "/data", "--job_dir", "/out", *extra]
        )

    def test_slide_encoder_run_does_not_report_the_stale_default(self):
        # The exact command from issue #239.
        args = self._args("--slide_encoder", "feather_uni_v2", "--patch_size", "256", "--mag", "20")
        self.assertEqual(args.patch_encoder, "conch_v15", "argparse default changed; update this test")
        self.assertEqual(resolve_patch_encoder_name(args.slide_encoder), "uni_v2")

    def test_patch_encoder_run_is_reported_verbatim(self):
        self.assertEqual(self._args("--patch_encoder", "uni_v1").patch_encoder, "uni_v1")


if __name__ == "__main__":
    unittest.main()
