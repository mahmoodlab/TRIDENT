import unittest
import torch

import sys; sys.path.append('../')
from trident.slide_encoder_models import *
from tests._test_gating import RUN_INTEGRATION_TESTS

"""
Test the forward pass of the slide encoders.
"""


def _has_prism2_decoder() -> bool:
    """PRISM2's 'diagnostic' embedding needs transformers 4.51.x -- see PRISM2SlideEncoder._build."""
    import transformers
    from packaging.version import Version
    return Version('4.51') <= Version(transformers.__version__) < Version('4.52')


PRISM2_DECODER_AVAILABLE = _has_prism2_decoder()

@unittest.skipUnless(
    RUN_INTEGRATION_TESTS,
    "Set TRIDENT_RUN_INTEGRATION_TESTS=1 to run heavy integration tests.",
)
class TestSlideEncoders(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _test_encoder_forward(self, encoder, batch, expected_precision):
        print("\033[95m" + f"Testing {encoder.__class__.__name__} forward pass" + "\033[0m")
        encoder = encoder.to(self.device)
        encoder.eval()
        self.assertEqual(encoder.precision, expected_precision)
        self.assertTrue(hasattr(encoder, 'model'))

        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device_type,
            dtype=encoder.precision,
            enabled=(self.device.type == "cuda"),
        ):
            output = encoder.forward(batch, device=self.device)

        self.assertIsNotNone(output)
        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(output.shape[-1] == encoder.embedding_dim)
        print("\033[94m"+ f"    {encoder.__class__.__name__} forward pass success with output shape {output.shape}" + "\033[0m")

    def test_prism_encoder_initialization(self):
        sample_batch = {
            'features': torch.randn(1, 100, 2560),
            'coords': torch.randn(1, 100, 2),
        }
        self._test_encoder_forward(PRISMSlideEncoder(), sample_batch, torch.float16)

    def test_prism2_encoder_initialization(self):
        # PRISM2 consumes class-token-only Virchow2 features (1280-dim).
        sample_batch = {
            'features': torch.randn(1, 100, 1280),
            'coords': torch.randn(1, 100, 2),
        }
        self._test_encoder_forward(PRISM2SlideEncoder(), sample_batch, torch.bfloat16)

    @unittest.skipUnless(PRISM2_DECODER_AVAILABLE,
                         "PRISM2's diagnostic embedding needs transformers 4.51.x (see README).")
    def test_prism2_diagnostic_embedding(self):
        sample_batch = {
            'features': torch.randn(1, 100, 1280),
            'coords': torch.randn(1, 100, 2),
        }
        encoder = PRISM2SlideEncoder(embedding_type='diagnostic')
        self.assertEqual(encoder.embedding_dim, 3072)
        self._test_encoder_forward(encoder, sample_batch, torch.bfloat16)

    def test_prism2_diagnostic_rejects_unsupported_transformers(self):
        # Must fail at build time, not deep inside the remote code's forward pass.
        if PRISM2_DECODER_AVAILABLE:
            self.skipTest("transformers 4.51.x supports the decoder; nothing to reject.")
        with self.assertRaises(Exception) as ctx:
            PRISM2SlideEncoder(embedding_type='diagnostic')
        self.assertIn('4.51', str(ctx.exception))

    def test_chief_encoder_initialization(self):
        sample_batch = {
            'features': torch.randn(1, 100, 768),
        }
        self._test_encoder_forward(CHIEFSlideEncoder(), sample_batch, torch.float32)

    def test_titan_encoder_initialization(self):
        sample_batch = {
            'features': torch.randn(1, 100, 768),
            'coords': torch.randint(0, 4096, (1, 100, 2)),
            'attributes': {'patch_size_level0': 512}
        }
        self._test_encoder_forward(TitanSlideEncoder(), sample_batch, torch.float16)
        
    def test_care_encoder_initialization(self):
        sample_batch = {
            'features': torch.randn(1, 100, 768),
            'coords': torch.randint(0, 512 * 100, (1, 100, 2)),
            'attributes': {'patch_size_level0': 512}
        }

        self._test_encoder_forward(CARESlideEncoder(), sample_batch, torch.float32)

    def test_gigapath_encoder_initialization(self):
        sample_batch = {
            'features': torch.randn(1, 100, 1536),
            'coords': torch.randn(1, 100, 2),
            'attributes': {'patch_size_level0': 224}
        }
        self._test_encoder_forward(GigaPathSlideEncoder(), sample_batch, torch.float16)

    def test_gigapath_flash_encoder_initialization(self):
        sample_batch = {
            'features': torch.randn(1, 100, 384),
            'coords': torch.randn(1, 100, 2),
            'attributes': {'patch_size_level0': 224}
        }
        self._test_encoder_forward(GigaPathFlashSlideEncoder(), sample_batch, torch.float16)

    def test_slide_encoder_factory_with_valid_names(self):
        print("\033[95m" + "Testing Slide Encoder Factory with valid names" + "\033[0m")
        # Keep this focused on dependency-light encoders.
        # Other encoder backends are tested in dedicated integration-style tests.
        for model_name, expected_class in [
            ('mean-conch_v15', MeanSlideEncoder),
            ('mean-uni_v1', MeanSlideEncoder),
        ]:
            encoder = encoder_factory(model_name)
            self.assertIsInstance(encoder, expected_class)

    def test_madeleine_encoder_initialization(self):
        sample_batch = {
            'features': torch.randn(1, 100, 512),
        }
        self._test_encoder_forward(MadeleineSlideEncoder(), sample_batch, torch.bfloat16)

    def test_slide_encoder_factory_invalid_name(self):
        print("\033[95m" + "Testing Slide Encoder Factory with invalid names" + "\033[0m")
        with self.assertRaises(ValueError):
            encoder_factory('invalid-model')
        with self.assertRaises((ValueError, KeyError)):
            encoder_factory('mean-blahblah')


if __name__ == "__main__":
    unittest.main()
