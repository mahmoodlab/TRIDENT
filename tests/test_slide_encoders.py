import unittest
import torch

import sys; sys.path.append('../')

# New imports to test 
from trident.slide_encoder_models.load import *

"""
Test the forward pass of the slide encoders.
"""

class TestSlideEncoders(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _test_encoder_forward(self, encoder, batch, expected_precision):
        print("\033[95m" + f"Testing {encoder.__class__.__name__} forward pass" + "\033[0m")
        encoder = encoder.to(self.device)
        encoder.eval()
        self.assertEqual(encoder.precision, expected_precision)
        self.assertTrue(hasattr(encoder, 'model'))

        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=encoder.precision):
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
        
    def test_gigapath_encoder_initialization(self):
        sample_batch = {
            'features': torch.randn(1, 100, 1536),
            'coords': torch.randn(1, 100, 2),
            'attributes': {'patch_size_level0': 224}
        }
        self._test_encoder_forward(GigaPathSlideEncoder(), sample_batch, torch.float16)

    def test_slide_encoder_factory_with_valid_names(self):
        print("\033[95m" + "Testing Slide Encoder Factory with valid names" + "\033[0m")
        # Test factory method for valid model names
        for model_name, expected_class in [
            ('mean-conch_v15', MeanSlideEncoder),
            ('mean-blahblah', MeanSlideEncoder),
            ('prism', PRISMSlideEncoder),
            ('chief', CHIEFSlideEncoder),
            ('gigapath', GigaPathSlideEncoder),
            ('titan', TitanSlideEncoder),
            ('madeleine', MadeleineSlideEncoder),
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


if __name__ == "__main__":
    unittest.main()
