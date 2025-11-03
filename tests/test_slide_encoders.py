import unittest
import torch

import sys; sys.path.append('../')
from trident.slide_encoder_models import *
from trident.slide_encoder_models.load import CustomSlideEncoder

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

    def test_prism_encoder_concatenation(self):
        """
        Test that PRISM encoder properly concatenates image_embedding and image_latents.
        This test uses a mock model to verify the concatenation logic without needing
        the actual PRISM model weights.
        """
        print("\033[95m" + "Testing PRISM encoder concatenation of image_embedding and image_latents" + "\033[0m")
        
        # Create a mock PRISM model
        class MockPRISMModel:
            def __init__(self):
                self.text_decoder = None
            
            def slide_representations(self, x):
                """Mock method that returns expected dict structure"""
                batch_size = x.shape[0]
                return {
                    'image_embedding': torch.randn(batch_size, 1280),
                    'image_latents': torch.randn(batch_size, 512, 1280)
                }
            
            def parameters(self):
                return []
            
            def eval(self):
                pass
        
        # Create custom encoder with mock model
        encoder = CustomSlideEncoder(
            enc_name='prism',
            model=MockPRISMModel(),
            precision=torch.float16,
            embedding_dim=1280
        )
        
        # Use PRISM's forward method
        encoder.forward = lambda batch, device='cuda': PRISMSlideEncoder.forward(encoder, batch, device)
        
        # Create sample batch
        sample_batch = {
            'features': torch.randn(1, 100, 2560),
            'coords': torch.randn(1, 100, 2),
        }
        
        # Run forward pass
        output = encoder.forward(sample_batch, device='cpu')
        
        # Verify output shape
        self.assertEqual(output.shape[0], 1, "Batch dimension should be 1")
        self.assertEqual(output.shape[1], 513, "Should have 513 embeddings (1 image_embedding + 512 image_latents)")
        self.assertEqual(output.shape[2], 1280, "Feature dimension should be 1280")
        self.assertEqual(output.shape[-1], encoder.embedding_dim, "Last dimension should match embedding_dim")
        
        print("\033[94m" + f"    PRISM concatenation test success with output shape {output.shape}" + "\033[0m")
        print("\033[94m" + f"    ✓ Concatenated 1 image_embedding + 512 image_latents = 513 total embeddings" + "\033[0m")


if __name__ == "__main__":
    unittest.main()
