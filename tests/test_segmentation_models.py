import torch
import numpy as np 
from PIL import Image
import unittest

try:
    import lovely_tensors; lovely_tensors.monkey_patch()
except:
    pass

import sys; sys.path.append('../')
from trident.segmentation_models import segmentation_model_factory 

"""
Test forward pass of the segmentation model(s).
"""

class TestSegmentationModels(unittest.TestCase):

    def setUp(self):
        pass

    def _test_forward(self, encoder_name):
        print("\033[95m" + f"Testing {encoder_name} forward pass" + "\033[0m")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        encoder = segmentation_model_factory(encoder_name).to(device)

        self.dummy_image = np.random.randint(0, 256, (encoder.input_size, encoder.input_size, 3), dtype=np.uint8)
        self.dummy_image = Image.fromarray(self.dummy_image)

        with torch.inference_mode():
            dummy_input = encoder.eval_transforms(self.dummy_image).unsqueeze(dim=0).to(device)
            output = encoder(dummy_input)

        self.assertIsNotNone(output)
        self.assertIsInstance(output, torch.Tensor)
        print("\033[94m"+ f"    {encoder_name} forward pass success with output {output}" + "\033[0m")

    def test_hest(self):
        self._test_forward('hest')
        
    def test_grandqc(self):
        self._test_forward('grandqc')

if __name__ == '__main__':
    unittest.main()
