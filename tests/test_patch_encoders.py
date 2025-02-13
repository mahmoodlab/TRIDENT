import torch
import numpy as np 
from PIL import Image
import unittest
try:
    import lovely_tensors; lovely_tensors.monkey_patch()
except:
    pass

import sys; sys.path.append('../')
from trident.patch_encoder_models.load import encoder_factory 

"""
Test forward pass of patch encoders
"""

class TestPatchEncoders(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dummy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        self.dummy_image = Image.fromarray(self.dummy_image)

    def _test_encoder_forward(self, encoder_name, **kwargs):
        print("\033[95m" + f"Testing {encoder_name} forward pass" + "\033[0m")
        if kwargs:
            print("\033[92m" + f"    With kwargs: {kwargs}" + "\033[0m")
        encoder = encoder_factory(encoder_name, **kwargs)
        encoder = encoder.to(self.device)
        encoder.eval()

        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=encoder.precision):
            dummy_input = encoder.eval_transforms(self.dummy_image).to(self.device).unsqueeze(dim=0)
            output = encoder(dummy_input)

        self.assertIsNotNone(output)
        self.assertIsInstance(output, torch.Tensor)
        print("\033[94m"+ f"    {encoder_name} forward pass success with output {output}" + "\033[0m")

    def test_conch_v1_forward(self):
        self._test_encoder_forward('conch_v1', with_proj = True, normalize = True)
        self._test_encoder_forward('conch_v1', with_proj = False, normalize = True)
        self._test_encoder_forward('conch_v1', with_proj = True, normalize = False)
        self._test_encoder_forward('conch_v1', with_proj = False, normalize = False)
        
    def test_conch_v15_forward(self):
        self._test_encoder_forward('conch_v15')

    def test_uni_v1_forward(self):
        self._test_encoder_forward('uni_v1')
        
    def test_uni_v2_forward(self):
        self._test_encoder_forward('uni_v2')

    def test_ctranspath_forward(self):
        self._test_encoder_forward('ctranspath')

    def test_phikon_forward(self):
        self._test_encoder_forward('phikon')
    
    def test_phikon_v2_forward(self):
        self._test_encoder_forward('phikon_v2')

    def test_resnet50_forward(self):
        self._test_encoder_forward('resnet50')

    def test_gigapath_forward(self):
        self._test_encoder_forward('gigapath')

    def test_virchow_forward(self):
        self._test_encoder_forward('virchow')

    def test_virchow2_forward(self):
        self._test_encoder_forward('virchow2')

    def test_hoptimus0_forward(self):
        self._test_encoder_forward('hoptimus0')
        
    def test_musk_forward(self):
        self._test_encoder_forward('musk')
    
    def test_hibou_l_forward(self):
        self._test_encoder_forward('hibou_l')
    
    def test_kaiko_forward(self):
        self._test_encoder_forward('kaiko-vits8')
        self._test_encoder_forward('kaiko-vits16')
        self._test_encoder_forward('kaiko-vitb8')
        self._test_encoder_forward('kaiko-vitb16')
        self._test_encoder_forward('kaiko-vitl14')
        
    def test_lunitvits8_forward(self):
        self._test_encoder_forward('lunit-vits8')
    
    

if __name__ == '__main__':
    unittest.main()
