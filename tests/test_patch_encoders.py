import torch
import numpy as np 
from PIL import Image
import unittest
try:
    import lovely_tensors; lovely_tensors.monkey_patch()
except:
    pass

import sys; sys.path.append('../')
from trident.patch_encoder_models import * 
from tests._test_gating import RUN_INTEGRATION_TESTS

"""
Test forward pass of patch encoders
"""

@unittest.skipUnless(
    RUN_INTEGRATION_TESTS,
    "Set TRIDENT_RUN_INTEGRATION_TESTS=1 to run heavy integration tests.",
)
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

        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device_type,
            dtype=encoder.precision,
            enabled=(self.device.type == "cuda"),
        ):
            dummy_input = encoder.eval_transforms(self.dummy_image).to(self.device).unsqueeze(dim=0)
            output = encoder(dummy_input)

        self.assertIsNotNone(output)
        self.assertIsInstance(output, torch.Tensor)
        print("\033[94m"+ f"    {encoder_name} forward pass success with output {output}" + "\033[0m")
        return output

    def _output_dim(self, encoder_name, **kwargs):
        """Run a forward pass and return the embedding dimension (last axis)."""
        output = self._test_encoder_forward(encoder_name, **kwargs)
        return output.shape[-1]

    def _test_encoder_resize(self, encoder_name, target_img_size):
        """
        Forward pass at a custom `target_img_size`, and assert the output
        embedding dimension is unchanged vs. the native resolution. The encoders
        pool over patch tokens, so a different input resolution changes the
        number of tokens but must NOT change the feature dimensionality
        (otherwise downstream feature files become incompatible).
        """
        native_dim = self._output_dim(encoder_name)
        resized_dim = self._output_dim(encoder_name, target_img_size=target_img_size)
        self.assertEqual(
            native_dim, resized_dim,
            f"{encoder_name}: embedding dim changed with target_img_size="
            f"{target_img_size} ({native_dim} -> {resized_dim})",
        )

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

    def test_hoptimus1_forward(self):
        self._test_encoder_forward('hoptimus1')
    
    def test_h0mini_forward(self):
        self._test_encoder_forward('h0-mini')
        self._test_encoder_forward('h0-mini', return_type="cls+mean")

    def test_musk_forward(self):
        self._test_encoder_forward('musk')
    
    def test_openmidnight_forward(self):
        self._test_encoder_forward('openmidnight')
    
    def test_gpfm_forward(self):
        self._test_encoder_forward('gpfm')
    
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
    
    def test_midnight12k_forward(self):
        self._test_encoder_forward('midnight12k')
        self._test_encoder_forward('midnight12k', return_type="cls+mean")

    def test_genbio_pathfm_forward(self):
        self._test_encoder_forward('genbio-pathfm')

    def test_gemma4_forward(self):
        self._test_encoder_forward('gemma4-e4b')
        self._test_encoder_forward('gemma4-26b')

    def test_gemma4_shape_and_batch(self):
        # Regression guard: pooling must reduce over tokens (not features) and
        # the encoder must accept batched input (TRIDENT extracts patches in batches).
        encoder = encoder_factory('gemma4-26b').to(self.device).eval()
        x = encoder.eval_transforms(self.dummy_image).to(self.device)
        with torch.inference_mode():
            out_b1 = encoder(x.unsqueeze(0))
            out_b2 = encoder(torch.stack([x, x]))
        self.assertEqual(tuple(out_b1.shape), (1, 1152))
        self.assertEqual(tuple(out_b2.shape), (2, 1152))

    # ------------------------------------------------------------------
    # Configurable input resolution (`target_img_size` / dynamic_img_size).
    # One representative encoder per distinct `_build` branch is exercised.
    # target_img_size must be a multiple of each model's patch size.
    # ------------------------------------------------------------------
    def test_uni_v1_resize(self):
        # patch_size 16 -> 448 is a multiple. Plain Resize/CenterCrop transform.
        self._test_encoder_resize('uni_v1', target_img_size=448)

    def test_uni_v2_resize(self):
        # patch_size 14 -> 448 is a multiple.
        self._test_encoder_resize('uni_v2', target_img_size=448)

    def test_virchow_resize(self):
        self._test_encoder_resize('virchow', target_img_size=448)

    def test_virchow2_resize(self):
        # Exercises the reg-token (output[:, 5:]) pooling path.
        self._test_encoder_resize('virchow2', target_img_size=448)

    def test_gigapath_resize(self):
        # Exercises the branched eval-transform (target_img_size is not None).
        self._test_encoder_resize('gigapath', target_img_size=448)

    def test_hoptimus0_resize(self):
        # Backbone default was dynamic_img_size=False; now flipped to True.
        self._test_encoder_resize('hoptimus0', target_img_size=448)

    def test_h0mini_resize(self):
        # Exercises the resolve_model_data_config rewrite path + num_prefix_tokens.
        self._test_encoder_resize('h0-mini', target_img_size=448)

    def test_gpfm_resize(self):
        self._test_encoder_resize('gpfm', target_img_size=448)

    def test_lunitvits8_resize(self):
        # patch_size 8; exercises the data_config crop_pct=1.0 rewrite path.
        self._test_encoder_resize('lunit-vits8', target_img_size=448)

    def test_kaiko_vitl14_resize(self):
        # Special case: backbone built at 518, eval transform at 224 by default.
        # patch_size 14 -> 448 is a multiple.
        self._test_encoder_resize('kaiko-vitl14', target_img_size=448)

    def test_kaiko_vits8_resize(self):
        # patch_size 8 -> 256 is a multiple.
        self._test_encoder_resize('kaiko-vits8', target_img_size=256)

    def test_resize_rejects_non_multiple(self):
        # 500 is not a multiple of UNI-v2's patch size (14): must fail loudly.
        with self.assertRaises(AssertionError):
            encoder_factory('uni_v2', target_img_size=500)

    def test_resize_rejects_unsupported_encoder_path(self):
        # conch_v15 has no target_img_size kwarg in its _build -> TypeError.
        with self.assertRaises(TypeError):
            encoder_factory('conch_v15', target_img_size=448)

if __name__ == '__main__':
    unittest.main()