import unittest

import torch

from trident.patch_encoder_models.load import BasePatchEncoder

"""
Regression tests for #243: `encoder_factory(name, normalize=True)` must work for every
patch encoder, not only those whose `_build()` accepts a `normalize` argument.
These use toy encoders so they run without downloading weights.
"""


class _ScaledIdentityEncoder(BasePatchEncoder):
    def _build(self):
        self.enc_name = "toy"
        return torch.nn.Identity(), None, torch.float32

    def forward(self, x):
        return 3.0 * self.model(x)


class _NativeNormalizeEncoder(BasePatchEncoder):
    def _build(self, normalize=False):
        self.enc_name = "toy-native"
        self.native_normalize = normalize
        return torch.nn.Identity(), None, torch.float32


class TestPatchEncoderNormalize(unittest.TestCase):
    def test_normalize_true_l2_normalizes_overridden_forward(self):
        x = torch.randn(4, 16)
        z = _ScaledIdentityEncoder(normalize=True)(x)
        torch.testing.assert_close(z.norm(dim=-1), torch.ones(4))
        torch.testing.assert_close(z, torch.nn.functional.normalize(x, dim=-1))

    def test_normalize_default_leaves_output_unchanged(self):
        x = torch.randn(4, 16)
        torch.testing.assert_close(_ScaledIdentityEncoder()(x), 3.0 * x)

    def test_native_normalize_is_forwarded_without_hook(self):
        enc = _NativeNormalizeEncoder(normalize=True)
        self.assertTrue(enc.native_normalize)
        self.assertEqual(len(enc._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
