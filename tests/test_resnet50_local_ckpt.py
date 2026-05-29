import os
import tempfile
import unittest

import torch

from trident.patch_encoder_models.load import ResNet50InferenceEncoder

"""
Regression test for loading ResNet50 from a *local* checkpoint.

`_get_weights_path()` returns a plain `str`, so the encoder must not assume a
`pathlib.Path` (e.g. calling `weights_path.suffix`). This test exercises both the
`.safetensors` and the `torch.save` local-checkpoint branches and requires no
network access (timm builds ResNet50 with `pretrained=False`).
"""


class TestResNet50LocalCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import timm

        cls._ref_state = timm.create_model(
            "resnet50", pretrained=False, num_classes=0
        ).state_dict()

    def _load_with_local_path(self, path):
        orig = ResNet50InferenceEncoder._get_weights_path
        ResNet50InferenceEncoder._get_weights_path = lambda self: path
        try:
            encoder = ResNet50InferenceEncoder()
            encoder.eval()
            with torch.inference_mode():
                out = encoder(torch.randn(2, 3, 224, 224))
            self.assertEqual(out.shape[0], 2)
        finally:
            ResNet50InferenceEncoder._get_weights_path = orig

    def test_load_from_local_safetensors(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.safetensors")
            save_file(self._ref_state, path)
            self._load_with_local_path(path)

    def test_load_from_local_torch_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pytorch_model.bin")
            torch.save(self._ref_state, path)
            self._load_with_local_path(path)


if __name__ == "__main__":
    unittest.main()
