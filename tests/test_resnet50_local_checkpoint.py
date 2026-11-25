"""
Regression test for #172: a local ResNet50 checkpoint stored as `.safetensors` (the format
timm / huggingface-cli download by default) must load when `weights_path` is a plain string,
instead of failing inside `_build` and surfacing the misleading "download pytorch_model.bin" error.

Fast tier: CPU only, no network, no model download (timm builds an untrained resnet50).
"""
import os
import tempfile
import unittest

import torch

from trident.patch_encoder_models import encoder_factory


class TestResNet50LocalCheckpoint(unittest.TestCase):

    # Mirrors the defaults of ResNet50InferenceEncoder._build so state-dict keys line up.
    TIMM_KWARGS = {"features_only": True, "out_indices": [3], "num_classes": 0}
    PROBE_VALUE = 0.123

    def setUp(self):
        import timm

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        model = timm.create_model("resnet50", pretrained=False, **self.TIMM_KWARGS)
        self.state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # Stamp one tensor so the test proves the file was really loaded, rather than
        # silently skipped by `load_state_dict(..., strict=False)`.
        self.probe_key = next(k for k in self.state_dict if k.endswith("conv1.weight"))
        self.state_dict[self.probe_key].fill_(self.PROBE_VALUE)

    def _assert_checkpoint_loaded(self, encoder):
        loaded = encoder.model.state_dict()[self.probe_key]
        expected = torch.full_like(loaded, self.PROBE_VALUE)
        self.assertTrue(
            torch.allclose(loaded, expected),
            "checkpoint tensors were not loaded into the encoder",
        )

    def test_safetensors_checkpoint_from_str_path(self):
        from safetensors.torch import save_file

        path = os.path.join(self.tmp.name, "model.safetensors")
        save_file(self.state_dict, path)

        # `weights_path` is a str here, exactly as it comes from local_ckpts.json / the CLI.
        encoder = encoder_factory("resnet50", weights_path=path)
        self._assert_checkpoint_loaded(encoder)

    def test_torch_checkpoint_from_str_path(self):
        path = os.path.join(self.tmp.name, "pytorch_model.bin")
        torch.save(self.state_dict, path)

        encoder = encoder_factory("resnet50", weights_path=path)
        self._assert_checkpoint_loaded(encoder)


if __name__ == "__main__":
    unittest.main()
