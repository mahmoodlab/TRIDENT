import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

import run_batch_of_slides as batch_mod
from trident.patch_encoder_models import (
    DigepathInferenceEncoder,
    RESIZE_SUPPORTED_PATCH_ENCODERS,
    encoder_factory,
    encoder_registry,
)
from trident.patch_encoder_models.load import BasePatchEncoder


class TestDigepathEncoder(unittest.TestCase):
    def test_public_registration(self):
        self.assertIs(encoder_registry["digepath"], DigepathInferenceEncoder)
        self.assertIn("digepath", RESIZE_SUPPORTED_PATCH_ENCODERS)

    def test_batch_cli_accepts_digepath(self):
        args = batch_mod.build_parser().parse_args([
            "--job_dir", "job",
            "--wsi_dir", "wsis",
            "--patch_encoder", "digepath",
        ])
        self.assertEqual(args.patch_encoder, "digepath")

    @patch.object(BasePatchEncoder, "_has_internet", True)
    @patch("timm.create_model")
    def test_hub_loading_uses_official_model(self, create_model):
        model = MagicMock()
        create_model.return_value = model

        encoder = encoder_factory("digepath")

        create_model.assert_called_once_with(
            "hf_hub:xtxx/Digepath",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        self.assertIs(encoder.model, model)
        self.assertEqual(encoder.precision, torch.float16)

    @patch("safetensors.torch.load_file")
    @patch("timm.create_model")
    def test_local_safetensors_loads_strictly_on_native_grid(self, create_model, load_file):
        model = MagicMock()
        state_dict = {"weight": torch.ones(1)}
        create_model.return_value = model
        load_file.return_value = state_dict

        with tempfile.NamedTemporaryFile(suffix=".safetensors") as checkpoint:
            encoder = encoder_factory(
                "digepath",
                weights_path=checkpoint.name,
                target_img_size=448,
            )

        create_model.assert_called_once_with(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        load_file.assert_called_once_with(checkpoint.name, device="cpu")
        model.load_state_dict.assert_called_once_with(state_dict, strict=True)

        transformed = encoder.eval_transforms(Image.new("RGB", (256, 256)))
        self.assertEqual(tuple(transformed.shape), (3, 448, 448))
        normalize = encoder.eval_transforms.transforms[-1]
        self.assertEqual(tuple(normalize.mean), (0.485, 0.456, 0.406))
        self.assertEqual(tuple(normalize.std), (0.229, 0.224, 0.225))

    def test_rejects_non_multiple_input_size_before_loading(self):
        with self.assertRaisesRegex(AssertionError, "multiple.*16"):
            encoder_factory("digepath", target_img_size=225)

    def test_missing_local_checkpoint_is_clear(self):
        missing = Path(tempfile.gettempdir()) / "missing-digepath-model.safetensors"
        with self.assertRaisesRegex(FileNotFoundError, "Expected checkpoint file"):
            encoder_factory("digepath", weights_path=str(missing))


if __name__ == "__main__":
    unittest.main()
