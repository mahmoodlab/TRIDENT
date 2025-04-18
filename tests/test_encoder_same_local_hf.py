import torch
import numpy as np 
from PIL import Image
import unittest
import json
from pathlib import Path

try:
    import lovely_tensors; lovely_tensors.monkey_patch()
except:
    pass

import sys; sys.path.append('../')
from trident.patch_encoder_models import *


class TestEncoderConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.dummy_image = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))

    def _load_encoder(self, encoder_name, source, weights_path=None, **kwargs):
        print(f"  🔧 Loading {encoder_name} ({source})")
        encoder = encoder_factory(encoder_name, weights_path=weights_path, **kwargs)
        encoder = encoder.to(self.device)
        encoder.eval()
        return encoder

    def _run_forward(self, encoder, encoder_name, source):
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=encoder.precision):
            dummy_input = encoder.eval_transforms(self.dummy_image).to(self.device).unsqueeze(dim=0)
            output = encoder(dummy_input)
        print(f"  📐 Output shape from {source}: {tuple(output.shape)}")
        return output

    def _compare_architecture(self, enc1, enc2):
        keys1 = set(enc1.state_dict().keys())
        keys2 = set(enc2.state_dict().keys())
        if keys1 != keys2:
            print("\033[1;33m⚠️ Architecture mismatch in keys:\033[0m")
            print("  Only in default :", keys1 - keys2)
            print("  Only in local   :", keys2 - keys1)
            return False
        return True

    def _compare_weights(self, enc1, enc2):
        diffs = []
        for k in enc1.state_dict().keys():
            w1 = enc1.state_dict()[k]
            w2 = enc2.state_dict()[k]
            if not torch.allclose(w1, w2, atol=1e-5, rtol=1e-4):
                abs_diff = (w1 - w2).abs()
                max_diff = abs_diff.max().item()
                mean_diff = abs_diff.mean().item()
                diffs.append((k, max_diff, mean_diff))
        if diffs:
            print("\033[1;33m⚠️ Weight differences found:\033[0m")
            for k, max_d, mean_d in sorted(diffs, key=lambda x: -x[1])[:10]:
                print(f"    🔍 {k:<50} max diff: {max_d:.4e}, mean diff: {mean_d:.4e}")
            return False
        return True


def generate_encoder_test(encoder_name, weights_path, **kwargs):
    def test(self):
        header = f"🧪 TEST: {encoder_name}"
        if kwargs:
            kwarg_str = ', '.join(f"{k}={v}" for k, v in kwargs.items())
            header += f" ({kwarg_str})"
        print(f"\n\033[1;36m{'=' * len(header)}\n{header}\n{'=' * len(header)}\033[0m")

        # Load models
        enc_default = self._load_encoder(encoder_name, source="default", **kwargs)
        enc_local = self._load_encoder(encoder_name, source="local checkpoint", weights_path=weights_path, **kwargs)

        # # Compare architecture
        # arch_match = self._compare_architecture(enc_default, enc_local)
        # self.assertTrue(arch_match, f"Architecture mismatch in {encoder_name}")

        # # Compare weights
        # weights_match = self._compare_weights(enc_default, enc_local)
        # self.assertTrue(weights_match, f"Weight mismatch in {encoder_name}")

        # Run inference
        out_default = self._run_forward(enc_default, encoder_name, source="default")
        out_local = self._run_forward(enc_local, encoder_name, source="local checkpoint")

        if torch.allclose(out_default, out_local, atol=1e-5, rtol=1e-4):
            print(f"\033[1;32m✅ Outputs match for {encoder_name}\033[0m")
        else:
            diff = (out_default - out_local).abs().max().item()
            print(f"\033[1;31m❌ Outputs do NOT match (max abs diff = {diff:.4e})\033[0m")
            self.fail(f"Output mismatch for {encoder_name} with kwargs={kwargs}")
    return test


# Dynamically register tests before unittest.main()
def register_tests():
    ckpt_path = Path('../trident/patch_encoder_models/local_ckpts_guillaume.json')
    with open(ckpt_path) as f:
        local_ckpts = json.load(f)

    # local ckpt not supported
    local_ckpts.pop('musk')
    local_ckpts.pop('custom_encoder')
    local_ckpts.pop('hibou_l')

    for encoder_name, path in local_ckpts.items():
        test_name = f"test_{encoder_name}"
        test_fn = generate_encoder_test(encoder_name, path)
        setattr(TestEncoderConsistency, test_name, test_fn)


register_tests()

if __name__ == '__main__':
    unittest.main()
