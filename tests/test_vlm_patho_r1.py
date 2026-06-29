"""
Real-model integration test for Patho-R1 through the TRIDENT vlm path.

This test loads the actual Patho-R1 (Qwen2.5-VL) weights and runs a real generation. It
is skipped automatically unless ALL of:
    * `transformers` exposes `Qwen2_5_VLForConditionalGeneration`,
    * a CUDA device is available (a 7B VLM on CPU is impractically slow),
    * a bundled test slide is present under `wsis/`, and
    * the env var `TRIDENT_TEST_VLM_REAL=1` is set (the weights are a multi-GB download,
      gated by a non-commercial license — opt in explicitly).

By default it exercises the lighter 3B variant; override with `TRIDENT_TEST_VLM_MODEL`
(e.g. `patho_r1_7b`).

Run it:
    TRIDENT_TEST_VLM_REAL=1 pytest tests/test_vlm_patho_r1.py -q -s
"""

import importlib.util
import os
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WSI_DIR = os.path.join(REPO_ROOT, "wsis")
_SLIDE_EXTS = (".tif", ".tiff", ".svs", ".ndpi")

_HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
_HAS_QWEN25VL = False
if _HAS_TRANSFORMERS:
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
        _HAS_QWEN25VL = True
    except Exception:
        _HAS_QWEN25VL = False

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False

_OPT_IN = os.environ.get("TRIDENT_TEST_VLM_REAL") == "1"
_MODEL = os.environ.get("TRIDENT_TEST_VLM_MODEL", "patho_r1_3b")


def _find_slide_dir():
    if not os.path.isdir(WSI_DIR):
        return None
    for f in os.listdir(WSI_DIR):
        if f.lower().endswith(_SLIDE_EXTS):
            return WSI_DIR
    return None


@unittest.skipUnless(_OPT_IN, "set TRIDENT_TEST_VLM_REAL=1 to run the real Patho-R1 download")
@unittest.skipUnless(_HAS_QWEN25VL, "transformers without Qwen2.5-VL support")
@unittest.skipUnless(_HAS_CUDA, "CUDA not available (a 7B VLM on CPU is too slow)")
@unittest.skipUnless(_find_slide_dir(), "No bundled test slide under wsis/")
class TestPathoR1Real(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from trident.vlm_models import vlm_factory
        cls.vlm = vlm_factory(_MODEL, max_new_tokens=64)
        cls.vlm.to("cuda:0").eval()
        cls.slide_dir = _find_slide_dir()
        cls.slide_file = [f for f in os.listdir(cls.slide_dir)
                          if f.lower().endswith(_SLIDE_EXTS)][0]

    def test_processor_and_left_padding(self):
        # Build assumptions: name set, processor present, left padding configured for
        # correct batched decoder-only generation.
        self.assertTrue(self.vlm.vlm_name.startswith("patho_r1_"))
        self.assertIsNotNone(self.vlm.processor)
        self.assertEqual(self.vlm.processor.tokenizer.padding_side, "left")

    def test_generate_on_real_image(self):
        from PIL import Image
        imgs = [Image.new("RGB", (224, 224), color=(180, 120, 150)) for _ in range(2)]
        answers = self.vlm.generate(imgs, "Describe this image briefly.")
        self.assertEqual(len(answers), 2)
        for a in answers:
            self.assertIsInstance(a, str)
            self.assertGreater(len(a.strip()), 0)

    def test_query_region_on_real_slide(self):
        from trident import load_wsi
        with load_wsi(slide_path=os.path.join(self.slide_dir, self.slide_file),
                      lazy_init=False) as s:
            answer = s.query_region(
                self.vlm, "What tissue is shown? Answer briefly.",
                location=(0, 0), size=256, mag=5,
            )
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer.strip()), 0)


if __name__ == "__main__":
    unittest.main()
