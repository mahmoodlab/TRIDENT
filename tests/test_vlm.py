"""
Unit tests for the vision-language model framework (task=vlm).

These tests are dependency-light: they exercise the model-agnostic plumbing
(`BaseVLM.generate` prompt broadcasting / alignment / trimming / stripping, the system
prompt, max_new_tokens routing, the registry/factory, and Patho-R1 variant validation)
WITHOUT downloading Patho-R1 or needing a slide. A tiny fake processor + model stand in
for the HF interface so the *real* generate() code path runs on CPU.
"""

import unittest
from types import SimpleNamespace

import torch
from PIL import Image

from trident.vlm_models import (
    vlm_registry,
    vlm_factory,
    BaseVLM,
    CustomInferenceVLM,
    PathoR1VLM,
)


# ---------------------------------------------------------------------------
# Fakes that implement just enough of the HF processor/model interface for the
# real BaseVLM.generate to run end-to-end on CPU.
# ---------------------------------------------------------------------------
class _FakeInputs(dict):
    """A dict that also supports `.to(device)` like a transformers BatchFeature."""
    def to(self, device):
        self._device = device
        return self


class _FakeProcessor:
    """
    Echoes prompts so tests can verify routing:
      * apply_chat_template -> records the messages, returns a marker string,
      * __call__            -> stores the per-row texts, returns fixed-width input_ids,
      * batch_decode        -> returns "  ANSWER::<prompt> <padding>  " per row.
    """
    PROMPT_LEN = 4

    def __init__(self):
        self.tokenizer = SimpleNamespace(padding_side="right")
        self.seen_messages = []
        self._texts = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.seen_messages.append(messages)
        user = [m for m in messages if m["role"] == "user"][0]
        text = [c["text"] for c in user["content"] if c.get("type") == "text"][0]
        return f"TEMPLATED::{text}"

    def __call__(self, text, images, padding=True, return_tensors="pt"):
        self._texts = list(text)
        bsz = len(text)
        input_ids = torch.zeros((bsz, self.PROMPT_LEN), dtype=torch.long)
        return _FakeInputs(input_ids=input_ids)

    def batch_decode(self, trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        # Reconstruct one answer per row from the prompt captured in __call__; surround with
        # whitespace so the test confirms generate() strips it.
        out = []
        for i in range(trimmed.shape[0]):
            prompt = self._texts[i].replace("TEMPLATED::", "")
            out.append(f"  ANSWER::{prompt}  ")
        return out


class _FakeModel(torch.nn.Module):
    """Minimal generative model: appends `max_new_tokens` tokens to the prompt ids."""
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)  # gives `.parameters()` a device to report
        self.last_max_new_tokens = None
        self.last_do_sample = None

    def generate(self, input_ids=None, max_new_tokens=None, do_sample=False, **kwargs):
        self.last_max_new_tokens = max_new_tokens
        self.last_do_sample = do_sample
        bsz, plen = input_ids.shape
        new = torch.arange(plen, plen + max_new_tokens).unsqueeze(0).repeat(bsz, 1)
        return torch.cat([input_ids, new], dim=1)


def _fake_vlm(system_prompt=None, max_new_tokens=8):
    proc = _FakeProcessor()
    model = _FakeModel()
    vlm = CustomInferenceVLM(
        vlm_name="fake_vlm", model=model, processor=proc,
        precision=torch.float32, max_new_tokens=max_new_tokens, system_prompt=system_prompt,
    )
    return vlm, proc, model


def _imgs(n):
    return [Image.new("RGB", (16, 16), color=(i * 10, 0, 0)) for i in range(n)]


class TestGenerateContract(unittest.TestCase):
    def test_single_prompt_broadcasts(self):
        vlm, _, _ = _fake_vlm()
        answers = vlm.generate(_imgs(3), "what is this?")
        self.assertEqual(len(answers), 3)
        for a in answers:
            self.assertEqual(a, "ANSWER::what is this?")  # stripped, prompt routed

    def test_prompt_list_alignment(self):
        vlm, _, _ = _fake_vlm()
        prompts = ["question A", "question B"]
        answers = vlm.generate(_imgs(2), prompts)
        self.assertEqual(answers, ["ANSWER::question A", "ANSWER::question B"])

    def test_mismatched_lengths_raise(self):
        vlm, _, _ = _fake_vlm()
        with self.assertRaises(ValueError):
            vlm.generate(_imgs(2), ["only one prompt"])

    def test_empty_returns_empty(self):
        vlm, proc, model = _fake_vlm()
        self.assertEqual(vlm.generate([], "anything"), [])
        # Short-circuits before touching the model/processor.
        self.assertEqual(proc.seen_messages, [])
        self.assertIsNone(model.last_max_new_tokens)

    def test_answers_are_stripped(self):
        vlm, _, _ = _fake_vlm()
        ans = vlm.generate(_imgs(1), "x")[0]
        self.assertEqual(ans, ans.strip())
        self.assertFalse(ans.startswith(" "))

    def test_system_prompt_prepended(self):
        vlm, proc, _ = _fake_vlm(system_prompt="You are a pathologist.")
        vlm.generate(_imgs(1), "describe")
        roles = [m["role"] for m in proc.seen_messages[0]]
        self.assertEqual(roles, ["system", "user"])
        self.assertEqual(proc.seen_messages[0][0]["content"], "You are a pathologist.")

    def test_no_system_prompt_by_default(self):
        vlm, proc, _ = _fake_vlm()
        vlm.generate(_imgs(1), "describe")
        roles = [m["role"] for m in proc.seen_messages[0]]
        self.assertEqual(roles, ["user"])

    def test_message_has_image_then_text(self):
        vlm, proc, _ = _fake_vlm()
        vlm.generate(_imgs(1), "describe")
        content = proc.seen_messages[0][0]["content"]
        self.assertEqual(content[0]["type"], "image")
        self.assertEqual(content[1], {"type": "text", "text": "describe"})

    def test_max_new_tokens_default_and_override(self):
        vlm, _, model = _fake_vlm(max_new_tokens=8)
        vlm.generate(_imgs(1), "x")
        self.assertEqual(model.last_max_new_tokens, 8)
        vlm.generate(_imgs(1), "x", max_new_tokens=3)
        self.assertEqual(model.last_max_new_tokens, 3)

    def test_greedy_decoding(self):
        vlm, _, model = _fake_vlm()
        vlm.generate(_imgs(1), "x")
        self.assertFalse(model.last_do_sample)  # deterministic by default


class TestRegistryFactory(unittest.TestCase):
    def test_registry_has_both_variants(self):
        self.assertIn("patho_r1_7b", vlm_registry)
        self.assertIn("patho_r1_3b", vlm_registry)

    def test_factory_unknown_raises(self):
        with self.assertRaises(ValueError):
            vlm_factory("does_not_exist")

    def test_registry_maps_to_basevlm_subclasses(self):
        for name, cls in vlm_registry.items():
            self.assertTrue(issubclass(cls, BaseVLM), name)
            self.assertTrue(issubclass(cls, PathoR1VLM), name)


class TestPathoR1Construction(unittest.TestCase):
    def test_invalid_variant_raises_without_download(self):
        # The variant check happens before super().__init__/_build, so no network needed.
        with self.assertRaises(ValueError):
            PathoR1VLM(variant="9b")

    def test_registry_classes_bind_variant(self):
        self.assertEqual(vlm_registry["patho_r1_7b"]._DEFAULT_VARIANT, "7b")
        self.assertEqual(vlm_registry["patho_r1_3b"]._DEFAULT_VARIANT, "3b")

    def test_repo_ids_resolved(self):
        self.assertEqual(PathoR1VLM._HF_REPOS["7b"], "WenchuanZhang/Patho-R1-7B")
        self.assertEqual(PathoR1VLM._HF_REPOS["3b"], "WenchuanZhang/Patho-R1-3B")


if __name__ == "__main__":
    unittest.main()
