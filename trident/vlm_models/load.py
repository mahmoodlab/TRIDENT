"""
Vision-language models (VLMs) for TRIDENT's ``vlm`` task — *interrogating* ROIs with
free-text prompts instead of producing fixed embeddings or cell polygons.

To add a model: subclass ``BaseVLM``, implement ``_build()`` (return
``(model, processor, precision)``), and rely on the default ``generate`` (which drives
any HF chat-template image-text-to-text model) or override it. Register the class in
``vlm_registry`` at the bottom of this file.

Upstream models wrapped here
----------------------------
* **Patho-R1** — Zhang et al., *"Patho-R1: A Multimodal Reinforcement Learning-Based
  Pathology Expert Reasoner"*, arXiv:2505.11404 (2025). A Qwen2.5-VL-based pathology
  reasoning VLM. Weights: https://huggingface.co/WenchuanZhang/Patho-R1-7B (and -3B),
  license **CC-BY-NC-ND-4.0** (non-commercial research use only).
"""

from typing import Dict, Any, Tuple, Callable, Optional, List, Union
from abc import abstractmethod

import torch
from PIL import Image


def vlm_factory(model_name: str, **kwargs) -> "BaseVLM":
    """
    Instantiate a vision-language model by name.

    Parameters:
        model_name (str): One of the keys in ``vlm_registry``.
        **kwargs: Forwarded to the model constructor (e.g. ``weights_path``,
            ``max_new_tokens``, ``device_map``).

    Returns:
        BaseVLM: The instantiated model wrapper.

    Raises:
        ValueError: If ``model_name`` is not registered.
    """
    if model_name in vlm_registry:
        return vlm_registry[model_name](**kwargs)
    raise ValueError(
        f"Unknown VLM '{model_name}'. Available: {sorted(vlm_registry.keys())}"
    )


class BaseVLM(torch.nn.Module):
    """
    Base wrapper for a generative vision-language model.

    Attributes:
        vlm_name (Optional[str]): Unique model identifier (used for output dir names).
        processor (Callable): Upstream processor / feature-extractor + tokenizer; owns
            image preprocessing and the chat template (taken verbatim from upstream).
        precision (torch.dtype): Weight / autocast precision used during generation.
        max_new_tokens (int): Default generation length cap.
        system_prompt (Optional[str]): Optional system message prepended to every query.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        max_new_tokens: int = 512,
        system_prompt: Optional[str] = None,
        **build_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.vlm_name: Optional[str] = None
        self.weights_path: Optional[str] = weights_path
        self.max_new_tokens: int = max_new_tokens
        self.system_prompt: Optional[str] = system_prompt
        self.model, self.processor, self.precision = self._build(**build_kwargs)

    @abstractmethod
    def _build(self, **build_kwargs: Dict[str, Any]) -> Tuple[torch.nn.Module, Callable, torch.dtype]:
        """Return ``(model, processor, precision)``."""
        pass

    def _build_messages(self, prompt: str) -> List[dict]:
        """Assemble a single-turn chat message with one image placeholder + the prompt."""
        messages: List[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        })
        return messages

    @torch.inference_mode()
    def generate(
        self,
        images: List[Image.Image],
        prompts: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """
        Answer ``prompts`` about ``images`` (one prompt per image, or a single prompt
        broadcast to all). Default implementation drives any HF chat-template
        image-text-to-text model (Qwen2.5-VL, LLaVA, …); override for bespoke APIs.

        Parameters:
            images (List[PIL.Image]): RGB ROI crops, one per query.
            prompts (str | List[str]): Question(s). A single str is applied to all images.
            max_new_tokens (int, optional): Override the instance default for this call.

        Returns:
            List[str]: One decoded answer per image (generation prompt stripped).
        """
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if len(prompts) != len(images):
            raise ValueError(
                f"Got {len(images)} images but {len(prompts)} prompts; they must align "
                f"(or pass a single prompt str to broadcast)."
            )
        if len(images) == 0:
            return []

        device = next(self.model.parameters()).device
        texts = [
            self.processor.apply_chat_template(
                self._build_messages(p), tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
        # One image placeholder per text, so a flat image list aligns with the batch.
        inputs = self.processor(
            text=texts, images=list(images),
            padding=True, return_tensors="pt",
        ).to(device)

        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False,
        )
        # Strip the prompt tokens so only the model's answer is decoded.
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        answers = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return [a.strip() for a in answers]


class CustomInferenceVLM(BaseVLM):
    """
    Wrap an already-instantiated model + processor, for quick experimentation without
    registering a new class. Uses the default (chat-template) ``generate``.
    """

    def __init__(
        self,
        vlm_name: str,
        model: torch.nn.Module,
        processor: Callable,
        precision: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ):
        torch.nn.Module.__init__(self)
        self.vlm_name = vlm_name
        self.weights_path = None
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.model = model
        self.processor = processor
        self.precision = precision

    def _build(self) -> Tuple[None, None, None]:
        return None, None, None


class PathoR1VLM(BaseVLM):
    """
    Wrapper around **Patho-R1** (Zhang et al., 2025): a Qwen2.5-VL-based pathology
    reasoning VLM for visual question answering / captioning / region interrogation.
    This class contains no model code; it only drives ``transformers``.

    Attribution
    -----------
    Zhang et al., *"Patho-R1: A Multimodal Reinforcement Learning-Based Pathology Expert
    Reasoner"*, arXiv:2505.11404 (2025).
    Weights (gated by license, **CC-BY-NC-ND-4.0**, non-commercial research only):
    https://huggingface.co/WenchuanZhang/Patho-R1-7B · https://huggingface.co/WenchuanZhang/Patho-R1-3B

    Requirements
    ------------
    * ``pip install "transformers>=4.49" accelerate qwen-vl-utils`` (Qwen2.5-VL support).
    * ~16 GB GPU memory for the 7B in bf16 (~7 GB for the 3B). Generation is
      autoregressive, so a batch sweep over many patches is far slower than the
      feed-forward encoders — prefer targeted ROIs.

    Args:
        variant (str): ``"7b"`` (default) or ``"3b"``.
        device_map (str | None): Passed to ``from_pretrained``. ``None`` (default) loads
            on CPU then relies on the caller's ``.to(device)``; pass ``"auto"`` to let
            accelerate place a model too large for one GPU.
    """

    _HF_REPOS = {
        "7b": "WenchuanZhang/Patho-R1-7B",
        "3b": "WenchuanZhang/Patho-R1-3B",
    }
    # Default variant; subclasses in the registry override it (so the registry maps
    # names -> classes, mirroring patch_segmenter_registry).
    _DEFAULT_VARIANT = "7b"

    def __init__(self, variant: Optional[str] = None, device_map: Optional[str] = None, **kwargs):
        self._variant = (variant or self._DEFAULT_VARIANT).lower()
        self._device_map = device_map
        if self._variant not in self._HF_REPOS:
            raise ValueError(f"Patho-R1 variant must be one of {list(self._HF_REPOS)}, got '{variant}'.")
        super().__init__(**kwargs)

    def _build(self) -> Tuple[torch.nn.Module, Callable, torch.dtype]:
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                "Patho-R1 needs a recent `transformers` with Qwen2.5-VL support. Install "
                "`pip install \"transformers>=4.49\" accelerate qwen-vl-utils`. "
                "Original error: " + str(e)
            )

        repo = self.weights_path or self._HF_REPOS[self._variant]
        precision = torch.bfloat16
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            repo, torch_dtype=precision, device_map=self._device_map,
        )
        model.eval()
        # left padding is required for correct batched generation with decoder-only LMs.
        processor = AutoProcessor.from_pretrained(repo)
        if getattr(processor, "tokenizer", None) is not None:
            processor.tokenizer.padding_side = "left"

        self.vlm_name = f"patho_r1_{self._variant}"
        return model, processor, precision


class PathoR17B(PathoR1VLM):
    """Patho-R1 7B (default). Registry alias for ``PathoR1VLM(variant='7b')``."""
    _DEFAULT_VARIANT = "7b"


class PathoR13B(PathoR1VLM):
    """Patho-R1 3B. Registry alias for ``PathoR1VLM(variant='3b')``."""
    _DEFAULT_VARIANT = "3b"


vlm_registry = {
    "patho_r1_7b": PathoR17B,
    "patho_r1_3b": PathoR13B,
}
