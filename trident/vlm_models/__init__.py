"""
Vision-language models for the ``vlm`` task — interrogating ROIs with free-text prompts.

Thin wrappers around upstream packages (e.g. Patho-R1 via ``transformers``); see
``trident.vlm_models.load`` for attribution and the model registry.
"""
from trident.vlm_models.load import (
    vlm_registry,
    vlm_factory,
    BaseVLM,
    CustomInferenceVLM,
    PathoR1VLM,
    PathoR17B,
    PathoR13B,
)

__all__ = [
    "vlm_registry",
    "vlm_factory",
    "BaseVLM",
    "CustomInferenceVLM",
    "PathoR1VLM",
    "PathoR17B",
    "PathoR13B",
]
