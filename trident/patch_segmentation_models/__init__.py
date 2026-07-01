"""
Cell / nuclei segmentation models for the ``patch_seg`` task.

Thin wrappers around upstream packages (HistoPlus, CellViT++, weave/SAM3); see
``trident.patch_segmentation_models.load`` for attribution and the model registry.
"""
from trident.patch_segmentation_models.load import (
    patch_segmenter_registry,
    patch_segmenter_factory,
    BasePatchSegmenter,
    CustomInferenceSegmenter,
    HistoPlusSegmenter,
    CellViTPlusPlusSegmenter,
    WeaveSegmenter,
)

__all__ = [
    "patch_segmenter_registry",
    "patch_segmenter_factory",
    "BasePatchSegmenter",
    "CustomInferenceSegmenter",
    "HistoPlusSegmenter",
    "CellViTPlusPlusSegmenter",
    "WeaveSegmenter",
]
