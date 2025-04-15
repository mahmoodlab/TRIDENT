from typing import Protocol

import numpy as np
import torch
from torch.utils.data import DataLoader

from trident.patch_encoder_models.load import BasePatchEncoder, CustomInferenceEncoder


class InferenceStrategy(Protocol):
    """
    An interface that supports arbitrary strategies for inference of
    image patch embedding models.
    """

    def forward(
        self,
        dataloader: DataLoader,
        patch_encoder: BasePatchEncoder | CustomInferenceEncoder,
        device: torch.device,
        precision: torch.dtype,
    ) -> np.ndarray: ...


class DefaultInferenceStrategy(InferenceStrategy):
    """
    This is the default inference strategy for embedding image patches.
    It sequentially processes one batch at a time on the specified device/GPU.
    Automatic mixed precision is enabled if `precision` != torch.float32.

    Args:
        dataloader (DataLoader):
            A dataloader that generates image patches.
        patch_encoder (BasePatchEncoder | CustomInferenceEncoder):
            The image patch embedding model.
        device (torch.device):
            Device to run feature extraction on (e.g., 'cuda:0').
        precision (torch.dtype):
            Precision of embedding model weights (e.g. torch.float32).

    Returns:
        embeddings (np.ndarray): The embeddings for the batch of image patches
    """

    def forward(
        self,
        dataloader: DataLoader,
        patch_encoder: BasePatchEncoder | CustomInferenceEncoder,
        device: torch.device,
        precision: torch.dtype,
    ) -> np.ndarray:
        features = []
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            with torch.autocast(
                device_type="cuda",
                dtype=precision,
                enabled=(precision != torch.float32),
            ):
                batch_features = patch_encoder(imgs)
            features.append(batch_features.cpu().numpy())
        features = np.concatenate(features, axis=0)
        return features
