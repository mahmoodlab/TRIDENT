from __future__ import annotations

import cv2
import numpy as np
import skimage.color as sk_color
import skimage.filters as sk_filters
import skimage.morphology as sk_morphology


def mask_rgb(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Mask an RGB image.

    Taken from https://github.com/TIO-IKIM/CellViT
    """
    assert (
        rgb.shape[:-1] == mask.shape
    ), "Mask and RGB shape are different. Cannot mask when source and mask have different dimension."
    mask3 = mask[..., None]
    positive = np.where(mask3, rgb, 0)
    negative = np.where((~mask3) & (rgb > 0), 255, 0)
    return np.clip(positive + negative, a_min=0, a_max=255).astype(np.uint8)


def apply_otsu_thresholding(tile: np.ndarray) -> np.ndarray:
    """
    Generate a binary mask by using Otsu thresholding.

    Taken from https://github.com/TIO-IKIM/CellViT
    """
    tile = tile.copy()

    # Remove black border padding in some images.
    black_pixels = np.all(tile == [0, 0, 0], axis=-1)
    tile[black_pixels] = [255, 255, 255]

    hsv_img = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    gray_mask = cv2.inRange(hsv_img, (0, 0, 70), (180, 10, 255))
    black_mask = cv2.inRange(hsv_img, (0, 0, 0), (180, 255, 85))

    # Set all gray/black pixels to white.
    full_tile_bg = tile.copy()
    full_tile_bg[(gray_mask | black_mask) > 0] = 255

    # First Otsu pass for larger artifacts.
    masked_image_gray = 255 * sk_color.rgb2gray(full_tile_bg)
    thresh = sk_filters.threshold_otsu(masked_image_gray)
    otsu_masking = masked_image_gray < thresh
    otsu_masking = sk_morphology.remove_small_objects(otsu_masking, 60)
    tile = mask_rgb(tile, otsu_masking).astype(np.uint8)

    # Second Otsu pass for smaller artifacts.
    masked_image_gray = 255 * sk_color.rgb2gray(tile)
    thresh = sk_filters.threshold_otsu(masked_image_gray)
    otsu_masking = masked_image_gray < thresh
    otsu_masking = sk_morphology.remove_small_holes(otsu_masking, 5000)
    otsu_thr = ~otsu_masking
    return otsu_thr.astype(np.uint8)

