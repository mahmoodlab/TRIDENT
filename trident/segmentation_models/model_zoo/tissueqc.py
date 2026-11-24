"""
Slide-level mask rules for TissueQC, which predicts background (0), tissue (1) and artifact (2).

``tissue_masks`` turns the stitched class map of a slide into two masks:

    default             tissue plus artifacts on tissue, enclosed holes filled
    artifacts removed   the default mask with artifact regions cut out

``mask_to_polygons`` traces them into level-0 polygons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    import geopandas as gpd

BACKGROUND, TISSUE, ARTIFACT = 0, 1, 2


def pool_classes(class_map: np.ndarray, factor: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Foreground and artifact masks on ``factor x factor`` blocks of a class map.

    A block is foreground when at least half of it is tissue or artifact, and artifact when at
    least half of it is artifact. Values above 2 (two edge tiles summed by the stitching) count as
    foreground.
    """
    h, w = class_map.shape[0] // factor, class_map.shape[1] // factor
    class_map = class_map[:h * factor, :w * factor]

    def at_least_half(mask: np.ndarray) -> np.ndarray:
        # INTER_AREA on a 0/1 uint8 map rounds the block mean, so 1 means at least half the block
        return cv2.resize(mask.view(np.uint8), (w, h), interpolation=cv2.INTER_AREA).astype(bool)

    return at_least_half(class_map != BACKGROUND), at_least_half(class_map == ARTIFACT)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill every hole that cannot be reached from outside the mask."""
    h, w = mask.shape
    outside = np.ones((h + 2, w + 2), np.uint8)
    outside[1:-1, 1:-1] = ~mask
    cv2.floodFill(outside, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2)
    return outside[1:-1, 1:-1] != 2


def artifact_on_tissue(artifact: np.ndarray, tissue: np.ndarray, min_tissue_border: float = 0.25,
                       ring_px: int = 2) -> np.ndarray:
    """
    Artifact regions with at least ``min_tissue_border`` of their outer border on tissue (folds,
    blur, ink on tissue). Regions bordered mostly by glass (ink, bubbles or mountant on bare glass)
    are left out.
    """
    n, labels = cv2.connectedComponents(artifact.astype(np.uint8), connectivity=8)
    if n == 1:
        return np.zeros_like(artifact)
    ring_labels = cv2.dilate(labels.astype(np.float32), np.ones((2 * ring_px + 1,) * 2, np.uint8)).astype(np.int64)
    ring = (ring_labels > 0) & ~artifact
    on_tissue = np.bincount(ring_labels[ring], weights=tissue[ring], minlength=n)
    ring_size = np.bincount(ring_labels[ring], minlength=n)
    keep = (ring_size > 0) & (on_tissue >= min_tissue_border * ring_size)
    keep[0] = False
    return keep[labels]


def tissue_masks(class_map: np.ndarray, holes_are_tissue: bool = True, map_mpp: float = 2.0,
                 mask_mpp: float = 8.0, min_artifact_width_um: float = 24.0,
                 min_artifact_area_mm2: float = 0.005) -> Tuple[np.ndarray, np.ndarray]:
    """
    Default and artifact-free tissue masks at ``mask_mpp`` um/px from a class map at ``map_mpp``.

    Only artifact regions at least ``min_artifact_width_um`` wide and ``min_artifact_area_mm2``
    large are cut out: thinner cracks and specks stay tissue, which keeps the mask unfragmented.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (default, artifacts removed) boolean masks.
    """
    foreground, artifact = pool_classes(class_map, max(1, round(mask_mpp / map_mpp)))
    tissue = foreground & ~artifact
    default = tissue | artifact_on_tissue(artifact, tissue)
    if holes_are_tissue:
        default = fill_holes(default)

    width_px = max(1, round(min_artifact_width_um / mask_mpp))
    cut = cv2.morphologyEx((artifact & default).astype(np.uint8), cv2.MORPH_OPEN, np.ones((width_px, width_px), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(cut, connectivity=8)
    large = stats[:, cv2.CC_STAT_AREA] * (mask_mpp / 1000) ** 2 >= min_artifact_area_mm2
    large[0] = False
    return default, default & ~large[labels]


def mask_to_polygons(mask: np.ndarray, mask_mpp: float, level0_mpp: float, min_tissue_area_mm2: float = 0.01,
                     min_hole_area_mm2: float = 0.005, tolerance_px: float = 0.5) -> "gpd.GeoDataFrame":
    """
    Tissue polygons with holes, in level-0 pixels, traced at the mask's own resolution.

    ``trident.IO.mask_to_gdf`` resizes the mask to a 2000 px edge and keeps holes of 4000 resized
    pixels or more (about 0.1 mm2), which drops nearly every artifact cut, so TissueQC traces its own.
    """
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Polygon

    px_mm2 = (mask_mpp / 1000) ** 2
    scale = mask_mpp / level0_mpp
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    def ring(contour):
        points = cv2.approxPolyDP(contour, tolerance_px, True).reshape(-1, 2)
        return points.astype(np.float64) * scale if len(points) >= 3 else None

    polygons = []
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i in np.flatnonzero(hierarchy[:, 3] == -1):
            if cv2.contourArea(contours[i]) * px_mm2 < min_tissue_area_mm2 or (exterior := ring(contours[i])) is None:
                continue
            holes, j = [], hierarchy[i, 2]
            while j != -1:
                if cv2.contourArea(contours[j]) * px_mm2 >= min_hole_area_mm2 and (hole := ring(contours[j])) is not None:
                    holes.append(hole)
                j = hierarchy[j, 0]
            polygon = Polygon(exterior, holes)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            polygons += [p for p in getattr(polygon, "geoms", [polygon]) if p.geom_type == "Polygon" and not p.is_empty]

    return gpd.GeoDataFrame(pd.DataFrame({"tissue_id": range(len(polygons))}), geometry=polygons)
