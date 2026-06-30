import os
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

if TYPE_CHECKING:
    import geopandas as gpd


def create_overlay(
    scores: np.ndarray,
    coords: np.ndarray,
    patch_size_level0: int,
    scale: np.ndarray,
    region_size: Tuple[int, int]
) -> np.ndarray:
    """
    Create the heatmap overlay based on scores and coordinates.
    
    Parameters:
        scores (np.ndarray):
            Normalized scores.
        coords (np.ndarray):
            Coordinates of patches.
        patch_size_level0 (int):
            Patch size at level 0.
        scale (np.ndarray):
            Scaling factors.
        region_size (Tuple[int, int]):
            Dimensions of the region.

    Returns:
        np.ndarray: Heatmap overlay.
    """
    patch_size = np.ceil(np.array([patch_size_level0, patch_size_level0]) * scale).astype(int)
    coords = np.ceil(coords * scale).astype(int)
    
    overlay = np.zeros(tuple(np.flip(region_size)), dtype=float)
    counter = np.zeros_like(overlay, dtype=np.uint16)
    
    for idx, coord in enumerate(coords):
        overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += scores[idx]
        counter[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += 1
    
    zero_mask = counter == 0
    overlay[~zero_mask] /= counter[~zero_mask]
    overlay[zero_mask] = np.nan  # Set areas with no data to NaN
    
    return overlay


def apply_colormap(overlay: np.ndarray, cmap_name: str) -> np.ndarray:
    """
    Apply a colormap to the heatmap overlay.
    
    Parameters:
        overlay (np.ndarray):
            Heatmap overlay.
        cmap_name (str):
            Colormap name.

    Returns:
        np.ndarray: Colored overlay image.
    """
    cmap = plt.get_cmap(cmap_name)
    overlay_colored = np.zeros((*overlay.shape, 3), dtype=np.uint8)
    valid_mask = ~np.isnan(overlay)
    colored_valid = (cmap(overlay[valid_mask]) * 255).astype(np.uint8)[:, :3]
    overlay_colored[valid_mask] = colored_valid
    return overlay_colored


def visualize_heatmap(
    wsi: Any,
    scores: np.ndarray,
    coords: np.ndarray,
    patch_size_level0: int,
    vis_level: Optional[int] = 2,
    cmap: str = 'coolwarm',
    normalize: bool = True,
    num_top_patches_to_save: int = -1,
    output_dir: Optional[str] = "output",
    vis_mag: Optional[int] = None,
    overlay_only: bool = False,
    filename: str = 'heatmap.png'
) -> str:
    """
    Generate a heatmap visualization overlayed on a whole slide image (WSI).
    
    Parameters:
        wsi (WSI):
            Whole slide image object.
        scores (np.ndarray):
            Scores associated with each coordinate.
        coords (np.ndarray):
            Coordinates of patches at level 0.
        patch_size_level0 (int):
            Patch size at level 0.
        vis_level (Optional[int]):
            Visualization level.
        cmap (str):
            Colormap to use for the heatmap.
        normalize (bool):
            Whether to normalize the scores.
        num_top_patches_to_save (int):
            Number of high-score patches to save. If set to -1, do not save any. Defaults to -1.
        output_dir (Optional[str]):
            Directory to save heatmap and top-k patches.
        vis_mag (Optional[int]):
            Visualization magnification. This overwrites `vis_level`.
        overlay_only (bool):
            Whether to save the overlay only. If True, saves the overlay on top of a downscaled version of the WSI.
            Defaults to False.
        filename (str):
            File will be saved in `output_dir`/`filename`.

    Returns:
        str: Path to the saved heatmap image.
    """

    if normalize:
        from scipy.stats import rankdata
        scores = rankdata(scores, 'average') / len(scores) * 100 / 100
    
    if vis_mag is None:
        downsample = wsi.level_downsamples[vis_level]
    else:
        src_mag = wsi.mag
        downsample = src_mag / vis_mag
        if not overlay_only:
            vis_level, _ = wsi.get_best_level_and_custom_downsample(downsample)
    
    scale = np.array([1 / downsample, 1 / downsample])
    region_size = tuple((np.array(wsi.level_dimensions[0]) * scale).astype(int))
    overlay = create_overlay(scores, coords, patch_size_level0, scale, region_size)

    overlay_colored = apply_colormap(overlay, cmap)
    
    if overlay_only:
        blended_img = overlay_colored
    else:
        img = wsi.read_region((0, 0), vis_level, wsi.level_dimensions[vis_level]).convert("RGB")
        img = img.resize(region_size, resample=Image.Resampling.BICUBIC)
        img = np.array(img)
        
        blended_img = cv2.addWeighted(img, 0.6, overlay_colored, 0.4, 0)
    
    blended_img = Image.fromarray(blended_img)

    os.makedirs(output_dir, exist_ok=True)
    heatmap_path = os.path.join(output_dir, filename)
    blended_img.save(heatmap_path)

    if num_top_patches_to_save > 0:
        topk_dir = os.path.join(output_dir, "topk_patches")
        os.makedirs(topk_dir, exist_ok=True)
        topk_indices = np.argsort(scores)[-num_top_patches_to_save:]
        for idx, i in enumerate(topk_indices):
            x, y = coords[i]
            patch = wsi.read_region((x, y), 0, (patch_size_level0, patch_size_level0))
            patch.save(os.path.join(topk_dir, f"top_{idx}_score_{scores[i]:.4f}.png"))

    return heatmap_path


# =============================================================================
# Polygon overlays (tissue contours, cell/nuclei instances, patch grids)
#
# Everything below renders *polygon geometries* on a slide raster, as opposed to
# the score/attention heatmaps above. ``render_overlay`` is the single drawing
# core; the helpers are thin format-adapters used by the segmentation pipeline,
# and ``WSI.overlay`` is the public, user-facing entrypoint that wraps them.
# =============================================================================

# Distinct, reproducible colors per class id (used as cv2 BGR tuples on the BGR canvas).
CELL_VIZ_PALETTE = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163), (255, 127, 0),
    (255, 215, 0), (166, 86, 40), (247, 129, 191), (153, 153, 153), (26, 188, 156),
    (52, 152, 219), (155, 89, 182), (241, 196, 15), (231, 76, 60), (149, 165, 166),
]


def cell_class_color(class_id: int) -> tuple:
    """Color (cv2 BGR-on-BGR-canvas) for a class id; stable across the overview and patches."""
    return CELL_VIZ_PALETTE[int(class_id) % len(CELL_VIZ_PALETTE)]


def _draw_cell_legend(canvas: np.ndarray, entries: List[tuple]) -> np.ndarray:
    """
    Draw a color->cell-type legend in the top-left corner of ``canvas`` (BGR, in place).

    Args:
        canvas (np.ndarray): BGR image to draw on.
        entries (List[tuple]): ordered ``(label, color)`` pairs (color same as the contours).
    """
    if not entries:
        return canvas
    h, w = canvas.shape[:2]
    fs = max(0.45, min(1.2, w / 1600.0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    tth = max(1, int(round(fs * 1.5)))
    sw = int(round(22 * fs))          # swatch side
    gap = int(round(8 * fs)) + 2
    row_h = sw + gap
    text_w = max(cv2.getTextSize(lbl, font, fs, tth)[0][0] for lbl, _ in entries)
    panel_w = min(sw + 3 * gap + text_w, w - 2 * gap)
    panel_h = row_h * len(entries) + gap
    x0, y0 = gap, gap
    cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 255, 255), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), max(1, tth // 2))
    y = y0 + gap
    for label, color in entries:
        cv2.rectangle(canvas, (x0 + gap, y), (x0 + gap + sw, y + sw), color, -1)
        cv2.rectangle(canvas, (x0 + gap, y), (x0 + gap + sw, y + sw), (0, 0, 0), 1)
        cv2.putText(canvas, label, (x0 + 2 * gap + sw, y + sw - int(round(5 * fs))),
                    font, fs, (0, 0, 0), tth, cv2.LINE_AA)
        y += row_h
    return canvas


def render_overlay(
    canvas: np.ndarray,
    polygons: List[tuple],
    *,
    mode: str = 'outline',
    thickness: int = 2,
    alpha: float = 0.4,
    hole_color: Optional[tuple] = None,
    legend: Optional[List[tuple]] = None,
) -> np.ndarray:
    """
    Shared rendering core for every overlay in TRIDENT (tissue contours, cell instances,
    patch grids, generic ``WSI.overlay``). Draws already-scaled polygons onto a BGR canvas,
    in place, and returns it. All higher-level overlay helpers route through this so the
    drawing logic lives in exactly one place.

    Args:
        canvas (np.ndarray): ``(H, W, 3)`` **BGR** uint8 image to draw on (modified in place).
        polygons (List[tuple]): one entry per polygon as ``(exterior, interiors, color)`` where
            ``exterior`` is an ``(K, 2)`` int32 array of pixel coords on ``canvas``, ``interiors``
            is a (possibly empty) list of ``(M, 2)`` int32 hole rings, and ``color`` is a BGR tuple.
        mode (str): ``'outline'`` draws polygon boundaries with ``cv2.polylines``; ``'fill'`` fills
            polygon interiors and alpha-blends them over the canvas (holes are left untinted).
        thickness (int): line thickness for outline mode. Defaults to 2.
        alpha (float): blend strength for fill mode in ``[0, 1]``. Defaults to 0.4.
        hole_color (tuple, optional): outline color for interior rings (holes). Defaults to each
            polygon's own color when ``None``. Only used in outline mode.
        legend (List[tuple], optional): ordered ``(label, color)`` pairs drawn as a corner legend.

    Returns:
        np.ndarray: ``canvas`` (same array, mutated).
    """
    if mode not in ('outline', 'fill'):
        raise ValueError(f"mode must be 'outline' or 'fill', got {mode!r}")

    if mode == 'outline':
        for exterior, interiors, color in polygons:
            cv2.polylines(canvas, [exterior], isClosed=True, color=color, thickness=thickness)
            for hole in interiors:
                cv2.polylines(canvas, [hole], isClosed=True,
                              color=hole_color if hole_color is not None else color,
                              thickness=thickness)
    else:  # fill
        layer = canvas.copy()
        # Group exteriors by color so each color is one batched fillPoly call (fast for many cells).
        by_color: dict = {}
        all_holes: List[np.ndarray] = []
        for exterior, interiors, color in polygons:
            by_color.setdefault(tuple(color), []).append(exterior)
            all_holes.extend(interiors)
        for color, exts in by_color.items():
            cv2.fillPoly(layer, exts, color)
        if all_holes:  # punch holes back to the untouched background before blending
            hole_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
            cv2.fillPoly(hole_mask, all_holes, 255)
            layer[hole_mask > 0] = canvas[hole_mask > 0]
        np.copyto(canvas, cv2.addWeighted(layer, alpha, canvas, 1.0 - alpha, 0.0))

    if legend:
        _draw_cell_legend(canvas, legend)
    return canvas


def overlay_instances_on_thumbnail(
    gdf: "gpd.GeoDataFrame",
    thumbnail: np.ndarray,
    saveto: str,
    scale: float,
) -> str:
    """
    Draw instance polygons (level-0 coords) onto a slide thumbnail for debugging and save
    it as a JPEG. Colors are assigned per class id, with a color->cell-type legend drawn in
    the corner. Mirrors `overlay_gdf_on_thumbnail` used by tissue segmentation.

    Args:
        gdf (gpd.GeoDataFrame): Instances with `class`/`class_name` columns and polygon
            `geometry` (level-0).
        thumbnail (np.ndarray): RGB thumbnail to draw on.
        saveto (str): Output `.jpg` path.
        scale (float): thumbnail-pixels-per-level0-pixel (i.e. thumb_width / wsi_width).

    Returns:
        str: `saveto`.
    """
    canvas = np.ascontiguousarray(thumbnail[..., ::-1])  # RGB -> BGR for cv2
    present = {}  # class_id -> class_name, for the legend
    polygons = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        class_id = int(row.get('class', 0))
        present.setdefault(class_id, row.get('class_name'))
        color = cell_class_color(class_id)
        polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
        for poly in polys:
            ext = (np.asarray(poly.exterior.coords) * scale).astype(np.int32)
            polygons.append((ext, [], color))

    entries = [
        (str(present[cid]) if present[cid] is not None else f"class {cid}", cell_class_color(cid))
        for cid in sorted(present)
    ]
    render_overlay(canvas, polygons, mode='outline', thickness=1, legend=entries)

    os.makedirs(os.path.dirname(saveto), exist_ok=True)
    cv2.imwrite(saveto, canvas)
    return saveto


def draw_instances_on_tile(
    tile_rgb: np.ndarray,
    instances_px: List[dict],
    class_names: Optional[List[str]],
    saveto: str,
    thickness: int = 2,
) -> str:
    """
    Draw per-cell instance contours (in *patch* pixel coords) on a full-resolution tile and
    save as JPEG. This is the readable debug artifact for cell segmentation: at this zoom the
    individual cells and their class colors are clearly visible. Color is keyed by class id.

    Args:
        tile_rgb (np.ndarray): `(H, W, 3)` RGB patch image.
        instances_px (List[dict]): instances with `contour` (K,2 patch-px) and `class_id`.
        class_names (List[str], optional): unused for drawing; kept for parity/legend hooks.
        saveto (str): output `.jpg` path.
        thickness (int, optional): contour line thickness. Defaults to 2.

    Returns:
        str: `saveto`.
    """
    canvas = np.ascontiguousarray(tile_rgb[..., ::-1])  # RGB -> BGR for cv2
    present = set()
    polygons = []
    for inst in instances_px:
        contour = np.asarray(inst['contour'], dtype=np.int32)
        if contour.ndim != 2 or contour.shape[0] < 3:
            continue
        class_id = int(inst['class_id'])
        present.add(class_id)
        polygons.append((contour, [], cell_class_color(class_id)))

    def _name(cid):
        if class_names is not None and 0 <= cid < len(class_names):
            return str(class_names[cid])
        return f"class {cid}"
    entries = [(_name(cid), cell_class_color(cid)) for cid in sorted(present)]
    render_overlay(canvas, polygons, mode='outline', thickness=thickness, legend=entries)

    os.makedirs(os.path.dirname(saveto), exist_ok=True)
    cv2.imwrite(saveto, canvas)
    return saveto


def overlay_gdf_on_thumbnail(
    gdf_contours, thumbnail, contours_saveto, scale, tissue_color=(0, 255, 0), hole_color=(0, 0, 255)
):
    """
    The `overlay_gdf_on_thumbnail` function overlays polygons from a GeoDataFrame onto a scaled
    thumbnail image using OpenCV. This is particularly useful for visualizing tissue regions and
    their boundaries on smaller representations of whole-slide images. Drawing is delegated to
    the shared :func:`render_overlay` core.

    Parameters:
    -----------
    gdf_contours : gpd.GeoDataFrame
        A GeoDataFrame containing the polygons to overlay, with a `geometry` column.
    thumbnail : np.ndarray
        The thumbnail image as an RGB NumPy array.
    contours_saveto : str
        The file path to save the annotated thumbnail.
    scale : float
        The scaling factor between the GeoDataFrame coordinates and the thumbnail resolution.
    tissue_color : tuple, optional
        The color (BGR format) for tissue polygons. Defaults to green `(0, 255, 0)`.
    hole_color : tuple, optional
        The color (BGR format) for hole polygons. Defaults to red `(0, 0, 255)`.

    Returns:
    --------
    None
        The function saves the annotated image to the specified file path.

    Example:
    --------
    >>> overlay_gdf_on_thumbnail(
    ...     gdf_contours=gdf,
    ...     thumbnail=thumbnail_img,
    ...     contours_saveto="annotated_thumbnail.png",
    ...     scale=0.5
    ... )
    """
    canvas = np.ascontiguousarray(thumbnail[..., ::-1])  # RGB -> BGR for cv2

    polygons = []
    for poly in gdf_contours.geometry:
        if poly is None or poly.is_empty or not poly.exterior:
            continue
        ext = (np.array(poly.exterior.coords) * scale).astype(np.int32)
        interiors = [(np.array(interior.coords) * scale).astype(np.int32) for interior in poly.interiors]
        polygons.append((ext, interiors, tissue_color))

    render_overlay(canvas, polygons, mode='outline', thickness=2, hole_color=hole_color)

    # Crop black borders of the annotated image
    nz = np.nonzero(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY))  # Non-zero pixel locations
    xmin, xmax, ymin, ymax = np.min(nz[1]), np.max(nz[1]), np.min(nz[0]), np.max(nz[0])
    cropped_annotated = canvas[ymin:ymax, xmin:xmax]

    # Save the annotated image (canvas is already BGR, ready for cv2.imwrite)
    os.makedirs(os.path.dirname(contours_saveto), exist_ok=True)
    cv2.imwrite(contours_saveto, cropped_annotated)
