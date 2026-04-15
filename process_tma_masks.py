"""
Process TMA core coordinates CSV and create block-level segmentation visualizations.

This script:
1. Combines all TMA_Map_block*.csv files into TMA_Map_Master.csv
2. Reads TMA core coordinates from CSV (center_x, center_y, radius)
3. Generates separate segmentation mask visualizations for each TMA type (folder) per block,
   organized in subfolders: block_{N}/{TMA_type}_segmentation.png
   Each visualization shows TMA cores with alpha=0.5 black overlay on non-TMA regions

Last updated: 2024-12-16
"""

import os
import re
import glob
import argparse
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import rotate, scale
from collections import defaultdict
from typing import Dict, Tuple, Optional, List
import math


def combine_tma_map_csvs(tma_maps_dir: str, output_path: str) -> pd.DataFrame:
    """
    Combine all TMA_Map_block*.csv files into a master CSV.
    Ensures unique mappings (case_id, block) pairs.
    
    Args:
        tma_maps_dir: Directory containing TMA_Map_block*.csv files
        output_path: Path to save the combined master CSV
        
    Returns:
        DataFrame with columns: tma_index, case_id, block
    """
    csv_files = sorted(glob.glob(os.path.join(tma_maps_dir, 'TMA_Map_block*.csv')))
    
    if not csv_files:
        raise ValueError(f"No TMA_Map_block*.csv files found in {tma_maps_dir}")
    
    all_data = []
    
    for csv_file in csv_files:
        # Extract block number from filename
        match = re.search(r'block(\d+)', os.path.basename(csv_file))
        if not match:
            print(f"Warning: Could not extract block number from {csv_file}, skipping")
            continue
        
        block_num = int(match.group(1))
        
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Standardize column names (handle case variations)
        df.columns = df.columns.str.strip()
        if 'core' in df.columns:
            df = df.rename(columns={'core': 'tma_index'})
        if 'Case ID' in df.columns:
            df = df.rename(columns={'Case ID': 'case_id'})
        elif 'case_id' in df.columns:
            pass  # Already correct
        else:
            print(f"Warning: Unexpected columns in {csv_file}: {df.columns.tolist()}")
            continue
        
        # Add block column
        df['block'] = block_num
        
        all_data.append(df)
    
    # Combine all dataframes
    master_df = pd.concat(all_data, ignore_index=True)
    
    # Ensure case_id is numeric (handle any string values)
    if 'case_id' in master_df.columns:
        # Convert to numeric, coercing errors to NaN
        master_df['case_id'] = pd.to_numeric(master_df['case_id'], errors='coerce')
    
    # Ensure unique mappings: (case_id, block) pairs should be unique
    # If duplicates exist, keep the first occurrence
    initial_count = len(master_df)
    master_df = master_df.drop_duplicates(subset=['case_id', 'block'], keep='first')
    if len(master_df) < initial_count:
        print(f"Warning: Removed {initial_count - len(master_df)} duplicate (case_id, block) pairs")
    
    # Also check for case_id appearing in multiple blocks and warn
    case_id_counts = master_df.groupby('case_id')['block'].nunique()
    multi_block_case_ids = case_id_counts[case_id_counts > 1]
    if len(multi_block_case_ids) > 0:
        print(f"Warning: {len(multi_block_case_ids)} case_ids appear in multiple blocks (this is expected for some cases)")
    
    # Save master CSV
    master_df.to_csv(output_path, index=False)
    print(f"Created master CSV with {len(master_df)} entries: {output_path}")
    print(f"  Unique case_ids: {master_df['case_id'].nunique()}")
    print(f"  Unique blocks: {master_df['block'].nunique()}")
    print(f"  Unique (case_id, block) pairs: {len(master_df)}")
    
    return master_df


def load_tma_coordinates_csv(csv_path: str) -> pd.DataFrame:
    """
    Load TMA core coordinates from CSV file.
    
    Expected CSV format should have columns for:
    - center_x, center_y (or x, y): center coordinates
    - radius: radius of the TMA core
    - case_id: case identifier (optional, may be in filename or separate column)
    
    Args:
        csv_path: Path to CSV file with TMA core coordinates
        
    Returns:
        DataFrame with TMA core coordinates
    """
    df = pd.read_csv(csv_path)
    
    # Standardize column names
    df.columns = df.columns.str.strip()
    
    # Try to identify coordinate columns (case-insensitive)
    col_lower = {col.lower(): col for col in df.columns}
    
    # Find center_x, center_y
    if 'center_x' in col_lower:
        center_x_col = col_lower['center_x']
    elif 'x' in col_lower:
        center_x_col = col_lower['x']
    else:
        raise ValueError(f"Could not find center_x or x column in CSV. Available columns: {df.columns.tolist()}")
    
    if 'center_y' in col_lower:
        center_y_col = col_lower['center_y']
    elif 'y' in col_lower:
        center_y_col = col_lower['y']
    else:
        raise ValueError(f"Could not find center_y or y column in CSV. Available columns: {df.columns.tolist()}")
    
    # Find radius column
    if 'radius' in col_lower:
        radius_col = col_lower['radius']
    elif 'r' in col_lower:
        radius_col = col_lower['r']
    else:
        raise ValueError(f"Could not find radius or r column in CSV. Available columns: {df.columns.tolist()}")
    
    # Rename to standard names
    df = df.rename(columns={
        center_x_col: 'center_x',
        center_y_col: 'center_y',
        radius_col: 'radius'
    })
    
    # Ensure numeric types
    df['center_x'] = pd.to_numeric(df['center_x'], errors='coerce')
    df['center_y'] = pd.to_numeric(df['center_y'], errors='coerce')
    df['radius'] = pd.to_numeric(df['radius'], errors='coerce')
    
    # Check for case_id column
    if 'case_id' not in df.columns:
        # Try to find case_id in other columns
        if 'case id' in col_lower:
            df = df.rename(columns={col_lower['case id']: 'case_id'})
        elif 'caseid' in col_lower:
            df = df.rename(columns={col_lower['caseid']: 'case_id'})
    
    # Remove rows with missing coordinates or radius
    initial_len = len(df)
    df = df.dropna(subset=['center_x', 'center_y', 'radius'])
    if len(df) < initial_len:
        print(f"Warning: Removed {initial_len - len(df)} rows with missing coordinates or radius")
    
    print(f"Loaded {len(df)} TMA cores from {csv_path}")
    print(f"  Coordinate range: x=[{df['center_x'].min():.1f}, {df['center_x'].max():.1f}], "
          f"y=[{df['center_y'].min():.1f}, {df['center_y'].max():.1f}]")
    print(f"  Radius range: [{df['radius'].min():.1f}, {df['radius'].max():.1f}]")
    
    return df


def create_circle_polygon(center_x: float, center_y: float, radius: float, num_points: int = 64) -> Polygon:
    """
    Create a circular polygon from center coordinates and radius.
    
    Args:
        center_x: X coordinate of center
        center_y: Y coordinate of center
        radius: Radius of the circle
        num_points: Number of points to approximate the circle
        
    Returns:
        Shapely Polygon representing the circle
    """
    angles = np.linspace(0, 2 * math.pi, num_points, endpoint=False)
    x_coords = center_x + radius * np.cos(angles)
    y_coords = center_y + radius * np.sin(angles)
    
    # Close the polygon
    coords = list(zip(x_coords, y_coords))
    coords.append(coords[0])  # Close the polygon
    
    return Polygon(coords)


def rotate_geodataframe_180_degrees(gdf: gpd.GeoDataFrame, wsi_width: float, wsi_height: float) -> gpd.GeoDataFrame:
    """
    Rotate all geometries in a GeoDataFrame 180 degrees around the WSI center.
    
    This is used to convert coordinates from original WSI space to QuPath's rotated space,
    matching the coordinate system used by GeoJSON exports from QuPath.
    
    Rotation formula: (x', y') = (wsi_width - x, wsi_height - y)
    This rotates around the WSI center point (wsi_width/2, wsi_height/2).
    
    Args:
        gdf: GeoDataFrame with geometries to rotate
        wsi_width: Width of the WSI at level 0
        wsi_height: Height of the WSI at level 0
        
    Returns:
        GeoDataFrame with rotated geometries
    """
    center_x = wsi_width / 2.0
    center_y = wsi_height / 2.0
    
    # Rotate each geometry 180 degrees around the WSI center
    rotated_geoms = []
    for geom in gdf.geometry:
        if isinstance(geom, Polygon):
            # Use shapely's rotate function with the WSI center as origin
            rotated_geom = rotate(geom, 180, origin=(center_x, center_y))
        elif isinstance(geom, MultiPolygon):
            rotated_polys = []
            for poly in geom.geoms:
                rotated_polys.append(rotate(poly, 180, origin=(center_x, center_y)))
            rotated_geom = MultiPolygon(rotated_polys)
        else:
            # For other geometry types, rotate directly
            rotated_geom = rotate(geom, 180, origin=(center_x, center_y))
        rotated_geoms.append(rotated_geom)
    
    # Create new GeoDataFrame with rotated geometries
    rotated_gdf = gdf.copy()
    rotated_gdf.geometry = rotated_geoms
    
    return rotated_gdf


def create_binary_mask_from_geometry(
    geometry: Polygon,
    wsi_width: int,
    wsi_height: int
) -> np.ndarray:
    """
    Create a binary mask from a geometry at WSI level 0 resolution.
    
    Args:
        geometry: Shapely Polygon geometry
        wsi_width: Width of WSI at level 0
        wsi_height: Height of WSI at level 0
        
    Returns:
        Binary mask as numpy array (uint8, 0=background, 255=foreground)
    """
    mask = np.zeros((wsi_height, wsi_width), dtype=np.uint8)
    
    if geometry.is_empty:
        return mask
    
    # Convert geometry coordinates to pixel coordinates
    if isinstance(geometry, Polygon):
        if hasattr(geometry, 'exterior') and geometry.exterior:
            exterior_coords = np.array(geometry.exterior.coords)
            # Convert to integer pixel coordinates
            exterior_coords = exterior_coords.astype(np.int32)
            
            # Handle interior rings (holes) if present
            if len(geometry.interiors) > 0:
                all_rings = [exterior_coords]
                for interior in geometry.interiors:
                    interior_coords = np.array(interior.coords).astype(np.int32)
                    all_rings.append(interior_coords)
                cv2.fillPoly(mask, all_rings, 255)
            else:
                cv2.fillPoly(mask, [exterior_coords], 255)
    
    elif isinstance(geometry, MultiPolygon):
        for poly in geometry.geoms:
            if poly.is_empty or not hasattr(poly, 'exterior') or not poly.exterior:
                continue
            exterior_coords = np.array(poly.exterior.coords).astype(np.int32)
            
            if len(poly.interiors) > 0:
                all_rings = [exterior_coords]
                for interior in poly.interiors:
                    interior_coords = np.array(interior.coords).astype(np.int32)
                    all_rings.append(interior_coords)
                cv2.fillPoly(mask, all_rings, 255)
            else:
                cv2.fillPoly(mask, [exterior_coords], 255)
    
    return mask


def create_masks_per_case_id(
    coordinates_df: pd.DataFrame,
    master_df: pd.DataFrame,
    wsi_dimensions: Tuple[int, int],
    thumbnail_dimensions: Tuple[int, int],
    output_dir: str,
    block: int,
    tma_marker: str,
    skip_existing: bool = False
) -> Dict[int, str]:
    """
    Create binary masks for each case_id using transformed coordinates, downsampled to thumbnail size.
    
    Args:
        coordinates_df: DataFrame with center_x, center_y, radius, and case_id
        master_df: Master DataFrame with tma_index, case_id, block columns
        wsi_dimensions: Tuple of (wsi_width, wsi_height) at level 0
        thumbnail_dimensions: Tuple of (thumb_width, thumb_height) for thumbnail size
        output_dir: Base output directory
        block: Block number
        tma_marker: TMA marker name (e.g., 'TMA_CD3', 'TMA_PDL1')
        skip_existing: If True, skip creating masks that already exist
        
    Returns:
        Dictionary mapping case_id to mask file path
    """
    wsi_width, wsi_height = wsi_dimensions
    thumb_width, thumb_height = thumbnail_dimensions
    
    # Calculate scale factors from WSI to thumbnail
    scale_x = thumb_width / wsi_width if wsi_width > 0 else 1.0
    scale_y = thumb_height / wsi_height if wsi_height > 0 else 1.0
    
    # Create lookup: case_id -> block (to filter by block)
    case_to_blocks = defaultdict(set)
    for _, row in master_df.iterrows():
        if pd.isna(row['case_id']):
            continue
        try:
            case_id = int(row['case_id'])
            block_num = int(row['block'])
            case_to_blocks[case_id].add(block_num)
        except (ValueError, TypeError):
            continue
    
    # Resolve conflicts: use first block if case_id appears in multiple blocks
    case_to_block = {}
    for case_id, blocks in case_to_blocks.items():
        case_to_block[case_id] = sorted(blocks)[0]
    
    # Group coordinates by case_id
    case_id_to_geoms = defaultdict(list)
    
    for _, coord_row in coordinates_df.iterrows():
        center_x = coord_row['center_x']
        center_y = coord_row['center_y']
        radius = coord_row['radius']
        
        # Get case_id
        case_id = None
        if 'case_id' in coord_row and not pd.isna(coord_row['case_id']):
            try:
                case_id = int(coord_row['case_id'])
            except (ValueError, TypeError):
                pass
        
        # Only process if case_id maps to this block
        if case_id is not None and case_id in case_to_block and case_to_block[case_id] == block:
            # Create circle polygon
            circle = create_circle_polygon(center_x, center_y, radius)
            case_id_to_geoms[case_id].append(circle)
    
    # Extract marker name without TMA_ prefix (e.g., "CD3" from "TMA_CD3")
    marker_name = tma_marker.replace('TMA_', '') if tma_marker.startswith('TMA_') else tma_marker
    
    # Create output directory for masks: {marker}/ (e.g., CD3/)
    masks_dir = os.path.join(output_dir, marker_name)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Create and save mask for each case_id
    case_id_to_mask_path = {}
    
    for case_id, geometries in case_id_to_geoms.items():
        # Combine all geometries for this case_id (in case there are multiple cores)
        if len(geometries) == 1:
            combined_geom = geometries[0]
        else:
            # Union all geometries
            combined_geom = geometries[0]
            for geom in geometries[1:]:
                combined_geom = combined_geom.union(geom)
        
        # Apply transformations: rotate 180 degrees, then reflect along y-axis
        transformed_geom = rotate(combined_geom, 180, origin=(wsi_width / 2.0, wsi_height / 2.0))
        transformed_geom = scale(transformed_geom, xfact=-1.0, yfact=1.0, origin=(wsi_width / 2.0, wsi_height / 2.0))
        
        # Scale geometry to thumbnail size
        scaled_geom = scale(transformed_geom, xfact=scale_x, yfact=scale_y, origin=(0, 0))
        
        # Create binary mask at thumbnail resolution
        mask = create_binary_mask_from_geometry(scaled_geom, thumb_width, thumb_height)
        
        # Save mask with new format: TumorCenter_{marker}_block{N}_case{case_id:03d}_mask.png
        mask_filename = f'TumorCenter_{marker_name}_block{block}_case{case_id:03d}_mask.png'
        mask_path = os.path.join(masks_dir, mask_filename)
        
        # Check if mask already exists
        if skip_existing and os.path.exists(mask_path):
            print(f"    Skipping existing mask for case_id {case_id}: {mask_path}")
            case_id_to_mask_path[case_id] = mask_path
            continue
        
        # Save as PNG (0=black/background, 255=white/foreground)
        mask_img = Image.fromarray(mask)
        mask_img.save(mask_path)
        
        case_id_to_mask_path[case_id] = mask_path
        
        print(f"    Created mask for case_id {case_id}: {mask_path} (foreground pixels: {np.sum(mask > 0)})")
    
    return case_id_to_mask_path


def reflect_geodataframe_along_y_axis(gdf: gpd.GeoDataFrame, wsi_width: float, wsi_height: float) -> gpd.GeoDataFrame:
    """
    Reflect all geometries in a GeoDataFrame along the y-axis at the center of the image.
    
    This performs a horizontal flip: x' = wsi_width - x, y' = y
    
    Args:
        gdf: GeoDataFrame with geometries to reflect
        wsi_width: Width of the WSI at level 0
        wsi_height: Height of the WSI at level 0 (used for center point)
        
    Returns:
        GeoDataFrame with reflected geometries
    """
    center_x = wsi_width / 2.0
    center_y = wsi_height / 2.0
    
    # Reflect each geometry along the y-axis (horizontal flip)
    # Using scale with xfact=-1 to flip horizontally around the center
    reflected_geoms = []
    for geom in gdf.geometry:
        if isinstance(geom, Polygon):
            # Scale with xfact=-1, yfact=1 to flip horizontally
            reflected_geom = scale(geom, xfact=-1.0, yfact=1.0, origin=(center_x, center_y))
        elif isinstance(geom, MultiPolygon):
            reflected_polys = []
            for poly in geom.geoms:
                reflected_polys.append(scale(poly, xfact=-1.0, yfact=1.0, origin=(center_x, center_y)))
            reflected_geom = MultiPolygon(reflected_polys)
        else:
            # For other geometry types, scale directly
            reflected_geom = scale(geom, xfact=-1.0, yfact=1.0, origin=(center_x, center_y))
        reflected_geoms.append(reflected_geom)
    
    # Create new GeoDataFrame with reflected geometries
    reflected_gdf = gdf.copy()
    reflected_gdf.geometry = reflected_geoms
    
    return reflected_gdf


def create_geodataframe_from_coordinates(
    coordinates_df: pd.DataFrame,
    master_df: pd.DataFrame
) -> Dict[int, Dict[str, gpd.GeoDataFrame]]:
    """
    Create GeoDataFrames from TMA core coordinates, grouped by block and folder.
    
    Args:
        coordinates_df: DataFrame with center_x, center_y, radius columns
        master_df: Master DataFrame with tma_index, case_id, block columns
        
    Returns:
        Dictionary mapping block number to dictionary mapping folder to GeoDataFrame
    """
    block_to_folder_to_gdf = defaultdict(lambda: defaultdict(list))
    
    # Create lookup: case_id -> block
    case_to_blocks = defaultdict(set)
    for _, row in master_df.iterrows():
        if pd.isna(row['case_id']):
            continue
        try:
            case_id = int(row['case_id'])
            block = int(row['block'])
            case_to_blocks[case_id].add(block)
        except (ValueError, TypeError):
            continue
    
    # Resolve conflicts: use first block if case_id appears in multiple blocks
    case_to_block = {}
    for case_id, blocks in case_to_blocks.items():
        case_to_block[case_id] = sorted(blocks)[0]
    
    # Process coordinates and create circles
    for _, coord_row in coordinates_df.iterrows():
        center_x = coord_row['center_x']
        center_y = coord_row['center_y']
        radius = coord_row['radius']
        
        # Get case_id from coordinates_df if available
        case_id = None
        if 'case_id' in coord_row and not pd.isna(coord_row['case_id']):
            try:
                case_id = int(coord_row['case_id'])
            except (ValueError, TypeError):
                pass
        
        # If case_id not in coordinates_df, we'll need to map by position or other method
        # For now, we'll create geometries for all coordinates and let the user
        # specify which folder/block they belong to
        
        # Create circle polygon
        circle = create_circle_polygon(center_x, center_y, radius)
        
        # Create a GeoDataFrame row
        gdf_row = gpd.GeoDataFrame([{
            'geometry': circle,
            'center_x': center_x,
            'center_y': center_y,
            'radius': radius,
            'case_id': case_id if case_id is not None else None
        }])
        
        # For now, we'll need to determine block and folder
        # Since we don't have that info directly, we'll create a single entry
        # The user may need to specify folder name or we can infer from context
        # For PDL1, we'll use 'TMA_PDL1' as the folder
        folder = 'TMA_PDL1'  # Default folder, can be made configurable
        
        if case_id is not None and case_id in case_to_block:
            block = case_to_block[case_id]
            block_to_folder_to_gdf[block][folder].append(gdf_row)
        else:
            # If no case_id mapping, we'll need to handle this differently
            # For now, skip or assign to a default block
            # This might need adjustment based on actual CSV structure
            pass
    
    # Combine GeoDataFrames for each block/folder combination
    result = {}
    for block, folder_to_gdfs in block_to_folder_to_gdf.items():
        result[block] = {}
        for folder, gdfs in folder_to_gdfs.items():
            if gdfs:
                combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
                result[block][folder] = combined_gdf
    
    return result


def map_coordinates_to_blocks(
    coordinates_df: pd.DataFrame,
    master_df: pd.DataFrame,
    folder_name: str = 'TMA_PDL1'
) -> Dict[int, gpd.GeoDataFrame]:
    """
    Map TMA core coordinates to blocks using case_id.
    
    Args:
        coordinates_df: DataFrame with center_x, center_y, radius, and optionally case_id
        master_df: Master DataFrame with tma_index, case_id, block columns
        folder_name: Folder name to use for grouping (e.g., 'TMA_PDL1')
        
    Returns:
        Dictionary mapping block number to GeoDataFrame with TMA core geometries
    """
    block_to_gdfs = defaultdict(list)
    
    # Create lookup: case_id -> block
    case_to_blocks = defaultdict(set)
    for _, row in master_df.iterrows():
        if pd.isna(row['case_id']):
            continue
        try:
            case_id = int(row['case_id'])
            block = int(row['block'])
            case_to_blocks[case_id].add(block)
        except (ValueError, TypeError):
            continue
    
    # Resolve conflicts: use first block if case_id appears in multiple blocks
    case_to_block = {}
    case_id_conflicts = []
    for case_id, blocks in case_to_blocks.items():
        if len(blocks) > 1:
            case_id_conflicts.append((case_id, sorted(blocks)))
            case_to_block[case_id] = sorted(blocks)[0]
        else:
            case_to_block[case_id] = list(blocks)[0]
    
    if case_id_conflicts:
        print(f"\nWarning: Found {len(case_id_conflicts)} case_ids appearing in multiple blocks:")
        for case_id, blocks in case_id_conflicts[:10]:
            print(f"  Case ID {case_id} appears in blocks: {blocks}")
        print("  (Using first block for these case_ids)\n")
    
    print(f"Created case_id to block mapping: {len(case_to_block)} case_ids mapped")
    
    # Process coordinates and create circles, mapping to blocks
    skipped_count = 0
    for _, coord_row in coordinates_df.iterrows():
        center_x = coord_row['center_x']
        center_y = coord_row['center_y']
        radius = coord_row['radius']
        
        # Get case_id
        case_id = None
        if 'case_id' in coord_row and not pd.isna(coord_row['case_id']):
            try:
                case_id = int(coord_row['case_id'])
            except (ValueError, TypeError):
                pass
        
        # Create circle polygon
        circle = create_circle_polygon(center_x, center_y, radius)
        
        # Map to block if case_id available
        if case_id is not None and case_id in case_to_block:
            block = case_to_block[case_id]
            gdf_row = gpd.GeoDataFrame([{
                'geometry': circle,
                'center_x': center_x,
                'center_y': center_y,
                'radius': radius,
                'case_id': case_id
            }])
            block_to_gdfs[block].append(gdf_row)
        else:
            skipped_count += 1
    
    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} cores without valid case_id mapping")
    
    # Combine GeoDataFrames for each block
    result = {}
    for block, gdfs in block_to_gdfs.items():
        if gdfs:
            combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
            result[block] = combined_gdf
    
    return result


# GeoJSON loading functions removed - now using CSV coordinates instead


def find_tma_slide_path(folder: str, block: int, tma_base_dir: str = '/data3/hancock', slide_type: str = 'TMA_TumorCenter') -> Optional[str]:
    """
    Find the TMA slide path for a given folder (marker) and block.
    
    Args:
        folder: TMA folder name (e.g., 'TMA_PDL1' -> 'PDL1')
        block: Block number
        tma_base_dir: Base directory for TMA slides
        slide_type: Type of slide directory (e.g., 'TMA_TumorCenter' or 'TMA_InvasionFront')
        
    Returns:
        Path to slide file or None if not found
    """
    # Map folder names to marker names (remove 'TMA_' prefix)
    marker = folder.replace('TMA_', '')
    
    # Try specified slide type first
    slide_path = os.path.join(tma_base_dir, slide_type, marker, f'TumorCenter_{marker}_block{block}.svs')
    if os.path.exists(slide_path):
        return slide_path
    
    # Try TMA_TumorCenter as fallback
    if slide_type != 'TMA_TumorCenter':
        slide_path = os.path.join(tma_base_dir, 'TMA_TumorCenter', marker, f'TumorCenter_{marker}_block{block}.svs')
        if os.path.exists(slide_path):
            return slide_path
    
    # Try TMA_InvasionFront as fallback
    if slide_type != 'TMA_InvasionFront':
        slide_path = os.path.join(tma_base_dir, 'TMA_InvasionFront', marker, f'InvasionFront_{marker}_block{block}.svs')
        if os.path.exists(slide_path):
            return slide_path
    
    return None


def load_wsi_thumbnail(slide_path: str, max_size: int = 2000) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
    """
    Load a WSI slide and generate a thumbnail.
    
    Args:
        slide_path: Path to WSI slide file
        max_size: Maximum dimension for thumbnail
        
    Returns:
        Tuple of (thumbnail as numpy array (RGB), (wsi_width, wsi_height)) or (None, None) if loading fails
    """
    try:
        from trident import load_wsi
        
        # Load WSI with lazy_init=False to ensure it's initialized
        wsi = load_wsi(slide_path, lazy_init=False)
        
        # Get WSI dimensions (level 0, full resolution)
        wsi_width = wsi.width
        wsi_height = wsi.height
        
        # Get thumbnail
        if wsi_width > wsi_height:
            thumb_width = max_size
            thumb_height = int(max_size * wsi_height / wsi_width)
        else:
            thumb_height = max_size
            thumb_width = int(max_size * wsi_width / wsi_height)
        
        thumbnail = wsi.get_thumbnail((thumb_width, thumb_height))
        thumb_array = np.array(thumbnail.convert('RGB'))
        
        return thumb_array, (wsi_width, wsi_height)
    except Exception as e:
        print(f"Warning: Could not load WSI {slide_path}: {e}")
        return None, None


def create_segmentation_visualization_with_thumbnail(
    gdf: gpd.GeoDataFrame,
    thumbnail_array: Optional[np.ndarray],
    wsi_dimensions: Optional[Tuple[int, int]],
    output_path: str,
    alpha: float = 0.5,
    dpi: int = 300
) -> None:
    """
    Create a visualization of TMA cores overlaid on thumbnail with gray mask on non-TMA regions.
    
    IMPORTANT: The TMA slides were rotated 180 degrees during import into QuPath.
    QuPath exports coordinates relative to the rotated/oriented view in QuPath.
    To align with the original WSI thumbnail, we need to account for this rotation.
    
    Transformation steps:
    1. Rotate coordinates back 180 degrees: (x', y') = (width - x, height - y)
    2. Scale from WSI level 0 to thumbnail dimensions
    3. Handle Y-axis conversion (GeoJSON Y-up to image Y-down)
    
    Args:
        gdf: GeoDataFrame containing TMA core polygons (coordinates in QuPath rotated view)
        thumbnail_array: Thumbnail image as numpy array (optional)
        wsi_dimensions: Tuple of (wsi_width, wsi_height) at level 0 (optional)
        output_path: Path to save the visualization
        alpha: Transparency for the gray mask overlay (0.0 to 1.0)
        dpi: Resolution for the output image
    """
    if gdf is None or len(gdf) == 0:
        print(f"Warning: No geometries to visualize for {output_path}")
        return
    
    # Use thumbnail if provided
    use_thumbnail = False
    wsi_width = None
    wsi_height = None
    if thumbnail_array is not None and wsi_dimensions is not None:
        thumb_height, thumb_width = thumbnail_array.shape[:2]
        wsi_width, wsi_height = wsi_dimensions
        
        # Calculate scale factors: geojson coordinates are in WSI level 0 pixel space
        # Scale them to thumbnail dimensions
        scale_x = thumb_width / wsi_width if wsi_width > 0 else 1.0
        scale_y = thumb_height / wsi_height if wsi_height > 0 else 1.0
        
        # Use the thumbnail as base
        canvas = thumbnail_array.copy()
        use_thumbnail = True
    else:
        # If no thumbnail, create a white background
        # Get bounds of all geometries for sizing
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        geo_width = bounds[2] - bounds[0]
        geo_height = bounds[3] - bounds[1]
        
        # Create a reasonable canvas size based on geometry bounds
        max_dimension = 2000
        if geo_width > geo_height:
            canvas_width = max_dimension
            canvas_height = int(max_dimension * geo_height / geo_width) if geo_width > 0 else max_dimension
        else:
            canvas_height = max_dimension
            canvas_width = int(max_dimension * geo_width / geo_height) if geo_height > 0 else max_dimension
        
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        scale_x = canvas_width / geo_width if geo_width > 0 else 1.0
        scale_y = canvas_height / geo_height if geo_height > 0 else 1.0
        thumb_height, thumb_width = canvas_height, canvas_width
        
        # If we have WSI dimensions, use them for rotation correction
        if wsi_dimensions is not None:
            wsi_width, wsi_height = wsi_dimensions
    
    # Create a mask for non-TMA regions
    # First, create a full mask covering the entire canvas
    mask = np.zeros((thumb_height, thumb_width), dtype=np.uint8)
    
    def transform_coords(coords_array, use_thumb, bounds_ref=None):
        """
        Transform coordinates from WSI level 0 pixel space to thumbnail space.
        
        IMPORTANT: The TMA slides were rotated 180 degrees during import into QuPath.
        QuPath exports coordinates relative to the rotated/oriented view in QuPath.
        To align with the original WSI thumbnail, we need to:
        1. Account for the 180-degree rotation: (x', y') = (width - x, height - y)
        2. Scale to thumbnail dimensions
        3. Handle Y-axis flip if needed (GeoJSON Y-up vs image Y-down)
        
        QuPath exports TMA cores via GsonTools as GeoJSON with coordinates in pixel space.
        The coordinates are relative to the QuPath view (which is rotated 180 degrees).
        """
        coords = coords_array.copy()
        
        if use_thumb:
            # Step 1: Account for 180-degree rotation in QuPath
            # Rotate coordinates back: (x', y') = (WSI_width - x, WSI_height - y)
            coords[:, 0] = wsi_width - coords[:, 0]
            coords[:, 1] = wsi_height - coords[:, 1]
            
            # Step 2: Scale from WSI level 0 pixel coordinates to thumbnail pixel coordinates
            coords[:, 0] = coords[:, 0] * scale_x
            coords[:, 1] = coords[:, 1] * scale_y
            
            # Step 3: Handle coordinate system conversion
            # GeoJSON uses Y-up, image coordinates use Y-down
            # Flip Y-axis: Y' = thumb_height - Y
            coords[:, 1] = thumb_height - coords[:, 1]
        else:
            # For white background, use bounds-based scaling
            if bounds_ref is None:
                bounds_ref = gdf.total_bounds
            
            # Account for 180-degree rotation if we have WSI dimensions
            # Rotate coordinates back: (x', y') = (WSI_width - x, WSI_height - y)
            if wsi_width is not None and wsi_height is not None:
                coords[:, 0] = wsi_width - coords[:, 0]
                coords[:, 1] = wsi_height - coords[:, 1]
            
            # Scale relative to bounds
            coords[:, 0] = (coords[:, 0] - bounds_ref[0]) * scale_x
            coords[:, 1] = (coords[:, 1] - bounds_ref[1]) * scale_y
            
            # Handle coordinate system conversion (GeoJSON Y-up to image Y-down)
            coords[:, 1] = thumb_height - coords[:, 1]
        
        return coords.astype(np.int32)
    
    # Fill TMA core regions with white (these will remain visible)
    for geom in gdf.geometry:
        if geom.is_empty:
            continue
        
        # Handle Polygon
        if isinstance(geom, Polygon):
            if hasattr(geom, 'exterior') and geom.exterior:
                exterior_coords = transform_coords(np.array(geom.exterior.coords), use_thumbnail)
                
                # Handle interior rings (holes) if present
                if len(geom.interiors) > 0:
                    all_rings = [exterior_coords]
                    for interior in geom.interiors:
                        interior_coords = transform_coords(np.array(interior.coords), use_thumbnail)
                        all_rings.append(interior_coords)
                    cv2.fillPoly(mask, all_rings, 255)
                else:
                    cv2.fillPoly(mask, [exterior_coords], 255)
        
        # Handle MultiPolygon
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                if poly.is_empty or not hasattr(poly, 'exterior') or not poly.exterior:
                    continue
                exterior_coords = transform_coords(np.array(poly.exterior.coords), use_thumbnail)
                
                # Handle interior rings (holes) if present
                if len(poly.interiors) > 0:
                    all_rings = [exterior_coords]
                    for interior in poly.interiors:
                        interior_coords = transform_coords(np.array(interior.coords), use_thumbnail)
                        all_rings.append(interior_coords)
                    cv2.fillPoly(mask, all_rings, 255)
                else:
                    cv2.fillPoly(mask, [exterior_coords], 255)
    
    # Create overlay: gray out non-TMA regions
    overlay = canvas.copy()
    # Apply gray mask to non-TMA regions (where mask is 0)
    gray_mask = mask == 0
    overlay[gray_mask] = (overlay[gray_mask] * (1 - alpha)).astype(np.uint8)
    
    # Draw TMA core boundaries
    for geom in gdf.geometry:
        if geom.is_empty:
            continue
        
        # Handle Polygon
        if isinstance(geom, Polygon):
            if hasattr(geom, 'exterior') and geom.exterior:
                exterior_coords = transform_coords(np.array(geom.exterior.coords), use_thumbnail)
                cv2.polylines(overlay, [exterior_coords], isClosed=True, color=(0, 0, 0), thickness=2)
                
                # Draw interior ring boundaries (holes)
                for interior in geom.interiors:
                    interior_coords = transform_coords(np.array(interior.coords), use_thumbnail)
                    cv2.polylines(overlay, [interior_coords], isClosed=True, color=(0, 0, 0), thickness=2)
        
        # Handle MultiPolygon
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                if poly.is_empty or not hasattr(poly, 'exterior') or not poly.exterior:
                    continue
                exterior_coords = transform_coords(np.array(poly.exterior.coords), use_thumbnail)
                cv2.polylines(overlay, [exterior_coords], isClosed=True, color=(0, 0, 0), thickness=2)
                
                # Draw interior ring boundaries (holes)
                for interior in poly.interiors:
                    interior_coords = transform_coords(np.array(interior.coords), use_thumbnail)
                    cv2.polylines(overlay, [interior_coords], isClosed=True, color=(0, 0, 0), thickness=2)
    
    # Save the result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_img = Image.fromarray(overlay)
    result_img.save(output_path, dpi=(dpi, dpi))
    
    print(f"Saved visualization: {output_path}")


def create_segmentation_visualization(
    gdf: gpd.GeoDataFrame,
    output_path: str,
    alpha: float = 0.5,
    dpi: int = 300
) -> None:
    """
    Create a visualization of TMA cores with black mask overlay on non-TMA regions.
    
    Args:
        gdf: GeoDataFrame containing TMA core polygons
        output_path: Path to save the visualization
        alpha: Transparency for the black mask overlay (0.0 to 1.0)
        dpi: Resolution for the output image
    """
    if gdf is None or len(gdf) == 0:
        print(f"Warning: No geometries to visualize for {output_path}")
        return
    
    # Get bounds of all geometries
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    
    # Calculate figure size maintaining aspect ratio
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    aspect_ratio = height / width if width > 0 else 1.0
    
    # Set reasonable figure size (max 12 inches on longest side)
    max_size = 12
    if aspect_ratio > 1:
        fig_height = max_size
        fig_width = max_size / aspect_ratio
    else:
        fig_width = max_size
        fig_height = max_size * aspect_ratio
    
    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    
    # Create a white background (representing the TMA slide)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    
    # Plot TMA cores in white/light color
    gdf.plot(ax=ax, color='white', edgecolor='black', linewidth=0.5, alpha=1.0)
    
    # Create a mask for non-TMA regions (everything outside the cores)
    # We'll create a rectangle covering the entire bounds, then subtract the cores
    from shapely.geometry import box
    full_bounds_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
    
    # Combine all TMA core geometries
    all_cores = gdf.geometry.unary_union
    
    # Calculate non-TMA regions (full bounds minus cores)
    try:
        non_tma_regions = full_bounds_box.difference(all_cores)
    except Exception as e:
        print(f"Warning: Could not compute difference for visualization: {e}")
        # Fallback: just show the cores without mask
        non_tma_regions = None
    
    # Overlay black mask on non-TMA regions
    if non_tma_regions is not None and not non_tma_regions.is_empty:
        # Handle different geometry types
        if isinstance(non_tma_regions, Polygon):
            mask_gdf = gpd.GeoDataFrame([1], geometry=[non_tma_regions])
            mask_gdf.plot(ax=ax, color='black', alpha=alpha, edgecolor='none')
        elif isinstance(non_tma_regions, MultiPolygon):
            for geom in non_tma_regions.geoms:
                if not geom.is_empty:
                    mask_gdf = gpd.GeoDataFrame([1], geometry=[geom])
                    mask_gdf.plot(ax=ax, color='black', alpha=alpha, edgecolor='none')
        else:
            # Try to convert to GeoDataFrame directly
            try:
                mask_gdf = gpd.GeoDataFrame([1], geometry=[non_tma_regions])
                mask_gdf.plot(ax=ax, color='black', alpha=alpha, edgecolor='none')
            except Exception as e:
                print(f"Warning: Could not plot non-TMA mask: {e}")
    
    # Remove axes for cleaner visualization
    ax.axis('off')
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved visualization: {output_path}")


def find_tma_folders_with_coordinates(base_dir: str = '/data/hancock_qupath') -> List[Tuple[str, str]]:
    """
    Find all TMA folders that contain tma_cores_coordinates.csv files.
    
    Args:
        base_dir: Base directory containing TMA folders
        
    Returns:
        List of tuples (folder_name, csv_path) for each TMA folder found
    """
    tma_folders = []
    
    if not os.path.exists(base_dir):
        print(f"Warning: Base directory not found: {base_dir}")
        return tma_folders
    
    # Look for folders starting with TMA_
    for item in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, item)
        if os.path.isdir(folder_path) and item.startswith('TMA_'):
            csv_path = os.path.join(folder_path, 'tma_cores_coordinates.csv')
            if os.path.exists(csv_path):
                tma_folders.append((item, csv_path))
                print(f"Found TMA folder: {item} -> {csv_path}")
    
    return sorted(tma_folders)


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Process TMA core coordinates and create masks/visualizations'
    )
    parser.add_argument(
        '--binary-mask-only',
        action='store_true',
        help='Only create binary masks per case_id, skip visualization overlays'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip creating masks that already exist'
    )
    args = parser.parse_args()
    
    # Paths - using CSV coordinates instead of geojson
    tma_maps_dir = '/data3/hancock/TMA_Maps'
    qupath_base_dir = '/data/hancock_qupath'
    output_dir = '/data3/hancock/TMA_Visualizations'
    slide_type = 'TMA_TumorCenter'  # Use TMA_TumorCenter slides
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Combine all TMA Map CSV files
    print("Step 1: Combining TMA Map CSV files...")
    master_csv_path = os.path.join(output_dir, 'TMA_Map_Master.csv')
    master_df = combine_tma_map_csvs(tma_maps_dir, master_csv_path)
    
    # Step 2: Find all TMA folders with coordinate CSV files
    print(f"\nStep 2: Finding TMA folders with coordinate CSV files in {qupath_base_dir}...")
    tma_folders = find_tma_folders_with_coordinates(qupath_base_dir)
    
    if not tma_folders:
        print(f"Warning: No TMA folders with tma_cores_coordinates.csv found in {qupath_base_dir}")
        return
    
    print(f"Found {len(tma_folders)} TMA folders to process")
    
    # Step 3: Process each TMA folder
    for folder_name, coordinates_csv_path in tma_folders:
        print(f"\n{'='*80}")
        print(f"Processing TMA folder: {folder_name}")
        print(f"{'='*80}")
        
        # Load TMA core coordinates from CSV
        print(f"\nLoading TMA core coordinates from {coordinates_csv_path}...")
        if not os.path.exists(coordinates_csv_path):
            print(f"Warning: Coordinates CSV not found: {coordinates_csv_path}, skipping")
            continue
        
        coordinates_df = load_tma_coordinates_csv(coordinates_csv_path)
        
        # Map coordinates to blocks
        print("\nMapping coordinates to blocks...")
        block_to_gdfs = map_coordinates_to_blocks(coordinates_df, master_df, folder_name=folder_name)
        print(f"Mapped coordinates to {len(block_to_gdfs)} blocks")
        for block, gdf in sorted(block_to_gdfs.items()):
            print(f"  Block {block}: {len(gdf)} TMA cores")
            if len(gdf) > 0:
                print(f"    Bounds: {gdf.total_bounds}")
        
        # Process each block
        if args.binary_mask_only:
            print("\nCreating binary masks only (skipping visualizations)...")
        else:
            print("\nGenerating segmentation visualizations...")
        
        for block in sorted(block_to_gdfs.keys()):
            print(f"\nProcessing block {block}...")
            gdf = block_to_gdfs[block]
            
            print(f"  Processing {folder_name}: {len(gdf)} TMA cores")
            if len(gdf) > 0:
                print(f"    Bounds: {gdf.total_bounds}")
            
            # Create output subdirectory for this block
            block_output_dir = os.path.join(output_dir, f'block_{block}')
            os.makedirs(block_output_dir, exist_ok=True)
            
            # Find TMA slide for this folder and block
            slide_path = find_tma_slide_path(folder_name, block, slide_type=slide_type)
            thumbnail_array = None
            wsi_dimensions = None
            
            if slide_path:
                print(f"    Loading slide: {slide_path}")
                thumbnail_array, wsi_dimensions = load_wsi_thumbnail(slide_path)
                if thumbnail_array is not None:
                    thumb_height, thumb_width = thumbnail_array.shape[:2]
                    thumbnail_dimensions = (thumb_width, thumb_height)
                    print(f"    Thumbnail loaded: {thumbnail_array.shape}, WSI dimensions: {wsi_dimensions}")
                    
                    # Create binary masks per case_id (downsampled to thumbnail size)
                    print("\n    Creating binary masks per case_id (thumbnail size)...")
                    case_id_to_mask_path = create_masks_per_case_id(
                        coordinates_df, master_df, wsi_dimensions, thumbnail_dimensions, output_dir, block, folder_name, args.skip_existing
                    )
                    print(f"    Created {len(case_id_to_mask_path)} binary masks at thumbnail resolution")
                    
                    # Only create visualization if not in binary-mask-only mode
                    if not args.binary_mask_only:
                        # Rotate geometries 180 degrees to match QuPath coordinate system
                        # CSV coordinates are in original WSI space, rotate to QuPath space
                        print("\n    Rotating geometries 180 degrees to match QuPath coordinate system...")
                        gdf = rotate_geodataframe_180_degrees(gdf, wsi_dimensions[0], wsi_dimensions[1])
                        # Reflect geometries along the y-axis (horizontal flip) at the center
                        print("    Reflecting geometries along y-axis at image center...")
                        gdf = reflect_geodataframe_along_y_axis(gdf, wsi_dimensions[0], wsi_dimensions[1])
                        
                        # Create visualization in block subfolder with thumbnail overlay
                        output_path = os.path.join(block_output_dir, f'{folder_name}_segmentation.png')
                        create_segmentation_visualization_with_thumbnail(
                            gdf, thumbnail_array, wsi_dimensions, output_path, alpha=0.5
                        )
                else:
                    print("    Warning: Could not load thumbnail from slide")
            else:
                print(f"    Warning: No slide found for {folder_name} block {block}")
                if not args.binary_mask_only:
                    print("      Cannot create visualization without slide")
    
    print(f"\n{'='*80}")
    print(f"Processing complete! Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

