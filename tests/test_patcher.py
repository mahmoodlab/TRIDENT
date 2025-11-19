#!/usr/bin/env python3
"""
Unit tests for WSIPatcher with visualization showing overlap.
"""

import sys
import os
import unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from PIL import Image
from huggingface_hub import snapshot_download
import geopandas as gpd
from shapely.geometry import Polygon
from trident.wsi_objects.ImageWSI import ImageWSI
from trident.wsi_objects.WSIPatcher import WSIPatcher

def visualize_patches_debug(patcher, patches_list=None, mask_gdf=None, max_dimension=1000):
    """
    Enhanced visualization with color-coded patches, labels, and overlap highlighting.
    Makes it easy to debug patch placement and identify overlapping regions.
    
    Args:
        patcher: WSIPatcher instance
        patches_list: List of (patch_array, x, y) tuples to display at bottom
        mask_gdf: Optional GeoDataFrame with tissue mask polygons to overlay
        max_dimension: Maximum dimension for thumbnail
    """
    # Calculate thumbnail size
    if patcher.width > patcher.height:
        thumbnail_width = max_dimension
        thumbnail_height = int(thumbnail_width * patcher.height / patcher.width)
    else:
        thumbnail_height = max_dimension
        thumbnail_width = int(thumbnail_height * patcher.width / patcher.height)
    
    downsample_factor = patcher.width / thumbnail_width
    thumbnail_patch_size = max(1, int(patcher.patch_size_src / downsample_factor))
    
    # Get thumbnail
    canvas = np.array(patcher.wsi.get_thumbnail((thumbnail_width, thumbnail_height))).astype(np.uint8)
    
    # Color palette for patches (distinct colors)
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
    ]
    
    # Store patch rectangles for overlap detection
    patch_rects = []
    
    # Draw patches with colors and labels
    tmp_coords = patcher.coords_only
    patcher.coords_only = True
    
    for patch_idx, (x, y) in enumerate(patcher):
        x_thumb = int(x / downsample_factor)
        y_thumb = int(y / downsample_factor)
        
        # Get color (cycle through palette)
        color = colors[patch_idx % len(colors)]
        
        # Store rectangle for overlap detection
        rect = {
            'x': x_thumb,
            'y': y_thumb,
            'w': thumbnail_patch_size,
            'h': thumbnail_patch_size,
            'idx': patch_idx,
            'color': color
        }
        patch_rects.append(rect)
        
        # Draw semi-transparent fill (to show overlap)
        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (x_thumb, y_thumb),
            (x_thumb + thumbnail_patch_size, y_thumb + thumbnail_patch_size),
            color,
            -1  # Filled
        )
        cv2.addWeighted(overlay, 0.2, canvas, 0.8, 0, canvas)
        
        # Draw border (slim)
        thickness = max(1, thumbnail_patch_size // 40)
        cv2.rectangle(
            canvas,
            (x_thumb, y_thumb),
            (x_thumb + thumbnail_patch_size, y_thumb + thumbnail_patch_size),
            color,
            thickness
        )
        
        # Add patch index label
        label = str(patch_idx)
        font_scale = max(0.4, thumbnail_patch_size / 200)
        thickness_text = max(1, int(font_scale * 2))
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text
        )
        
        # Position label at top-left of patch
        label_x = x_thumb + 5
        label_y = y_thumb + text_height + 5
        
        # Draw label background for readability
        cv2.rectangle(
            canvas,
            (label_x - 2, label_y - text_height - 2),
            (label_x + text_width + 2, label_y + baseline + 2),
            (0, 0, 0),
            -1
        )
        
        # Draw label text
        cv2.putText(
            canvas,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness_text
        )
    
    patcher.coords_only = tmp_coords
    
    # Overlay tissue mask if provided
    if mask_gdf is not None:
        downsample_factor = patcher.width / thumbnail_width
        for idx, polygon in enumerate(mask_gdf.geometry):
            # Get polygon exterior coordinates (Shapely Polygon)
            coords = np.array(polygon.exterior.coords)
            
            # Scale to thumbnail coordinates
            thumb_coords = (coords / downsample_factor).astype(np.int32)
            
            # Draw polygon outline in cyan-green
            mask_color = (0, 255, 200)  # Cyan-green (BGR format for OpenCV)
            cv2.polylines(
                canvas,
                [thumb_coords],
                isClosed=True,
                color=mask_color,
                thickness=2
            )
            
            # Fill with semi-transparent overlay
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [thumb_coords], mask_color)
            cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
    
    # Highlight overlapping regions
    if patcher.overlap > 0:
        for i, rect1 in enumerate(patch_rects):
            for rect2 in patch_rects[i+1:]:
                # Check if rectangles overlap
                x_overlap = max(0, min(rect1['x'] + rect1['w'], rect2['x'] + rect2['w']) - max(rect1['x'], rect2['x']))
                y_overlap = max(0, min(rect1['y'] + rect1['h'], rect2['y'] + rect2['h']) - max(rect1['y'], rect2['y']))
                
                if x_overlap > 0 and y_overlap > 0:
                    # Draw overlap region in white (slim border)
                    overlap_x = max(rect1['x'], rect2['x'])
                    overlap_y = max(rect1['y'], rect2['y'])
                    cv2.rectangle(
                        canvas,
                        (overlap_x, overlap_y),
                        (overlap_x + x_overlap, overlap_y + y_overlap),
                        (255, 255, 255),  # White for overlap
                        1  # Slim border
                    )
    
    # Add info text
    text_area_height = 180 if mask_gdf is not None else 150
    text_x_offset = int(thumbnail_width * 0.02)
    text_y_spacing = 25
    
    # Darken area for text
    canvas[:text_area_height, :400] = (canvas[:text_area_height, :400] * 0.4).astype(np.uint8)
    
    # Add text info
    info_lines = [
        f'Patches: {len(patcher)}',
        f'Grid: {patcher.get_cols_rows()[0]}x{patcher.get_cols_rows()[1]}',
        f'Patch: {patcher.patch_size_target}px, Overlap: {patcher.overlap}px',
        f'Step: {patcher.patch_size_src - patcher.overlap_src}px',
        f'Colors = different patches',
        f'White boxes = overlap regions'
    ]
    if mask_gdf is not None:
        info_lines.append(f'Cyan regions = tissue mask ({len(mask_gdf)} polygons)')
    
    for i, line in enumerate(info_lines):
        cv2.putText(
            canvas,
            line,
            (text_x_offset, text_y_spacing * (i + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    # Add patches at the bottom if provided
    if patches_list and len(patches_list) > 0:
        # Calculate grid for patches
        num_patches = len(patches_list)
        cols = min(10, num_patches)  # Max 10 columns
        rows = (num_patches + cols - 1) // cols  # Ceiling division
        
        # Patch display size (resize patches to fit)
        patch_display_size = 64  # Size to display each patch
        patch_spacing = 4
        grid_width = cols * (patch_display_size + patch_spacing) - patch_spacing
        grid_height = rows * (patch_display_size + patch_spacing) - patch_spacing
        
        # Create patch grid
        patch_grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 240  # Light gray background
        
        for idx, (patch, x, y) in enumerate(patches_list):
            row = idx // cols
            col = idx % cols
            
            # Convert patch to numpy array if needed
            if isinstance(patch, Image.Image):
                patch_array = np.array(patch.convert('RGB'))
            elif isinstance(patch, np.ndarray):
                patch_array = patch.copy()
                # Ensure 3 channels
                if len(patch_array.shape) == 2:
                    patch_array = cv2.cvtColor(patch_array, cv2.COLOR_GRAY2RGB)
                elif patch_array.shape[2] == 4:
                    patch_array = patch_array[:, :, :3]  # Remove alpha channel
            else:
                patch_array = np.array(patch)
            
            # Resize patch to display size
            patch_resized = cv2.resize(patch_array, (patch_display_size, patch_display_size))
            
            # Place patch in grid
            y_start = row * (patch_display_size + patch_spacing)
            x_start = col * (patch_display_size + patch_spacing)
            patch_grid[y_start:y_start+patch_display_size, x_start:x_start+patch_display_size] = patch_resized
            
            # Add patch index label
            label = str(idx)
            font_scale = 0.4
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            label_x = x_start + 2
            label_y = y_start + text_height + 2
            
            # Label background
            cv2.rectangle(
                patch_grid,
                (label_x - 1, label_y - text_height - 1),
                (label_x + text_width + 1, label_y + baseline + 1),
                (0, 0, 0),
                -1
            )
            # Label text
            cv2.putText(
                patch_grid,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        # Combine canvas and patch grid
        canvas_height, canvas_width = canvas.shape[:2]
        combined_height = canvas_height + grid_height + 20  # 20px spacing
        combined_width = max(canvas_width, grid_width)
        
        combined = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
        combined[:canvas_height, :canvas_width] = canvas
        
        # Center the patch grid
        grid_x_start = (combined_width - grid_width) // 2
        combined[canvas_height + 10:canvas_height + 10 + grid_height, grid_x_start:grid_x_start + grid_width] = patch_grid
        
        # Add separator line
        cv2.line(
            combined,
            (0, canvas_height + 5),
            (combined_width, canvas_height + 5),
            (100, 100, 100),
            2
        )
        
        # Add label for patch grid
        label_text = f"Extracted Patches ({num_patches} total)"
        cv2.putText(
            combined,
            label_text,
            (10, canvas_height + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        
        return Image.fromarray(combined)
    
    return Image.fromarray(canvas)


def create_dummy_tissue_mask(width, height):
    """
    Create a dummy tissue mask as a GeoDataFrame with a few polygons.
    Covers parts of the image to simulate tissue regions.
    """
    # Create polygons covering different regions of the image
    # These are in pixel coordinates (level 0)
    polygons = [
        # Top-left region
        Polygon([
            (width * 0.1, height * 0.1),
            (width * 0.4, height * 0.1),
            (width * 0.4, height * 0.4),
            (width * 0.1, height * 0.4),
            (width * 0.1, height * 0.1)
        ]),
        # Center region
        Polygon([
            (width * 0.3, height * 0.3),
            (width * 0.7, height * 0.3),
            (width * 0.7, height * 0.7),
            (width * 0.3, height * 0.7),
            (width * 0.3, height * 0.3)
        ]),
        # Bottom-right region
        Polygon([
            (width * 0.6, height * 0.6),
            (width * 0.9, height * 0.6),
            (width * 0.9, height * 0.9),
            (width * 0.6, height * 0.9),
            (width * 0.6, height * 0.6)
        ]),
    ]
    
    gdf = gpd.GeoDataFrame(geometry=polygons)
    gdf.set_crs("EPSG:3857", inplace=True)  # Same CRS as trident uses
    return gdf


class TestWSIPatcher(unittest.TestCase):
    """Unit tests for WSIPatcher with visualization."""
    
    HF_REPO = "MahmoodLab/unit-testing"
    IMAGE_FILENAME = "Cat_August_2010-4.jpg"
    TEST_DATA_DIR = Path(__file__).parent / "test_patcher_artifacts"
    PATCH_SIZE = 512
    TARGET_MAG = 20
    MPP = 0.5
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment by downloading test image."""
        cls.TEST_DATA_DIR.mkdir(exist_ok=True)
        cls.image_path = cls.TEST_DATA_DIR / cls.IMAGE_FILENAME
        
        if not cls.image_path.exists():
            snapshot_download(
                repo_id=cls.HF_REPO,
                repo_type="dataset",
                local_dir=str(cls.TEST_DATA_DIR),
                allow_patterns=[cls.IMAGE_FILENAME]
            )
        
        # Verify image exists
        assert cls.image_path.exists(), f"Failed to download {cls.IMAGE_FILENAME}"
        
        # Load WSI once for all tests
        cls.wsi = ImageWSI(slide_path=str(cls.image_path), mpp=cls.MPP, lazy_init=False)
    
    def test_overlap(self):
        """Test WSIPatcher with different overlap values."""
        overlap_values = [0, 64, 128]
        
        for overlap in overlap_values:
            with self.subTest(overlap=overlap):
                # Create patcher
                patcher = WSIPatcher(
                    wsi=self.wsi,
                    patch_size=self.PATCH_SIZE,
                    src_mag=self.wsi.mag,
                    dst_mag=self.TARGET_MAG,
                    overlap=overlap,
                    coords_only=False,
                    pil=False,
                    threshold=0.5
                )
                
                # Basic assertions
                self.assertGreater(len(patcher), 0, "Should generate at least one patch")
                cols, rows = patcher.get_cols_rows()
                self.assertGreater(cols, 0)
                self.assertGreater(rows, 0)
                
                # Extract all patches
                all_patches = []
                patch_coords = []
                for i, (patch, x, y) in enumerate(patcher):
                    all_patches.append((patch, x, y))
                    patch_coords.append((i, x, y, patch.shape))
                    # Verify patch shape
                    self.assertEqual(len(patch.shape), 3, "Patch should be 3D (H, W, C)")
                
                self.assertEqual(len(all_patches), len(patcher), "Should extract all patches")
                
                # Verify overlap calculation
                if overlap > 0:
                    step = patcher.patch_size_src - patcher.overlap_src
                    self.assertGreater(step, 0, "Step size should be positive")
                    
                    # Verify step sizes between consecutive patches in same row/column
                    if len(patch_coords) >= 2:
                        for i in range(min(5, len(patch_coords) - 1)):
                            idx1, x1, y1, _ = patch_coords[i]
                            idx2, x2, y2, _ = patch_coords[i + 1]
                            if y1 == y2:  # Same row
                                step_x = x2 - x1
                                self.assertEqual(step_x, step, f"Step size mismatch in row")
                            if x1 == x2:  # Same column
                                step_y = y2 - y1
                                self.assertEqual(step_y, step, f"Step size mismatch in column")
                
                # Generate visualization
                vis_img = visualize_patches_debug(patcher, patches_list=all_patches)
                self.assertIsInstance(vis_img, Image.Image, "Visualization should return PIL Image")
                
                # Save visualization
                output_path = self.TEST_DATA_DIR / f"test_patcher_overlap_{overlap}px.png"
                vis_img.save(output_path)
                self.assertTrue(output_path.exists(), "Visualization should be saved")
    
    def test_mask(self):
        """Test WSIPatcher with a tissue mask."""
        # Create dummy tissue mask
        mask_gdf = create_dummy_tissue_mask(self.wsi.width, self.wsi.height)
        self.assertEqual(len(mask_gdf), 3, "Should have 3 mask polygons")
        
        overlap = 64
        patcher = WSIPatcher(
            wsi=self.wsi,
            patch_size=self.PATCH_SIZE,
            src_mag=self.wsi.mag,
            dst_mag=self.TARGET_MAG,
            overlap=overlap,
            mask=mask_gdf,
            coords_only=False,
            pil=False,
            threshold=0.2
        )
        
        # With mask, should have fewer or equal patches than without mask
        patcher_no_mask = WSIPatcher(
            wsi=self.wsi,
            patch_size=self.PATCH_SIZE,
            src_mag=self.wsi.mag,
            dst_mag=self.TARGET_MAG,
            overlap=overlap,
            coords_only=False,
            pil=False,
            threshold=0.5
        )
        self.assertLessEqual(len(patcher), len(patcher_no_mask), 
                           "Mask should filter patches")
        
        # Extract all patches
        all_patches = []
        for i, (patch, x, y) in enumerate(patcher):
            all_patches.append((patch, x, y))
            self.assertEqual(len(patch.shape), 3, "Patch should be 3D")
        
        self.assertEqual(len(all_patches), len(patcher), "Should extract all patches")
        
        # Generate visualization with mask overlay
        vis_img = visualize_patches_debug(patcher, patches_list=all_patches, mask_gdf=mask_gdf)
        self.assertIsInstance(vis_img, Image.Image, "Visualization should return PIL Image")
        
        # Save visualization
        output_path = self.TEST_DATA_DIR / "test_patcher_with_mask.png"
        vis_img.save(output_path)
        self.assertTrue(output_path.exists(), "Visualization should be saved")


if __name__ == "__main__":
    unittest.main()

