from __future__ import annotations

from typing import Tuple
import warnings
import cv2
import numpy as np
import geopandas as gpd
from shapely import Polygon
from PIL import Image

class WSIPatcher:
    """ Iterator class to handle patching, patch scaling and tissue mask intersection """
    
    def __init__(
        self, 
        wsi, 
        patch_size: int, 
        src_pixel_size: float = None,
        dst_pixel_size: float = None,
        src_mag: int = None,
        dst_mag: int = None,
        overlap: int = 0,
        mask: gpd.GeoDataFrame = None,
        coords_only = False,
        custom_coords = None,
        threshold = 0.,
        pil=False,
    ):
        """ Initialize patcher, compute number of (masked) rows, columns.

        Args:
            wsi (WSI): wsi to patch
            patch_size (int): patch width/height in pixel on the slide after rescaling
            src_pixel_size (float, optional): pixel size in um/px of the slide before rescaling. Defaults to None.
            dst_pixel_size (float, optional): pixel size in um/px of the slide after rescaling. Defaults to None.
	    src_mag (int, optional): level0 magnification of the slide before rescaling. Defaults to None.
            dst_mag (int, optional): target magnification of the slide after rescaling. Defaults to None.
            overlap (int, optional): Overlap between patches in pixels. Defaults to 0. 
            mask (gpd.GeoDataFrame, optional): geopandas dataframe of Polygons. Defaults to None.
            coords_only (bool, optional): whenever to extract only the coordinates insteaf of coordinates + tile. Default to False.
            threshold (float, optional): minimum proportion of the patch under tissue to be kept.
                This argument is ignored if mask=None, passing threshold=0 will be faster. Defaults to 0.15
            pil (bool, optional): whenever to get patches as `PIL.Image` (numpy array by default). Defaults to False
        """
        self.wsi = wsi
        self.overlap = overlap
        self.width, self.height = self.wsi.get_dimensions()
        self.patch_size_target = patch_size
        self.mask = mask
        self.i = 0
        self.coords_only = coords_only
        self.custom_coords = custom_coords
        self.pil = pil
        self.dst_mag = dst_mag
        
        # set src magnification and pixel size. 
        if src_pixel_size is not None:
            self.src_pixel_size = src_pixel_size
        else:
            self.src_pixel_size = 10 / src_mag

        if dst_pixel_size is not None:
            self.dst_pixel_size = dst_pixel_size
        else:
            self.dst_pixel_size = 10 / dst_mag

        self.downsample = self.dst_pixel_size / self.src_pixel_size
        self.patch_size_src = round(patch_size * self.downsample)
        self.overlap_src = round(overlap * self.downsample)
        
        self.level, self.patch_size_level, self.overlap_level = self._prepare()  
        
        if custom_coords is None: 
            self.cols, self.rows = self._compute_cols_rows()
            
            col_rows = np.array([
                [col, row] 
                for col in range(self.cols) 
                for row in range(self.rows)
            ])
            coords = np.array([self._colrow_to_xy(xy[0], xy[1]) for xy in col_rows])
        else:
            if round(custom_coords[0][0]) != custom_coords[0][0]:
                raise ValueError("custom_coords must be a (N, 2) array of int")
            coords = custom_coords
        if self.mask is not None:
            self.valid_patches_nb, self.valid_coords = self._compute_masked(coords, threshold)
        else:
            self.valid_patches_nb, self.valid_coords = len(coords), coords
            
    def _colrow_to_xy(self, col, row):
        """ Convert col row of a tile to its top-left coordinates before rescaling (x, y) """
        x = col * (self.patch_size_src) - self.overlap_src * np.clip(col - 1, 0, None)
        y = row * (self.patch_size_src) - self.overlap_src * np.clip(row - 1, 0, None)
        return (x, y)   
            
    def _xy_to_colrow(self, x, y):
        """Convert x, y coordinates to col, row indices."""
        if x == 0:
            col = 0
        else:
            col = ((x - self.patch_size_src) // (self.patch_size_src - self.overlap_src)) + 1
        
        if y == 0:
            row = 0
        else:
            row = ((y - self.patch_size_src) // (self.patch_size_src - self.overlap_src)) + 1
        
        return col, row

    def _compute_masked(self, coords, threshold, simplify_shape=True) -> None:
        """ Compute tiles which overlap with > threshold with the tissue """
        
		# Filter coordinates by bounding boxes of mask polygons
        if simplify_shape:
            mask = self.mask.simplify(tolerance=self.patch_size_target / 4, preserve_topology=True)
        else:
            mask = self.mask
        bounding_boxes = mask.geometry.bounds
        bbox_masks = []
        for _, bbox in bounding_boxes.iterrows():
            bbox_mask = (
                (coords[:, 0] >= bbox['minx'] - self.patch_size_src) & (coords[:, 0] <= bbox['maxx'] + self.patch_size_src) & 
                (coords[:, 1] >= bbox['miny'] - self.patch_size_src) & (coords[:, 1] <= bbox['maxy'] + self.patch_size_src)
            )
            bbox_masks.append(bbox_mask)

        if len(bbox_masks) > 0:
            bbox_mask = np.vstack(bbox_masks).any(axis=0)
        else:
            bbox_mask = np.zeros(len(coords), dtype=bool)
            
        
        union_mask = mask.union_all()

        squares = [
            Polygon([
                (xy[0], xy[1]), 
                (xy[0] + self.patch_size_src, xy[1]), 
                (xy[0] + self.patch_size_src, xy[1] + self.patch_size_src), 
                (xy[0], xy[1] + self.patch_size_src)]) 
            for xy in coords[bbox_mask]
        ]
        if threshold == 0:
            valid_mask = gpd.GeoSeries(squares).intersects(union_mask).values
        else:
            gdf = gpd.GeoSeries(squares)
            areas = gdf.area
            valid_mask = gdf.intersection(union_mask).area >= threshold * areas
            
        full_mask = bbox_mask
        full_mask[bbox_mask] &= valid_mask 

        valid_patches_nb = full_mask.sum()
        self.valid_mask = full_mask
        valid_coords = coords[full_mask]
        return valid_patches_nb, valid_coords
        
    def __len__(self):
        return self.valid_patches_nb
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i >= self.valid_patches_nb:
            raise StopIteration
        x = self.__getitem__(self.i)
        self.i += 1
        return x
    
    def __getitem__(self, index):
        if 0 <= index < len(self):
            xy = self.valid_coords[index]
            x, y = xy[0], xy[1]
            if self.coords_only:
                return x, y
            tile, x, y = self.get_tile_xy(x, y)
            return tile, x, y
        else:
            raise IndexError("Index out of range")
        
    def _prepare(self) -> None:
        level, _ = self.wsi.get_best_level_and_custom_downsample(self.downsample, tolerance=0.1)
        level_downsample = int(self.wsi.level_downsamples[level])
        patch_size_level = round(self.patch_size_src / level_downsample)
        overlap_level = round(self.overlap_src / level_downsample)
        return level, patch_size_level, overlap_level
    
    def get_cols_rows(self) -> Tuple[int, int]:
        """ Get the number of columns and rows in the associated WSI

        Returns:
            Tuple[int, int]: (nb_columns, nb_rows)
        """
        return self.cols, self.rows
      
    def get_tile_xy(self, x: int, y: int) -> Tuple[np.ndarray, int, int]:

        tile = self.wsi.read_region(
            location=(x, y),
            level=self.level,
            size=(self.patch_size_level, self.patch_size_level),
            read_as='pil' if self.pil else 'numpy'
        )

        if self.patch_size_target is not None:
            if self.pil:
                tile = tile.resize((self.patch_size_target, self.patch_size_target))
            else:
                tile = cv2.resize(tile, (self.patch_size_target, self.patch_size_target))[:, :, :3]

        assert x < self.width and y < self.height
        return tile, x, y
    
    def get_tile(self, col: int, row: int) -> Tuple[np.ndarray, int, int]:
        """ get tile at position (column, row)

        Args:
            col (int): column
            row (int): row

        Returns:
            Tuple[np.ndarray, int, int]: (tile, pixel x of top-left corner (before rescaling), pixel_y of top-left corner (before rescaling))
        """
        if self.custom_coords is not None:
            raise ValueError("Can't use get_tile as 'custom_coords' was passed to the constructor")
            
        x, y = self._colrow_to_xy(col, row)
        return self.get_tile_xy(x, y)
    
    def _compute_cols_rows(self) -> Tuple[int, int]:
        col = 0
        row = 0
        x, y = self._colrow_to_xy(col, row)
        while x < self.width:
            col += 1
            x, _ = self._colrow_to_xy(col, row)
        cols = col
        while y < self.height:
            row += 1
            _, y = self._colrow_to_xy(col, row)
        rows = row
        return cols, rows 
    

    def visualize(self) -> Image.Image:
        """ 
        The `visualize` function of the class `WSI` overlays patch coordinates computed by the WSIPatcher
        onto a scaled thumbnail of the WSI. It creates a visualization of the patcher coordinates 
        and returns it as an image.

        Returns
        -------
        Image.Image
            Patch visualization

        Example:
        --------
        >>> img = wsi_patcher.visualize()
        >>> img.save('test_vis.jpg')
        """
        max_dimension = 1000
        if self.width > self.height:
            thumbnail_width = max_dimension
            thumbnail_height = int(thumbnail_width * self.height / self.width)
        else:
            thumbnail_height = max_dimension
            thumbnail_width = int(thumbnail_height * self.width / self.height)

        downsample_factor = self.width / thumbnail_width

        thumbnail_patch_size = max(1, int(self.patch_size_src / downsample_factor))

        # Get thumbnail in right format
        canvas = np.array(self.wsi.get_thumbnail((thumbnail_width, thumbnail_height))).astype(np.uint8)

        tmp_coords = self.coords_only
        self.coords_only = True
        # Draw rectangles for patches
        for (x, y) in self:
            x, y = int(x/downsample_factor), int(y/downsample_factor)
            thickness = max(1, thumbnail_patch_size // 10)
            canvas = cv2.rectangle(
                canvas, 
                (x, y), 
                (x + thumbnail_patch_size, y + thumbnail_patch_size), 
                (255, 0, 0), 
                thickness
            )

        self.coords_only = tmp_coords

        # Add annotations
        text_area_height = 130
        text_x_offset = int(thumbnail_width * 0.03)  # Offset as 3% of thumbnail width
        text_y_spacing = 25  # Vertical spacing between lines of text

        canvas[:text_area_height, :300] = (
            canvas[:text_area_height, :300] * 0.5
        ).astype(np.uint8)

        patch_mpp_mag = f"{self.dst_mag}x" if self.dst_mag is not None else f"{self.dst_pixel_size}um/px"

        cv2.putText(canvas, f'{len(self)} patches', (text_x_offset, text_y_spacing), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(canvas, f'width={self.width}, height={self.height}', (text_x_offset, text_y_spacing * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(canvas, f'mpp={self.wsi.mpp}, mag={self.wsi.mag}', (text_x_offset, text_y_spacing * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, f'patch={self.patch_size_target} w. overlap={self.overlap} @ {patch_mpp_mag}', (text_x_offset, text_y_spacing * 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return Image.fromarray(canvas)
    
class OpenSlideWSIPatcher(WSIPatcher):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "OpenSlideWSIPatcher is deprecated and will be removed in a future release. "
            "Please use WSIPatcher instead.",
            category=DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)