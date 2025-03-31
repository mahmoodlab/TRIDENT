from __future__ import annotations
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import geopandas as gpd

from trident.wsi_objects.WSI import WSI

try:
    import cupy as cp
except:
    print('Couldnt import cupy. Please install cupy.')
    exit()


class CuCIMWSI(WSI):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _lazy_initialize(self) -> None:
        """
        Perform lazy initialization by loading the WSI file and its metadata using CuCIM.

        Raises:
        -------
        FileNotFoundError:
            If the WSI file or tissue segmentation mask cannot be found.
        Exception:
            If there is an error while initializing the WSI.

        Notes:
        ------
        This method sets the following attributes after initialization:
        - `width` and `height` of the WSI.
        - `mpp` (microns per pixel) and `mag` (magnification level).
        - `gdf_contours` if a tissue segmentation mask is provided.
        """

        from cucim import CuImage

        if not self.lazy_init:
            try:
                self.img = CuImage(self.slide_path)
                self.dimensions = (self.img.size()[0], self.img.size()[1])  
                self.width, self.height = self.dimensions
                self.level_count = self.img.resolutions['level_count']
                self.level_downsamples = self.img.resolutions['level_downsamples']
                self.level_dimensions = self.img.resolutions['level_dimensions']
                self.properties = self.img.metadata

                if self.mpp is None:
                    self.mpp = self._fetch_mpp(self.custom_mpp_keys)
                self.mag = self._fetch_magnification(self.custom_mpp_keys)
                self.lazy_init = True

                if self.tissue_seg_path is not None:
                    try:
                        self.gdf_contours = gpd.read_file(self.tissue_seg_path)
                    except FileNotFoundError:
                        raise FileNotFoundError(f"Tissue segmentation file not found: {self.tissue_seg_path}")
            except Exception as e:
                raise Exception(f"Error initializing WSI: {e}")

    def _fetch_mpp(self, custom_keys: dict = None) -> Optional[float]:
        """
        Fetch the microns per pixel (MPP) from CuImage metadata.

        Parameters
        ----------
        custom_keys : dict, optional
            Optional dictionary with keys for 'mpp_x' and 'mpp_y' metadata fields to check first.

        Returns
        -------
        float or None
            Average MPP if found, else None.
        """
        import json

        def try_parse(val):
            try:
                return float(val)
            except:
                return None

        # CuCIM metadata can be a JSON string
        metadata = self.img.metadata
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        # Flatten nested CuCIM metadata for convenience
        flat_meta = {}
        def flatten(d, parent_key=''):
            for k, v in d.items():
                key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    flatten(v, key)
                else:
                    flat_meta[key.lower()] = v
        flatten(metadata)

        # Check custom keys first if provided
        mpp_x = mpp_y = None
        if custom_keys:
            if 'mpp_x' in custom_keys:
                mpp_x = try_parse(flat_meta.get(custom_keys['mpp_x'].lower()))
            if 'mpp_y' in custom_keys:
                mpp_y = try_parse(flat_meta.get(custom_keys['mpp_y'].lower()))

        # Standard fallback keys used in SVS, NDPI, MRXS, etc.
        fallback_keys = [
            'openslide.mpp-x', 'openslide.mpp-y',
            'tiff.resolution-x', 'tiff.resolution-y',
            'mpp', 'spacing', 'microns_per_pixel',
            'aperio.mpp', 'hamamatsu.mpp',
            'metadata.resolutions.level[0].spacing',
            'metadata.resolutions.level[0].physical_size.0',
        ]

        for key in fallback_keys:
            if mpp_x is None and key in flat_meta:
                mpp_x = try_parse(flat_meta[key])
            elif mpp_y is None and key in flat_meta:
                mpp_y = try_parse(flat_meta[key])
            if mpp_x is not None and mpp_y is not None:
                break

        # Use same value for both axes if only one was found
        if mpp_x is not None and mpp_y is None:
            mpp_y = mpp_x
        if mpp_y is not None and mpp_x is None:
            mpp_x = mpp_y

        if mpp_x is not None and mpp_y is not None:
            return float((mpp_x + mpp_y) / 2)

        return None

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        """
        Generate a thumbnail image of the WSI.

        Args:
        -----
        size : tuple[int, int]
            A tuple specifying the desired width and height of the thumbnail.

        Returns:
        --------
        Image.Image:
            The thumbnail as a PIL Image in RGB format.
        """
        target_width, target_height = size

        # Compute desired downsample factor and level
        downsample_x = self.width / target_width
        downsample_y = self.height / target_height
        desired_downsample = max(downsample_x, downsample_y)
        level, _ = self.get_best_level_and_custom_downsample(desired_downsample)

        # Read region at selected level
        region = self.img.read_region(
            location=(0, 0),
            size=self.level_dimensions[level],
            level=level
        )
        np_img = np.asarray(region)

        # Drop alpha if present
        if np_img.shape[-1] == 4:
            np_img = np_img[..., :3]

        # Convert to PIL and resize to final size
        pil_img = Image.fromarray(np_img).convert("RGB")
        pil_img = pil_img.resize(size, resample=Image.BILINEAR)

        return pil_img

    def read_region(
        self, 
        location: Tuple[int, int], 
        level: int, 
        size: Tuple[int, int],
        device: str = 'cpu',
    ) -> np.ndarray:
        """
        The `read_region` function from the class `CuCIMWSI` extracts a specific region from the WSI as a NumPy array.

        Args:
        -----
        location : Tuple[int, int]
            Coordinates (x, y) of the top-left corner of the region.
        level : int
            Pyramid level to read from.
        size : Tuple[int, int]
            Width and height of the region.
        device : str
            Device used to run read the region. Defaults to cpu. 

        Returns:
        --------
        np.ndarray:
            The extracted region as a NumPy array.

        Example:
        --------
        >>> region = wsi.read_region((0, 0), level=0, size=(512, 512))
        >>> print(region.shape)
        (512, 512, 3)
        """

        region = self.img.read_region(location=location, level=level, size=size, device='cpu')
        region = cp.asnumpy(region) 

        if region.shape[-1] == 4:
            region = region[..., :3]

        return region

    def read_region_pil(
        self, 
        location: Tuple[int, int], 
        level: int, 
        size: Tuple[int, int],
        device: str = 'cpu',
    ) -> Image.Image:
        """
        The `read_region_pil` function from the class `CuCIMWSI` extracts a specific region from the WSI as a PIL Image.

        Args:
        -----
        location : Tuple[int, int]
            Coordinates (x, y) of the top-left corner of the region.
        level : int
            Pyramid level to read from.
        size : Tuple[int, int]
            Width and height of the region.
        device : str
            Device used to run read the region. Defaults to cpu. 

        Returns:
        --------
        Image.Image:
            The extracted region as a PIL Image in RGB format.

        Example:
        --------
        >>> region = wsi.read_region_pil((0, 0), level=0, size=(512, 512))
        >>> region.show()
        """
        region = self.read_region(location=location, level=level, size=size, device=device)
        region = Image.fromarray(region).convert("RGB")
        return region

    def get_dimensions(self) -> Tuple[int, int]:
        """
        The `get_dimensions` function from the class `OpenSlideWSI` Retrieve the dimensions of the WSI.

        Returns:
        --------
        Tuple[int, int]:
            A tuple containing the width and height of the WSI in pixels.

        Example:
        --------
        >>> wsi.get_dimensions()
        (100000, 80000)
        """
        return self.dimensions
