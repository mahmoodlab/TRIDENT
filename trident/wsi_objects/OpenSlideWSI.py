from __future__ import annotations
import numpy as np
import openslide
from PIL import Image
from typing import List, Tuple, Union
import geopandas as gpd

from trident.wsi_objects.WSI import WSI


class OpenSlideWSI(WSI):

    def __init__(self, **kwargs) -> None:
        """
        
        Example:
        --------
        >>> wsi = OpenSlideWSI("path/to/wsi.svs", lazy_init=False)
        >>> print(wsi)
        <width=100000, height=80000, backend=OpenSlideWSI, mpp=0.25, mag=40>
        """
        super().__init__(**kwargs)

    def _lazy_initialize(self) -> None:
        """
        The `_lazy_initialize` function from the class `OpenSlideWSI` Perform lazy initialization by loading the WSI file and its metadata.

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
        if not self.lazy_init:
            try:
                self.img = openslide.OpenSlide(self.slide_path)
                # set openslide attrs as self
                self.dimensions = self.get_dimensions()
                self.width, self.height = self.dimensions
                self.level_count = self.img.level_count
                self.level_downsamples = self.img.level_downsamples
                self.level_dimensions = self.img.level_dimensions
                self.properties = self.img.properties
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

    def _fetch_mpp(self, custom_mpp_keys: List[str] | None = None) -> float | None:
        """
        The `fetch_mpp` function from the class `OpenSlideWSI` retrieves the microns per pixel (MPP) value for the WSI.

        Args:
        -----
        custom_mpp_keys : List[str], optional
            Custom keys for extracting MPP. Defaults to common OpenSlide keys.

        Returns:
        --------
        float | None:
            The MPP value, or None if it cannot be determined.

        Notes:
        ------
        If the MPP value is unavailable, this method attempts to calculate it 
        from the slide's resolution metadata.
        """
        mpp_x = None
        mpp_keys = [
            openslide.PROPERTY_NAME_MPP_X,
            'openslide.mirax.MPP',
            'aperio.MPP',
            'hamamatsu.XResolution',
            'openslide.comment'
        ]

        if custom_mpp_keys:
            mpp_keys.extend(custom_mpp_keys)

        # Search for mpp_x in the available keys
        for key in mpp_keys:
            if key in self.img.properties:
                try:
                    mpp_x = float(self.img.properties[key])
                    break
                except ValueError:
                    continue

        # Convert pixel resolution to mpp 
        if mpp_x is None:
            x_resolution = self.img.properties.get('tiff.XResolution', None)
            unit = self.img.properties.get('tiff.ResolutionUnit', None)
            if not x_resolution or not unit:
                return None
            if unit == 'CENTIMETER' or unit == 'centimeter':
                mpp_x = 10000 / float(x_resolution)  # 1 cm = 10,000 microns
            elif unit == 'INCH':
                mpp_x = 25400 / float(x_resolution)  # 1 inch = 25,400 microns
            else:
                return None  # Unsupported unit -- add more conditions is needed. 
            
        mpp_x = round(mpp_x, 4)

        return mpp_x

    def read_region(
        self, 
        location: Tuple[int, int], 
        level: int, 
        size: Tuple[int, int],
        device: str = 'cpu',
        read_as: str = 'pil',
    ) -> Union[np.ndarray, Image.Image]:
        """
        Extract a specific region from the whole-slide image (WSI) as a NumPy array or PIL image.

        Args:
        -----
        location : Tuple[int, int]
            (x, y) coordinates of the top-left corner of the region to extract.
        level : int
            Pyramid level to read from.
        size : Tuple[int, int]
            (width, height) of the region to extract.
        device : str, optional
            Unused. Present for API compatibility with CuCIM. Defaults to 'cpu'.
        read_as : str, optional
            Format to return the region in. Options are:
            - 'numpy': returns a NumPy array
            - 'pil': returns a PIL Image object (default)

        Returns:
        --------
        Union[np.ndarray, PIL.Image.Image]
            The extracted region in the specified format.

        Example:
        --------
        >>> region = wsi.read_region((0, 0), level=0, size=(512, 512))
        >>> print(region.shape)
        (512, 512, 3)
        """
        region = self.img.read_region(location, level, size).convert('RGB')

        if read_as == 'numpy':
            return np.array(region)
        
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
        return self.img.dimensions

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

        Example:
        --------
        >>> thumbnail = wsi.get_thumbnail((256, 256))
        >>> thumbnail.show()
        """
        return self.img.get_thumbnail(size).convert('RGB')
