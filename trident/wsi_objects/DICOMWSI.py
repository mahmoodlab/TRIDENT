from __future__ import annotations
import numpy as np
from wsidicom import WsiDicom
from PIL import Image
from typing import List, Tuple, Union, Optional

from trident.wsi_objects.WSI import WSI, ReadMode


class DICOMWSI(WSI):

    def __init__(self, slide_path, **kwargs) -> None:
        """
        Initialize a DICOMWSI instance for DICOM whole-slide images.

        Parameters
        ----------
        slide_path : str
            Path to the DICOM WSI file or directory.
        **kwargs : dict
            Additional keyword arguments forwarded to the base `WSI` class.
            - lazy_init (bool, default=True): Whether to defer loading WSI and metadata.

        Example
        -------
        >>> wsi = DICOMWSI(slide_path="path/to/wsi", lazy_init=False)
        >>> print(wsi)
        <width=100000, height=80000, backend=DICOMWSI, mpp=0.25, mag=40>
        """
        super().__init__(slide_path, **kwargs)

    def _lazy_initialize(self) -> None:
        """
        Lazily initialize the WSI using the DICOM backend.

        This method opens a whole-slide image using the wsidicom backend, extracting
        key metadata including dimensions, magnification, and multiresolution pyramid
        information.

        Raises
        ------
        FileNotFoundError
            If the DICOM WSI file cannot be found.
        Exception
            If an unexpected error occurs during WSI initialization.

        Notes
        -----
        After initialization, the following attributes are set:
        - `width` and `height`: spatial dimensions of the base level.
        - `dimensions`: (width, height) tuple from the highest resolution.
        - `level_count`: number of resolution levels in the image pyramid.
        - `level_downsamples`: downsampling factors for each level.
        - `level_dimensions`: image dimensions at each level.
        - `mpp`: microns per pixel.
        - `mag`: estimated magnification level.
        """
        super()._lazy_initialize()

        if not self.lazy_init:
            try:
                self.img = WsiDicom.open(self.slide_path)
                self.dimensions = self.get_dimensions()
                self.width, self.height = self.dimensions
                self.level_count = len(self.img.levels)
                self.level_downsamples = self.get_downsamples()
                self.level_dimensions = [level.size.to_tuple() for level in self.img.levels]
                self.mpp = self.img.mpp.to_tuple()[0]
                self.mag = self._fetch_magnification(self.custom_mpp_keys)
                self.lazy_init = True

            except Exception as e:
                raise RuntimeError(f"Failed to initialize WSI with DICOM backend: {e}") from e

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: ReadMode = 'pil',
    ) -> Union[Image.Image, np.ndarray]:
        """
        Extract a specific region from the DICOM whole-slide image.

        Parameters
        ----------
        location : Tuple[int, int]
            (x, y) coordinates of the top-left corner of the region to extract, relative to the base level.
        level : int
            Pyramid level to read from.
        size : Tuple[int, int]
            (width, height) of the region to extract.
        read_as : {'pil', 'numpy'}, optional
            Output format for the region:
            - 'pil': returns a PIL Image (default)
            - 'numpy': returns a NumPy array (H, W, 3)

        Returns
        -------
        Union[PIL.Image.Image, np.ndarray]
            Extracted image region in the specified format.

        Raises
        ------
        ValueError
            If `read_as` is not one of 'pil' or 'numpy'.

        Notes
        -----
        The `location` is automatically converted to the coordinate system of the requested pyramid level.
        """
        # 'location' in wsidicom is relative to specified level as opposed to base level like in OpenSlide
        location_ = (int(location[0] / self.level_downsamples[level]), int(location[1] / self.level_downsamples[level]))

        # Get slide dimensions for the requested level
        level_shape = self.level_dimensions[level]
        x, y = location_
        w, h = size

        # Calculate the region inside the slide
        x_end = min(x + w, level_shape[0])
        y_end = min(y + h, level_shape[1])
        x_start = max(x, 0)
        y_start = max(y, 0)

        # Read the valid region
        region_w = max(0, x_end - x_start)
        region_h = max(0, y_end - y_start)
        region = None
        if region_w > 0 and region_h > 0:
            region = self.img.read_region((x_start, y_start), level, (region_w, region_h))
            region = np.array(region)
        else:
            region = np.zeros((h, w, 3), dtype=np.uint8)

        # Prepare output and place the valid region
        output = np.zeros((h, w, 3), dtype=np.uint8)
        x_off = x_start - x
        y_off = y_start - y
        output[y_off:y_off+region_h, x_off:x_off+region_w] = region[:region_h, :region_w]


        if read_as == 'pil':
            return Image.fromarray(output).convert("RGB")
        elif read_as == 'numpy':
            return output
        else:
            raise ValueError(f"Invalid `read_as` value: {read_as}. Must be 'pil', 'numpy'.")

    def get_dimensions(self) -> Tuple[int, int]:
        """
        Return the dimensions (width, height) of the DICOM WSI at the highest resolution.

        Returns
        -------
        tuple of int
            (width, height) in pixels.
        """
        return self.img.size.to_tuple()
    
    def get_downsamples(self) -> List[float]:
        """
        Get the downsampling factors for each pyramid level in the DICOM WSI.

        Returns
        -------
        list of float
            Downsampling factors relative to the highest resolution level.
        """
        base_mpp = self.img.mpp
        downsamples = [np.floor((level.mpp / base_mpp).to_tuple()[0]) for level in self.img.levels]
        return downsamples

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        """
        Generate a thumbnail of the DICOM WSI.

        Parameters
        ----------
        size : tuple of int
            Desired (width, height) of the thumbnail.

        Returns
        -------
        PIL.Image.Image
            RGB thumbnail as a PIL Image.
        """
        return self.img.read_thumbnail(size).convert('RGB')
