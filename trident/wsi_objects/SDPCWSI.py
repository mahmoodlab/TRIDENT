from __future__ import annotations
import numpy as np
from PIL import Image
from typing import Tuple, Union
from trident.wsi_objects.WSI import WSI, ReadMode


class SDPCWSI(WSI):

    def __init__(self, slide_path, **kwargs) -> None:
        """
        Initialize an SDPCWSI instance.

        Parameters:
            slide_path (str):
                Path to the WSI file.
            **kwargs (dict):
                Keyword arguments forwarded to the base `WSI` class. Most important key is:
                - lazy_init (bool, default=True): Whether to defer loading WSI and metadata.

        Please refer to WSI constructor for all parameters. 

        Example
        -------
        >>> wsi = SDPCWSI(slide_path="path/to/wsi.svs", lazy_init=False)
        >>> print(wsi)
        <width=100000, height=80000, backend=SDPCWSI, mpp=0.25, mag=40>
        """
        super().__init__(slide_path, **kwargs)

    def _lazy_initialize(self) -> None:
        """
        Lazily initialize the WSI using OpenSDPC.

        This method opens a whole-slide image using the OpenSdpc backend, extracting
        key metadata including dimensions, magnification, and multiresolution pyramid
        information. If a tissue segmentation mask is provided, it is also loaded.

        Raises:
            FileNotFoundError:
                If the WSI file or the tissue segmentation mask cannot be found.
            Exception:
                If an unexpected error occurs during WSI initialization.

        Notes:
        After initialization, the following attributes are set:
        - `width` and `height`: spatial dimensions of the base level.
        - `dimensions`: (width, height) tuple from the highest resolution.
        - `level_count`: number of resolution levels in the image pyramid.
        - `level_downsamples`: downsampling factors for each level.
        - `level_dimensions`: image dimensions at each level.
        - `properties`: None for Sdpc Format.
        - `mpp`: microns per pixel, inferred if not manually specified.
        - `mag`: estimated magnification level.
        - `gdf_contours`: loaded from `tissue_seg_path` if provided.
        """

        try:
            import opensdpc
        except ImportError:
            raise ImportError("The opensdpc package is not installed. Please install it using `pip install opensdpc`.")
        
        super()._lazy_initialize()

        if not self._initialized:
            try:
                self.img = opensdpc.OpenSdpc(self.slide_path)
                self.dimensions = self.get_dimensions()
                self.width, self.height = self.dimensions
                self.level_count = self.img.level_count
                self.level_downsamples = self.img.level_downsamples
                self.level_dimensions = self.img.level_dimensions
                self.properties = None
                if self.mpp is None:
                    self.mpp = self.img.readSdpc(self.slide_path).contents.picHead.contents.ruler
                self.mag = self.img.scan_magnification
                self._initialized = True

            except Exception as e:
                raise RuntimeError(f"Failed to initialize WSI with OpenSdpc: {e}") from e

    def _get_closed_thumbnail_level(self, size: Tuple[int, int]) -> int:
        """
        Determine the most appropriate pyramid level for generating a thumbnail
        of the specified size.

        Parameters:
            size (tuple[int, int]):
                Desired (width, height) of the thumbnail.

        Returns:
            int: Pyramid level index that best matches the requested thumbnail size.

        Notes:
        This method selects the highest resolution level where both dimensions
        are greater than or equal to the requested size. If no such level exists,
        it returns the lowest resolution level (highest index).
        """
        for level in range(self.level_count):
            level_width, level_height = self.level_dimensions[level]
            if level_width <= size[0] and level_height <= size[1]:
                return max(0, level - 1)
        return self.level_count - 1

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: ReadMode = 'pil',
    ) -> Union[Image.Image, np.ndarray]:
        """
        Extract a specific region from the whole-slide image (WSI).

        Parameters:
            location (Tuple[int, int]):
                (x, y) coordinates of the top-left corner of the region to extract.
            level (int):
                Pyramid level to read from.
            size (Tuple[int, int]):
                (width, height) of the region to extract.
            read_as ({'pil', 'numpy'}, optional):
                Output format for the region:
                - 'pil': returns a PIL Image (default)
                - 'numpy': returns a NumPy array (H, W, 3)

        Returns:
            Union[PIL.Image.Image, np.ndarray]: Extracted image region in the specified format.

        Raises:
            ValueError:
                If `read_as` is not one of 'pil' or 'numpy'.

        Example
        -------
        >>> region = wsi.read_region((0, 0), level=0, size=(512, 512), read_as='numpy')
        >>> print(region.shape)
        (512, 512, 3)
        """
        region = self.img.read_region(location, level, size).convert('RGB')

        if read_as == 'pil':
            return region
        elif read_as == 'numpy':
            return np.array(region)
        else:
            raise ValueError(f"Invalid `read_as` value: {read_as}. Must be 'pil', 'numpy'.")

    def get_dimensions(self) -> Tuple[int, int]:
        """
        Return the dimensions (width, height) of the WSI.

        Returns:
            tuple[int, int]: (width, height) in pixels.
        """
        return self.img.level_dimensions[0]

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        """
        Generate a thumbnail of the WSI.

        Parameters:
            size (tuple[int, int]):
                Desired (width, height) of the thumbnail.

        Returns:
            PIL.Image.Image: RGB thumbnail as a PIL Image.
        """
        closest_level = self._get_closed_thumbnail_level(size)
        level_width, level_height = self.level_dimensions[closest_level]
        thumbnail = self.read_region((0, 0), closest_level, (level_width, level_height), read_as='pil')
        thumbnail = thumbnail.resize(size, Image.LANCZOS)
        return thumbnail
