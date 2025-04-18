from __future__ import annotations
import numpy as np
from PIL import Image
from typing import Tuple, Union

from trident.wsi_objects.WSI import WSI, ReadMode


class ImageWSI(WSI):

    def __init__(self, **kwargs) -> None:
        """
        Initialize a WSI object from a standard image file (e.g., PNG, JPEG, etc.).

        Parameters
        ----------
        slide_path : str
            Path to the image file.
        mpp : float
            Microns per pixel. Required since standard image formats do not store this metadata.
        name : str, optional
            Optional name for the slide.
        lazy_init : bool, default=True
            Whether to defer initialization until the WSI is accessed.

        Raises
        ------
        ValueError
            If the required 'mpp' argument is not provided.

        Example
        -------
        >>> wsi = ImageWSI(slide_path="path/to/image.png", lazy_init=False, mpp=0.51)
        >>> print(wsi)
        <width=5120, height=3840, backend=ImageWSI, mpp=0.51, mag=20>
        """
        mpp = kwargs.get("mpp")
        if mpp is None:
            raise ValueError(
                "Missing required argument `mpp`. Standard image formats do not contain microns-per-pixel "
                "information, so you must specify it manually via the `ImageWSI` constructor."
            )
        
        #enable loading large images.
        from PIL import PngImagePlugin
        PngImagePlugin.MAX_TEXT_CHUNK = 2**30  # ~1GB
        PngImagePlugin.MAX_TEXT_MEMORY = 2**30
        PngImagePlugin.MAX_IMAGE_PIXELS = None  # Optional: disables large image warning

        self.img = None
        super().__init__(**kwargs)

    def _lazy_initialize(self) -> None:
        """
        Lazily initialize the WSI using a standard image file (e.g., JPEG, PNG, etc.).

        This method loads the image using PIL and extracts relevant metadata such as
        dimensions and magnification. It assumes a single-resolution image (no pyramid).
        If a tissue segmentation mask is available, it is also loaded.

        Raises
        ------
        FileNotFoundError
            If the WSI file or tissue segmentation mask is not found.
        Exception
            If an unexpected error occurs during initialization.

        Notes
        -----
        After initialization, the following attributes are set:
        - `width` and `height`: dimensions of the image.
        - `dimensions`: (width, height) tuple of the image.
        - `level_downsamples`: set to `[1]` (single resolution).
        - `level_dimensions`: set to a list containing the image dimensions.
        - `level_count`: set to `1`.
        - `mag`: estimated magnification level.
        - `gdf_contours`: loaded from `tissue_seg_path`, if available.
        """

        super()._lazy_initialize()

        if not self.lazy_init:
            try:
                self._ensure_image_open()
                self.level_downsamples = [1]
                self.dimensions = (self.img.width, self.img.height)
                self.width, self.height = self.dimensions[0], self.dimensions[1]
                self.mag = self._fetch_magnification(self.custom_mpp_keys)
                self.dimensions = self.img.size
                self.level_dimensions = [(self.img.width, self.img.height)]
                self.level_count = 1
                self.lazy_init = True

            except Exception as e:
                raise Exception(f"Error initializing WSI with PIL.Image: {e}")

    def _ensure_image_open(self):
        if self.img is None:
            self.img = Image.open(self.slide_path).convert("RGB")

    def get_dimensions(self):
        return self.dimensions

    def get_thumbnail(self, size):
        """
        Generate a thumbnail of the image.

        Parameters
        ----------
        size : tuple of int
            Desired thumbnail size (width, height).

        Returns
        -------
        PIL.Image.Image
            RGB thumbnail image.
        """
        self._ensure_image_open()
        img = self.img.copy()
        img.thumbnail(size)
        return img

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: ReadMode = 'pil',
    ) -> Union[Image.Image, np.ndarray]:
        """
        Extract a specific region from a single-resolution image (e.g., JPEG, PNG, TIFF).

        Parameters
        ----------
        location : Tuple[int, int]
            (x, y) coordinates of the top-left corner of the region to extract.
        level : int
            Pyramid level to read from. Only level 0 is supported for non-pyramidal images.
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
            If `level` is not 0 or if `read_as` is not one of the supported options.

        Example
        -------
        >>> region = wsi.read_region((0, 0), level=0, size=(512, 512), read_as='numpy')
        >>> print(region.shape)
        (512, 512, 3)
        """
        if level != 0:
            raise ValueError("ImageWSI only supports reading at level=0 (no pyramid levels).")

        self._ensure_image_open()
        region = self.img.crop((
            location[0],
            location[1],
            location[0] + size[0],
            location[1] + size[1]
        )).convert('RGB')

        if read_as == 'pil':
            return region
        elif read_as == 'numpy':
            return np.array(region)
        else:
            raise ValueError(f"Invalid `read_as` value: {read_as}. Must be 'pil' or 'numpy'.")

    def segment_tissue(self, **kwargs):
        out = super().segment_tissue(**kwargs)
        self.close()
        return out
    
    def extract_tissue_coords(self, **kwargs):
        out = super().extract_tissue_coords(**kwargs)
        self.close()
        return out

    def visualize_coords(self, **kwargs):
        out = super().visualize_coords(**kwargs)
        self.close()
        return out

    def extract_patch_features(self, **kwargs):
        out = super().extract_patch_features(**kwargs)
        self.close()
        return out

    def extract_slide_features(self, **kwargs):
        out = super().extract_slide_features(**kwargs)
        self.close()
        return out

    def close(self):
        """
        Close the internal image object to free memory. These can take several GB in RAM.
        """
        if self.img is not None:
            self.img.close()
            self.img = None
