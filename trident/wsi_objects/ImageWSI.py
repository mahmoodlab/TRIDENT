from __future__ import annotations
import numpy as np
from PIL import Image
import geopandas as gpd

from trident.wsi_objects.WSI import WSI


class ImageWSI(WSI):

    def __init__(self, **kwargs) -> None:
        """
        
        Example:
        --------
        >>> wsi = ImageWSI("path/to/wsi.png", lazy_init=False, mpp=0.51)
        >>> print(wsi)
        <width=100000, height=80000, backend=ImageWSI, mpp=0.51, mag=20>
        """

        mpp = kwargs.get("mpp")
        if mpp is None:
            raise ValueError("Missing argument 'mpp'. Please provide the micron per pixel. In ImageWSI, mpp must be provided as it cannot be stored in metadata.")
        self.img = None
        super().__init__(**kwargs)

    def _lazy_initialize(self) -> None:
        """
        The `_lazy_initialize` function from the class `ImageWSI` Perform lazy initialization by loading the WSI file and its metadata.

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
                self._ensure_image_open()
                self.level_downsamples = [1]
                self.width, self.height = self.dimensions
                self.mag = self._fetch_magnification(self.custom_mpp_keys)
                self.lazy_init = True

                if self.tissue_seg_path is not None:
                    try:
                        self.gdf_contours = gpd.read_file(self.tissue_seg_path)
                    except FileNotFoundError:
                        raise FileNotFoundError(f"Tissue segmentation file not found: {self.tissue_seg_path}")
            except Exception as e:
                raise Exception(f"Error initializing WSI with PIL.Image: {e}")

    def _ensure_image_open(self):
        if self.img is None:
            self.img = Image.open(self.slide_path)

    def get_dimensions(self):
        return self.dimensions

    @property
    def level_dimensions(self):
        self._ensure_image_open()
        return [(self.img.width, self.img.height)]

    @property
    def dimensions(self):
        self._ensure_image_open()
        return self.img.size

    @property
    def level_count(self):
        return 1

    def get_thumbnail(self, size):
        self._ensure_image_open()
        img = self.img.copy()
        img.thumbnail(size)
        return img

    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> np.ndarray:
        """
        Returns a region as a NumPy array.
        """
        pil_image = self.read_region_pil(location, level, size)
        return np.array(pil_image)

    def read_region_pil(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> Image.Image:
        """
        Returns a region as a PIL image in RGBA mode.
        """
        self._ensure_image_open()
        region = self.img.crop((location[0], location[1], location[0] + size[0], location[1] + size[1]))
        return region.convert("RGB")

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
        if self.img is not None:
            self.img.close()
            self.img = None

