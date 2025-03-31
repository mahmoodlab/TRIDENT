from __future__ import annotations
import numpy as np
from PIL import Image
import geopandas as gpd
from typing import List, Tuple, Optional, Union
import torch 

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

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        device: str = 'cpu',
        read_as: str = 'pil',
    ) -> Union[np.ndarray, torch.Tensor, Image.Image]:
        """
        Extract a specific region from the whole-slide image (WSI), returning it as a NumPy array,
        Torch tensor, or PIL image.

        Args:
        -----
        location : Tuple[int, int]
            (x, y) coordinates of the top-left corner of the region to extract.
        level : int
            Pyramid level to read from.
        size : Tuple[int, int]
            (width, height) of the region to extract.
        device : str, optional
            Device used for post-processing. Defaults to 'cpu'. Not used during reading.
        read_as : str, optional
            Format to return the region in. Options are:
            - 'numpy': returns a NumPy array
            - 'torch': returns a Torch tensor (on GPU or CPU)
            - 'pil': returns a PIL Image object (default)

        Returns:
        --------
        Union[np.ndarray, torch.Tensor, PIL.Image.Image]
            The extracted region in the specified format.

        Example:
        --------
        >>> region = wsi.read_region((0, 0), level=0, size=(512, 512), read_as='numpy')
        >>> print(region.shape)
        (512, 512, 3)
        """
        self._ensure_image_open()
        region = self.img.crop((location[0], location[1], location[0] + size[0], location[1] + size[1]))

        if read_as == 'pil':
            return region
        elif read_as == 'numpy':
            return np.array(region)
        elif read_as == 'torch':
            array = np.array(region)
            return torch.from_numpy(array).to(device)
        
        raise ValueError(f"Unsupported read_as value: {read_as}")
    
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

