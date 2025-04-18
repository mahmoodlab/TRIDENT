from __future__ import annotations
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union

from trident.wsi_objects.WSI import WSI, ReadMode


class CuCIMWSI(WSI):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _lazy_initialize(self) -> None:
        """
        Lazily load the whole-slide image (WSI) and its metadata using CuCIM.

        This method performs deferred initialization by reading the WSI file
        only when needed. It also retrieves key metadata such as dimensions,
        magnification, and microns-per-pixel (MPP). If a tissue segmentation
        mask is available, it is also loaded.

        Raises
        ------
        ImportError
            If `cupy` and/or `cucim` are not installed.
        FileNotFoundError
            If the WSI file or required segmentation mask is missing.
        Exception
            For any other errors that occur while initializing the WSI.

        Notes
        -----
        After initialization, the following attributes are set:
        - `width` and `height`: spatial dimensions of the WSI.
        - `mpp`: microns per pixel, inferred if not already set.
        - `mag`: estimated magnification level of the image.
        - `level_count`, `level_downsamples`, and `level_dimensions`: multiresolution pyramid metadata.
        - `properties`: raw metadata from the image.
        - `gdf_contours`: tissue mask contours, if applicable.
        """

        super()._lazy_initialize()

        try:
            from cucim import CuImage
            import cupy as cp
        except ImportError as e:
            raise ImportError(
                "Required dependencies not found: `cupy` and/or `cucim`.\n"
                "Please install them with:\n"
                "  pip install cucim cupy-cuda12x\n"
                "Make sure `cupy-cuda12x` matches your local CUDA version.\n"
                "Links:\n"
                "  cucim: https://docs.rapids.ai/install/\n"
                "  cupy: https://docs.cupy.dev/en/stable/install.html"
            ) from e

        if not self.lazy_init:
            try:
                self.img = CuImage(self.slide_path)
                self.dimensions = (self.img.size()[1], self.img.size()[0])  # width, height are reverted compared to openslide!!
                self.width, self.height = self.dimensions
                self.level_count = self.img.resolutions['level_count']
                self.level_downsamples = self.img.resolutions['level_downsamples']
                self.level_dimensions = self.img.resolutions['level_dimensions']
                self.properties = self.img.metadata
                if self.mpp is None:
                    self.mpp = self._fetch_mpp(self.custom_mpp_keys)
                self.mag = self._fetch_magnification(self.custom_mpp_keys)
                self.lazy_init = True

            except Exception as e:
                raise RuntimeError(f"Failed to initialize WSI using CuCIM: {e}") from e

    def _fetch_mpp(self, custom_keys: dict = None) -> float:
        """
        Fetch the microns per pixel (MPP) from CuImage metadata.

        Parameters
        ----------
        custom_keys : dict, optional
            Optional dictionary with keys for 'mpp_x' and 'mpp_y' metadata fields to check first.

        Returns
        -------
        float
            MPP value in microns per pixel.

        Raises
        ------
        ValueError
            If MPP cannot be determined from metadata.
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
        
        raise ValueError(
            f"Unable to extract MPP from CuCIM metadata for: '{self.slide_path}'.\n"
            "Suggestions:\n"
            "- Provide `custom_keys` with metadata key mappings for 'mpp_x' and 'mpp_y'.\n"
            "- Set the MPP manually when constructing the CuCIMWSI object."
        )

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

        # Compute the size to read at that level
        level_width, level_height = self.level_dimensions[level]

        # Read region at (0, 0) in target level
        region = self.read_region(
            location=(0, 0),
            size=(level_width, level_height),
            level=level
        ).convert("RGB")
        # region = region.resize((size[1], size[0]), resample=Image.BILINEAR)
        region = region.resize(size, resample=Image.BILINEAR)

        return region

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: ReadMode = 'pil',
    ) -> Union[Image.Image, np.ndarray]:
        """
        Extract a specific region from the whole-slide image (WSI) using CuCIM.

        Parameters
        ----------
        location : Tuple[int, int]
            (x, y) coordinates of the top-left corner of the region to extract.
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
            The extracted region in the specified format.

        Raises
        ------
        ValueError
            If `read_as` is not one of the supported options.

        Example
        -------
        >>> region = wsi.read_region((1000, 1000), level=0, size=(512, 512), read_as='pil')
        >>> region.show()
        """

        import cupy as cp

        region = self.img.read_region(location=location, level=level, size=size, device='cpu')
        region = cp.asnumpy(region)  # Convert from CuPy to NumPy

        if read_as == 'numpy':
            return region
        elif read_as == 'pil':
            return Image.fromarray(region).convert("RGB")
        else:
            raise ValueError(f"Invalid `read_as` value: {read_as}. Must be 'pil' or 'numpy'.")
        
    def get_dimensions(self) -> Tuple[int, int]:
        """
        Return the (width, height) dimensions of the CuCIM-managed WSI.

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

    def segment_tissue(self, **kwargs) -> str:
        out = super().segment_tissue(**kwargs)
        self.close()
        return out
    
    def extract_tissue_coords(self, **kwargs) -> str:
        out = super().extract_tissue_coords(**kwargs)
        self.close()
        return out

    def visualize_coords(self, **kwargs) -> str:
        out = super().visualize_coords(**kwargs)
        self.close()
        return out

    def extract_patch_features(self, **kwargs) -> str:
        out = super().extract_patch_features(**kwargs)
        self.close()
        return out

    def extract_slide_features(self, **kwargs) -> str:
        out = super().extract_slide_features(**kwargs)
        self.close()
        return out

    def close(self):
        if self.img is not None:
            self.img.close()
            self.img = None
            self.lazy_init = False
