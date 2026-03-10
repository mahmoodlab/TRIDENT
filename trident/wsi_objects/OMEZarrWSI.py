import ngff_zarr as nz
from typing import Tuple, Union, Optional, Any
from trident.wsi_objects.WSI import WSI, ReadMode
from cf_units import Unit as cf_Unit
from PIL import Image
import numpy as np

import dask

class OMEZarrWSI(WSI):
    """
    WSI implementation for reading zarrfiles following the OME specification.
    """
    def __init__(self, slide_path: str, **kwargs: Any) -> None:
        """
        Initialize a OMEZarr instance for OME-Zarr whole-slide images.

        Parameters
        ----------
        slide_path : str
            Path to an .ome.zarr multiscale file.
        **kwargs : dict
            Additional keyword arguments forwarded to the base `WSI` class.
            - lazy_init (bool, default=True): Whether to defer loading WSI and metadata.

        Example
        -------
        >>> wsi = OMEZarrWSI(slide_path="path/to/wsi", lazy_init=False)
        >>> print(wsi)
        <width=100000, height=80000, backend=OMEZarrWSI, mpp=0.25, mag=40>
        """
        super().__init__(slide_path, **kwargs)

    def _lazy_initialize(self) -> None:
        """
        Lazily initialize the WSI using ngff-zarr.

        This method opens a whole-slide image using the ngff-zarr backend, extracting
        key metadata including dimensions, magnification, and multiresolution pyramid
        information. If a tissue segmentation mask is provided, it is also loaded.

        Raises
        ------
        FileNotFoundError
            If the WSI file or the tissue segmentation mask cannot be found.
        RuntimeError
            If an unexpected error occurs during WSI initialization. Including if there 
            are not 3 dimensions in an image, as read_region depends on this property.

        Notes
        -----
        After initialization, the following attributes are set:
        - `width` and `height`: spatial dimensions of the base level.
        - `dimensions`: (width, height) tuple from the highest resolution.
        - `level_count`: number of resolution levels in the image pyramid.
        - `level_downsamples`: downsampling factors for each level.
        - `level_dimensions`: image dimensions at each level.
        - `properties`: metadata dictionary from OpenSlide.
        - `mpp`: microns per pixel, inferred if not manually specified.
        - `mag`: estimated magnification level (via WSI.py).
        - `gdf_contours`: loaded from `tissue_seg_path` if provided (via WSI.py).
        """

        super()._lazy_initialize()

        _get_W_and_H = lambda ngffimg: (ngffimg.data.shape[-1], ngffimg.data.shape[-2])

        if not self._initialized:
            try:
                self.img = nz.from_ngff_zarr(self.slide_path) # Multiscales dataclass from ngff-zarr
 
                toplevel_image = self.img.images[0] # a possibly cyx shape NgffImage object
                assert len(toplevel_image.data.shape) == 3, "Err, read_region expects 3 dimensional image data"

                self.dimensions = _get_W_and_H(toplevel_image) # based on cyx array storage x -> width, y -> height
                self.width, self.height = self.dimensions
                self.level_count = len(self.img.images)
                self.level_dimensions = tuple(map(_get_W_and_H, self.img.images))
                self.level_downsamples = self._fetch_downsamples()
                if self.mpp is None:
                    self.mpp = self._fetch_mpp()
                self.mag = self._fetch_magnification()
                self.properties = self.img.metadata # Properties here are limited to OME rather than the whole zarrfile

                self._initialized = True

            except Exception as e:
                raise RuntimeError(f"Failed to initialize WSI with ngff-zarr: {e}") from e

    def _fetch_mpp(self):
        """
        Retrieve microns per pixel (MPP) from OME Zarr metadata. Conforming to the OME zarr
        specification requires scale and UDUNITS-2, so custom_mpp_keys not requried.

        Returns
        -------
        np.float64
            MPP value in microns per pixel.
        """
        scale, scale_unit = self.img.images[0].scale['x'], self.img.images[0].axes_units['x']
        return cf_Unit(scale_unit).convert(scale, cf_Unit('micrometers')) # mpp for the x axis at the highest res image
    
    def _fetch_downsamples(self):
        return tuple( [1.] 
            + [(self.img.images[0].data.shape[-1] / ngff_img.data.shape[-1]) for ngff_img in self.img.images[1:]]
        )

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: ReadMode = 'pil',
    ) -> Union[Image.Image, np.ndarray]:
        """
        Extract a specific region from the whole-slide image (WSI).

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
            Extracted image region in the specified format.

        Raises
        ------
        ValueError
            If `read_as` is not one of 'pil' or 'numpy'.

        Examples
        --------
        >>> region = wsi.read_region((0, 0), level=0, size=(512, 512), read_as='numpy')
        >>> print(region.shape)
        (512, 512, 3)
        """
        # 'location' is relative to the level as calls are made to the data array
        location_ = (int(location[0] / self.level_downsamples[level]), int(location[1] / self.level_downsamples[level]))

        x, y = location_
        width_size, height_size = size

        # prevent deadlock that occurs when reading while nested in pytorch's distributed operations
        with dask.config.set(scheduler='synchronous'):
            # imgs are ordered cyx, so [: -> c, y:y+height_size, x:x+width_size, ]
            # also convert cyx to desired H,W,C
            region = self.img.images[level].data[:, y:y+height_size, x:x+width_size].compute().transpose(1, 2, 0)

        if read_as == 'pil':
            return Image.fromarray(region).convert('RGB')
        elif read_as == 'numpy':
            return region
        else:
            raise ValueError(f"Invalid `read_as` value: {read_as}. Must be 'pil', 'numpy'.")
    
    def get_dimensions(self) -> Tuple[int, int]:
        """
        Return the dimensions (width, height) of the WSI.

        Returns
        -------
        tuple of int
            (width, height) in pixels.
        """
        return self.dimensions
    
    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        """
        Generate a thumbnail of the WSI.

        Parameters
        ----------
        size : tuple of int
            Desired (width, height) of the thumbnail.

        Returns
        -------
        PIL.Image.Image
            RGB thumbnail as a PIL Image.
        """
        width, height = size
        # takes the average ratio between the thumbsize and the object's (level dimension) size then applies abs(x - 1) so min finds
        # the size ratio closest to 1
        get_dim_to_size_adjusted_ratio = lambda x: abs((((x[0]/width) + (x[1]/height)) / 2) - 1)
        # get the min index rather than value
        closest_level = min(range(self.level_count), key=lambda i: list(map(get_dim_to_size_adjusted_ratio, self.level_dimensions))[i])

        thumbimg_data = self.img.images[closest_level].data.compute().transpose(1, 2, 0)
        return Image.fromarray(thumbimg_data).convert('RGB').resize(size)