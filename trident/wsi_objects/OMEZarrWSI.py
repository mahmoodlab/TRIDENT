from typing import Tuple, Union, Any
from trident.wsi_objects.WSI import WSI, ReadMode
from PIL import Image
import numpy as np
import warnings

try:
    from zarr import open as zarr_open
    from dask.config import set as dask_config_set
    from ngff_zarr import from_ngff_zarr
    from cf_units import Unit as cf_Unit

    _HAS_OME_ZARR = True
    _EXCEPT_MESSAGE = None
except ImportError as e: # ModuleNotFoundError is likely
    _HAS_OME_ZARR = False
    _EXCEPT_MESSAGE = e


class OMEZarrWSI(WSI):
    """
    WSI implementation for reading zarrfiles following the OME specification.
    """

    def __init__(self, slide_path: str, **kwargs: Any) -> None:
        """
        Initialize a OMEZarr instance for OME-Zarr whole-slide images.

        Parameters:
            slide_path (str):
                Path to an .zarr OME multiscale file.
            **kwargs (dict):
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

        Raises:
            FileNotFoundError:
                If the WSI file or the tissue segmentation mask cannot be found.
            RuntimeError:
                If an unexpected error occurs during WSI initialization. Including if there
                are not 3 dimensions in an image, as read_region depends on this property.

        Notes:
        After initialization, the following attributes are set:
        - `width` and `height`: spatial dimensions of the base level.
        - `dimensions`: (width, height) tuple from the highest resolution.
        - `level_count`: number of resolution levels in the image pyramid.
        - `level_downsamples`: downsampling factors for each level.
        - `level_dimensions`: image dimensions at each level.
        - `properties`: metadata object from ngff-zarr.
        - `mpp`: microns per pixel, inferred if not manually specified.
        - `mag`: estimated magnification level (via WSI.py).
        - `gdf_contours`: loaded from `tissue_seg_path` if provided (via WSI.py).
        """

        super()._lazy_initialize()

        if not self._initialized:

            if not _HAS_OME_ZARR:
                raise ImportError(
                    "ngff-zarr, zarr, dask, and cf_units are required for omezarr support. "
                    "Install them with pip, or pip install .[omezarr] when installing TRIDENT. "
                    f"When trying to import, got message {_EXCEPT_MESSAGE}"
                )

            try:
                self.img = from_ngff_zarr(
                    self.slide_path
                )  # Multiscales dataclass from ngff-zarr

                idx_tuple, dimname_tuple = self._fetch_dimension_metadata()
                self._idx_x, self._idx_y, self._idx_c = idx_tuple
                self._xname, self._yname, self._cname = dimname_tuple

                self._transpose_order = (self._idx_y, self._idx_x, self._idx_c)

                # x -> width, y -> height
                _get_W_and_H = lambda ngffimg: (
                    ngffimg.data.shape[self._idx_x],
                    ngffimg.data.shape[self._idx_y],
                )
                self.dimensions = _get_W_and_H(
                    self.img.images[0]
                )  # use the top level image (largest resolution)

                self.width, self.height = self.dimensions
                self.level_count = len(self.img.images)
                self.level_dimensions = tuple(map(_get_W_and_H, self.img.images))
                self.level_downsamples = self._fetch_downsamples()
                if self.mpp is None:
                    self.mpp = self._fetch_mpp()
                self.mag = self._fetch_magnification()
                try:
                    self.properties = dict(
                        zarr_open(self.slide_path).attrs
                    )  # get the whole zarr.json object
                except:
                    self.properties = None

                self._initialized = True

            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize WSI with ngff-zarr: {e}"
                ) from e

    def _fetch_mpp(self):
        """
        Retrieve microns per pixel (MPP) from OME Zarr metadata. The OME spec
        has a designated axes unit property in UDUNITS-2, so custom_mpp_keys not requried.

        Returns:
            np.float64: MPP value in microns per pixel.
        """
        try:
            scale, scale_unit = (
                self.img.images[0].scale[self._xname],
                self.img.images[0].axes_units[self._xname],
            )
            return cf_Unit(scale_unit).convert(
                scale, cf_Unit("micrometers")
            )  # mpp for the x axis at the highest res image
        except:
            raise ValueError(
                f"Unable to extract MPP from slide metadata: '{self.slide_path}'.\n"
                "Suggestions:\n"
                "- Set the unit in the x/width axes metadata of the OME-Zarr Multiscales "
                "(likely having to update the corresponding scale property).\n"
                "- Set the MPP explicitly via the class constructor.\n"
                "- If using the `run_batch_of_slides.py` script, pass the MPP via the "
                "`--custom_list_of_wsis` argument in a CSV file. Refer to TRIDENT/README/Q&A."
            )


    def _fetch_downsamples(self):
        """
        Calculate the downsampling factors for each resolution level.

        For OME-Zarr, x and y downsampling should be consistent across pyramid
        levels. Some stores can drift slightly due to rounding, so we compute
        both x and y ratios, validate they match within tolerance, and return
        the average when they differ.

        Returns:
            Tuple[float]: Downsample factors for each level in the image pyramid.
        """
        base_x = self.img.images[0].data.shape[self._idx_x]
        base_y = self.img.images[0].data.shape[self._idx_y]

        rtol = 1e-3

        downsamples: list[float] = [1.0]
        for ngff_img in self.img.images[1:]:
            lvl_x = ngff_img.data.shape[self._idx_x]
            lvl_y = ngff_img.data.shape[self._idx_y]

            down_x = base_x / lvl_x
            down_y = base_y / lvl_y

            if not np.isclose(down_x, down_y, rtol=rtol, atol=0.0):
                warnings.warn(
                    "OMEZarrWSI: x/y downsample mismatch between pyramid levels "
                    f"(down_x={down_x:.6f}, down_y={down_y:.6f}); using average.",
                    RuntimeWarning,
                )

            downsamples.append((down_x + down_y) / 2.0)

        return tuple(downsamples)

    def _fetch_dimension_metadata(self):
        """
        Parse dimension metadata to identify spatial and channel axes.

        Extracts and maps the indices and original string names for the x-axis, 
        y-axis, and channel dimensions from the image metadata.

        Returns:
            Tuple[Tuple[int, int, int], Tuple[str, str, str]]:
                A pair of tuples containing the integer indices (idx_x, idx_y, idx_c) and the matched string names
                (x_name, y_name, c_name), respectively.

        Raises:
            AssertionError:
                If the image does not have exactly 3 dimensions or contains unrecognized dimension names.
            ValueError:
                If the dimensions do not consist of exactly one X-type, one Y-type, and one C-type axis.
        """

        dimnames = self.img.metadata.dimension_names
        possible_dimnames_lowercase = {"x", "y", "c", "width", "height", 'channel'}

        strlower = lambda x: x.lower()
        assert (len(dimnames) == 3) and (
            set(map(strlower, dimnames)).issubset(possible_dimnames_lowercase)
        ), f"Err, read_region expects 3 dimensional image data with {possible_dimnames_lowercase} dim names, found {dimnames}"

        try:
            _xname = next(d for d in dimnames if d.lower() in {"x", "width"})
            _yname = next(d for d in dimnames if d.lower() in {"y", "height"})
            _cname = next(d for d in dimnames if d.lower() in {"c", "channel"})
        except:
            raise ValueError(
                "Err, expecting one of each space/channel type dim in "
                f"{possible_dimnames_lowercase}, found {dimnames}."
            )

        _dimname_to_index = {
            name: i for i, name in enumerate(self.img.metadata.dimension_names)
        }
        _idx_x, _idx_y, _idx_c = (
            _dimname_to_index[_xname],
            _dimname_to_index[_yname],
            _dimname_to_index[_cname],
        )

        return (_idx_x, _idx_y, _idx_c), (_xname, _yname, _cname)

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: ReadMode = "pil",
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
        # 'location' is relative to the level as calls are made to the data array
        downsample_factor = self.level_downsamples[level]
        location_ = (
            int(location[0] / downsample_factor),
            int(location[1] / downsample_factor),
        )

        x, y = location_
        width_size, height_size = size

        region_as_slice = [None, None, None]
        region_as_slice[self._idx_y] = slice(y, y + height_size)
        region_as_slice[self._idx_x] = slice(x, x + width_size)
        region_as_slice[self._idx_c] = slice(None)
        region_as_slice = tuple(region_as_slice)

        # prevent deadlock that occurs when reading while nested in pytorch's distributed operations
        with dask_config_set(scheduler="synchronous"):
            region = (
                self.img.images[level]
                .data[region_as_slice]
                .compute()
                .transpose(self._transpose_order)
            )

        if read_as == "pil":
            return Image.fromarray(region).convert("RGB")
        elif read_as == "numpy":
            return region
        else:
            raise ValueError(
                f"Invalid `read_as` value: {read_as}. Must be 'pil', 'numpy'."
            )

    def get_dimensions(self) -> Tuple[int, int]:
        """
        Return the dimensions (width, height) of the WSI.

        Returns:
            tuple[int, int]: (width, height) in pixels.
        """
        return self.dimensions

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        """
        Generate a thumbnail of the WSI.

        Parameters:
            size (tuple[int, int]):
                Desired (width, height) of the thumbnail.

        Returns:
            PIL.Image.Image: RGB thumbnail as a PIL Image.
        """
        width, height = size
        # takes the average ratio between the thumbsize and the object's (level dimension) size then applies abs(x - 1) so min finds
        # the size ratio closest to 1
        get_dim_to_size_adjusted_ratio = lambda x: abs(
            (((x[0] / width) + (x[1] / height)) / 2) - 1
        )
        # get the min index rather than value
        closest_level = min(
            range(self.level_count),
            key=lambda i: list(
                map(get_dim_to_size_adjusted_ratio, self.level_dimensions)
            )[i],
        )

        thumbimg_data = (
            self.img.images[closest_level]
            .data.compute()
            .transpose(self._transpose_order)
        )
        return Image.fromarray(thumbimg_data).convert("RGB").resize(size)
