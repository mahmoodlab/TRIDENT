from __future__ import annotations
import numpy as np
import openslide
from PIL import Image
import os 
import warnings
import torch 
from typing import List, Tuple, Optional
from torch.utils.data import DataLoader

from trident.wsi_objects.WSIPatcher import *
from trident.wsi_objects.WSIPatcherDataset import WSIPatcherDataset
from trident.IO import (
    save_h5, read_coords, read_coords_legacy,
    mask_to_gdf, overlay_gdf_on_thumbnail, get_num_workers
)
class OpenSlideWSI:
    """
    The `OpenSlideWSI` class provides an interface to work with Whole Slide Images (WSIs) using OpenSlide. 
    It supports lazy initialization, metadata extraction, patching, and advanced operations such as 
    tissue segmentation and feature extraction. The class handles various WSI file formats and 
    offers utilities for integration with AI models.

    Attributes:
    -----------
    slide_path : str
        Path to the WSI file.
    name : str
        Name of the WSI (inferred from the file path if not provided).
    custom_mpp_keys : dict
        Custom keys for extracting microns per pixel (MPP) and magnification metadata.
    lazy_init : bool
        Indicates whether lazy initialization is used.
    tissue_seg_path : str
        Path to a tissue segmentation mask (if available).
    width : int
        Width of the WSI in pixels (lazy initialized).
    height : int
        Height of the WSI in pixels (lazy initialized).
    mpp : float
        Microns per pixel (lazy initialized).
    mag : float
        Magnification level (lazy initialized).
    """

    def __init__(
        self,
        slide_path: str,
        name: Optional[str] = None,
        tissue_seg_path: Optional[str] = None,
        custom_mpp_keys: Optional[List[str]] = None,
        lazy_init: bool = True,
        mpp: Optional[float] = None
    ):
        """
        Initialize the `OpenSlideWSI` object for working with a Whole Slide Image (WSI).

        Args:
        -----
        slide_path : str
            Path to the WSI file.
        name : str, optional
            Optional name for the WSI. Defaults to the filename (without extension).
        tissue_seg_path : str, optional
            Path to the tissue segmentation mask file. Defaults to None.
        custom_mpp_keys : dict, optional
            Custom keys for extracting MPP and magnification metadata. Defaults to None.
        lazy_init : bool, optional
            If True, defer loading the WSI until required. Defaults to True.
        mpp: float, optional
            If not None, will be the reference micron per pixel (mpp). Handy when mpp is not provided in the WSI.

        Example:
        --------
        >>> wsi = OpenSlideWSI("path/to/wsi.svs", lazy_init=False)
        >>> print(wsi)
        <width=100000, height=80000, backend=OpenSlideWSI, mpp=0.25, mag=40>
        """
        self.slide_path = slide_path
        if name is None:
            self.name, self.ext = os.path.splitext(os.path.basename(slide_path)) 
        else:
            self.name, self.ext = os.path.splitext(name)
        self.tissue_seg_path = tissue_seg_path
        self.custom_mpp_keys = custom_mpp_keys

        self.width, self.height = None, None  # Placeholder dimensions
        self.mpp = mpp  # Placeholder microns per pixel. Defaults will be None unless specified in constructor. 
        self.mag = None  # Placeholder magnification
        self.lazy_init = lazy_init  # Initialize immediately if lazy_init is False

        if not self.lazy_init:
            self._lazy_initialize()
        else: 
            self.lazy_init = not self.lazy_init

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
                        # self.mask = cv2.imread(self.tissue_seg_path, cv2.IMREAD_GRAYSCALE)
                        self.gdf_contours = gpd.read_file(self.tissue_seg_path)
                    except FileNotFoundError:
                        raise FileNotFoundError(f"Tissue segmentation file not found: {self.tissue_seg_path}")
            except Exception as e:
                raise Exception(f"Error initializing WSI: {e}")

    def __repr__(self) -> str:
        if self.lazy_init:
            return f"<width={self.width}, height={self.height}, backend={self.__class__.__name__}, mpp={self.mpp}, mag={self.mag}>"
        else:
            return f"<name={self.name}>"
    
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

    def read_region(
        self, 
        location: Tuple[int, int], 
        level: int, 
        size: Tuple[int, int]
    ) -> np.ndarray:
        """
        The `read_region` function from the class `OpenSlideWSI` Extract a specific region from the WSI as a NumPy array.

        Args:
        -----
        location : Tuple[int, int]
            Coordinates (x, y) of the top-left corner of the region.
        level : int
            Pyramid level to read from.
        size : Tuple[int, int]
            Width and height of the region.

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
        return np.array(self.read_region_pil(location, level, size))
    
    def read_region_pil(
        self, 
        location: Tuple[int, int], 
        level: int, 
        size: Tuple[int, int]
    ) -> Image.Image:
        """
        The `read_region_pil` function from the class `OpenSlideWSI` Extract a specific region from the WSI as a PIL Image.

        Args:
        -----
        location : Tuple[int, int]
            Coordinates (x, y) of the top-left corner of the region.
        level : int
            Pyramid level to read from.
        size : Tuple[int, int]
            Width and height of the region.

        Returns:
        --------
        Image.Image:
            The extracted region as a PIL Image in RGB format.

        Example:
        --------
        >>> region = wsi.read_region_pil((0, 0), level=0, size=(512, 512))
        >>> region.show()
        """
        return self.img.read_region(location, level, size).convert('RGB')

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

    def create_patcher(
        self, 
        patch_size: int, 
        src_pixel_size: float | None = None, 
        dst_pixel_size: float | None = None,
        src_mag: int | None = None, 
        dst_mag: int | None = None,
        overlap: int = 0, 
        mask: gpd.GeoDataFrame | None = None, 
        coords_only: bool = False, 
        custom_coords: np.ndarray | None = None,
        threshold: float = 0.15,
        pil: bool = False
    ) -> OpenSlideWSIPatcher:
        """
        The `create_patcher` function from the class `OpenSlideWSI` Create a patcher object for extracting patches from the WSI.

        Args:
        -----
        patch_size : int
            Size of each patch in pixels.
        src_pixel_size : float, optional
            Source pixel size. Defaults to None.
        dst_pixel_size : float, optional
            Destination pixel size. Defaults to None.
        ...

        Returns:
        --------
        WSIPatcher:
            An object for extracting patches.

        Example:
        --------
        >>> patcher = wsi.create_patcher(patch_size=512, src_pixel_size=0.25, dst_pixel_size=0.5)
        >>> for patch in patcher:
        ...     process(patch)
        """
        return OpenSlideWSIPatcher(
            self, patch_size, src_pixel_size, dst_pixel_size, src_mag, dst_mag,
            overlap, mask, coords_only, custom_coords, threshold, pil
        )
    
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

    def _fetch_magnification(self, custom_mpp_keys: List[str] | None = None) -> int | None:
        """
        The `_fetch_magnification` function of the class `OpenSlideWSI` calculates the magnification level 
        of the WSI based on the microns per pixel (MPP) value or other metadata. The magnification levels are 
        approximated to commonly used values such as 80x, 40x, 20x, etc. If the MPP is unavailable or insufficient 
        for calculation, it attempts to fallback to metadata-based values.

        Args:
        -----
        custom_mpp_keys : List[str] | None, optional
            Custom keys to search for MPP values in the WSI properties. Defaults to None.

        Returns:
        --------
        int | None:
            The approximated magnification level, or None if the magnification could not be determined.

        Raises:
        -------
        ValueError:
            If the identified MPP is too low for valid magnification values.

        Example:
        --------
        >>> mag = wsi._fetch_magnification()
        >>> print(mag)
        40
        """
        try:
            if self.mpp is None:
                mpp_x = self._fetch_mpp(custom_mpp_keys)
            else:
                mpp_x = self.mpp

            if mpp_x is not None:
                if mpp_x < 0.16:
                    return 80
                elif mpp_x < 0.2:
                    return 60
                elif mpp_x < 0.3:
                    return 40
                elif mpp_x < 0.6:
                    return 20
                elif mpp_x < 1.2:
                    return 10
                elif mpp_x < 2.4:
                    return 5
                else:
                    raise ValueError(f"Identified mpp is too low: mpp={mpp_x}")

            # Use metadata-based magnification as a fallback if mpp_x is not available
            magnification = self.img.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
            if magnification is not None:
                return int(magnification)

        except openslide.OpenSlideError as e:
            print(f"Error: Failed to process WSI properties. {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        # Return None if magnification couldn't be determined
        return None

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def segment_tissue(
        self,
        segmentation_model: torch.nn.Module,
        target_mag: int = 10,
        holes_are_tissue: bool = True,
        job_dir: str | None = None,
        batch_size: int = 16,
    ) -> str:
        """
        The `segment_tissue` function of the class `OpenSlideWSI` segments tissue regions in the WSI using 
        a specified segmentation model. It processes the WSI at a target magnification level, optionally 
        treating holes in the mask as tissue. The segmented regions are saved as thumbnails and GeoJSON contours.

        Args:
        -----
        segmentation_model : torch.nn.Module
            The model used for tissue segmentation.
        target_mag : int, optional
            Target magnification level for segmentation. Defaults to 10.
        holes_are_tissue : bool, optional
            Whether to treat holes in the mask as tissue. Defaults to True.
        job_dir : str | None, optional
            Directory to save the segmentation results. Defaults to None.
        batch_size : int, optional
            Batch size for processing patches. Defaults to 16.

        Returns:
        --------
        str:
            The absolute path to where the segmentation as GeoJSON is saved. 

        Example:
        --------
        >>> wsi.segment_tissue(segmentation_model, target_mag=10, job_dir="output_dir")
        >>> # Results saved in "output_dir"
        """

        self._lazy_initialize()
        max_dimension = 1000
        if self.width > self.height:
            thumbnail_width = max_dimension
            thumbnail_height = int(thumbnail_width * self.height / self.width)
        else:
            thumbnail_height = max_dimension
            thumbnail_width = int(thumbnail_height * self.width / self.height)
        thumbnail = self.get_thumbnail((thumbnail_width, thumbnail_height))

        # Get patch iterator
        destination_mpp = 10 / target_mag
        patcher = self.create_patcher(
            patch_size = segmentation_model.input_size,
            src_pixel_size = self.mpp,
            dst_pixel_size = destination_mpp
        )
        precision = segmentation_model.precision
        device = segmentation_model.device
        eval_transforms = segmentation_model.eval_transforms
        dataset = WSIPatcherDataset(patcher, eval_transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=get_num_workers(batch_size), pin_memory=True)

        mpp_reduction_factor = self.mpp / destination_mpp
        width, height = self.get_dimensions()
        width, height = int(width * mpp_reduction_factor), int(height * mpp_reduction_factor)
        predicted_mask = np.zeros((height, width), dtype=np.uint8)

        for imgs, (xcoords, ycoords) in dataloader:

            imgs = imgs.to(device, dtype=precision)  # Move to device and match dtype
            with torch.autocast(device_type=device.split(":")[0], dtype=precision, enabled=(precision != torch.float32)):
                preds = segmentation_model(imgs).cpu().numpy()

            x_starts = np.round(xcoords.numpy() * mpp_reduction_factor).astype(int)
            y_starts = np.round(ycoords.numpy() * mpp_reduction_factor).astype(int)
            x_ends = np.clip(x_starts + segmentation_model.input_size, 0, width)
            y_ends = np.clip(y_starts + segmentation_model.input_size, 0, height)
            
            for i in range(len(preds)):
                x_start, x_end = x_starts[i], x_ends[i]
                y_start, y_end = y_starts[i], y_ends[i]
                patch_pred = preds[i][:y_end - y_start, :x_end - x_start]
                predicted_mask[y_start:y_end, x_start:x_end] += patch_pred
        
        # Post-process the mask
        predicted_mask = (predicted_mask > 0).astype(np.uint8) * 255

        # Fill holes if desired
        # if not holes_are_tissue:
        holes, _ = cv2.findContours(predicted_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for hole in holes:
            cv2.drawContours(predicted_mask, [hole], 0, 255, -1)

        # Save thumbnail image
        thumbnail_saveto = os.path.join(job_dir, 'thumbnails', f'{self.name}.jpg')
        os.makedirs(os.path.dirname(thumbnail_saveto), exist_ok=True)
        thumbnail.save(thumbnail_saveto)

        # Save geopandas contours
        gdf_saveto = os.path.join(job_dir, 'contours_geojson', f'{self.name}.geojson')
        os.makedirs(os.path.dirname(gdf_saveto), exist_ok=True)
        gdf_contours = mask_to_gdf(
            mask=predicted_mask,
            max_nb_holes=0 if holes_are_tissue else 5,
            min_contour_area=1000,
            pixel_size=self.mpp,
            contour_scale=1/mpp_reduction_factor
        )
        gdf_contours.set_crs("EPSG:3857", inplace=True)  # used to silent warning // Web Mercator
        gdf_contours.to_file(gdf_saveto, driver="GeoJSON")
        self.gdf_contours = gdf_contours
        self.tissue_seg_path = gdf_saveto

        # Draw the contours on the thumbnail image
        contours_saveto = os.path.join(job_dir, 'contours', f'{self.name}.jpg')
        annotated = np.array(thumbnail)
        overlay_gdf_on_thumbnail(gdf_contours, annotated, contours_saveto, thumbnail_width / self.width)

        return gdf_saveto

    def get_best_level_and_custom_downsample(
        self,
        downsample: float,
        tolerance: float = 0.01
    ) -> Tuple[int, float]:
        """
        The `get_best_level_and_custom_downsample` function of the class `OpenSlideWSI` determines the best level 
        and custom downsample factor to approximate a desired downsample value. It identifies the most suitable 
        resolution level of the WSI and calculates any additional scaling required.

        Args:
        -----
        downsample : float
            The desired downsample factor.
        tolerance : float, optional
            Tolerance for rounding differences. Defaults to 0.01.

        Returns:
        --------
        Tuple[int, float]:
            The closest resolution level and the custom downsample factor.

        Raises:
        -------
        ValueError:
            If no suitable resolution level is found for the specified downsample factor.

        Example:
        --------
        >>> level, custom_downsample = wsi.get_best_level_and_custom_downsample(2.5)
        >>> print(level, custom_downsample)
        2, 1.1
        """
        level_downsamples = self.level_downsamples

        # First, check for an exact match within tolerance
        for level, level_downsample in enumerate(level_downsamples):
            if abs(level_downsample - downsample) <= tolerance:
                return level, 1.0  # Exact match, no custom downsampling needed

        if downsample >= level_downsamples[0]:
            # Downsampling: find the highest level_downsample less than or equal to the desired downsample
            closest_level = None
            closest_downsample = None
            for level, level_downsample in enumerate(level_downsamples):
                if level_downsample <= downsample:
                    closest_level = level
                    closest_downsample = level_downsample
                else:
                    break  # Since level_downsamples are sorted, no need to check further
            if closest_level is not None:
                custom_downsample = downsample / closest_downsample
                return closest_level, custom_downsample
        else:
            # Upsampling: find the smallest level_downsample greater than or equal to the desired downsample
            for level, level_downsample in enumerate(level_downsamples):
                if level_downsample >= downsample:
                    custom_downsample = level_downsample / downsample
                    return level, custom_downsample

        # If no suitable level is found, raise an error
        raise ValueError(f"No suitable level found for downsample {downsample}.")

    def extract_tissue_coords(
        self,
        target_mag: int,
        patch_size: int,
        save_coords: str,
        overlap: int = 0,
        min_tissue_proportion: float  = 0.,
    ) -> str:
        """
        The `extract_tissue_coords` function of the class `OpenSlideWSI` extracts patch coordinates 
        from tissue regions in the WSI. It generates coordinates of patches at the specified 
        magnification and saves the results in an HDF5 file.

        Args:
        -----
        target_mag : int
            Target magnification level for the patches.
        patch_size : int
            Size of each patch at the target magnification.
        save_coords : str
            Directory path to save the extracted coordinates.
        overlap : int, optional
            Overlap between patches in pixels. Defaults to 0.
        min_tissue_proportion: float, optional 
            Minimum proportion of the patch under tissue to be kept. Defaults to 0. 

        Returns:
        --------
        str:
            The absolute file path to the saved HDF5 file containing the patch coordinates.

        Example:
        --------
        >>> coords_path = wsi.extract_tissue_coords(20, 256, "output_coords", overlap=32)
        >>> print(coords_path)
        output_coords/patches/sample_name_patches.h5
        """

        self._lazy_initialize()

        patcher = self.create_patcher(
            patch_size=patch_size,
            src_mag=self.mag,
            dst_mag=target_mag,
            mask=self.gdf_contours,
            coords_only=True,
            overlap=overlap,
            threshold=min_tissue_proportion,
        )

        coords_to_keep = [(x, y) for x, y in patcher]

        # Prepare assets for saving
        assets = {'coords' : np.array(coords_to_keep)}
        attributes = {
            'patch_size': patch_size, # Reference frame: patch_level
            'patch_size_level0': patch_size * self.mag // target_mag, # Reference frame: level0
            'level0_magnification': self.mag,
            'target_magnification': target_mag,
            'overlap': overlap,
            'name': self.name,
            'savetodir': save_coords
        }

        # Save the assets and attributes to an hdf5 file
        os.makedirs(os.path.join(save_coords, 'patches'), exist_ok=True)
        out_fname = os.path.join(save_coords, 'patches', str(self.name) + '_patches.h5')
        save_h5(out_fname,
                assets = assets,
                attributes = {'coords': attributes},
                mode='w')
        
        return out_fname

    def visualize_coords(self, coords_path: str, save_patch_viz: str) -> str:
        """
        The `visualize_coords` function of the class `OpenSlideWSI` overlays patch coordinates 
        onto a scaled thumbnail of the WSI. It creates a visualization of the extracted patches 
        and saves it as an image file.

        Args:
        -----
        coords_path : str
            Path to the file containing the patch coordinates.
        save_patch_viz : str
            Directory path to save the visualization image.

        Returns:
        --------
        str:
            The file path to the saved visualization image.

        Example:
        --------
        >>> viz_path = wsi.visualize_coords("output_coords/sample_name_patches.h5", "output_viz")
        >>> print(viz_path)
        output_viz/sample_name.png
        """

        try:
            coords_attrs, coords = read_coords(coords_path)  # Coords are ALWAYS wrt. level 0 of the slide.
            patch_size = coords_attrs.get('patch_size', None)
            level0_magnification = coords_attrs.get('level0_magnification', None)
            target_magnification = coords_attrs.get('target_magnification', None)
            overlap = coords_attrs.get('overlap', 'NA')
            
            if None in (patch_size, level0_magnification, target_magnification):
                raise KeyError('Missing essential attributes in coords_attrs.')
        except (KeyError, FileNotFoundError, ValueError) as e:
            warnings.warn(f"Cannot read using Trident coords format ({str(e)}). Trying with CLAM/Fishing-Rod.")
            patch_size, patch_level, custom_downsample, coords = read_coords_legacy(coords_path)
            level0_magnification = self.mag
            target_magnification = int(self.mag / (self.level_downsamples[patch_level] * custom_downsample))

        patcher = self.create_patcher(
            patch_size=patch_size,
            src_mag=level0_magnification,
            dst_mag=target_magnification,
            custom_coords=coords,
            coords_only=True
        )

        max_dimension = 1000
        if self.width > self.height:
            thumbnail_width = max_dimension
            thumbnail_height = int(thumbnail_width * self.height / self.width)
        else:
            thumbnail_height = max_dimension
            thumbnail_width = int(thumbnail_height * self.width / self.height)

        downsample_factor = self.width / thumbnail_width

        patch_size_src = round(patch_size * level0_magnification / target_magnification)
        thumbnail_patch_size = max(1, int(patch_size_src / downsample_factor))

        # Get thumbnail in right format
        canvas = np.array(self.get_thumbnail((thumbnail_width, thumbnail_height))).astype(np.uint8)

        # Draw rectangles for patches
        for (x, y) in patcher:
            x, y = int(x/downsample_factor), int(y/downsample_factor)
            thickness = max(1, thumbnail_patch_size // 10)
            canvas = cv2.rectangle(
                canvas, 
                (x, y), 
                (x + thumbnail_patch_size, y + thumbnail_patch_size), 
                (255, 0, 0), 
                thickness
            )

        # Add annotations
        text_area_height = 130
        text_x_offset = int(thumbnail_width * 0.03)  # Offset as 3% of thumbnail width
        text_y_spacing = 25  # Vertical spacing between lines of text

        canvas[:text_area_height, :300] = (
            canvas[:text_area_height, :300] * 0.5
        ).astype(np.uint8)

        cv2.putText(canvas, f'{len(coords)} patches', (text_x_offset, text_y_spacing), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(canvas, f'width={self.width}, height={self.height}', (text_x_offset, text_y_spacing * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, f'mpp={self.mpp}, mag={self.mag}', (text_x_offset, text_y_spacing * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, f'patch={patch_size} w. overlap={overlap} @ {target_magnification}x', (text_x_offset, text_y_spacing * 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save visualization
        os.makedirs(save_patch_viz, exist_ok=True)
        viz_coords_path = os.path.join(save_patch_viz, f'{os.path.splitext(os.path.basename(self.name))[0]}.jpg')
        Image.fromarray(canvas).save(viz_coords_path)

        return viz_coords_path

    @torch.inference_mode()
    def extract_patch_features(
        self,
        patch_encoder: torch.nn.Module,
        coords_path: str,
        save_features: str,
        device: str = 'cuda:0',
        saveas: str = 'h5',
        batch_limit: int = 512
    ) -> str:
        """
        The `extract_features` function of the class `OpenSlideWSI` extracts feature embeddings 
        from the WSI using a specified patch encoder. It processes the patches as specified 
        in the coordinates file and saves the features in the desired format.

        Args:
        -----
        patch_encoder : torch.nn.Module
            The model used for feature extraction.
        coords_path : str
            Path to the file containing patch coordinates.
        save_features : str
            Directory path to save the extracted features.
        device : str, optional
            Device to run feature extraction on (e.g., 'cuda:0'). Defaults to 'cuda:0'.
        saveas : str, optional
            Format to save the features ('h5' or 'pt'). Defaults to 'h5'.
        batch_limit : int, optional
            Maximum batch size for feature extraction. Defaults to 512.

        Returns:
        --------
        str:
            The absolute file path to the saved feature file in the specified format.

        Example:
        --------
        >>> features_path = wsi.extract_features(patch_encoder, "output_coords/sample_name_patches.h5", "output_features")
        >>> print(features_path)
        output_features/sample_name.h5
        """
        self._lazy_initialize()

        precision = patch_encoder.precision
        patch_transforms = patch_encoder.eval_transforms

        try:
            coords_attrs, coords = read_coords(coords_path)
            patch_size = coords_attrs.get('patch_size', None)
            level0_magnification = coords_attrs.get('level0_magnification', None)
            target_magnification = coords_attrs.get('target_magnification', None)            
            if None in (patch_size, level0_magnification, target_magnification):
                raise KeyError('Missing attributes in coords_attrs.')
        except (KeyError, FileNotFoundError, ValueError) as e:
            warnings.warn(f"Cannot read using Trident coords format ({str(e)}). Trying with CLAM/Fishing-Rod.")
            patch_size, patch_level, custom_downsample, coords = read_coords_legacy(coords_path)
            level0_magnification = self.mag
            target_magnification = int(self.mag / (self.level_downsamples[patch_level] * custom_downsample))

        patcher = self.create_patcher(
            patch_size=patch_size,
            src_mag=level0_magnification,
            dst_mag=target_magnification,
            custom_coords=coords,
            coords_only=False,
            pil=True
        )
        dataset = WSIPatcherDataset(patcher, patch_transforms)
        dataloader = DataLoader(dataset, batch_size=batch_limit, num_workers=get_num_workers(batch_limit), pin_memory=True)

        features = []
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            with torch.autocast(device_type='cuda', dtype=precision, enabled=(precision != torch.float32)):
                batch_features = patch_encoder(imgs)  
            features.append(batch_features.cpu().numpy())

        # Concatenate features
        features = np.concatenate(features, axis=0)

        # Save the features to disk
        os.makedirs(save_features, exist_ok=True)
        if saveas == 'h5':
            save_h5(os.path.join(save_features, f'{self.name}.{saveas}'),
                    assets = {
                        'features' : features,
                        'coords': coords,
                    },
                    attributes = {
                        'features': {'name': self.name, 'savetodir': save_features},
                        'coords': coords_attrs
                    },
                    mode='w')
        elif saveas == 'pt':
            torch.save(features, os.path.join(save_features, f'{self.name}.{saveas}'))
        else:
            raise ValueError(f'Invalid save_features_as: {saveas}. Only "h5" and "pt" are supported.')

        return os.path.join(save_features, f'{self.name}.{saveas}')

    @torch.inference_mode()
    def extract_slide_features(
        self,
        patch_features_path: str,
        slide_encoder: torch.nn.Module,
        save_features: str,
        device: str = 'cuda',
    ) -> str:
        """
        Extract slide-level features by encoding patch-level features using a pretrained slide encoder.

        This function processes patch-level features extracted from a whole-slide image (WSI) and
        generates a single feature vector representing the entire slide. The extracted features are
        saved to a specified directory in HDF5 format.

        Args:
            patch_features_path (str): Path to the HDF5 file containing patch-level features and coordinates.
            slide_encoder (torch.nn.Module): Pretrained slide encoder model for generating slide-level features.
            save_features (str): Directory where the extracted slide features will be saved.
            device (str, optional): Device to run computations on (e.g., 'cuda', 'cpu'). Defaults to 'cuda'.

        Returns:
            str: The absolute path to the slide-level features.

        Workflow:
            1. Load the pretrained slide encoder model and set it to evaluation mode.
            2. Load patch-level features and corresponding coordinates from the provided HDF5 file.
            3. Convert patch-level features into a tensor and move it to the specified device.
            4. Generate slide-level features using the slide encoder, with automatic mixed precision if supported.
            5. Save the slide-level features and associated metadata (e.g., coordinates) in an HDF5 file.
            6. Return the path to the saved slide features.

        Notes:
            - The `patch_features_path` must point to a valid HDF5 file containing datasets named `features` and `coords`.
            - The saved HDF5 file includes both the slide-level features and metadata such as patch coordinates.
            - Automatic mixed precision is enabled if the slide encoder supports precision lower than `torch.float32`.

        Raises:
            FileNotFoundError: If the `patch_features_path` does not exist.
            RuntimeError: If there is an issue with the slide encoder or tensor operations.

        Example:
            >>> slide_features = extract_slide_features(
            ...     patch_features_path='path/to/patch_features.h5',
            ...     slide_encoder=pretrained_model,
            ...     save_features='output/slide_features',
            ...     device='cuda'
            ... )
            >>> print(slide_features.shape)  # Outputs the shape of the slide-level feature vector.
        """
        import h5py

        # Set the slide encoder model to device and eval
        slide_encoder.to(device)
        slide_encoder.eval()
        
        # Load patch-level features from h5 file
        with h5py.File(patch_features_path, 'r') as f:
            coords = f['coords'][:]
            patch_features = f['features'][:]
            coords_attrs = dict(f['coords'].attrs)

        # Convert slide_features to tensor
        patch_features = torch.from_numpy(patch_features).float().to(device)
        patch_features = patch_features.unsqueeze(0)  # Add batch dimension

        coords = torch.from_numpy(coords).to(device)
        coords = coords.unsqueeze(0)  # Add batch dimension

        # Prepare input batch dictionary
        batch = {
            'features': patch_features,
            'coords': coords,
            'attributes': coords_attrs
        }

        # Generate slide-level features
        with torch.autocast(device_type='cuda', enabled=(slide_encoder.precision != torch.float32)):
            features = slide_encoder(batch, device)
        features = features.float().cpu().numpy().squeeze()

        # Save slide-level features if save path is provided
        os.makedirs(save_features, exist_ok=True)
        save_path = os.path.join(save_features, f'{self.name}.h5')

        save_h5(os.path.join(save_features, f'{self.name}.h5'),
                    assets = {
                        'features' : features,
                        'coords': coords.cpu().numpy().squeeze(),
                    },
                    attributes = {
                        'features': {'name': self.name, 'savetodir': save_features},
                        'coords': coords_attrs
                    },
                    mode='w')

        return save_path
