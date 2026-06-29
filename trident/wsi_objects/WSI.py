from __future__ import annotations
import numpy as np
import os 
import warnings
import multiprocessing as mp
import torch 
from typing import List, Tuple, Optional, Literal, Union
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from trident.segmentation_models.load import SegmentationModel
from trident.wsi_objects.WSIPatcher import *
from trident.wsi_objects.WSIPatcherDataset import WSIPatcherDataset
from trident.IO import (
    save_h5, read_coords,
    mask_to_gdf, overlay_gdf_on_thumbnail, get_num_workers, coords_to_h5,
    save_cell_segmentation_h5, overlay_instances_on_thumbnail, splitext
)

ReadMode = Literal['pil', 'numpy']


_WARNED_CTX_FALLBACKS: set[str] = set()
_SPAWN_PICKLING_MSGS = (
    'ctypes objects containing pointers cannot be pickled',
    "Can't pickle",
    "PicklingError",
)


def _warn_ctx_fallback_once(key: str, message: str) -> None:
    if key not in _WARNED_CTX_FALLBACKS:
        warnings.warn(message)
        _WARNED_CTX_FALLBACKS.add(key)


def _dataloader_context_candidates(num_workers: int):
    if not num_workers or num_workers <= 0:
        return [None]

    candidates = []
    # Prefer fork on POSIX to avoid spawn pickling issues with complex objects.
    for method in ('fork', 'spawn'):
        try:
            if method in mp.get_all_start_methods():
                ctx = mp.get_context(method)
                if ctx not in candidates:
                    candidates.append(ctx)
        except (ValueError, AttributeError):
            continue

    candidates.append(None)
    return candidates


def _run_with_dataloader_ctx_fallback(run_fn, num_workers: int, warn_key: str, warn_msg: str, fail_label: str):
    """
    Try `run_fn(ctx)` for each candidate multiprocessing context. Only
    pickling-related errors are swallowed (so we can fall back to the next
    candidate, e.g. 'fork' or single-process). Any other error propagates.
    """
    last_err = None
    for ctx in _dataloader_context_candidates(num_workers):
        try:
            return run_fn(ctx)
        except Exception as err:
            is_pickling_issue = any(msg in str(err) for msg in _SPAWN_PICKLING_MSGS)
            # `ctx is None` is the single-process fallback: nothing left to try.
            if ctx is None or not is_pickling_issue:
                raise
            last_err = err
            _warn_ctx_fallback_once(warn_key, warn_msg)

    raise last_err if last_err is not None else RuntimeError(f'Failed to build {fail_label}')


class WSI:
    """
    The `WSI` class provides an interface to work with Whole Slide Images (WSIs). 
    It supports lazy initialization, metadata extraction, tissue segmentation,
    patching, and feature extraction. The class handles various WSI file formats and 
    offers utilities for integration with AI models.

    Attributes:
        slide_path (str):
            Path to the WSI file.
        name (str):
            Name of the WSI (inferred from the file path if not provided).
        custom_mpp_keys (dict):
            Custom keys for extracting microns per pixel (MPP) and magnification metadata.
        lazy_init (bool):
            User preference indicating whether initialization should be deferred.
        _initialized (bool):
            Internal runtime flag indicating whether the backend has been initialized.
        tissue_seg_path (str):
            Path to a tissue segmentation mask (if available).
        width (int):
            Width of the WSI in pixels (set during lazy initialization).
        height (int):
            Height of the WSI in pixels (set during lazy initialization).
        dimensions (Tuple[int, int]):
            (width, height) tuple of the WSI (set during lazy initialization).
        mpp (float):
            Microns per pixel (set during lazy initialization or inferred).
        mag (float):
            Estimated magnification level (set during lazy initialization or inferred).
        level_count (int):
            Number of resolution levels in the WSI (set during lazy initialization).
        level_downsamples (List[float]):
            Downsampling factors for each pyramid level (set during lazy initialization).
        level_dimensions (List[Tuple[int, int]]):
            Dimensions of the WSI at each pyramid level (set during lazy initialization).
        properties (dict):
            Metadata properties extracted from the image backend (set during lazy initialization).
        img (Any):
            Backend-specific image object used for reading regions (set during lazy initialization).
        gdf_contours (geopandas.GeoDataFrame):
            Tissue segmentation mask as a GeoDataFrame, if available (set during lazy initialization).
    """

    def __init__(
        self,
        slide_path: str,
        name: Optional[str] = None,
        tissue_seg_path: Optional[str] = None,
        custom_mpp_keys: Optional[List[str]] = None,
        lazy_init: bool = True,
        mpp: Optional[float] = None,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize the `WSI` object for working with a Whole Slide Image (WSI).

        Parameters:
            slide_path (str):
                Path to the WSI file.
            name (str, optional):
                Optional name for the WSI. Defaults to the filename (without extension).
            tissue_seg_path (str, optional):
                Path to the tissue segmentation mask file. Defaults to None.
            custom_mpp_keys (Optional[List[str]]):
                Custom keys for extracting MPP and magnification metadata. Defaults to None.
            lazy_init (bool, optional):
                If True, defer loading the WSI until required. Defaults to True.
            mpp (float, optional):
                If not None, will be the reference micron per pixel (mpp). Handy when mpp is not provided in the WSI.
            max_workers (Optional[int]):
                Maximum number of workers for data loading.

        """
        self.slide_path = slide_path
        if name is None:
            self.name, self.ext = splitext(os.path.basename(slide_path)) 
        else:
            self.name, self.ext = splitext(name)
        self.tissue_seg_path = tissue_seg_path
        self.custom_mpp_keys = custom_mpp_keys

        self.width, self.height = None, None  # Placeholder dimensions
        self.mpp = mpp  # Placeholder microns per pixel. Defaults will be None unless specified in constructor. 
        self.mag = None  # Placeholder magnification
        # Public configuration flag (do not mutate at runtime).
        self.lazy_init = lazy_init
        # Internal runtime state flag.
        self._initialized = False
        self.max_workers = max_workers

        if not self.lazy_init:
            self._lazy_initialize()

    def __repr__(self) -> str:
        if self._initialized:
            return f"<width={self.width}, height={self.height}, backend={self.__class__.__name__}, mpp={self.mpp}, mag={self.mag}>"
        else:
            return f"<name={self.name}>"

    def __enter__(self) -> "WSI":
        """Enable use as a context manager (`with ... as wsi`)."""
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Always release resources when leaving a context."""
        self.release()
        return False
    
    def _lazy_initialize(self) -> None:
        """
        Perform lazy initialization of internal attributes for the WSI interface.

        This method is intended to be called by subclasses of `WSI`, and should not be used directly.
        It sets default values for key image attributes and optionally loads a tissue segmentation mask
        if a path is provided. Subclasses must override this method to implement backend-specific behavior.

        Raises:
            FileNotFoundError:
                If the tissue segmentation mask file is provided but cannot be found.

        Notes:
        This method sets the following attributes:
        - `img`, `dimensions`, `width`, `height`: placeholder image properties (set to None).
        - `level_count`, `level_downsamples`, `level_dimensions`: multiresolution placeholders (None).
        - `properties`, `mag`: metadata and magnification (None).
        - `gdf_contours`: loaded from `tissue_seg_path` if available.
        """

        if not self._initialized:
            self.img = None
            self.dimensions = None
            self.width, self.height = None, None
            self.level_count = None
            self.level_downsamples = None
            self.level_dimensions = None
            self.properties = None
            self.mag = None
            if self.tissue_seg_path is not None:
                import geopandas as gpd
                try:
                    self.gdf_contours = gpd.read_file(self.tissue_seg_path)
                except FileNotFoundError:
                    raise FileNotFoundError(f"Tissue segmentation file not found: {self.tissue_seg_path}")

    def create_patcher(
        self, 
        patch_size: int, 
        src_pixel_size: Optional[float] = None, 
        dst_pixel_size: Optional[float] = None, 
        src_mag: Optional[int] = None, 
        dst_mag: Optional[int] = None, 
        overlap: int = 0, 
        mask: Optional[gpd.GeoDataFrame] = None,
        coords_only: bool = False, 
        custom_coords:  Optional[np.ndarray] = None,
        threshold: float = 0.15,
        pil: bool = False,
    ) -> WSIPatcher:
        """
        Create a patcher object for extracting patches from the WSI.

        Parameters:
            patch_size (int):
                Size of each patch in pixels.
            src_pixel_size (float, optional):
                Source pixel size. Defaults to None.
            dst_pixel_size (float, optional):
                Destination pixel size. Defaults to None.
            src_mag (int, optional):
                Source magnification. Defaults to None.
            dst_mag (int, optional):
                Destination magnification. Defaults to None.
            overlap (int, optional):
                Overlap between patches in pixels. Defaults to 0.
            mask (Optional[gpd.GeoDataFrame]):
                Mask for patching. Defaults to None.
            coords_only (bool, optional):
                Whether to only return coordinates. Defaults to False.
            custom_coords (Optional[np.ndarray]):
                Custom coordinates to use. Defaults to None.
            threshold (float, optional):
                Threshold for tissue detection. Defaults to 0.15.
            pil (bool, optional):
                Whether to use PIL for image reading. Defaults to False.

        Returns:
            WSIPatcher: An object for extracting patches.

        Example
        -------
        >>> patcher = wsi.create_patcher(patch_size=512, src_pixel_size=0.25, dst_pixel_size=0.5)
        >>> for patch in patcher:
        ...     process(patch)
        """
        return WSIPatcher(
            self, patch_size, src_pixel_size, dst_pixel_size, src_mag, dst_mag,
            overlap, mask, coords_only, custom_coords, threshold, pil
        )
    
    def _fetch_magnification(self, custom_mpp_keys: Optional[List[str]] = None) -> int:
        """
        Calculate the magnification level of the WSI based on the microns per pixel (MPP) value or other metadata.
        The magnification levels are 
        approximated to commonly used values such as 80x, 40x, 20x, etc. If the MPP is unavailable or insufficient 
        for calculation, it attempts to fallback to metadata-based values.

        Parameters:
            custom_mpp_keys (Optional[List[str]], optional):
                Custom keys to search for MPP values in the WSI properties. Defaults to None.

        Returns:
            Optional[int]: The approximated magnification level, or None if the magnification could not be determined.

        Raises:
            ValueError:
                If the identified MPP is too low for valid magnification values.

        Example
        -------
        >>> mag = wsi._fetch_magnification()
        >>> print(mag)
        40
        """
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
                raise ValueError(f"Identified mpp is very low: mpp={mpp_x}. Most WSIs are at 20x, 40x magnification.")

    def _segment_semantic(
        self, 
        segmentation_model: SegmentationModel,
        target_mag: int, 
        verbose: bool,
        device: str,
        batch_size: int,
        collate_fn,
        num_workers: Optional[int],
        inference_fn
    ):
        """
        Segment semantic regions in the WSI using a specified segmentation model.

        Parameters:
            segmentation_model (SegmentationModel):
                Model to use for segmentation.
            target_mag (int):
                Perform segmentation at this magnification.
            verbose (bool, optional):
                Whether to print segmentation progress. Defaults to False.
            device (str):
                The computation device to use (e.g., 'cuda:0' for GPU or 'cpu' for CPU).
            batch_size (int, optional):
                Batch size for processing patches. Defaults to 16.
            collate_fn (optional):
                Custom collate function used in the dataloader. It must return a dictionary containing at least
                `xcoords` and `ycoords` (level-0 coordinates), and `img` if `inference_fn` is not provided.
            num_workers (Optional[int], optional):
                Number of workers to use for the tile dataloader. If None, the number of workers is automatically
                inferred. Defaults to None.
            inference_fn (optional):
                Function used during inference. Called as `inference_fn(model, batch, device)` where `batch` is the
                batch returned by `collate_fn` (if provided) or `(img, (xcoords, ycoords))` otherwise. Must return a
                tensor with shape `(B, H, W)` and dtype `uint8`.

        Returns:
            Tuple[np.ndarray, float]: A downscaled H x W np.ndarray containing class predictions and its downscale factor.
        """
        # Get patch iterator
        destination_mpp = 10 / target_mag
        patcher = self.create_patcher(
            patch_size = segmentation_model.input_size,
            src_pixel_size = self.mpp,
            dst_pixel_size = destination_mpp,
            mask=self.gdf_contours if hasattr(self, "gdf_contours") else None
        )
        precision = segmentation_model.precision
        eval_transforms = segmentation_model.eval_transforms
        dataset = WSIPatcherDataset(patcher, eval_transforms)
        inferred_workers = get_num_workers(batch_size, max_workers=self.max_workers) if num_workers is None else num_workers

        dataloader_kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=inferred_workers,
            pin_memory=False,
        )

        mpp_reduction_factor = self.mpp / destination_mpp
        width, height = self.get_dimensions()
        width, height = int(round(width * mpp_reduction_factor)), int(round(height * mpp_reduction_factor))

        def _process_batches(ctx):
            dl_kwargs = dict(dataloader_kwargs)
            if ctx is not None:
                dl_kwargs['multiprocessing_context'] = ctx
            dataloader = DataLoader(**dl_kwargs)
            iterator = tqdm(dataloader) if verbose else dataloader
            local_mask = np.zeros((height, width), dtype=np.uint8)

            for batch in iterator:

                with torch.autocast(device_type=device.split(":")[0], dtype=precision, enabled=(precision != torch.float32)):
                    if collate_fn is not None:
                        if 'xcoords' not in batch or 'ycoords' not in batch:
                            raise ValueError(f"collate_fn must return level 0 patch coordinates in 'xcoords' and 'ycoords'")
                        xcoords, ycoords = torch.tensor(batch['xcoords']), torch.tensor(batch['ycoords'])
                        if inference_fn is None:
                            if 'img' not in batch:
                                raise ValueError(f"collate_fn must return the raw tile in 'img' if inference_fn is not provided.")
                            imgs = batch['img']
                    else:
                        imgs, (xcoords, ycoords) = batch

                    if inference_fn is not None:
                        preds = inference_fn(segmentation_model, batch, device).cpu().numpy()
                    else:
                        imgs = imgs.to(device, dtype=precision)  # Move to device and match dtype
                        preds = segmentation_model(imgs).cpu().numpy()

                x_starts = np.clip(np.round(xcoords.numpy() * mpp_reduction_factor).astype(int), 0, width - 1) # clip for starts
                y_starts = np.clip(np.round(ycoords.numpy() * mpp_reduction_factor).astype(int), 0, height - 1)
                x_ends = np.clip(x_starts + segmentation_model.input_size, 0, width)
                y_ends = np.clip(y_starts + segmentation_model.input_size, 0, height)
                
                for i in range(len(preds)):
                    x_start, x_end = x_starts[i], x_ends[i]
                    y_start, y_end = y_starts[i], y_ends[i]
                    if x_start >= x_end or y_start >= y_end: # invalid patch
                        continue
                    patch_pred = preds[i][:y_end - y_start, :x_end - x_start]
                    local_mask[y_start:y_end, x_start:x_end] += patch_pred
            return local_mask

        predicted_mask = _run_with_dataloader_ctx_fallback(
            _process_batches,
            inferred_workers,
            'segmentation_spawn_fallback',
            "[WSI] Falling back to a fork-based DataLoader context for segmentation due to pickling limits.",
            'segmentation dataloader',
        )
        return predicted_mask, mpp_reduction_factor

    @torch.inference_mode()
    def segment_tissue(
        self,
        segmentation_model: SegmentationModel,
        target_mag: int = 10,
        holes_are_tissue: bool = True,
        job_dir: Optional[str] = None,
        batch_size: int = 16,
        device: str = 'cuda:0',
        verbose=False,
        num_workers=None
    ) -> Union[str, gpd.GeoDataFrame]:
        """
        Segment tissue regions in the WSI using a specified segmentation model.
        It processes the WSI at a target magnification level, optionally 
        treating holes in the mask as tissue. The segmented regions are saved as thumbnails and GeoJSON contours.

        Parameters:
            segmentation_model (SegmentationModel):
                The model used for tissue segmentation.
            target_mag (int, optional):
                Target magnification level for segmentation. Defaults to 10.
            holes_are_tissue (bool, optional):
                Whether to treat holes in the mask as tissue. Defaults to True.
            job_dir (Optional[str], optional):
                Directory to save the segmentation results. If None, this method directly returns the contours as a
                GeoDataFrame without saving files. Defaults to None.
            batch_size (int, optional):
                Batch size for processing patches. Defaults to 16.
            device (str):
                The computation device to use (e.g., 'cuda:0' for GPU or 'cpu' for CPU).
            verbose (bool, optional):
                Whether to print segmentation progress. Defaults to False.
            num_workers (Optional[int], optional):
                Number of workers to use for the tile dataloader. If None, the number of workers is automatically
                inferred. Defaults to None.

        Returns:
            Union[str, gpd.GeoDataFrame]: The absolute path to the GeoJSON if `job_dir` is not None; otherwise a GeoDataFrame.

        Example
        -------
        >>> wsi.segment_tissue(segmentation_model, target_mag=10, job_dir="output_dir")
        >>> # Results saved in "output_dir"
        """

        self._lazy_initialize()
        segmentation_model.to(device)
        max_dimension = 1000
        if self.width > self.height:
            thumbnail_width = max_dimension
            thumbnail_height = int(thumbnail_width * self.height / self.width)
        else:
            thumbnail_height = max_dimension
            thumbnail_width = int(thumbnail_height * self.width / self.height)
        thumbnail = self.get_thumbnail((thumbnail_width, thumbnail_height))

        # dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

        predicted_mask, mpp_reduction_factor = self._segment_semantic(
            segmentation_model,
            target_mag,
            verbose,
            device,
            batch_size,
            None,
            num_workers,
            None
        )
        
        # Post-process the mask
        predicted_mask = (predicted_mask > 0).astype(np.uint8) * 255

        # # Fill holes if desired
        # if not holes_are_tissue:
        #     holes, _ = cv2.findContours(predicted_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        #     for hole in holes:
        #         cv2.drawContours(predicted_mask, [hole], 0, 255, -1)

        gdf_contours = mask_to_gdf(
            mask=predicted_mask,
            max_nb_holes=0 if holes_are_tissue else 20,
            min_contour_area=1000,
            pixel_size=self.mpp,
            contour_scale=1/mpp_reduction_factor
        )
        if job_dir is not None:

            # Save thumbnail image
            thumbnail_saveto = os.path.join(job_dir, 'thumbnails', f'{self.name}.jpg')
            os.makedirs(os.path.dirname(thumbnail_saveto), exist_ok=True)
            thumbnail.save(thumbnail_saveto)

            # Save geopandas contours
            gdf_saveto = os.path.join(job_dir, 'contours_geojson', f'{self.name}.geojson')
            os.makedirs(os.path.dirname(gdf_saveto), exist_ok=True)
            gdf_contours.set_crs("EPSG:3857", inplace=True)  # used to silent warning // Web Mercator
            gdf_contours.to_file(gdf_saveto, driver="GeoJSON")
            self.gdf_contours = gdf_contours
            self.tissue_seg_path = gdf_saveto

            # Draw the contours on the thumbnail image
            contours_saveto = os.path.join(job_dir, 'contours', f'{self.name}.jpg')
            annotated = np.array(thumbnail)
            overlay_gdf_on_thumbnail(gdf_contours, annotated, contours_saveto, thumbnail_width / self.width)

            return gdf_saveto
        else:
            return gdf_contours

    @torch.inference_mode()
    def segment_semantic(
        self,
        segmentation_model: SegmentationModel,
        target_mag: int = 10,
        batch_size: int = 16,
        device: str = 'cuda:0',
        verbose=False,
        num_workers=None,
        collate_fn=None,
        inference_fn=None,
        return_contours=False
    ) -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, gpd.GeoDataFrame]]:
        """
        Segment semantic regions in the WSI using a specified segmentation model.

        Parameters:
            segmentation_model (SegmentationModel):
                The model used for tissue segmentation.
            target_mag (int, optional):
                Target magnification level for segmentation. Defaults to 10.
            batch_size (int, optional):
                Batch size for processing patches. Defaults to 16.
            device (str):
                The computation device to use (e.g., 'cuda:0' for GPU or 'cpu' for CPU).
            verbose (bool, optional):
                Whether to print segmentation progress. Defaults to False.
            num_workers (Optional[int], optional):
                Number of workers to use for the tile dataloader. If None, the number of workers is automatically
                inferred. Defaults to None.
            collate_fn (optional):
                Custom collate function used in the dataloader. It must return a dictionary containing at least
                `xcoords` and `ycoords` (level-0 coordinates), and `img` if `inference_fn` is not provided.
            inference_fn (optional):
                Function used during inference. Called as `inference_fn(model, batch, device)` where `batch` is the
                batch returned by `collate_fn` (if provided) or `(img, (xcoords, ycoords))` otherwise. Must return a
                tensor with shape `(B, H, W)` and dtype `uint8`.
            return_contours (bool, optional):
                Whether to return the contours of each class in a GeoDataFrame. Defaults to False.

        Returns:
            Union[Tuple[np.ndarray, float], Tuple[np.ndarray, float, gpd.GeoDataFrame]]:
                A downscaled H x W np.ndarray containing class predictions and its downscale factor. If
                `return_contours` is True, also returns the contours of each class in a GeoDataFrame.

        Example
        -------
        >>> wsi.segment_tissue(segmentation_model, target_mag=10, job_dir="output_dir")
        >>> # Results saved in "output_dir"
        """
        import pandas as pd
        import geopandas as gpd

        self._lazy_initialize()
        segmentation_model.to(device)

        predicted_mask, mpp_reduction_factor = self._segment_semantic(
            segmentation_model,
            target_mag,
            verbose,
            device,
            batch_size,
            collate_fn,
            num_workers,
            inference_fn
        )

        if not return_contours:
            return predicted_mask, mpp_reduction_factor

        gdfs = []
        unique_labels = np.unique(predicted_mask)
        for unique_label in unique_labels:
            if unique_label == 0:
                continue

            gdf_contours = mask_to_gdf(
                mask=(predicted_mask == unique_label).astype(np.uint8),
                max_nb_holes=20,
                min_contour_area=1000,
                pixel_size=self.mpp,
                contour_scale=1/mpp_reduction_factor
            )
            gdfs.append(gdf_contours)
        
        if len(gdfs) > 0:
            gdf = pd.concat(gdfs)
        else:
            gdf = gpd.GeoDataFrame()

        return predicted_mask, mpp_reduction_factor, gdf
        

    def get_best_level_and_custom_downsample(
        self,
        downsample: float,
        tolerance: float = 0.01
    ) -> Tuple[int, float]:
        """
        Determine the best level and custom downsample factor to approximate a desired downsample value.

        Parameters:
            downsample (float):
                The desired downsample factor.
            tolerance (float, optional):
                Tolerance for rounding differences. Defaults to 0.01.

        Returns:
            Tuple[int, float]: The closest resolution level and the custom downsample factor.

        Raises:
            ValueError:
                If no suitable resolution level is found for the specified downsample factor.

        Example
        -------
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
        Extract patch coordinates from tissue regions in the WSI.
        It generates coordinates of patches at the specified 
        magnification and saves the results in an HDF5 file.

        Parameters:
            target_mag (int):
                Target magnification level for the patches.
            patch_size (int):
                Size of each patch at the target magnification.
            save_coords (str):
                Directory path to save the extracted coordinates.
            overlap (int, optional):
                Overlap between patches in pixels. Defaults to 0.
            min_tissue_proportion (float, optional):
                Minimum proportion of the patch under tissue to be kept. Defaults to 0.

        Returns:
            str: The absolute file path to the saved HDF5 file containing the patch coordinates.

        Example
        -------
        >>> coords_path = wsi.extract_tissue_coords(20, 256, "output_coords", overlap=32)
        >>> print(coords_path)
        output_coords/patches/sample_name_patches.h5
        """

        self._lazy_initialize()

        patcher = self.create_patcher(
            patch_size=patch_size,
            src_mag=self.mag,
            dst_mag=target_mag,
            mask=self.gdf_contours if hasattr(self, "gdf_contours") else None,
            coords_only=True,
            overlap=overlap,
            threshold=min_tissue_proportion,
        )

        coords_to_keep = [(x, y) for x, y in patcher]

        os.makedirs(os.path.join(save_coords, 'patches'), exist_ok=True)
        out_fname = os.path.join(save_coords, 'patches', str(self.name) + '_patches.h5')
        coords_to_h5(coords_to_keep, out_fname, patch_size, self.mag, target_mag,
                     save_coords, self.width, self.height, self.name, overlap)
        return out_fname

    def visualize_coords(self, coords_path: str, save_patch_viz: str) -> str:
        """
        Overlay patch coordinates onto a scaled thumbnail of the WSI.
        
        Parameters:
            coords_path (str):
                Path to the file containing the patch coordinates.
            save_patch_viz (str):
                Directory path to save the visualization image.

        Returns:
            str: The file path to the saved visualization image.

        Example
        -------
        >>> viz_path = wsi.visualize_coords("output_coords/sample_name_patches.h5", "output_viz")
        >>> print(viz_path)
        output_viz/sample_name.png
        """

        self._lazy_initialize()

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
            patcher = WSIPatcher.from_legacy_coords_file(self, coords_path, coords_only=True)
        
        else:
            patcher = self.create_patcher(
                patch_size=patch_size,
                src_mag=level0_magnification,
                dst_mag=target_magnification,
                custom_coords=coords,
                coords_only=True
            )

        img =  patcher.visualize()

        # Save visualization
        os.makedirs(save_patch_viz, exist_ok=True)
        viz_coords_path = os.path.join(save_patch_viz, f'{self.name}.jpg')
        img.save(viz_coords_path)
        return viz_coords_path

    def dump_patches(
        self,
        coords_path: str,
        save_patches_dir: str,
        max_patches: int = 0,
        image_format: str = "png",
        jpeg_quality: int = 90,
    ) -> str:
        """
        Dump patch images to disk for debugging/inspection.

        This reads a Trident coords H5 file (or legacy coords if needed), iterates the
        corresponding patches, and writes them under `save_patches_dir/<slide_name>/`.

        Parameters:
            coords_path (str):
                Path to a coords .h5 file produced by TRIDENT.
            save_patches_dir (str):
                Output directory to store patch images.
            max_patches (int, optional):
                If > 0, cap the number of patches written. Defaults to 0 (no cap).
            image_format ({"png", "jpg"}, optional):
                Image format to write. Defaults to "png".
            jpeg_quality (int, optional):
                JPEG quality (1-100). Only used when image_format="jpg". Defaults to 90.

        Returns:
            str: Directory where patches were written.
        """
        self._lazy_initialize()

        image_format = image_format.lower().strip()
        if image_format not in {"png", "jpg"}:
            raise ValueError(f"Unsupported image_format='{image_format}'. Expected 'png' or 'jpg'.")
        if not (1 <= int(jpeg_quality) <= 100):
            raise ValueError(f"jpeg_quality must be between 1 and 100, got {jpeg_quality}.")

        try:
            coords_attrs, coords = read_coords(coords_path)  # coords are level-0
            patch_size = coords_attrs.get("patch_size", None)
            level0_magnification = coords_attrs.get("level0_magnification", None)
            target_magnification = coords_attrs.get("target_magnification", None)
            overlap = coords_attrs.get("overlap", 0)
            if None in (patch_size, level0_magnification, target_magnification):
                raise KeyError("Missing essential attributes in coords_attrs.")
        except (KeyError, FileNotFoundError, ValueError) as e:
            warnings.warn(
                f"Cannot read using Trident coords format ({str(e)}). Trying with CLAM/Fishing-Rod."
            )
            patcher = WSIPatcher.from_legacy_coords_file(self, coords_path, coords_only=False, pil=True)
        else:
            patcher = self.create_patcher(
                patch_size=patch_size,
                src_mag=level0_magnification,
                dst_mag=target_magnification,
                custom_coords=coords,
                coords_only=False,
                overlap=int(overlap) if overlap is not None else 0,
                pil=True,
            )

        out_dir = os.path.join(save_patches_dir, self.name)
        os.makedirs(out_dir, exist_ok=True)

        written = 0
        for tile, x, y in patcher:
            # tile is a PIL Image when pil=True
            out_fp = os.path.join(out_dir, f"{written:06d}_x{x}_y{y}.{image_format}")
            if image_format == "jpg":
                tile.save(out_fp, format="JPEG", quality=int(jpeg_quality), optimize=True)
            else:
                tile.save(out_fp)
            written += 1
            if max_patches and written >= max_patches:
                break

        return out_dir

    @torch.inference_mode()
    def extract_patch_features(
        self,
        patch_encoder: torch.nn.Module,
        coords_path: str,
        save_features: str,
        device: str = 'cuda:0',
        saveas: str = 'h5',
        batch_limit: int = 512,
        verbose: bool = False
    ) -> str:
        """
        Extract feature embeddings from the WSI using a specified patch encoder.

        Parameters:
            patch_encoder (torch.nn.Module):
                The model used for feature extraction.
            coords_path (str):
                Path to the file containing patch coordinates.
            save_features (str):
                Directory path to save the extracted features.
            device (str, optional):
                Device to run feature extraction on (e.g., 'cuda:0'). Defaults to 'cuda:0'.
            saveas (str, optional):
                Format to save the features ('h5' or 'pt'). Defaults to 'h5'.
            batch_limit (int, optional):
                Maximum batch size for feature extraction. Defaults to 512.
            verbose (bool, optional):
                Whether to print patch embedding progress. Defaults to False.

        Returns:
            str: The absolute file path to the saved feature file in the specified format.

        Example
        -------
        >>> features_path = wsi.extract_features(patch_encoder, "output_coords/sample_name_patches.h5", "output_features")
        >>> print(features_path)
        output_features/sample_name.h5
        """

        self._lazy_initialize()
        patch_encoder.to(device)
        patch_encoder.eval()
        precision = getattr(patch_encoder, 'precision', torch.float32)
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
            patcher = WSIPatcher.from_legacy_coords_file(self, coords_path, coords_only=True, pil=True)

        else:
            patcher = self.create_patcher(
                patch_size=patch_size,
                src_mag=level0_magnification,
                dst_mag=target_magnification,
                custom_coords=coords,
                coords_only=False,
                pil=True,
            )  


        dataset = WSIPatcherDataset(patcher, patch_transforms)
        if len(dataset) == 0:
            warnings.warn(
                f"No patch coordinates available for slide '{self.name}'. Saving empty features."
            )
            coords_attrs = coords_attrs if 'coords_attrs' in locals() else {}
            coords = np.empty((0, 2), dtype=np.int64)
            embedding_dim = getattr(patch_encoder, "embedding_dim", None)
            if embedding_dim is None:
                features = np.empty((0,), dtype=np.float32)
            else:
                features = np.empty((0, int(embedding_dim)), dtype=np.float32)
            os.makedirs(save_features, exist_ok=True)
            if saveas == 'h5':
                model_name = patch_encoder.enc_name if hasattr(patch_encoder, 'enc_name') else None
                save_h5(
                    os.path.join(save_features, f'{self.name}.{saveas}'),
                    assets={
                        'features': features,
                        'coords': coords,
                    },
                    attributes={
                        'features': {'name': self.name, 'savetodir': save_features, 'encoder': model_name},
                        'coords': coords_attrs,
                    },
                    mode='w'
                )
            elif saveas == 'pt':
                torch.save(features, os.path.join(save_features, f'{self.name}.{saveas}'))
            else:
                raise ValueError(f'Invalid save_features_as: {saveas}. Only "h5" and "pt" are supported.')
            return os.path.join(save_features, f'{self.name}.{saveas}')

        inferred_workers = get_num_workers(batch_limit, max_workers=self.max_workers)
        dataloader_kwargs = dict(
            dataset=dataset,
            batch_size=batch_limit,
            num_workers=inferred_workers,
            pin_memory=False,
        )

        def _collect_features(ctx):
            dl_kwargs = dict(dataloader_kwargs)
            if ctx is not None:
                dl_kwargs['multiprocessing_context'] = ctx
            dataloader = DataLoader(**dl_kwargs)
            iterator = tqdm(dataloader) if verbose else dataloader
            collected = []
            for imgs, _ in iterator:
                imgs = imgs.to(device)
                with torch.autocast(
                    device_type=device.split(":")[0],
                    dtype=precision,
                    enabled=(precision != torch.float32),
                ):
                    batch_features = patch_encoder(imgs)
                collected.append(batch_features.cpu().numpy())
            return collected

        features_batches = _run_with_dataloader_ctx_fallback(
            _collect_features,
            inferred_workers,
            'feature_spawn_fallback',
            "[WSI] Falling back to fork-based DataLoader workers for feature extraction due to pickling limits.",
            'feature extraction dataloader',
        )

        # Concatenate features
        features = np.concatenate(features_batches, axis=0)

        # Save the features to disk
        os.makedirs(save_features, exist_ok=True)
        if saveas == 'h5':
            model_name = patch_encoder.enc_name if hasattr(patch_encoder, 'enc_name') else None
            save_h5(os.path.join(save_features, f'{self.name}.{saveas}'),
                    assets = {
                        'features' : features,
                        'coords': coords,
                    },
                    attributes = {
                        'features': {'name': self.name, 'savetodir': save_features, 'encoder': model_name},
                        'coords': coords_attrs
                    },
                    mode='w')
        elif saveas == 'pt':
            torch.save(features, os.path.join(save_features, f'{self.name}.{saveas}'))
        else:
            raise ValueError(f'Invalid save_features_as: {saveas}. Only "h5" and "pt" are supported.')

        return os.path.join(save_features, f'{self.name}.{saveas}')

    @torch.inference_mode()
    def segment_patches(
        self,
        patch_segmenter: torch.nn.Module,
        coords_path: str,
        save_dir: str,
        device: str = 'cuda:0',
        batch_limit: int = 4,
        save_viz: Optional[str] = None,
        verbose: bool = False,
    ) -> str:
        """
        Run a patch-segmentation model (e.g. HistoPlus, CellViT++) over the tissue patches
        of the WSI and stitch the per-patch cell/object instances into slide-level artifacts.

        This mirrors `extract_patch_features`: it iterates the same patch coordinates and
        batches patches through TRIDENT's dataloader, but instead of a feature vector per
        patch it collects per-cell instances (polygon + class + confidence) translated into
        level-0 coordinates, and writes three artifacts aligned with the rest of the pipeline:

            * ``<save_dir>/<slide>.geojson`` — per-cell polygons with ``class``/``class_name``/
              ``confidence`` properties (QuPath-loadable, like the tissue contours).
            * ``<save_dir>/<slide>.h5`` — compact ragged storage of contours + centroids +
              class ids + confidences (mirrors the patch-feature ``.h5`` convention).
            * ``<save_viz>/<slide>.jpg`` — optional debug overlay of polygons on a thumbnail.

        Parameters:
            patch_segmenter (torch.nn.Module):
                A `BasePatchSegmenter` exposing `eval_transforms`, `class_names`, and
                `predict_patches(imgs) -> list-per-image of instance dicts` (contours in
                input-patch pixel coords).
            coords_path (str): Path to the patch-coordinate `.h5` from the coords task.
            save_dir (str): Directory for the GeoJSON + HDF5 outputs.
            device (str, optional): Compute device. Defaults to 'cuda:0'.
            batch_limit (int, optional): Max patches per batch. Defaults to 4.
            save_viz (str, optional): If set, directory for the debug overlay JPEG.
            verbose (bool, optional): Show a progress bar. Defaults to False.

        Returns:
            str: Absolute path to the saved GeoJSON.
        """
        import json
        import geopandas as gpd
        from shapely import Polygon

        self._lazy_initialize()
        patch_segmenter.to(device)
        patch_segmenter.eval()
        patch_transforms = patch_segmenter.eval_transforms
        class_names = getattr(patch_segmenter, 'class_names', None)
        model_name = getattr(patch_segmenter, 'seg_name', None)

        coords_attrs, coords = read_coords(coords_path)
        patch_size = coords_attrs.get('patch_size', None)
        level0_magnification = coords_attrs.get('level0_magnification', None)
        target_magnification = coords_attrs.get('target_magnification', None)
        if None in (patch_size, level0_magnification, target_magnification):
            raise KeyError('Missing attributes in coords_attrs.')

        # Level-0 pixels spanned by one patch edge. Patches are read at target_magnification,
        # so each covers patch_size * downsample level-0 pixels. The model returns contours in
        # the *input image* pixel space, so the level-0 scale is patch_extent / input_edge.
        downsample = level0_magnification / target_magnification
        patch_extent_level0 = patch_size * downsample

        patcher = self.create_patcher(
            patch_size=patch_size,
            src_mag=level0_magnification,
            dst_mag=target_magnification,
            custom_coords=coords,
            coords_only=False,
            pil=True,
        )

        os.makedirs(save_dir, exist_ok=True)
        geojson_path = os.path.join(save_dir, f'{self.name}.geojson')
        h5_path = os.path.join(save_dir, f'{self.name}.h5')

        h5_attrs = {
            'model': model_name or 'unknown',
            'class_names': json.dumps(class_names) if class_names is not None else '[]',
            'level0_magnification': float(level0_magnification),
            'target_magnification': float(target_magnification),
            'patch_size': int(patch_size),
            'mpp': float(self.mpp) if self.mpp is not None else -1.0,
        }

        dataset = WSIPatcherDataset(patcher, patch_transforms)
        if len(dataset) == 0:
            warnings.warn(
                f"No patch coordinates available for slide '{self.name}'. Saving empty outputs."
            )
            empty = gpd.GeoDataFrame(columns=['class', 'class_name', 'confidence', 'geometry'], geometry='geometry')
            empty.set_crs("EPSG:3857", inplace=True)
            empty.to_file(geojson_path, driver="GeoJSON")
            save_cell_segmentation_h5(h5_path, [], h5_attrs)
            return geojson_path

        inferred_workers = get_num_workers(batch_limit, max_workers=self.max_workers)
        dataloader_kwargs = dict(
            dataset=dataset,
            batch_size=batch_limit,
            num_workers=inferred_workers,
            pin_memory=False,
        )

        def _collect_instances(ctx):
            dl_kwargs = dict(dataloader_kwargs)
            if ctx is not None:
                dl_kwargs['multiprocessing_context'] = ctx
            dataloader = DataLoader(**dl_kwargs)
            iterator = tqdm(dataloader) if verbose else dataloader
            collected = []
            for imgs, (xs, ys) in iterator:
                imgs = imgs.to(device)
                input_edge = imgs.shape[-1]
                scale = patch_extent_level0 / input_edge  # level-0 px per input px
                scale_holder[0] = scale
                # Precision is the model's responsibility (instance models such as
                # HistoPlus manage their own AMP / output casting); the semantic default
                # path applies autocast internally. So no outer autocast here.
                per_image = patch_segmenter.predict_patches(imgs)
                xs = xs.numpy()
                ys = ys.numpy()
                for i, instances in enumerate(per_image):
                    origin = np.array([int(xs[i]), int(ys[i])], dtype=np.float64)
                    for inst in instances:
                        contour_l0 = np.asarray(inst['contour'], dtype=np.float64) * scale + origin
                        centroid_l0 = np.asarray(inst['centroid'], dtype=np.float64) * scale + origin
                        collected.append({
                            'contour': contour_l0,
                            'centroid': centroid_l0,
                            'class_id': int(inst['class_id']),
                            'class_name': inst.get('class_name'),
                            'confidence': float(inst.get('confidence', 1.0)),
                            'origin': (int(xs[i]), int(ys[i])),
                        })
            return collected

        scale_holder = [patch_extent_level0 / patch_size]  # updated with the true input edge
        instances = _run_with_dataloader_ctx_fallback(
            _collect_instances,
            inferred_workers,
            'patch_seg_spawn_fallback',
            "[WSI] Falling back to fork-based DataLoader workers for patch segmentation due to pickling limits.",
            'patch segmentation dataloader',
        )

        # 1) GeoJSON of per-cell polygons (level-0 coords).
        records = []
        for inst in instances:
            contour = inst['contour']
            if len(contour) < 3:
                continue
            polygon = Polygon(contour)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            records.append({
                'class': inst['class_id'],
                'class_name': inst['class_name'],
                'confidence': inst['confidence'],
                'geometry': polygon,
            })
        gdf = gpd.GeoDataFrame(
            records, columns=['class', 'class_name', 'confidence', 'geometry'], geometry='geometry'
        )
        gdf.set_crs("EPSG:3857", inplace=True)
        gdf.to_file(geojson_path, driver="GeoJSON")

        # 2) Compact HDF5 of all instances.
        save_cell_segmentation_h5(h5_path, instances, h5_attrs)

        # 3) Optional debug visualization. Two complementary artifacts:
        #    (a) a slide-thumbnail overview showing where cells are (global density), and
        #    (b) full-resolution overlays of the most cell-dense patches, where individual
        #        cells are actually visible (they vanish at thumbnail scale).
        # The GeoJSON + HDF5 above are the real deliverables and are already on disk, so a
        # visualization failure must NOT fail the whole task (it would mark the slide as
        # errored despite the cell segmentation having succeeded). Best-effort, warn only.
        if save_viz is not None and len(instances) > 0:
            try:
                max_dimension = 2000
                if self.width >= self.height:
                    thumb_w = max_dimension
                    thumb_h = int(thumb_w * self.height / self.width)
                else:
                    thumb_h = max_dimension
                    thumb_w = int(thumb_h * self.width / self.height)
                thumbnail = np.array(self.get_thumbnail((thumb_w, thumb_h)))
                overlay_instances_on_thumbnail(
                    gdf, thumbnail, os.path.join(save_viz, f'{self.name}_overview.jpg'),
                    thumb_w / self.width,
                )
                self._save_patch_segmentation_viz(
                    instances, scale_holder[0], patch_size, level0_magnification,
                    target_magnification, class_names, save_viz, max_patches=8,
                )
            except Exception as e:
                warnings.warn(
                    f"[WSI] Cell-segmentation visualization failed for '{self.name}' "
                    f"({type(e).__name__}: {e}). The GeoJSON and HDF5 outputs were written "
                    f"successfully; only the optional --seg_viz overlay was skipped."
                )

        return geojson_path

    def _save_patch_segmentation_viz(
        self, instances, scale, patch_size, level0_mag, target_mag,
        class_names, save_viz, max_patches=8,
    ) -> None:
        """
        Render the most cell-dense patches at full resolution with their instance contours
        overlaid, saved under ``<save_viz>/<slide>/<x>_<y>.jpg``. This is the useful debug
        artifact for cell segmentation (cells are too small to see on a slide thumbnail).
        """
        from collections import defaultdict
        import numpy as np
        from trident.IO import draw_instances_on_tile

        by_patch = defaultdict(list)
        for inst in instances:
            by_patch[inst['origin']].append(inst)
        top = sorted(by_patch.items(), key=lambda kv: -len(kv[1]))[:max_patches]
        if not top:
            return

        out_dir = os.path.join(save_viz, self.name)
        os.makedirs(out_dir, exist_ok=True)
        sample_coords = np.array([list(origin) for origin, _ in top], dtype=int)
        patcher = self.create_patcher(
            patch_size=patch_size, src_mag=level0_mag, dst_mag=target_mag,
            custom_coords=sample_coords, coords_only=False, pil=True,
        )
        for idx, (origin, patch_instances) in enumerate(top):
            tile, x, y = patcher[idx]
            origin_arr = np.array(origin, dtype=np.float64)
            px_instances = [{
                'contour': (np.asarray(inst['contour'], dtype=np.float64) - origin_arr) / scale,
                'class_id': inst['class_id'],
            } for inst in patch_instances]
            draw_instances_on_tile(
                np.array(tile), px_instances, class_names,
                os.path.join(out_dir, f'{int(x)}_{int(y)}.jpg'),
            )

    @torch.inference_mode()
    def query_region(
        self,
        vlm: torch.nn.Module,
        prompt: str,
        location: Tuple[int, int],
        size: int,
        mag: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Interactively interrogate a single ROI with a vision-language model: crop the
        region, ask ``prompt``, and return the model's free-text answer.

        This is the interactive counterpart to ``query_patches`` (batch over coords). It
        reuses the same magnification -> pyramid-level cropping logic as patching by
        building a one-coordinate patcher, so ``location`` is given in **level-0 pixels**
        and ``size`` is the square ROI edge in **pixels at the requested ``mag``**.

        Parameters:
            vlm (torch.nn.Module): A ``BaseVLM`` exposing ``generate(images, prompts)``.
            prompt (str): The free-text question to ask about the ROI.
            location (Tuple[int, int]): (x, y) top-left of the ROI in level-0 pixels.
            size (int): Square ROI edge length, in pixels at ``mag``.
            mag (float, optional): Magnification to read the ROI at. Defaults to the
                slide's native magnification (``self.mag``).
            max_new_tokens (int, optional): Override the model's default answer length.

        Returns:
            str: The model's answer.

        Example
        -------
        >>> wsi.query_region(vlm, "Describe the tissue and any tumor present.",
        ...                  location=(10240, 8192), size=512, mag=20)
        'The field shows invasive ductal carcinoma with ...'
        """
        self._lazy_initialize()
        src_mag = self.mag
        dst_mag = mag if mag is not None else self.mag
        patcher = self.create_patcher(
            patch_size=size, src_mag=src_mag, dst_mag=dst_mag,
            custom_coords=np.array([[int(location[0]), int(location[1])]]),
            coords_only=False, pil=True,
        )
        tile, _, _ = patcher[0]
        return vlm.generate([tile], [prompt], max_new_tokens=max_new_tokens)[0]

    @torch.inference_mode()
    def query_patches(
        self,
        vlm: torch.nn.Module,
        coords_path: str,
        prompt: str,
        save_dir: str,
        device: str = 'cuda:0',
        batch_limit: int = 4,
        max_new_tokens: Optional[int] = None,
        verbose: bool = False,
    ) -> str:
        """
        Run a vision-language model over the tissue patches of the WSI, asking the same
        ``prompt`` of every patch, and save the per-patch answers.

        Batch analog of ``query_region`` and structural twin of ``segment_patches``: it
        iterates the same patch coordinates, but instead of cell instances it collects one
        free-text answer per patch. Two artifacts are written, aligned with the rest of the
        pipeline:

            * ``<save_dir>/<slide>.json`` — list of ``{x, y, prompt, answer}`` (level-0 coords).
            * ``<save_dir>/<slide>.geojson`` — one patch-box polygon per coord carrying the
              ``answer`` (and ``prompt``) as properties, loadable as annotations in QuPath.

        Generation is autoregressive and therefore much slower than feed-forward feature
        extraction; this sweeps *every* patch, so prefer a tight coords set (or the
        interactive ``query_region``) for large slides.

        Parameters:
            vlm (torch.nn.Module): A ``BaseVLM`` exposing ``generate(images, prompts)``.
            coords_path (str): Path to the patch-coordinate ``.h5`` from the coords task.
            prompt (str): The question asked of every patch.
            save_dir (str): Directory for the JSON + GeoJSON outputs.
            device (str, optional): Compute device. Defaults to 'cuda:0'.
            batch_limit (int, optional): Patches per generation batch. Defaults to 4.
            max_new_tokens (int, optional): Override the model's default answer length.
            verbose (bool, optional): Show a progress bar. Defaults to False.

        Returns:
            str: Absolute path to the saved JSON.
        """
        import json
        import geopandas as gpd
        from shapely import Polygon

        self._lazy_initialize()
        vlm.to(device)
        vlm.eval()
        model_name = getattr(vlm, 'vlm_name', None)

        coords_attrs, coords = read_coords(coords_path)
        patch_size = coords_attrs.get('patch_size', None)
        level0_magnification = coords_attrs.get('level0_magnification', None)
        target_magnification = coords_attrs.get('target_magnification', None)
        if None in (patch_size, level0_magnification, target_magnification):
            raise KeyError('Missing attributes in coords_attrs.')

        # Level-0 pixels spanned by one patch edge (for drawing boxes in slide coords).
        downsample = level0_magnification / target_magnification
        patch_extent_level0 = patch_size * downsample

        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, f'{self.name}.json')
        geojson_path = os.path.join(save_dir, f'{self.name}.geojson')

        patcher = self.create_patcher(
            patch_size=patch_size, src_mag=level0_magnification,
            dst_mag=target_magnification, custom_coords=coords,
            coords_only=False, pil=True,
        )

        records: List[dict] = []
        if len(patcher) > 0:
            batch_imgs: List[Image.Image] = []
            batch_xy: List[Tuple[int, int]] = []
            indices = range(len(patcher))
            iterator = tqdm(indices, desc=f'Querying {self.name}') if verbose else indices

            def _flush():
                if not batch_imgs:
                    return
                answers = vlm.generate(batch_imgs, prompt, max_new_tokens=max_new_tokens)
                for (x, y), answer in zip(batch_xy, answers):
                    records.append({'x': int(x), 'y': int(y), 'prompt': prompt, 'answer': answer})
                batch_imgs.clear()
                batch_xy.clear()

            for idx in iterator:
                tile, x, y = patcher[idx]
                batch_imgs.append(tile)
                batch_xy.append((x, y))
                if len(batch_imgs) >= batch_limit:
                    _flush()
            _flush()

        # 1) JSON of per-patch answers.
        with open(json_path, 'w') as f:
            json.dump({'model': model_name, 'prompt': prompt, 'answers': records}, f, indent=2)

        # 2) GeoJSON of patch boxes carrying the answer (QuPath-loadable annotations).
        geo_records = []
        for rec in records:
            x, y = rec['x'], rec['y']
            box = Polygon([
                (x, y), (x + patch_extent_level0, y),
                (x + patch_extent_level0, y + patch_extent_level0), (x, y + patch_extent_level0),
            ])
            geo_records.append({'prompt': rec['prompt'], 'answer': rec['answer'], 'geometry': box})
        gdf = gpd.GeoDataFrame(
            geo_records, columns=['prompt', 'answer', 'geometry'], geometry='geometry'
        )
        gdf.set_crs("EPSG:3857", inplace=True)
        gdf.to_file(geojson_path, driver="GeoJSON")

        return json_path

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

        Parameters:
            patch_features_path (str):
                Path to the HDF5 file containing patch-level features and coordinates.
            slide_encoder (torch.nn.Module):
                Pretrained slide encoder model for generating slide-level features.
            save_features (str):
                Directory where the extracted slide features will be saved.
            device (str, optional):
                Device to run computations on (e.g., 'cuda', 'cpu'). Defaults to 'cuda'.

        Returns:
            str: The absolute path to the slide-level features.

        Workflow:
            1. Load the pretrained slide encoder model and set it to evaluation mode.
            2. Load patch-level features and corresponding coordinates from the provided HDF5 file.
            3. Convert patch-level features into a tensor and move it to the specified device.
            4. Generate slide-level features using the slide encoder, with automatic mixed precision if supported.
            5. Save the slide-level features and associated metadata (e.g., coordinates) in an HDF5 file.
            6. Return the path to the saved slide features.

        Raises:
            FileNotFoundError:
                If the `patch_features_path` does not exist.
            RuntimeError:
                If there is an issue with the slide encoder or tensor operations.

        Example
        -------
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

        if patch_features.size == 0 or (patch_features.ndim > 0 and patch_features.shape[0] == 0):
            warnings.warn(
                f"No patch features available for slide '{self.name}'. Saving empty slide features."
            )
            os.makedirs(save_features, exist_ok=True)
            save_path = os.path.join(save_features, f'{self.name}.h5')
            save_h5(
                save_path,
                assets={
                    'features': np.empty((0,), dtype=np.float32),
                    'coords': np.asarray(coords),
                },
                attributes={
                    'features': {'name': self.name, 'savetodir': save_features},
                    'coords': coords_attrs,
                },
                mode='w'
            )
            return save_path

        # Convert slide_features to tensor
        patch_features = torch.from_numpy(patch_features).float().to(device)
        patch_features = patch_features.unsqueeze(0)  # Add batch dimension

        coords = torch.from_numpy(coords).to(device)
        if torch.is_floating_point(coords):
            coords = torch.round(coords).to(torch.int64)
        else:
            coords = coords.to(torch.int64)
        coords = coords.unsqueeze(0)  # Add batch dimension

        try:
            if "patch_size_level0" in coords_attrs:
                coords_attrs["patch_size_level0"] = int(coords_attrs["patch_size_level0"])
        except Exception:
            pass

        # Prepare input batch dictionary
        batch = {
            'features': patch_features,
            'coords': coords,
            'attributes': coords_attrs
        }

        # Generate slide-level features
        precision = getattr(slide_encoder, "precision", torch.float32)
        with torch.autocast(
            device_type=device.split(":")[0],
            enabled=(precision != torch.float32),
        ):
            features = slide_encoder(batch, device)
        features = features.float().cpu().numpy().squeeze()

        # Save slide-level features if save path is provided
        os.makedirs(save_features, exist_ok=True)
        save_path = os.path.join(save_features, f'{self.name}.h5')

        save_h5(os.path.join(save_features, f'{self.name}.h5'),
                    assets = {
                        'features' : features,
                        'coords': coords.cpu().numpy().squeeze().astype(np.int64, copy=False),
                    },
                    attributes = {
                        'features': {'name': self.name, 'savetodir': save_features},
                        'coords': coords_attrs
                    },
                    mode='w')

        return save_path

    def release(self) -> None:
        """
        Release internal data (CPU/GPU/memory) and clear heavy references in the WSI instance.
        Call this method after you're done processing to avoid memory/GPU leaks.
        """
        # Clear backend image object
        if hasattr(self, "close"):
            self.close()

        if hasattr(self, "img"):
            try:
                if hasattr(self.img, "close"):
                    self.img.close()
            except Exception:
                pass
            self.img = None

        # The path is lightweight and needed for subsequent tasks
        if hasattr(self, "gdf_contours"):
            self.gdf_contours = None

        self._initialized = False
