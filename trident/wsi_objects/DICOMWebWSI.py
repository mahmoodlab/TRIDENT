# trident/wsi_objects/DICOMWebWSI.py
from __future__ import annotations
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union, Any
import requests
from io import BytesIO
import os
import numpy as np
import threading
import inspect
import multiprocessing

from trident.wsi_objects.WSI import WSI, ReadMode


class DICOMWebWSI(WSI):
    """
    WSI implementation for accessing DICOM images via DICOMweb protocol.
    Supports WADO-RS (Web Access to DICOM Objects - RESTful Services).

    Automatically handles authentication for:
    - Google Cloud Healthcare API (uses Application Default Credentials)
    - Other DICOMweb servers (basic auth via environment variables)
    """

    def __init__(self, slide_path: str, **kwargs: Any) -> None:
        """Initialize a DICOMWebWSI instance."""
        self.auth_config = kwargs.pop('auth', None)
        self.headers = kwargs.pop('headers', {})

        # Parse and validate the DICOMweb path
        self.dicomweb_url = self._parse_dicomweb_path(slide_path)

        # Auto-detect and configure authentication
        if self.auth_config is None:
            self.auth_config = self._setup_authentication(self.dicomweb_url)

        # Use process-safe storage for sessions (not thread-local)
        # Each process (including workers) will have its own session
        self._process_id = None
        self._session = None

        self.series_metadata = None
        self.instances = []
        self.frame_cache = {}

        # For DICOMweb, derive a better name from the series UID if no name provided
        if 'name' not in kwargs and '/series/' in slide_path:
            series_start = slide_path.rfind('/series/') + 8
            series_end = slide_path.find('/', series_start)
            if series_end == -1:
                series_end = len(slide_path)
            series_uid = slide_path[series_start:series_end]

            name_parts = series_uid.split('.')
            kwargs['name'] = '.'.join(name_parts[-2:]) if len(name_parts) > 1 else series_uid

        super().__init__(slide_path, **kwargs)

    @property
    def session(self):
        """
        Get or create a process-local session.
        Each process (main or worker) gets its own session with fresh auth.
        """
        current_process_id = os.getpid()

        # Check if we need to create a new session for this process
        if self._session is None or self._process_id != current_process_id:
            self._process_id = current_process_id
            self._session = requests.Session()

            if self.auth_config:
                # Determine if we need to instantiate the auth or use it directly
                if inspect.isclass(self.auth_config):
                    # It's a class, instantiate it fresh in this process
                    self._session.auth = self.auth_config()
                elif isinstance(self.auth_config, tuple):
                    # It's a tuple (username, password) for basic auth
                    self._session.auth = self.auth_config
                else:
                    # It's already an auth instance - try to use it
                    # But this might not work across processes
                    self._session.auth = self.auth_config

            self._session.headers.update(self.headers)

        return self._session

    def _setup_authentication(self, url: str) -> Optional[Any]:
        """
        Auto-detect and setup authentication based on URL.
        Returns auth config (class or tuple) that can be recreated in workers.
        """
        # Check if this is a GCP Healthcare API URL
        if 'healthcare.googleapis.com' in url:
            return self._get_gcp_auth_class()

        # Check for basic auth in environment variables
        username = os.getenv('DICOMWEB_USERNAME')
        password = os.getenv('DICOMWEB_PASSWORD')
        if username and password:
            print(f"Using basic authentication from environment variables")
            return (username, password)

        # No authentication needed
        return None

    def _get_gcp_auth_class(self):
        """
        Return the GCPAuth class (not instance) for lazy instantiation in workers.
        """
        try:
            from google.auth import default
            from google.auth.transport.requests import Request
        except ImportError:
            raise ImportError(
                "Google Cloud authentication requires google-auth. Install with:\n"
                "  pip install google-auth google-auth-httplib2"
            )

        print("Using Google Cloud Healthcare API authentication")

        # Define the auth class
        class GCPAuth(requests.auth.AuthBase):
            """Authentication handler for GCP Healthcare API."""

            def __init__(self):
                try:
                    self.credentials, _ = default(
                        scopes=['https://www.googleapis.com/auth/cloud-healthcare']
                    )
                    self.credentials.refresh(Request())
                except Exception as e:
                    raise RuntimeError(
                        "Failed to authenticate with Google Cloud. Ensure either:\n"
                        "  1. GOOGLE_APPLICATION_CREDENTIALS is set, or\n"
                        "  2. You've run 'gcloud auth application-default login'\n"
                        f"Error: {e}"
                    )

            def __call__(self, r):
                """Attach Bearer token to request, refreshing if needed."""
                if not self.credentials.valid:
                    self.credentials.refresh(Request())
                r.headers['Authorization'] = f'Bearer {self.credentials.token}'
                return r

        # Return the class itself, not an instance
        return GCPAuth

    def _parse_dicomweb_path(self, path: str) -> str:
        """
        Parse and validate DICOMweb path.
        Must be at series level. Instance-level URLs are trimmed with a warning.
        Study-level URLs are rejected.
        """
        if path.startswith('dicomweb://'):
            path = path.replace('dicomweb://', 'https://')
        elif not path.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid DICOMweb path: {path}")

        # Check for study level (missing series)
        if '/studies/' in path and '/series/' not in path:
            raise ValueError(
                f"DICOMweb URL must be at series level, not study level.\n"
                f"Expected format:\n"
                f"  https://server/studies/{{studyUID}}/series/{{seriesUID}}\n"
                f"Got:\n"
                f"  {path}\n"
                f"\nA study may contain multiple series. Please specify which series to load."
            )

        # Check if series is present
        if '/series/' not in path:
            raise ValueError(
                f"DICOMweb URL must point to a series.\n"
                f"Expected format:\n"
                f"  https://server/studies/{{studyUID}}/series/{{seriesUID}}\n"
                f"Got:\n"
                f"  {path}"
            )

        # Check for instance level (needs trimming)
        if '/instances/' in path:
            series_end = path.find('/instances/')
            trimmed_path = path[:series_end]
            instance_part = path[series_end:]

            print(f"⚠ Warning: Instance-level URL provided.")
            print(f"  Trimmed: {instance_part}")
            print(f"  Loading entire series instead: {trimmed_path}")
            print(f"  (All pyramid levels in the series will be loaded automatically)")

            path = trimmed_path

        # Check for other endpoints that should be removed
        unwanted_endpoints = ['/frames/', '/metadata', '/rendered', '/thumbnail']
        for endpoint in unwanted_endpoints:
            if endpoint in path:
                endpoint_start = path.find(endpoint)
                trimmed_path = path[:endpoint_start]

                print(f"⚠ Warning: URL contains endpoint '{endpoint}'")
                print(f"  Trimmed to series level: {trimmed_path}")

                path = trimmed_path
                break

        # Ensure no trailing slash
        path = path.rstrip('/')

        # Final validation
        if '/series/' in path:
            series_index = path.rfind('/series/')
            after_series = path[series_index + 8:]
            if not after_series:
                raise ValueError(
                    f"DICOMweb URL ends with /series/ but no series UID provided.\n"
                    f"Expected format:\n"
                    f"  https://server/studies/{{studyUID}}/series/{{seriesUID}}\n"
                    f"Got:\n"
                    f"  {path}"
                )

        return path

    def _lazy_initialize(self) -> None:
        """
        Lazily initialize DICOMweb connection and fetch metadata.
        """
        super()._lazy_initialize()

        if not self.lazy_init:
            try:
                # Fetch all instances in the series
                self._fetch_series_metadata()

                # Sort instances by size to create pyramid (largest = level 0)
                self._organize_pyramid_levels()

                # Get dimensions from highest resolution instance AFTER organizing
                # IMPORTANT: self.instances stores dimensions as:
                #   'cols' = WIDTH (x-axis, horizontal)
                #   'rows' = HEIGHT (y-axis, vertical)
                self.width = self.instances[0]['cols']  # WIDTH
                self.height = self.instances[0]['rows']  # HEIGHT
                self.dimensions = (self.width, self.height)  # (WIDTH, HEIGHT)

                print(f"\n✓ Final dimensions: {self.width} (width) × {self.height} (height)")

                # Extract pixel spacing for MPP calculation
                if self.mpp is None:
                    self.mpp = self._fetch_mpp_from_dicom()

                # Extract or calculate magnification
                self.mag = self._fetch_magnification(self.custom_mpp_keys)

                # Print detailed pyramid info
                self.print_pyramid_info()

                self.lazy_init = True

            except Exception as e:
                raise RuntimeError(f"Failed to initialize DICOMweb WSI: {e}") from e

    def _parse_dicom_json(self, json_data: dict) -> dict:
        """Parse DICOM JSON format to simple dictionary."""
        result = {}

        for tag, value in json_data.items():
            if 'Value' in value:
                val = value['Value']

                # Map common tags to readable names
                if tag == '00080018':  # SOP Instance UID
                    result['SOPInstanceUID'] = val[0] if len(val) == 1 else val
                elif tag == '00280010':  # Rows (tile height)
                    result['Rows'] = val[0] if len(val) == 1 else val
                elif tag == '00280011':  # Columns (tile width)
                    result['Columns'] = val[0] if len(val) == 1 else val
                elif tag == '00480006':  # TotalPixelMatrixRows
                    result['TotalPixelMatrixRows'] = val[0] if len(val) == 1 else val
                elif tag == '00480007':  # TotalPixelMatrixColumns
                    result['TotalPixelMatrixColumns'] = val[0] if len(val) == 1 else val
                elif tag == '00280008':  # NumberOfFrames
                    result['NumberOfFrames'] = val[0] if len(val) == 1 else val
                elif tag == '00280030':  # Pixel Spacing
                    result['PixelSpacing'] = val
                elif tag == '00080008':  # ImageType
                    result['ImageType'] = val
                elif tag == '52009229':  # Shared Functional Groups Sequence
                    result['SharedFunctionalGroups'] = val
                else:
                    # Store with VR or tag as key
                    vr = value.get('vr', '')
                    key = vr if vr else tag
                    result[key] = val[0] if len(val) == 1 else val

        return result

    def _organize_pyramid_levels(self) -> None:
        """
        Organize instances into pyramid levels.
        Filters to keep only actual pyramid levels (excludes LABEL, OVERVIEW, THUMBNAIL).
        """
        # Filter instances to keep only actual pyramid levels
        pyramid_instances = []

        for instance in self.instances:
            raw_json = instance['raw_json']

            # Get ImageType (0008,0008)
            image_type = raw_json.get('00080008', {}).get('Value', [])
            image_type_str = ' '.join(str(t) for t in image_type)

            # Exclude LABEL, OVERVIEW, THUMBNAIL
            if any(keyword in image_type_str.upper() for keyword in ['LABEL', 'OVERVIEW', 'THUMBNAIL']):
                continue

            pyramid_instances.append(instance)

        if not pyramid_instances:
            print("Warning: No pyramid instances found after filtering, using all instances")
            pyramid_instances = self.instances

        # Sort by pixel count (descending) - largest is level 0
        pyramid_instances.sort(key=lambda x: x['pixels'], reverse=True)

        self.instances = pyramid_instances
        self.level_count = len(self.instances)
        self.level_dimensions = []
        self.level_downsamples = []

        # Calculate downsamples relative to level 0
        if self.instances:
            base_width = self.instances[0]['cols']

            for instance in self.instances:
                width = instance['cols']
                height = instance['rows']
                downsample = base_width / width

                self.level_dimensions.append((width, height))
                self.level_downsamples.append(downsample)

    def _fetch_magnification(self, custom_mpp_keys: Optional[list] = None) -> Optional[float]:
        """
        Extract magnification from DICOM metadata.

        Priority order:
        1. Calculate from MPP (most reliable for WSI)
        2. ObjectiveLensPower from DICOM (with validation)
        3. Fall back to parent class method
        """
        highest_res = self.instances[0]
        raw_json = highest_res['raw_json']

        # Calculate from MPP if available (PREFERRED METHOD)
        # Standard reference: 40x = 0.25 µm/pixel, 20x = 0.5 µm/pixel, etc.
        calculated_mag = None
        if self.mpp:
            calculated_mag = 10.0 / self.mpp  # 40x at 0.25µm, 20x at 0.5µm

        # Try to get from Optical Path Sequence (0048,0105)
        dicom_mag = None
        if '00480105' in raw_json:
            optical_path_seq = raw_json['00480105'].get('Value', [])
            for optical_path in optical_path_seq:
                # Objective Lens Power (0048,0106)
                if '00480106' in optical_path:
                    lens_power = optical_path['00480106'].get('Value', [None])[0]
                    if lens_power:
                        dicom_mag = float(lens_power)
                        break

        # Decide which to use
        if calculated_mag and dicom_mag:
            # Both available - validate consistency
            ratio = calculated_mag / dicom_mag

            # If they're within 20% of each other, trust DICOM
            if 0.8 <= ratio <= 1.2:
                print(f"Using magnification from DICOM metadata: {dicom_mag}x (MPP-based: {calculated_mag:.1f}x)")
                return dicom_mag
            else:
                # Significant mismatch - trust MPP calculation
                print(
                    f"DICOM magnification ({dicom_mag}x) inconsistent with MPP-based calculation ({calculated_mag:.1f}x)")
                print(f"Using MPP-based magnification: {calculated_mag:.1f}x")
                return calculated_mag

        elif calculated_mag:
            # Only MPP available
            print(f"Calculated magnification from MPP ({self.mpp:.4f} µm/pixel): {calculated_mag:.1f}x")
            return calculated_mag

        elif dicom_mag:
            # Only DICOM available
            print(f"Using magnification from DICOM metadata: {dicom_mag}x")
            return dicom_mag

        # Fall back to parent class method
        return super()._fetch_magnification(custom_mpp_keys)

    def _fetch_frame_google_cloud(self, instance_url: str, frame_number: int) -> Optional[Image.Image]:
        """
        Fetch a frame from Google Cloud Healthcare API using the rendered endpoint.

        This is simpler than parsing multipart responses - Google's rendered
        endpoint returns standard image formats (JPEG/PNG).
        """
        # Try rendered endpoint first (simpler, returns JPEG/PNG)
        rendered_url = f"{instance_url}/frames/{frame_number}/rendered"
        headers = {'Accept': 'image/jpeg, image/png'}

        try:
            response = self.session.get(rendered_url, headers=headers)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [404, 406]:
                # Frame doesn't exist (edge case)
                return None

            # If rendered fails, try multipart format
            frame_url = f"{instance_url}/frames/{frame_number}"
            headers = {
                'Accept': 'multipart/related; transfer-syntax=1.2.840.10008.1.2.1; type="application/octet-stream"'
            }

            try:
                response = self.session.get(frame_url, headers=headers)
                response.raise_for_status()
                return self._parse_multipart_frame(response)
            except:
                raise e  # Re-raise original error

    def _fetch_mpp_from_dicom(self) -> float:
        """
        Extract microns per pixel from highest resolution DICOM instance.
        Prints the source for transparency.
        """
        highest_res = self.instances[0]
        metadata = highest_res['metadata']
        raw_json = highest_res['raw_json']

        # Check Shared Functional Groups Sequence (WSI-specific)
        if '52009229' in raw_json:
            shared_groups = raw_json['52009229'].get('Value', [])
            for group in shared_groups:
                # Pixel Measures Sequence (0028,9110)
                if '00289110' in group:
                    pixel_measure_seq = group['00289110'].get('Value', [])
                    for measure in pixel_measure_seq:
                        # Pixel Spacing (0028,0030)
                        if '00280030' in measure:
                            pixel_spacing = measure['00280030'].get('Value', [])
                            if pixel_spacing and len(pixel_spacing) >= 2:
                                # Convert mm to microns, average row and column
                                row_spacing_mm = float(pixel_spacing[0])
                                col_spacing_mm = float(pixel_spacing[1])
                                mpp = (row_spacing_mm + col_spacing_mm) / 2 * 1000
                                print(f"Extracted MPP from Shared Functional Groups: {mpp:.4f} µm/pixel")
                                print(f"  Pixel Spacing: [{row_spacing_mm}, {col_spacing_mm}] mm")
                                return mpp

        # Fallback: Direct Pixel Spacing (0028,0030)
        pixel_spacing = metadata.get('PixelSpacing')
        if pixel_spacing:
            row_spacing_mm = float(pixel_spacing[0])
            col_spacing_mm = float(pixel_spacing[1])
            mpp = (row_spacing_mm + col_spacing_mm) / 2 * 1000
            print(f"Extracted MPP from Pixel Spacing: {mpp:.4f} µm/pixel")
            print(f"  Pixel Spacing: [{row_spacing_mm}, {col_spacing_mm}] mm")
            return mpp

        # Fallback: Imager Pixel Spacing (0018,1164)
        imager_pixel_spacing = metadata.get('ImagerPixelSpacing')
        if imager_pixel_spacing:
            row_spacing_mm = float(imager_pixel_spacing[0])
            col_spacing_mm = float(imager_pixel_spacing[1])
            mpp = (row_spacing_mm + col_spacing_mm) / 2 * 1000
            print(f"Extracted MPP from Imager Pixel Spacing: {mpp:.4f} µm/pixel")
            return mpp

        raise ValueError(f"Unable to extract MPP from DICOM metadata")

    def _fetch_series_metadata(self) -> None:
        """Fetch metadata for all instances in the series."""
        metadata_url = f"{self.dicomweb_url}/metadata"
        response = self.session.get(metadata_url)
        response.raise_for_status()

        metadata_list = response.json()

        print(f"\n📊 Parsing {len(metadata_list)} DICOM instances...")

        for idx, instance_json in enumerate(metadata_list):
            parsed = self._parse_dicom_json(instance_json)

            # Get SOP Instance UID
            sop_uid = parsed.get('SOPInstanceUID', '')

            # Get dimensions - DICOM standard tags
            # 00480006 = TotalPixelMatrixRows (HEIGHT)
            # 00480007 = TotalPixelMatrixColumns (WIDTH)
            # 00280010 = Rows (tile HEIGHT)
            # 00280011 = Columns (tile WIDTH)

            total_rows = instance_json.get('00480006', {}).get('Value', [0])[0] if '00480006' in instance_json else 0
            total_cols = instance_json.get('00480007', {}).get('Value', [0])[0] if '00480007' in instance_json else 0

            # Fallback to regular Rows/Columns if TotalPixelMatrix not available
            if total_rows == 0:
                total_rows = instance_json.get('00280010', {}).get('Value', [0])[0]
            if total_cols == 0:
                total_cols = instance_json.get('00280011', {}).get('Value', [0])[0]

            # Get tile dimensions
            tile_rows = instance_json.get('00280010', {}).get('Value', [0])[0]
            tile_cols = instance_json.get('00280011', {}).get('Value', [0])[0]

            # Get number of frames
            num_frames = instance_json.get('00280008', {}).get('Value', [1])[0]

            # VALIDATION: Tile dimensions should not exceed total dimensions
            # If they do, the metadata is likely swapped
            swapped = False
            if tile_cols > total_cols or tile_rows > total_rows:
                print(f"\n  ⚠️  Instance {idx}: Detected dimension swap!")
                print(f"      Before: total={total_cols}×{total_rows}, tile={tile_cols}×{tile_rows}")

                # Try swapping tile dimensions
                tile_rows, tile_cols = tile_cols, tile_rows

                # If still invalid, swap total dimensions too
                if tile_cols > total_cols or tile_rows > total_rows:
                    total_rows, total_cols = total_cols, total_rows

                print(f"      After:  total={total_cols}×{total_rows}, tile={tile_cols}×{tile_rows}")
                swapped = True

            # Calculate expected grid size
            if tile_cols > 0 and tile_rows > 0:
                expected_tiles_h = (int(total_cols) + int(tile_cols) - 1) // int(tile_cols)
                expected_tiles_v = (int(total_rows) + int(tile_rows) - 1) // int(tile_rows)
                expected_total = expected_tiles_h * expected_tiles_v
            else:
                expected_total = 1

            print(f"\n  Instance {idx} (Level {idx}):")
            print(f"    SOP UID: {sop_uid}")
            print(f"    Total dimensions: {total_cols} × {total_rows}")
            print(f"    Tile size: {tile_cols} × {tile_rows}")
            print(f"    Frames in DICOM: {num_frames}")
            print(f"    Expected grid: {expected_tiles_h} × {expected_tiles_v} = {expected_total} tiles")
            if swapped:
                print(f"    ⚠️  Dimensions were auto-corrected")

            if num_frames != expected_total:
                print(f"    ⚠️  MISMATCH: DICOM has {num_frames} frames but grid expects {expected_total}")

            self.instances.append({
                'sop_instance_uid': sop_uid,
                'rows': int(total_rows),  # HEIGHT
                'cols': int(total_cols),  # WIDTH
                'tile_rows': int(tile_rows),  # Tile HEIGHT
                'tile_cols': int(tile_cols),  # Tile WIDTH
                'num_frames': int(num_frames),
                'pixels': int(total_rows) * int(total_cols),
                'metadata': parsed,
                'raw_json': instance_json
            })

    def get_best_level_for_mag(self, target_mag: float, tolerance: float = 0.1) -> int:
        """
        Get the pyramid level closest to the target magnification.

        Parameters
        ----------
        target_mag : float
            Target magnification (e.g., 20 for 20x)
        tolerance : float
            Acceptable tolerance as a fraction (default 0.1 = 10%)

        Returns
        -------
        int
            Best pyramid level for the target magnification

        Examples
        --------
        >>> level = wsi.get_best_level_for_mag(20)  # Get level closest to 20x
        """
        if not self.mag:
            raise ValueError("Magnification not available for this WSI")

        # Calculate target MPP from target magnification
        target_mpp = 10.0 / target_mag  # 40x = 0.25, 20x = 0.5, etc.

        # Find closest level
        best_level = 0
        best_diff = float('inf')

        for level in range(self.level_count):
            # Calculate MPP at this level
            level_mpp = self.mpp * self.level_downsamples[level]
            diff = abs(level_mpp - target_mpp)

            if diff < best_diff:
                best_diff = diff
                best_level = level

        # Check if within tolerance
        level_mpp = self.mpp * self.level_downsamples[best_level]
        level_mag = 10.0 / level_mpp

        if abs(level_mag - target_mag) / target_mag > tolerance:
            print(f"Warning: Closest level {best_level} is at {level_mag:.1f}x, "
                  f"which differs from target {target_mag}x by "
                  f"{abs(level_mag - target_mag) / target_mag * 100:.1f}%")

        return best_level

    def get_mpp_for_level(self, level: int) -> float:
        """
        Get the microns per pixel for a specific pyramid level.

        Parameters
        ----------
        level : int
            Pyramid level

        Returns
        -------
        float
            MPP at the specified level
        """
        if level >= self.level_count:
            raise ValueError(f"Invalid level {level}, max is {self.level_count - 1}")

        return self.mpp * self.level_downsamples[level]

    def get_mag_for_level(self, level: int) -> float:
        """
        Get the effective magnification for a specific pyramid level.

        Parameters
        ----------
        level : int
            Pyramid level

        Returns
        -------
        float
            Effective magnification at the specified level
        """
        level_mpp = self.get_mpp_for_level(level)
        return 10.0 / level_mpp

    def print_pyramid_info(self) -> None:
        """Print detailed pyramid information including magnifications."""
        print(f"\n{'=' * 70}")
        print(f"WSI Pyramid Information:")
        print(f"{'=' * 70}")
        print(f"  Full dimensions (level 0): {self.dimensions[0]} (w) × {self.dimensions[1]} (h)")
        print(f"  Base MPP: {self.mpp:.4f} µm/pixel")
        print(f"  Base magnification: {self.mag:.1f}x")
        print(f"  Number of levels: {self.level_count}")
        print(f"\nPyramid levels:")
        print(f"  {'Level':<7} {'Width':<8} {'Height':<8} {'Downsample':<12} {'MPP':<15} {'Mag':<10} {'Tiles':<10}")
        print(f"  {'-' * 7} {'-' * 8} {'-' * 8} {'-' * 12} {'-' * 15} {'-' * 10} {'-' * 10}")

        for level in range(self.level_count):
            width, height = self.level_dimensions[level]
            downsample = self.level_downsamples[level]
            mpp = self.get_mpp_for_level(level)
            mag = self.get_mag_for_level(level)
            tiles = self.instances[level]['num_frames']

            print(f"  {level:<7} {width:<8} {height:<8} {downsample:6.2f}×      "
                  f"{mpp:6.4f} µm/px   {mag:6.1f}×    {tiles:6d}")

        # Sanity check: width should generally be close to height for typical slides
        aspect_ratio = self.width / self.height
        if aspect_ratio > 3 or aspect_ratio < 0.33:
            print(f"\n⚠️  WARNING: Unusual aspect ratio ({aspect_ratio:.2f})")
            print(f"    This might indicate swapped dimensions.")
            print(f"    If thumbnail looks rotated, dimensions may need correction.")

        print(f"{'=' * 70}\n")

    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int], read_as: ReadMode = 'pil', ) -> \
    Union[Image.Image, np.ndarray]:
        """
        Extract a region from the DICOM image via DICOMweb.
        Handles both tiled and non-tiled DICOM instances.
        """
        if level >= self.level_count:
            raise ValueError(f"Invalid level {level}, max is {self.level_count - 1}")

        instance = self.instances[level]
        sop_uid = instance['sop_instance_uid']
        tile_width = instance['tile_cols']
        tile_height = instance['tile_rows']
        num_frames = instance['num_frames']

        x, y = location
        req_width, req_height = size

        # Build instance-specific URL
        instance_url = f"{self.dicomweb_url}/instances/{sop_uid}"

        # Detect if we're using Google Cloud Healthcare API
        is_google_cloud = 'healthcare.googleapis.com' in self.dicomweb_url

        # SPECIAL CASE: Single-frame levels (non-tiled)
        # If there's only 1 frame, it contains the entire level image
        if num_frames == 1:
            frame_url = f"{instance_url}/frames/1"
            if is_google_cloud:
                frame_url = f"{instance_url}/frames/1/rendered"

            headers = {'Accept': 'image/jpeg, image/png'}

            try:
                response = self.session.get(frame_url, headers=headers)
                response.raise_for_status()

                # Get the full level image
                full_image = Image.open(BytesIO(response.content))

                # Crop to requested region
                cropped = full_image.crop((x, y, x + req_width, y + req_height))

                if read_as == 'pil':
                    return cropped.convert('RGB')
                elif read_as == 'numpy':
                    return np.array(cropped)
                else:
                    raise ValueError(f"Invalid read_as: {read_as}")

            except Exception as e:
                print(f"\n⚠ Failed to read single-frame level {level}: {e}")
                # Fall back to white canvas
                if read_as == 'pil':
                    return Image.new('RGB', size, (255, 255, 255))
                else:
                    return np.ones((req_height, req_width, 3), dtype=np.uint8) * 255

        # MULTI-FRAME CASE: Standard tiled DICOM
        # Calculate which tiles we need
        start_tile_col = x // tile_width
        start_tile_row = y // tile_height
        end_tile_col = (x + req_width - 1) // tile_width
        end_tile_row = (y + req_height - 1) // tile_height

        # Calculate tiles per row
        total_cols = instance['cols']
        tiles_per_row = (total_cols + tile_width - 1) // tile_width

        # Create canvas for the result
        canvas = Image.new('RGB', size, (255, 255, 255))

        # Track success/failure
        tiles_fetched = 0
        tiles_failed = 0

        # Fetch and composite tiles
        for tile_row in range(start_tile_row, end_tile_row + 1):
            for tile_col in range(start_tile_col, end_tile_col + 1):
                # Calculate frame number (1-based)
                frame_number = tile_row * tiles_per_row + tile_col + 1

                # Skip if frame number exceeds available frames
                if frame_number > num_frames:
                    tiles_failed += 1
                    continue

                # Check cache
                cache_key = (sop_uid, frame_number)

                if cache_key not in self.frame_cache:
                    # Build frame URL based on server type
                    if is_google_cloud:
                        frame_url = f"{instance_url}/frames/{frame_number}/rendered"
                        headers = {'Accept': 'image/jpeg, image/png'}
                    else:
                        frame_url = f"{instance_url}/frames/{frame_number}"
                        headers = {'Accept': 'image/jpeg, image/png, application/octet-stream'}

                    try:
                        response = self.session.get(frame_url, headers=headers)
                        response.raise_for_status()

                        tile_img = Image.open(BytesIO(response.content))

                        # VALIDATION: Check if tile is valid
                        tile_array = np.array(tile_img)
                        if tile_array.size > 0:
                            mean_val = tile_array.mean()

                            # Skip suspiciously dark tiles (likely corrupt)
                            if mean_val < 20:
                                if tiles_failed <= 3:
                                    print(f"  Frame {frame_number} is black/corrupt (mean={mean_val:.1f}), skipping")
                                tiles_failed += 1
                                continue

                        self.frame_cache[cache_key] = tile_img
                        tiles_fetched += 1

                    except requests.exceptions.HTTPError as e:
                        tiles_failed += 1

                        # If frame doesn't exist, skip silently
                        if e.response.status_code in [400, 404, 406]:
                            if tiles_failed <= 3:
                                print(f"  Frame {frame_number} unavailable (status {e.response.status_code})")
                            continue

                        # For other errors, print and raise
                        print(f"\n⚠ Frame fetch failed:")
                        print(f"  Frame: {frame_number}")
                        print(f"  Status: {e.response.status_code}")
                        print(f"  Response: {e.response.text[:200]}")
                        raise

                    except Exception as e:
                        print(f"\n⚠ Unexpected error fetching frame {frame_number}: {e}")
                        tiles_failed += 1
                        continue
                else:
                    tile_img = self.frame_cache[cache_key]
                    tiles_fetched += 1

                # Calculate where this tile goes in our canvas
                tile_x = tile_col * tile_width
                tile_y = tile_row * tile_height

                # Calculate the region within the tile we need
                src_x = max(0, x - tile_x)
                src_y = max(0, y - tile_y)
                src_x2 = min(tile_width, x + req_width - tile_x)
                src_y2 = min(tile_height, y + req_height - tile_y)

                # Calculate where to paste in the canvas
                dst_x = max(0, tile_x - x)
                dst_y = max(0, tile_y - y)

                # Crop the tile and paste into canvas
                tile_crop = tile_img.crop((src_x, src_y, src_x2, src_y2))
                canvas.paste(tile_crop, (dst_x, dst_y))

        # Warn if no tiles were fetched
        total_tiles = (end_tile_row - start_tile_row + 1) * (end_tile_col - start_tile_col + 1)
        if tiles_fetched == 0:
            print(f"\n⚠ WARNING: Failed to fetch any tiles!")
            print(f"  Level: {level}")
            print(f"  Tiles requested: {total_tiles}")
            print(f"  Available frames: {num_frames}")

        if read_as == 'pil':
            return canvas.convert('RGB')
        elif read_as == 'numpy':
            return np.array(canvas)
        else:
            raise ValueError(f"Invalid read_as: {read_as}")

    def get_thumbnail(self, size: Tuple[int, int]) -> Image.Image:
        """
        Generate thumbnail from an appropriate resolution level.

        Avoids lowest levels which may have sparse tiling issues.
        Uses a level with good coverage and reasonable performance.
        """
        target_width, target_height = size

        # For DICOMweb with potentially sparse tiling at low levels,
        # explicitly avoid the lowest 2 levels which often have issues
        max_level_to_use = min(3, self.level_count - 1)

        # Find a level that's reasonable for thumbnail generation
        # Target: 2-4x the thumbnail size for good quality
        best_level = None

        for level in range(max_level_to_use, -1, -1):
            level_width, level_height = self.level_dimensions[level]

            # Skip if this level is too small (< 200 pixels in any dimension)
            if level_width < 200 or level_height < 200:
                continue

            # Check if this level is suitable
            # We want something 2-4x larger than thumbnail size
            width_ratio = level_width / target_width
            height_ratio = level_height / target_height

            if width_ratio >= 2 or height_ratio >= 2:
                best_level = level
                break

        # Fallback: if no suitable level found, use level with most tiles
        if best_level is None:
            # Use the level with the most coverage (avoid single-tile levels)
            best_level = 0
            for level in range(1, min(4, self.level_count)):
                if self.instances[level]['num_frames'] > 10:
                    best_level = level
                    break

        print(f"Generating thumbnail from level {best_level} ({self.level_dimensions[best_level]})")

        # Read the full image at this level
        level_width, level_height = self.level_dimensions[best_level]

        try:
            # Read entire level
            region = self.read_region(
                (0, 0),
                best_level,
                (level_width, level_height),
                read_as='pil'
            )

            # Resize to requested thumbnail size while preserving aspect ratio
            # Calculate scaling to fit within target size
            width_scale = target_width / level_width
            height_scale = target_height / level_height
            scale = min(width_scale, height_scale)

            new_width = int(level_width * scale)
            new_height = int(level_height * scale)

            thumbnail = region.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # If not square thumbnail requested, return as-is
            if target_width == target_height:
                # Create square canvas and paste (for segmentation which expects square)
                canvas = Image.new('RGB', (target_width, target_height), (255, 255, 255))

                # Center the thumbnail
                x_offset = (target_width - new_width) // 2
                y_offset = (target_height - new_height) // 2
                canvas.paste(thumbnail, (x_offset, y_offset))

                return canvas
            else:
                return thumbnail

        except Exception as e:
            print(f"Failed to generate thumbnail from level {best_level}: {e}")

            # Fallback: try level 2 or 3 (usually safe)
            fallback_level = min(2, self.level_count - 1)
            print(f"Falling back to level {fallback_level}")

            level_width, level_height = self.level_dimensions[fallback_level]
            region = self.read_region(
                (0, 0),
                fallback_level,
                (level_width, level_height),
                read_as='pil'
            )
            return region.resize(size, Image.Resampling.LANCZOS)

    def get_dimensions(self) -> Tuple[int, int]:
        """Return dimensions of the highest resolution level."""
        return self.dimensions

    def close(self):
        """Close the session and clear cache."""
        if self._session is not None:
            try:
                self._session.close()
            except:
                pass
            self._session = None
            self._process_id = None

        self.frame_cache.clear()
        self.lazy_init = False
