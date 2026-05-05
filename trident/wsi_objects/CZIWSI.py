from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
from PIL import Image

from trident.wsi_objects.WSI import WSI, ReadMode

try:
    from pylibCZIrw import czi as pylibczi

    _HAS_PYLIBCZI = True
    _PYLIBCZI_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:  # pragma: no cover
    _HAS_PYLIBCZI = False
    _PYLIBCZI_IMPORT_ERROR = e


class CZIWSI(WSI):
    """
    WSI implementation for reading Zeiss CZI slides using `pylibCZIrw`.

    CZI slides may have a non-zero and even negative origin in the global coordinate
    system. TRIDENT's `WSI` interface expects the top-left of the slide to be (0, 0),
    so this backend translates coordinates by the slide's `total_bounding_rectangle`.
    """

    def __init__(self, slide_path: str, **kwargs: Any) -> None:
        """
        Initialize a `CZIWSI` instance.

        Parameters:
            slide_path (str):
                Path to a `.czi` file.
            **kwargs (dict):
                Keyword arguments forwarded to the base `WSI` class. Most important key is:
                - lazy_init (bool, default=True): Whether to defer loading WSI and metadata.
                - mpp (float, optional): If provided, overrides metadata-derived pixel size.
        """
        self.img = None
        self._ctx = None
        self._x0 = 0
        self._y0 = 0
        super().__init__(slide_path, **kwargs)

    def release(self) -> None:
        """Close the underlying CZI reader and release resources."""
        if self._ctx is not None:
            try:
                self._ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._ctx = None
            self.img = None
        self._initialized = False

    def _lazy_initialize(self) -> None:
        """
        Lazily initialize the WSI using `pylibCZIrw`.

        Raises:
            ImportError:
                If `pylibCZIrw` is not installed.
            RuntimeError:
                If the slide cannot be opened or required metadata cannot be read.
        """
        super()._lazy_initialize()

        if self._initialized:
            return

        if not _HAS_PYLIBCZI:
            raise ImportError(
                "pylibCZIrw is required for CZI support. Install it with "
                "`pip install pylibCZIrw` (or `pip install .[czi]` if available). "
                f"Import error was: {_PYLIBCZI_IMPORT_ERROR}"
            )

        try:
            # Keep the document open for random-access reads.
            self._ctx = pylibczi.open_czi(self.slide_path)
            self.img = self._ctx.__enter__()

            rect = self.img.total_bounding_rectangle
            # Note: rect can have negative x/y.
            self._x0, self._y0 = int(rect.x), int(rect.y)
            self.dimensions = (int(rect.w), int(rect.h))
            self.width, self.height = self.dimensions

            # Expose a synthetic pyramid similar to other backends. CZI reading is performed
            # via roi + zoom, which can use internal pyramid subblocks when available.
            downsamples = [1.0]
            w, h = self.dimensions
            while min(w, h) > 512:
                downsamples.append(downsamples[-1] * 2.0)
                w = int(np.ceil(w / 2.0))
                h = int(np.ceil(h / 2.0))
            self.level_downsamples = tuple(downsamples)
            self.level_count = len(self.level_downsamples)
            self.level_dimensions = tuple(
                (int(np.ceil(self.width / d)), int(np.ceil(self.height / d)))
                for d in self.level_downsamples
            )

            # Store raw metadata dict when available.
            try:
                self.properties = self.img.metadata
            except Exception:
                self.properties = None

            if self.mpp is None:
                self.mpp = self._fetch_mpp()
            self.mag = self._fetch_magnification()

            self._initialized = True
        except Exception as e:
            self.release()
            raise RuntimeError(f"Failed to initialize WSI with pylibCZIrw: {e}") from e

    def _fetch_mpp(self) -> float:
        """
        Retrieve microns-per-pixel (MPP) from CZI scaling metadata.

        Returns:
            float: MPP value in microns per pixel (average of X/Y).

        Raises:
            ValueError:
                If scaling metadata cannot be found.
        """
        md = getattr(self.img, "metadata", None)
        if not isinstance(md, dict):
            raise ValueError(f"Unable to extract MPP from slide metadata: '{self.slide_path}'.")

        try:
            scaling = md["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
            # Expect list of dicts like {'@Id': 'X', 'Value': '1.1E-07', ...}
            x_val = None
            y_val = None
            for item in scaling:
                if not isinstance(item, dict):
                    continue
                axis = item.get("@Id")
                val = item.get("Value")
                if axis == "X":
                    x_val = float(val)
                elif axis == "Y":
                    y_val = float(val)
            if x_val is None or y_val is None:
                raise KeyError("Missing X/Y scaling values")
            # `Value` is expressed in meters per pixel in common CZI metadata.
            return float(((x_val + y_val) / 2.0) * 1e6)
        except Exception as e:
            raise ValueError(
                f"Unable to extract MPP from slide metadata: '{self.slide_path}'. "
                "Set `mpp` explicitly when constructing the CZIWSI object."
            ) from e

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: ReadMode = "pil",
    ) -> Union[Image.Image, np.ndarray]:
        """
        Extract a region from the CZI slide.

        Parameters:
            location (Tuple[int, int]):
                (x, y) coordinates in TRIDENT level-0 coordinate system (top-left is (0, 0)).
            level (int):
                Pyramid level to read from.
            size (Tuple[int, int]):
                (width, height) of the region to extract at the requested level.
            read_as ({'pil', 'numpy'}, optional):
                Output format.

        Returns:
            Union[PIL.Image.Image, np.ndarray]: Extracted image region in the specified format.
        """
        self._lazy_initialize()

        if level < 0 or level >= self.level_count:
            raise ValueError(f"Invalid level={level}. Must be in [0, {self.level_count - 1}].")

        downsample = float(self.level_downsamples[level])
        zoom = 1.0 / downsample

        x, y = int(location[0]), int(location[1])
        w, h = int(size[0]), int(size[1])
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid size={size}.")

        # Convert TRIDENT coords (0..width/height) to CZI global coordinates.
        roi = (self._x0 + x, self._y0 + y, int(np.ceil(w * downsample)), int(np.ceil(h * downsample)))

        # Read using default plane (C=0, Z=0, T=0). This can be extended later.
        arr = self.img.read(roi=roi, zoom=zoom, plane={"C": 0, "Z": 0, "T": 0})

        # pylibCZIrw returns BGR for color; convert to RGB for consistency with OpenSlideWSI.
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, :3]

        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
            # For safety; many CZIs are uint8 already.
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # BGR -> RGB
        arr = arr[:, :, ::-1].copy()

        if read_as == "numpy":
            return arr
        if read_as == "pil":
            return Image.fromarray(arr).convert("RGB")
        raise ValueError(f"Invalid `read_as` value: {read_as}. Must be 'pil' or 'numpy'.")

    def get_dimensions(self) -> Tuple[int, int]:
        """Return the dimensions (width, height) of the WSI."""
        self._lazy_initialize()
        return self.dimensions

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        """
        Generate a thumbnail of the slide.

        Parameters:
            size (tuple[int, int]):
                Desired (width, height) of the thumbnail.

        Returns:
            PIL.Image.Image: RGB thumbnail as a PIL Image.
        """
        self._lazy_initialize()
        target_w, target_h = size
        if target_w <= 0 or target_h <= 0:
            raise ValueError(f"Invalid thumbnail size={size}.")

        downsample = max(self.width / target_w, self.height / target_h, 1.0)
        zoom = 1.0 / downsample
        roi = (self._x0, self._y0, self.width, self.height)
        arr = self.img.read(roi=roi, zoom=zoom, plane={"C": 0, "Z": 0, "T": 0})
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr = arr[:, :, ::-1].copy()  # BGR -> RGB
        return Image.fromarray(arr).convert("RGB").resize(size, resample=Image.Resampling.BICUBIC)

