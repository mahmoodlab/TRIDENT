from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from trident.wsi_objects.WSI import WSI, ReadMode


class ISyntaxWSI(WSI):
    """Read native Philips iSyntax slides using the optional pyisyntax backend."""

    def __init__(self, slide_path: str, **kwargs: Any) -> None:
        self.img = None
        super().__init__(slide_path, **kwargs)

    def _lazy_initialize(self) -> None:
        """Open the slide and populate WSI metadata on first use."""
        super()._lazy_initialize()
        if self._initialized:
            return

        try:
            from isyntax import ISyntax
        except ImportError as e:
            raise ImportError(
                "pyisyntax is required for iSyntax support. Install it with "
                "`pip install 'pyisyntax>=0.1.7'` or `pip install '.[isyntax]'`."
            ) from e

        try:
            self.img = ISyntax.open(self.slide_path)
            self.dimensions = self.img.dimensions
            self.width, self.height = self.dimensions
            self.level_count = self.img.level_count
            self.level_downsamples = self.img.level_downsamples
            self.level_dimensions = self.img.level_dimensions
            self.properties = {
                "isyntax.mpp_x": self.img.mpp_x,
                "isyntax.mpp_y": self.img.mpp_y,
                "isyntax.offset_x": self.img.offset_x,
                "isyntax.offset_y": self.img.offset_y,
            }
            if self.mpp is None:
                self.mpp = self._fetch_mpp(self.custom_mpp_keys)
            self.mag = self._fetch_magnification(self.custom_mpp_keys)
            self._initialized = True
        except Exception as e:
            self.close()
            raise RuntimeError(f"Failed to initialize WSI with pyisyntax: {e}") from e

    def _fetch_mpp(self, custom_mpp_keys: Optional[List[str]] = None) -> float:
        """Return the X-axis pixel size in microns, as with OpenSlideWSI."""
        mpp_keys = ["isyntax.mpp_x"]
        if custom_mpp_keys:
            mpp_keys.extend(custom_mpp_keys)
        for key in mpp_keys:
            try:
                mpp = float(self.properties[key])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(mpp) and mpp > 0:
                return mpp
        raise ValueError(
            f"Unable to extract MPP from iSyntax metadata: '{self.slide_path}'. "
            "Set `mpp` explicitly via the constructor or a custom WSI CSV."
        )

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: ReadMode = "pil",
    ) -> Union[Image.Image, np.ndarray]:
        """
        Read an RGB region at a level-0 (x, y) location.

        `size` is (width, height) in pixels at the requested pyramid level.
        """
        self._lazy_initialize()
        if level < 0 or level >= self.level_count:
            raise ValueError(f"Invalid level={level}. Must be in [0, {self.level_count - 1}].")

        # pyisyntax expects coordinates relative to the requested level.
        # Macro-image offsets do not apply to WSI-relative coordinates.
        downsample = self.level_downsamples[level]
        x = int(location[0] // downsample)
        y = int(location[1] // downsample)
        w, h = int(size[0]), int(size[1])
        region = self.img.read_region(x, y, w, h, level=level)[:, :, :3]
        if read_as == "numpy":
            return region
        if read_as == "pil":
            return Image.fromarray(region)
        raise ValueError(f"Invalid `read_as` value: {read_as}. Must be 'pil' or 'numpy'.")

    def get_dimensions(self) -> Tuple[int, int]:
        """Return the dimensions (width, height) at level 0."""
        self._lazy_initialize()
        return self.dimensions

    def get_thumbnail(self, size: Tuple[int, int]) -> Image.Image:
        """Return an RGB thumbnail fitting within `size`, preserving aspect ratio."""
        self._lazy_initialize()
        downsample = max(self.width / size[0], self.height / size[1])
        level = 0
        for i, level_downsample in enumerate(self.level_downsamples):
            if level_downsample <= downsample:
                level = i
        thumbnail = self.read_region((0, 0), level, self.level_dimensions[level])
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        return thumbnail

    def close(self) -> None:
        """Close the backend handle and allow the slide to be reopened on use."""
        if self.img is not None:
            self.img.close()
            self.img = None
        self._initialized = False

    def __del__(self) -> None:
        self.close()
