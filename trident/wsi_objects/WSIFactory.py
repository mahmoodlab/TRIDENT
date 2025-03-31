import os 
import importlib.util

from trident.wsi_objects.OpenSlideWSI import OpenSlideWSI
from trident.wsi_objects.ImageWSI import ImageWSI
from trident.wsi_objects.CuCIMWSI import CuCIMWSI


# Global variable for OpenSlide-supported extensions
OPENSLIDE_EXTENSIONS = {'.svs', '.tif', '.tiff', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs'}


def load_wsi(slide_path: str, **kwargs):
    """
    Load a whole-slide image using CuCIM-based or OpenSlide-based WSI reader.

    Args:
        slide_path (str): Path to the slide image.
        **kwargs: Additional keyword arguments passed to the WSI class.

    Returns:
        Instance of CuCIMWSI or OpenSlideWSI.
    """
    _, ext = os.path.splitext(slide_path)
    ext = ext.lower()

    if ext in OPENSLIDE_EXTENSIONS:
        cucim_available = importlib.util.find_spec("cucim") is not None
        if cucim_available:
            return CuCIMWSI(slide_path=slide_path, **kwargs)
        else:
            return OpenSlideWSI(slide_path=slide_path, **kwargs)
    else:
        return ImageWSI(slide_path=slide_path, **kwargs)
