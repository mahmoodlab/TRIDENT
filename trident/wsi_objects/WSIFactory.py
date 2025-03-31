import os 
import importlib.util

from trident.wsi_objects.OpenSlideWSI import OpenSlideWSI
from trident.wsi_objects.ImageWSI import ImageWSI
from trident.wsi_objects.CuCIMWSI import CuCIMWSI


# Global variable for OpenSlide-supported extensions
OPENSLIDE_EXTENSIONS = {'.svs', '.tif', '.tiff', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs'}

def load_wsi(slide_path: str, **kwargs):
    """
    Load a whole-slide image using CuCIM for .svs files (if available),
    otherwise use OpenSlide or fallback to ImageWSI.

    Args:
        slide_path (str): Path to the whole-slide image.
        **kwargs: Additional arguments passed to the WSI reader.

    Returns:
        CuCIMWSI, OpenSlideWSI, or ImageWSI instance.
    """
    _, ext = os.path.splitext(slide_path)
    ext = ext.lower()

    if ext == '.svs':
        cucim_available = importlib.util.find_spec("cucim") is not None
        if cucim_available:
            return CuCIMWSI(slide_path=slide_path, **kwargs)

    if ext in OPENSLIDE_EXTENSIONS:
        return OpenSlideWSI(slide_path=slide_path, **kwargs)
    else:
        return ImageWSI(slide_path=slide_path, **kwargs)
