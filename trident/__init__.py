from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("trident")
except PackageNotFoundError:
    __version__ = "unknown"

from trident.wsi_objects.OpenSlideWSI import OpenSlideWSI
from trident.wsi_objects.CuCIMWSI import CuCIMWSI
from trident.wsi_objects.ImageWSI import ImageWSI
from trident.wsi_objects.SDPCWSI import SDPCWSI
from trident.wsi_objects.WSIFactory import load_wsi, WSIReaderType
from trident.wsi_objects.WSIPatcher import OpenSlideWSIPatcher, WSIPatcher
from trident.wsi_objects.WSIPatcherDataset import WSIPatcherDataset
from trident.wsi_objects.DICOMWebWSI import DICOMWebWSI

from trident.Visualization import visualize_heatmap

from trident.Processor import Processor

from trident.Converter import AnyToTiffConverter

from trident.Maintenance import deprecated

__all__ = [
    "Processor",
    "load_wsi",
    "OpenSlideWSI", 
    "ImageWSI",
    "CuCIMWSI",
    "SDPCWSI",
    "WSIPatcher",
    "OpenSlideWSIPatcher",
    "WSIPatcherDataset",
    "visualize_heatmap",
    "AnyToTiffConverter",
    "deprecated",
    "WSIReaderType",
    "DICOMWebWSI"
]
