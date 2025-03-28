from __future__ import annotations

import torch
import os
import json
from typing import List
import h5py
import numpy as np
import cv2
import pandas as pd
from geopandas import gpd
from shapely import Polygon


def get_weights_path(encoder_type, encoder_name):
    """
    Retrieve the path to the weights file for a given model name.

    This function looks up the path to the weights file in a local checkpoint
    registry (local_ckpts.json). If the path in the registry is absolute, it
    returns that path. If the path is relative, it joins the relative path with
    the provided weights_root directory.

    Args:
        weights_root (str): The root directory where weights files are stored.
        name (str): The name of the model whose weights path is to be retrieved.

    Returns:
        str: The absolute path to the weights file.
    """
    root = os.path.join(os.path.dirname(__file__), f"{encoder_type}_encoder_models")
    assert encoder_type in ['patch', 'slide'], f"Encoder type must be 'patch' or 'slide', not '{encoder_type}'"
    registry_path = os.path.join(root, "local_ckpts.json")
    with open(registry_path, "r") as f:
        registry = json.load(f)
    path = registry.get(encoder_name)
    if not path:
        raise ValueError(f"Please specify the weights path to '{encoder_name}' in '{registry_path}'")
    path = path if os.path.isabs(path) else os.path.abspath(os.path.join(root, 'model_zoo', path)) # Make path absolute
    if not os.path.exists(path):
        print(f"WARNING: Path at '{path}' does not exist. Please double-check the registry in '{registry_path}'")
    return path


################################################################################

def create_lock(path, suffix = None):
    """
    The `create_lock` function creates a lock file to signal that a particular file or process 
    is currently being worked on. This is especially useful in multiprocessing or distributed 
    systems to avoid conflicts or multiple processes working on the same resource.

    Parameters:
    -----------
    path : str
        The path to the file or resource being locked.
    suffix : str, optional
        An additional suffix to append to the lock file name. This allows for creating distinct 
        lock files for similar resources. Defaults to None.

    Returns:
    --------
    None
        The function creates a `.lock` file in the specified path and does not return anything.

    Example:
    --------
    >>> create_lock("/path/to/resource")
    >>> # Creates a file named "/path/to/resource.lock" to indicate the resource is locked.
    """
    if suffix is not None:
        path = f"{path}_{suffix}"
    lock_file = f"{path}.lock"
    with open(lock_file, 'w') as f:
        f.write("")

#####################

def remove_lock(path, suffix = None):
    """
    The `remove_lock` function removes a lock file, signaling that the file or process 
    is no longer in use and is available for other operations.

    Parameters:
    -----------
    path : str
        The path to the file or resource whose lock needs to be removed.
    suffix : str, optional
        An additional suffix to identify the lock file. Defaults to None.

    Returns:
    --------
    None
        The function deletes the `.lock` file associated with the resource.

    Example:
    --------
    >>> remove_lock("/path/to/resource")
    >>> # Removes the file "/path/to/resource.lock", indicating the resource is unlocked.
    """
    if suffix is not None:
        path = f"{path}_{suffix}"
    lock_file = f"{path}.lock"
    os.remove(lock_file)

#####################

def is_locked(path, suffix = None):
    """
    The `is_locked` function checks if a resource is currently locked by verifying 
    the existence of a `.lock` file.

    Parameters:
    -----------
    path : str
        The path to the file or resource to check for a lock.
    suffix : str, optional
        An additional suffix to identify the lock file. Defaults to None.

    Returns:
    --------
    bool
        True if the `.lock` file exists, indicating the resource is locked. False otherwise.

    Example:
    --------
    >>> is_locked("/path/to/resource")
    False
    >>> create_lock("/path/to/resource")
    >>> is_locked("/path/to/resource")
    True
    """
    if suffix is not None:
        path = f"{path}_{suffix}"
    return os.path.exists(f"{path}.lock")


###########################################################################
def update_log(path_to_log, key, message):
    """
    The `update_log` function appends or updates a message in a log file. It is useful for tracking 
    progress or recording errors during a long-running process.

    Parameters:
    -----------
    path_to_log : str
        The path to the log file where messages will be written.
    key : str
        A unique identifier for the log entry, such as a slide name or file ID.
    message : str
        The message to log, such as a status update or error message.

    Returns:
    --------
    None
        The function writes to the log file in-place.

    Example:
    --------
    >>> update_log("processing.log", "slide1", "Processing completed")
    >>> # Appends or updates "slide1: Processing completed" in the log file.
    """    
    # Create log if it doesn't exist
    if not os.path.exists(path_to_log):
        with open(path_to_log, 'w') as f:
            f.write(f'{key}: {message}\n')
            return
        
    # If slide id already in log, delete the message and add the new one
    if os.path.exists(path_to_log):
        with open(path_to_log, 'r') as f:
            lines = f.readlines()
        with open(path_to_log, 'w') as f:
            for line in lines:
                if not line.split(':')[0] == key:
                    f.write(line)
            f.write(f'{key}: {message}\n')
        return
    
################################################################################

def save_h5(save_path, assets, attributes = None, mode = 'w'):
    """
    The `save_h5` function saves a dictionary of assets to an HDF5 file. This is commonly used to store 
    large datasets or hierarchical data structures in a compact and organized format.

    Parameters:
    -----------
    save_path : str
        The path where the HDF5 file will be saved.
    assets : dict
        A dictionary containing the data to save. Keys represent dataset names, and values are NumPy arrays.
    attributes : dict, optional
        A dictionary mapping dataset names to additional metadata (attributes) to save alongside the data. Defaults to None.
    mode : str, optional
        The file mode for opening the HDF5 file. Options include 'w' (write) and 'a' (append). Defaults to 'w'.

    Returns:
    --------
    None
        The function writes data and attributes to the specified HDF5 file.

    Example:
    --------
    >>> assets = {'data': np.array([1, 2, 3]), 'labels': np.array([0, 1, 1])}
    >>> attributes = {'data': {'description': 'Numerical data'}}
    >>> save_h5("output.h5", assets, attributes)
    >>> # Saves datasets and attributes to "output.h5".
    """

    with h5py.File(save_path, mode) as file:
        for key, val in assets.items():
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (1, ) + data_shape[1:]
                maxshape = (None, ) + data_shape[1:]
                dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                dset[:] = val
                if attributes is not None:
                    if key in attributes.keys():
                        for attr_key, attr_val in attributes[key].items():
                            try:
                                # Serialize if the attribute value is a dictionary
                                if isinstance(attr_val, dict):
                                    attr_val = json.dumps(attr_val)
                                # Serialize Nones
                                elif attr_val is None:
                                    attr_val = 'None'
                                dset.attrs[attr_key] = attr_val
                            except:
                                raise Exception(f'WARNING: Could not save attribute {attr_key} with value {attr_val} for asset {key}')
                                
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = val

################################################################################

class JSONsaver(json.JSONEncoder):
    """
    The `JSONsaver` class extends the `json.JSONEncoder` to handle objects that are typically 
    unserializable by the standard JSON encoder. It provides support for custom types, including 
    NumPy arrays, ranges, PyTorch data types, and callable objects.

    This class is particularly useful when saving complex configurations or datasets to JSON files, 
    ensuring that all objects are serialized correctly or replaced with representative strings.

    Methods:
    --------
    default(obj):
        Overrides the default serialization behavior to handle custom types.

    Parameters:
    -----------
    json.JSONEncoder : class
        Inherits from Python's built-in `json.JSONEncoder`.

    Example:
    --------
    >>> data = {
    ...     "array": np.array([1.2, 3.4]),
    ...     "range": range(10),
    ...     "torch_dtype": torch.float32,
    ...     "lambda_func": lambda x: x**2
    ... }
    >>> with open("output.json", "w") as f:
    ...     json.dump(data, f, cls=JSONsaver)
    >>> # Successfully saves all objects to "output.json".
    """
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, range):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return str(obj)
        elif obj in [torch.float16, torch.float32, torch.bfloat16]:
            return str(obj)
        elif callable(obj):
            if hasattr(obj, '__name__'):
                if obj.__name__ == '<lambda>':
                    return f'CALLABLE.{id(obj)}' # Unique identifier for lambda functions
                else:   
                    return f'CALLABLE.{obj.__name__}'
            else:
                return f'CALLABLE.{str(obj)}'
        else:
            print(f"WARNING: Could not serialize object {obj}")
            return super().default(obj)
        

def read_coords(coords_path):
    """
    The `read_coords` function reads patch coordinates from an HDF5 file, along with any user-defined 
    attributes stored during the patching process. This function is essential for workflows that rely 
    on spatial metadata, such as patch-based analysis in computational pathology.

    Parameters:
    -----------
    coords_path : str
        The path to the HDF5 file containing patch coordinates and attributes.

    Returns:
    --------
    attrs : dict
        A dictionary of user-defined attributes stored during patching.
    coords : np.array
        An array of patch coordinates at level 0.

    Example:
    --------
    >>> attrs, coords = read_coords("patch_coords.h5")
    >>> print(attrs)
    {'patch_size': 256, 'target_mag': 20}
    >>> print(coords)
    [[0, 0], [0, 256], [256, 0], ...]
    """
    with h5py.File(coords_path, 'r') as f:
        attrs = dict(f['coords'].attrs)
        coords = f['coords'][:]
    return attrs, coords


def read_coords_legacy(coords_path):
    """
    The `read_coords_legacy` function reads legacy patch coordinates from an HDF5 file. This function 
    is designed for compatibility with older patching tools such as CLAM or Fishing-Rod, which used 
    a different structure for storing patching metadata.

    Parameters:
    -----------
    coords_path : str
        The path to the HDF5 file containing legacy patch coordinates and metadata.

    Returns:
    --------
    patch_size : int
        The target patch size at the desired magnification.
    patch_level : int
        The patch level used when reading the slide.
    custom_downsample : int
        Any additional downsampling applied to the patches.
    coords : np.array
        An array of patch coordinates.

    Example:
    --------
    >>> patch_size, patch_level, custom_downsample, coords = read_coords_legacy("legacy_coords.h5")
    >>> print(patch_size, patch_level, custom_downsample)
    256, 1, 2
    >>> print(coords)
    [[0, 0], [256, 0], [0, 256], ...]
    """
    with h5py.File(coords_path, 'r') as f:
        patch_size = f['coords'].attrs['patch_size']
        patch_level = f['coords'].attrs['patch_level']
        custom_downsample = f['coords'].attrs.get('custom_downsample', 1)
        coords = f['coords'][:]
    return patch_size, patch_level, custom_downsample, coords


def mask_to_gdf(
    mask: np.ndarray,
    keep_ids: List[int] = [],
    exclude_ids: List[int] = [],
    max_nb_holes: int = 0,
    min_contour_area: float = 1000,
    pixel_size: float = 1,
    contour_scale: float = 1.0
) -> gpd.GeoDataFrame:
    """
    Convert a binary mask into a GeoDataFrame of polygons representing detected regions.

    This function processes a binary mask to identify contours, filter them based on specified parameters,
    and scale them to the desired dimensions. The output is a GeoDataFrame where each row corresponds 
    to a detected region, with polygons representing the tissue contours and their associated holes.

    Args:
        mask (np.ndarray): The binary mask to process, where non-zero regions represent areas of interest.
        keep_ids (List[int], optional): A list of contour indices to keep. Defaults to an empty list (keep all).
        exclude_ids (List[int], optional): A list of contour indices to exclude. Defaults to an empty list.
        max_nb_holes (int, optional): The maximum number of holes to retain for each contour. 
            Use 0 to retain no holes. Defaults to 0.
        min_contour_area (float, optional): Minimum area (in pixels) for a contour to be retained. Defaults to 1000.
        pixel_size (float, optional): Pixel size of level 0. Defaults to 1.
        contour_scale (float, optional): Scaling factor for the output polygons. Defaults to 1.0.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons for the detected regions. The GeoDataFrame
        includes a `tissue_id` column (integer ID for each region) and a `geometry` column (polygons).

    Raises:
        Exception: If no valid contours are detected in the mask.

    Example:
        >>> mask = np.array([[0, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=np.uint8)
        >>> gdf = mask_to_gdf(mask, min_contour_area=500, pixel_size=0.5)
        >>> print(gdf)

    Notes:
        - The function internally downsamples the input mask for efficiency before finding contours.
        - The resulting polygons are scaled back to the original resolution using the `contour_scale` parameter.
        - Holes in contours are also detected and included in the resulting polygons.
    """

    TARGET_EDGE_SIZE = 2000
    scale = TARGET_EDGE_SIZE / mask.shape[0]

    downscaled_mask = cv2.resize(mask, (round(mask.shape[1] * scale), round(mask.shape[0] * scale)))

    # Find and filter contours
    mode = cv2.RETR_TREE if max_nb_holes == 0 else cv2.RETR_CCOMP
    contours, hierarchy = cv2.findContours(downscaled_mask, mode, cv2.CHAIN_APPROX_NONE)

    if hierarchy is None:
        hierarchy = np.array([])
    else:
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

    filter_params = {
        'filter_color_mode': 'none',
        'max_n_holes': max_nb_holes,
        'a_t': min_contour_area * pixel_size ** 2,
        'min_hole_area': 4000 * pixel_size ** 2
    }

    if filter_params: 
        foreground_contours, hole_contours = filter_contours(contours, hierarchy, filter_params, pixel_size)  # Necessary for filtering out artifacts

    if len(foreground_contours) == 0:
        print(f"Warning: No contour were detected. Contour GeoJSON will be empty.")
        return gpd.GeoDataFrame(columns=['tissue_id', 'geometry'])
    else:
        contours_tissue = scale_contours(foreground_contours, contour_scale / scale, is_nested=False)
        contours_holes = scale_contours(hole_contours, contour_scale / scale, is_nested=True)

    if len(keep_ids) > 0:
        contour_ids = set(keep_ids) - set(exclude_ids)
    else:
        contour_ids = set(np.arange(len(contours_tissue))) - set(exclude_ids)

    tissue_ids = [i for i in contour_ids]
    polygons = []
    for i in contour_ids:
        holes = [contours_holes[i][j].squeeze(1) for j in range(len(contours_holes[i]))] if len(contours_holes[i]) > 0 else None
        polygon = Polygon(contours_tissue[i].squeeze(1), holes=holes)
        if not polygon.is_valid:
            if not polygon.is_valid:
                polygon = make_valid(polygon)
        polygons.append(polygon)
    
    gdf_contours = gpd.GeoDataFrame(pd.DataFrame(tissue_ids, columns=['tissue_id']), geometry=polygons)
    
    return gdf_contours


def filter_contours(contours, hierarchy, filter_params, pixel_size):
    """
    The `filter_contours` function processes a list of contours and their hierarchy, filtering 
    them based on specified criteria such as minimum area and hole limits. This function is 
    typically used in digital pathology workflows to isolate meaningful tissue regions.

    Original implementation from: https://github.com/mahmoodlab/CLAM/blob/f1e93945d5f5ac6ed077cb020ed01cf984780a77/wsi_core/WholeSlideImage.py#L97

    Parameters:
    -----------
    contours : list
        A list of contours representing detected regions.
    hierarchy : np.ndarray
        The hierarchy of the contours, used to identify relationships (e.g., parent-child).
    filter_params : dict
        A dictionary containing filtering criteria. Expected keys include:
        - `filter_color_mode`: Mode for filtering based on color (currently unsupported).
        - `max_n_holes`: Maximum number of holes to retain.
        - `a_t`: Minimum area threshold for contours.
        - `min_hole_area`: Minimum area threshold for holes.
    pixel_size : float
        The pixel size at level 0, used to scale areas.

    Returns:
    --------
    tuple:
        A tuple containing:
        - Filtered foreground contours (list)
        - Corresponding hole contours (list)

    Example:
    --------
    >>> filter_params = {
    ...     "filter_color_mode": "none",
    ...     "max_n_holes": 5,
    ...     "a_t": 500,
    ...     "min_hole_area": 100
    ... }
    >>> fg_contours, hole_contours = filter_contours(contours, hierarchy, filter_params, pixel_size=0.5)
    """
    if not hierarchy.size:
        return [], []

    # Find indices of foreground contours (parent == -1)
    foreground_indices = np.flatnonzero(hierarchy[:, 1] == -1)
    filtered_foregrounds = []
    filtered_holes = []

    # Loop through each foreground contour
    for cont_idx in foreground_indices:

        contour = contours[cont_idx]
        hole_indices = np.flatnonzero(hierarchy[:, 1] == cont_idx)

        # Calculate area of the contour (foreground area minus holes)
        contour_area = cv2.contourArea(contour)
        hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in hole_indices]
        net_area = (contour_area - sum(hole_areas)) * (pixel_size ** 2)

        # Skip contours with negligible area
        if net_area <= 0 or net_area <= filter_params['a_t']:
            continue

        # Filter based on color mode if applicable
        if filter_params.get('filter_color_mode') not in [None, 'none']:
            raise Exception("Unsupported filter_color_mode")

        # Append valid contours
        filtered_foregrounds.append(contour)

        # Filter and limit the number of holes
        valid_holes = [
            contours[hole_idx]
            for hole_idx in hole_indices
            if cv2.contourArea(contours[hole_idx]) * (pixel_size ** 2) > filter_params['min_hole_area']
        ]
        valid_holes = sorted(valid_holes, key=cv2.contourArea, reverse=True)[:filter_params['max_n_holes']]
        filtered_holes.append(valid_holes)

    return filtered_foregrounds, filtered_holes


def make_valid(polygon):
    """
    The `make_valid` function attempts to fix invalid polygons by applying small buffer operations. 
    This is particularly useful in cases where geometric operations result in self-intersecting 
    or malformed polygons.

    Parameters:
    -----------
    polygon : shapely.geometry.Polygon
        The input polygon that may be invalid.

    Returns:
    --------
    shapely.geometry.Polygon
        A valid polygon object.

    Raises:
    -------
    Exception:
        If the function fails to create a valid polygon after several attempts.

    Example:
    --------
    >>> invalid_polygon = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])  # Self-intersecting
    >>> valid_polygon = make_valid(invalid_polygon)
    >>> print(valid_polygon.is_valid)
    True
    """
    
    for i in [0, 0.1, -0.1, 0.2]:
        new_polygon = polygon.buffer(i)
        if isinstance(new_polygon, Polygon) and new_polygon.is_valid:
            return new_polygon
    raise Exception("Failed to make a valid polygon")


def scale_contours(contours, scale, is_nested=False):
    """
    The `scale_contours` function scales the dimensions of contours or nested contours (e.g., holes) 
    by a specified factor. This is useful for resizing detected regions in masks or GeoDataFrames.

    Parameters:
    -----------
    contours : list
        A list of contours (or nested lists for holes) to be scaled.
    scale : float
        The scaling factor to apply.
    is_nested : bool, optional
        Indicates whether the input is a nested list of contours (e.g., for holes). Defaults to False.

    Returns:
    --------
    list:
        A list of scaled contours or nested contours.

    Example:
    --------
    >>> contours = [np.array([[0, 0], [1, 1], [1, 0]])]
    >>> scaled_contours = scale_contours(contours, scale=2.0)
    >>> print(scaled_contours)
    [array([[0, 0], [2, 2], [2, 0]])]
    """
    if is_nested:
        return [[np.array(hole * scale, dtype='int32') for hole in holes] for holes in contours]
    return [np.array(cont * scale, dtype='int32') for cont in contours]


def overlay_gdf_on_thumbnail(
    gdf_contours, thumbnail, contours_saveto, scale, tissue_color=(0, 255, 0), hole_color=(255, 0, 0)
):
    """
    The `overlay_gdf_on_thumbnail` function overlays polygons from a GeoDataFrame onto a scaled 
    thumbnail image using OpenCV. This is particularly useful for visualizing tissue regions and 
    their boundaries on smaller representations of whole-slide images.

    Parameters:
    -----------
    gdf_contours : gpd.GeoDataFrame
        A GeoDataFrame containing the polygons to overlay, with a `geometry` column.
    thumbnail : np.ndarray
        The thumbnail image as a NumPy array.
    contours_saveto : str
        The file path to save the annotated thumbnail.
    scale : float
        The scaling factor between the GeoDataFrame coordinates and the thumbnail resolution.
    tissue_color : tuple, optional
        The color (BGR format) for tissue polygons. Defaults to green `(0, 255, 0)`.
    hole_color : tuple, optional
        The color (BGR format) for hole polygons. Defaults to red `(255, 0, 0)`.

    Returns:
    --------
    None
        The function saves the annotated image to the specified file path.

    Example:
    --------
    >>> overlay_gdf_on_thumbnail(
    ...     gdf_contours=gdf, 
    ...     thumbnail=thumbnail_img, 
    ...     contours_saveto="annotated_thumbnail.png", 
    ...     scale=0.5
    ... )
    """

    for poly in gdf_contours.geometry:
        if poly.is_empty:
            continue

        # Draw tissue boundary
        if poly.exterior:
            exterior_coords = (np.array(poly.exterior.coords) * scale).astype(np.int32)
            cv2.polylines(thumbnail, [exterior_coords], isClosed=True, color=tissue_color, thickness=2)

        # Draw holes (if any) in a different color
        if poly.interiors:
            for interior in poly.interiors:
                interior_coords = (np.array(interior.coords) * scale).astype(np.int32)
                cv2.polylines(thumbnail, [interior_coords], isClosed=True, color=hole_color, thickness=2)

    # Crop black borders of the annotated image
    nz = np.nonzero(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY))  # Non-zero pixel locations
    xmin, xmax, ymin, ymax = np.min(nz[1]), np.max(nz[1]), np.min(nz[0]), np.max(nz[0])
    cropped_annotated = thumbnail[ymin:ymax, xmin:xmax]
 
    # Save the annotated image
    os.makedirs(os.path.dirname(contours_saveto), exist_ok=True)
    cropped_annotated = cv2.cvtColor(cropped_annotated, cv2.COLOR_BGR2RGB)
    cv2.imwrite(contours_saveto, cropped_annotated)

# .tools.register_tool(imports=["import numpy as np"])
def get_num_workers(batch_size: int, 
                    factor: float = 0.75, 
                    fallback: int = 16, 
                    max_workers: int | None = None) -> int:
    """
    The `get_num_workers` function calculates the optimal number of workers for a PyTorch DataLoader, 
    balancing system resources and workload. This ensures efficient data loading while avoiding 
    resource overutilization.

    Parameters:
    -----------
    batch_size : int
        The batch size for the DataLoader. This is used to limit the number of workers.
    factor : float, optional
        The fraction of available CPU cores to use. Defaults to 0.75 (75% of available cores).
    fallback : int, optional
        The default number of workers to use if the system's CPU core count cannot be determined. Defaults to 16.
    max_workers : int or None, optional
        The maximum number of workers allowed. Defaults to `2 * batch_size` if not provided.

    Returns:
    --------
    int
        The calculated number of workers for the DataLoader.

    Example:
    --------
    >>> num_workers = get_num_workers(batch_size=64, factor=0.5)
    >>> print(num_workers)
    8

    Notes:
    ------
    - The number of workers is clipped to a minimum of 1 to ensure multiprocessing is not disabled.
    - The maximum number of workers defaults to `2 * batch_size` unless explicitly specified.
    - The function ensures compatibility with systems where `os.cpu_count()` may return `None`.
    - On Windows systems, the number of workers is always set to 0 to ensure compatibility with PyTorch datasets whose attributes may not be serializable.
    """

    # Disable pytorch multiprocessing on Windows
    if os.name == 'nt':
        return 0
    
    num_cores = os.cpu_count() or fallback
    num_workers = int(factor * num_cores)  # Use a fraction of available cores
    max_workers = max_workers or (2 * batch_size)  # Optional cap
    num_workers = np.clip(num_workers, 1, max_workers)
    return int(num_workers)
