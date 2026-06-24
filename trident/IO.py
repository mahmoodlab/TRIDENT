from __future__ import annotations

import torch
import socket
import os
import time
import json
from typing import List, Optional, Union, Tuple
import h5py
import numpy as np
import cv2
import pandas as pd
from geopandas import gpd
from shapely import Polygon


COMPOUND_EXTENSIONS = {'.ome.tif', '.ome.tiff', '.ome.zarr'}


def splitext(path: str) -> tuple:
    """Like os.path.splitext but handles compound extensions (e.g., .ome.tif)."""
    path_lower = path.lower()
    for ext in COMPOUND_EXTENSIONS:
        if path_lower.endswith(ext):
            return path[:-len(ext)], path[-len(ext):]
    return os.path.splitext(path)


ENV_TRIDENT_HOME = "TRIDENT_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"
_cache_dir: Optional[str] = None


def collect_valid_slides(
    wsi_dir: str,
    custom_list_path: Optional[str] = None,
    wsi_ext: Optional[List[str]] = None,
    search_nested: bool = False,
    max_workers: int = 8,
    return_relative_paths: bool = False
) -> Union[List[str], Tuple[List[str], List[str]]]:
    """
    Retrieve all valid WSI file paths from a directory, optionally filtered by a custom list.

    Parameters:
        wsi_dir (str):
            Path to the directory containing WSIs.
        custom_list_path (Optional[str]):
            Path to a CSV file with 'wsi' column of relative slide paths.
        wsi_ext (Optional[List[str]]):
            Allowed file extensions.
        search_nested (bool):
            Whether to search subdirectories.
        max_workers (int):
            Threads to use when checking file existence.
        return_relative_paths (bool):
            Whether to also return relative paths.

    Returns:
        Union[List[str], Tuple[List[str], List[str]]]:
            Full paths to valid WSIs, or (full paths, relative paths) if return_relative_paths is True.

    Raises:
        ValueError:
            If custom CSV is invalid or files not found.
    """
    valid_rel_paths: List[str] = []

    if custom_list_path is not None:
        from concurrent.futures import ThreadPoolExecutor

        wsi_df = pd.read_csv(custom_list_path)
        if 'wsi' not in wsi_df.columns:
            raise ValueError("CSV must contain a column named 'wsi'.")

        rel_paths = wsi_df['wsi'].dropna().astype(str).tolist()
        if not rel_paths:
            raise ValueError(f"No valid slides found in the custom list at {custom_list_path}.")

        def exists_fn(rel_path: str) -> bool:
            return os.path.exists(os.path.join(wsi_dir, rel_path))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(exists_fn, rel_paths))

        for rel_path, exists in zip(rel_paths, results):
            if not exists:
                raise ValueError(
                    f"Slide '{rel_path}' not found in '{wsi_dir}'. "
                    "If the folder is nested, ensure 'wsi' column contains relative paths."
                )

        valid_rel_paths = rel_paths

    else:
        if wsi_ext is None:
            from trident.Converter import PIL_EXTENSIONS, OPENSLIDE_EXTENSIONS
            wsi_ext = list(PIL_EXTENSIONS) + list(OPENSLIDE_EXTENSIONS)

        wsi_ext = [ext.lower() for ext in wsi_ext]

        def matches_ext(filename: str) -> bool:
            return any(filename.lower().endswith(ext) for ext in wsi_ext)

        if search_nested:
            for root, _, files in os.walk(wsi_dir):
                for f in files:
                    if matches_ext(f):
                        rel_path = os.path.relpath(os.path.join(root, f), wsi_dir)
                        valid_rel_paths.append(rel_path)
        else:
            valid_rel_paths = [
                f for f in os.listdir(wsi_dir)
                if matches_ext(f)
            ]

        valid_rel_paths.sort()

    full_paths = [os.path.join(wsi_dir, rel) for rel in valid_rel_paths]

    return (full_paths, valid_rel_paths) if return_relative_paths else full_paths


def get_dir() -> str:
    r"""
    Get Trident cache directory used for storing downloaded models & weights.
    If :func:`~trident.hub.set_dir` is not called, default path is ``$TRIDENT_HOME`` where
    environment variable ``$TRIDENT_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesystem layout, with a default value ``~/.cache`` if the environment
    variable is not set.
    """

    if _cache_dir is not None:
        return _cache_dir
    return _get_trident_home()


def set_dir(d: Union[str, os.PathLike]) -> None:
    r"""
    Optionally set the Trident cache directory used to save downloaded models & weights.
    
    Parameters:
        d (Union[str, os.PathLike]):
            Path to a local folder to save downloaded models & weights.
    """
    global _cache_dir
    _cache_dir = os.path.expanduser(d)


def _get_trident_home() -> str:
    trident_home = os.path.expanduser(
        os.getenv(
            ENV_TRIDENT_HOME,
            os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "trident"),
        )
    )
    return trident_home


def has_internet_connection(timeout: float = 3.0) -> bool:
    endpoint = os.environ.get("HF_ENDPOINT", "huggingface.co")
    
    if endpoint.startswith(("http://", "https://")):
        from urllib.parse import urlparse
        endpoint = urlparse(endpoint).netloc
    
    try:
        # Fast socket-level check
        with socket.create_connection((endpoint, 443), timeout=timeout):
            return True
    except OSError:
        pass

    try:
        # Fallback HTTP-level check (if requests is available)
        import requests
        url = f"https://{endpoint}" if not endpoint.startswith(("http://", "https://")) else endpoint
        r = requests.head(url, timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False


def  get_weights_path(model_type: str, encoder_name: str) -> str:
    """
    Retrieve the path to the weights file for a given model name.
    This function looks up the path to the weights file in a local checkpoint
    registry (local_ckpts.json). If the path in the registry is absolute, it
    returns that path. If the path is relative, it joins the relative path with
    the provided weights_root directory.
    
    Parameters:
        model_type (str):
            The type of model ('patch', 'slide', or 'seg').
        encoder_name (str):
            The name of the model whose weights path is to be retrieved.

    Returns:
        str: The absolute path to the weights file.
    """

    assert model_type in ['patch', 'slide', 'seg'], f"Encoder type must be 'patch' or 'slide' or 'seg', not '{model_type}'"

    if model_type == 'patch' or model_type == 'slide':
        root = os.path.join(os.path.dirname(__file__), f"{model_type}_encoder_models")
    else:
        root = os.path.join(os.path.dirname(__file__), "segmentation_models")

    registry_path = os.path.join(root, "local_ckpts.json")
    with open(registry_path, "r") as f:
        registry = json.load(f)

    path = registry.get(encoder_name)    
    if path:
        path = path if os.path.isabs(path) else os.path.abspath(os.path.join(root, 'model_zoo', path)) # Make path absolute
        if not os.path.exists(path):
            path = ""

    return path


def create_lock(path: str, suffix: Optional[str] = None) -> None:
    """
    Create a lock file to signal that a particular file or process 
    is currently being worked on. This is especially useful in multiprocessing or distributed 
    systems to avoid conflicts or multiple processes working on the same resource.

    Parameters:
        path (str):
            The path to the file or resource being locked.
        suffix (str, optional):
            An additional suffix to append to the lock file name. This allows for creating distinct
            lock files for similar resources. Defaults to None.

    Example
    -------
    >>> create_lock("/path/to/resource")
    >>> # Creates a file named "/path/to/resource.lock" to indicate the resource is locked.
    """
    if suffix is not None:
        path = f"{path}_{suffix}"
    lock_file = f"{path}.lock"
    with open(lock_file, 'w') as f:
        # Write metadata to allow safe dead-lock cleanup.
        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_at": time.time(),
        }
        f.write(json.dumps(payload))

#####################

def remove_lock(path: str, suffix: Optional[str] = None) -> None:
    """
    Remove a lock file, signaling that the file or process 
    is no longer in use and is available for other operations.

    Parameters:
        path (str):
            The path to the file or resource whose lock needs to be removed.
        suffix (str, optional):
            An additional suffix to identify the lock file. Defaults to None.

    Example
    -------
    >>> remove_lock("/path/to/resource")
    >>> # Removes the file "/path/to/resource.lock", indicating the resource is unlocked.
    """
    if suffix is not None:
        path = f"{path}_{suffix}"
    lock_file = f"{path}.lock"
    # Locks are best-effort markers; be tolerant if it is already gone.
    try:
        if os.path.exists(lock_file):
            os.remove(lock_file)
    except Exception:
        pass

#####################

def is_locked(path: str, suffix: Optional[str] = None) -> bool:
    """
    Check if a resource is currently locked by verifying 
    the existence of a `.lock` file.

    Parameters:
        path (str):
            The path to the file or resource to check for a lock.
        suffix (str, optional):
            An additional suffix to identify the lock file. Defaults to None.

    Returns:
        bool: True if the `.lock` file exists (resource is locked), False otherwise.

    Example
    -------
    >>> is_locked("/path/to/resource")
    False
    >>> create_lock("/path/to/resource")
    >>> is_locked("/path/to/resource")
    True
    """
    if suffix is not None:
        path = f"{path}_{suffix}"
    return os.path.exists(f"{path}.lock")


def _pid_is_running(pid: int) -> bool:
    """
    Check whether a PID is alive on this host.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # PID exists but we may not have permission; assume it's running.
        return True
    except Exception:
        return False
    return True


def clear_dead_locks(root_dir: str, *, legacy_max_age_seconds: float = 24 * 3600) -> dict:
    """
    Remove stale `.lock` files under `root_dir`.

    Rules (conservative):
    - If `<output>` exists next to `<output>.lock`, the lock is stale → remove it.
    - If the lock file contains JSON with a PID and hostname matching this host:
        - remove only if PID is not running.
    - Otherwise (legacy empty lock / unreadable / other-host):
        - remove only if older than `legacy_max_age_seconds`.

    Returns stats: {"scanned": int, "removed": int, "kept": int}.
    """
    scanned = removed = kept = 0
    now = time.time()
    host = socket.gethostname()

    if not os.path.isdir(root_dir):
        return {"scanned": 0, "removed": 0, "kept": 0}

    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            if not fn.endswith(".lock"):
                continue
            lock_fp = os.path.join(dirpath, fn)
            scanned += 1

            target_fp = lock_fp[:-5]  # strip ".lock"
            try:
                if os.path.exists(target_fp):
                    os.remove(lock_fp)
                    removed += 1
                    continue

                # Try to parse metadata.
                pid = None
                lock_host = None
                created_at = None
                try:
                    with open(lock_fp, "r") as f:
                        raw = (f.read() or "").strip()
                    if raw:
                        data = json.loads(raw)
                        pid = data.get("pid")
                        lock_host = data.get("hostname")
                        created_at = data.get("created_at")
                except Exception:
                    pid = None
                    lock_host = None
                    created_at = None

                if pid is not None and lock_host == host:
                    try:
                        pid_int = int(pid)
                    except Exception:
                        pid_int = None
                    if pid_int is not None and not _pid_is_running(pid_int):
                        os.remove(lock_fp)
                        removed += 1
                        continue
                    kept += 1
                    continue

                # Legacy/unknown-host: age-based cleanup.
                try:
                    mtime = os.path.getmtime(lock_fp)
                except Exception:
                    mtime = None

                age_ref = None
                if created_at is not None:
                    try:
                        age_ref = float(created_at)
                    except Exception:
                        age_ref = None
                if age_ref is None and mtime is not None:
                    age_ref = float(mtime)

                if age_ref is not None and (now - age_ref) >= legacy_max_age_seconds:
                    os.remove(lock_fp)
                    removed += 1
                else:
                    kept += 1
            except Exception:
                kept += 1

    return {"scanned": scanned, "removed": removed, "kept": kept}


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
            print(f"[WARNING] Could not serialize object {obj}")
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


def coords_to_h5(
    coords: List[List[int]], 
    save_path, 
    patch_size, 
    src_mag, 
    target_mag, 
    save_coords, 
    width, 
    height, 
    name,
    overlap
):
    """ Save tissue coordinates to .h5 """
    coords_array = np.asarray(coords, dtype=np.int64)
    if coords_array.size == 0:
        coords_array = coords_array.reshape(0, 2)
    elif coords_array.ndim == 1 and coords_array.shape[0] == 2:
        coords_array = coords_array.reshape(1, 2)
    elif coords_array.ndim != 2 or coords_array.shape[1] != 2:
        raise ValueError(
            f"coords must have shape (N, 2). Got shape {coords_array.shape}."
        )

    # Prepare assets for saving
    assets = {'coords' : coords_array}
    attributes = {
        'patch_size': patch_size, # Reference frame: patch_level
        'patch_size_level0': patch_size * src_mag // target_mag, # Reference frame: level0
        'level0_magnification': src_mag,
        'target_magnification': target_mag,
        'overlap': overlap,
        'name': name,
        'savetodir': save_coords,
        'level0_width': width,
        'level0_height': height
    }

    # Save the assets and attributes to an hdf5 file
    save_h5(save_path,
            assets = assets,
            attributes = {'coords': attributes},
            mode='w')


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
        print(f"[Warning] No contour were detected. Contour GeoJSON will be empty.")
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


def mask_to_instances(
    class_map: np.ndarray,
    class_names: Optional[List[str]] = None,
    min_contour_area: float = 16.0,
) -> List[dict]:
    """
    Convert a single patch's dense class-index map into a list of instances, in *patch*
    pixel coordinates. Used as the default (semantic) `predict_patches` implementation:
    each connected component of every non-background class becomes one instance.

    Args:
        class_map (np.ndarray): `(H, W)` integer class indices for one patch. 0 = background.
        class_names (List[str], optional): Names per class index, for the `class_name` field.
        min_contour_area (float, optional): Minimum contour area (patch px) to keep. Defaults to 16.

    Returns:
        List[dict]: instances with keys `contour` (K,2 float patch-px), `class_id` (int),
            `class_name` (str|None), `confidence` (float, 1.0 for semantic), `centroid` (2,).
    """
    instances: List[dict] = []
    for class_id in np.unique(class_map):
        if class_id == 0:  # background
            continue
        binary = (class_map == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_contour_area:
                continue
            pts = contour.squeeze(1).astype(np.float64)
            if pts.ndim != 2 or len(pts) < 3:
                continue
            name = None
            if class_names is not None and 0 <= int(class_id) < len(class_names):
                name = class_names[int(class_id)]
            instances.append({
                "contour": pts,
                "class_id": int(class_id),
                "class_name": name,
                "confidence": 1.0,
                "centroid": pts.mean(axis=0),
            })
    return instances


def save_cell_segmentation_h5(save_path: str, instances: List[dict], attributes: dict) -> str:
    """
    Save slide-level cell/object instances to a compact HDF5 file.

    Polygons have variable length, so they are stored ragged: a single flat `(M, 2)`
    `contours` array plus a `contour_offsets` index of length `N+1` (instance `i` spans
    `contours[offsets[i]:offsets[i+1]]`). This mirrors the patch-feature `.h5` convention
    (datasets + attributes via `save_h5`) and is far smaller than dense per-patch masks.

    Args:
        save_path (str): Destination `.h5` path.
        instances (List[dict]): Records with level-0 `contour` (K,2), `class_id`,
            `confidence`, `centroid` (2,).
        attributes (dict): Metadata stored on the `cells` group (e.g. model name,
            class_mapping JSON, source coords attrs).

    Returns:
        str: `save_path`.
    """
    n = len(instances)
    if n:
        contours = [np.asarray(inst["contour"], dtype=np.float32).reshape(-1, 2) for inst in instances]
        lengths = np.array([len(c) for c in contours], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        flat_contours = np.concatenate(contours, axis=0).astype(np.float32)
        class_ids = np.array([inst["class_id"] for inst in instances], dtype=np.int32)
        confidences = np.array([inst["confidence"] for inst in instances], dtype=np.float32)
        centroids = np.stack([np.asarray(inst["centroid"], dtype=np.float32) for inst in instances])
    else:
        flat_contours = np.empty((0, 2), dtype=np.float32)
        offsets = np.zeros((1,), dtype=np.int64)
        class_ids = np.empty((0,), dtype=np.int32)
        confidences = np.empty((0,), dtype=np.float32)
        centroids = np.empty((0, 2), dtype=np.float32)

    with h5py.File(save_path, 'w') as f:
        grp = f.create_group('cells')
        grp.create_dataset('contours', data=flat_contours)
        grp.create_dataset('contour_offsets', data=offsets)
        grp.create_dataset('class_ids', data=class_ids)
        grp.create_dataset('confidences', data=confidences)
        grp.create_dataset('centroids', data=centroids)
        for key, val in (attributes or {}).items():
            grp.attrs[key] = val
    return save_path


# Distinct, reproducible colors per class id (used as cv2 BGR tuples on the BGR canvas).
CELL_VIZ_PALETTE = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163), (255, 127, 0),
    (255, 215, 0), (166, 86, 40), (247, 129, 191), (153, 153, 153), (26, 188, 156),
    (52, 152, 219), (155, 89, 182), (241, 196, 15), (231, 76, 60), (149, 165, 166),
]


def cell_class_color(class_id: int) -> tuple:
    """Color (cv2 BGR-on-BGR-canvas) for a class id; stable across the overview and patches."""
    return CELL_VIZ_PALETTE[int(class_id) % len(CELL_VIZ_PALETTE)]


def _draw_cell_legend(canvas: np.ndarray, entries: List[tuple]) -> np.ndarray:
    """
    Draw a color->cell-type legend in the top-left corner of ``canvas`` (BGR, in place).

    Args:
        canvas (np.ndarray): BGR image to draw on.
        entries (List[tuple]): ordered ``(label, color)`` pairs (color same as the contours).
    """
    if not entries:
        return canvas
    h, w = canvas.shape[:2]
    fs = max(0.45, min(1.2, w / 1600.0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    tth = max(1, int(round(fs * 1.5)))
    sw = int(round(22 * fs))          # swatch side
    gap = int(round(8 * fs)) + 2
    row_h = sw + gap
    text_w = max(cv2.getTextSize(lbl, font, fs, tth)[0][0] for lbl, _ in entries)
    panel_w = min(sw + 3 * gap + text_w, w - 2 * gap)
    panel_h = row_h * len(entries) + gap
    x0, y0 = gap, gap
    cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 255, 255), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), max(1, tth // 2))
    y = y0 + gap
    for label, color in entries:
        cv2.rectangle(canvas, (x0 + gap, y), (x0 + gap + sw, y + sw), color, -1)
        cv2.rectangle(canvas, (x0 + gap, y), (x0 + gap + sw, y + sw), (0, 0, 0), 1)
        cv2.putText(canvas, label, (x0 + 2 * gap + sw, y + sw - int(round(5 * fs))),
                    font, fs, (0, 0, 0), tth, cv2.LINE_AA)
        y += row_h
    return canvas


def overlay_instances_on_thumbnail(
    gdf: "gpd.GeoDataFrame",
    thumbnail: np.ndarray,
    saveto: str,
    scale: float,
) -> str:
    """
    Draw instance polygons (level-0 coords) onto a slide thumbnail for debugging and save
    it as a JPEG. Colors are assigned per class id, with a color->cell-type legend drawn in
    the corner. Mirrors `overlay_gdf_on_thumbnail` used by tissue segmentation.

    Args:
        gdf (gpd.GeoDataFrame): Instances with `class`/`class_name` columns and polygon
            `geometry` (level-0).
        thumbnail (np.ndarray): RGB thumbnail to draw on.
        saveto (str): Output `.jpg` path.
        scale (float): thumbnail-pixels-per-level0-pixel (i.e. thumb_width / wsi_width).

    Returns:
        str: `saveto`.
    """
    canvas = np.ascontiguousarray(thumbnail[..., ::-1])  # RGB -> BGR for cv2
    present = {}  # class_id -> class_name, for the legend
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        class_id = int(row.get('class', 0))
        present.setdefault(class_id, row.get('class_name'))
        color = cell_class_color(class_id)
        polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
        for poly in polys:
            ext = (np.asarray(poly.exterior.coords) * scale).astype(np.int32)
            cv2.polylines(canvas, [ext], isClosed=True, color=color, thickness=1)

    entries = [
        (str(present[cid]) if present[cid] is not None else f"class {cid}", cell_class_color(cid))
        for cid in sorted(present)
    ]
    _draw_cell_legend(canvas, entries)

    os.makedirs(os.path.dirname(saveto), exist_ok=True)
    cv2.imwrite(saveto, canvas)
    return saveto


def draw_instances_on_tile(
    tile_rgb: np.ndarray,
    instances_px: List[dict],
    class_names: Optional[List[str]],
    saveto: str,
    thickness: int = 2,
) -> str:
    """
    Draw per-cell instance contours (in *patch* pixel coords) on a full-resolution tile and
    save as JPEG. This is the readable debug artifact for cell segmentation: at this zoom the
    individual cells and their class colors are clearly visible. Color is keyed by class id.

    Args:
        tile_rgb (np.ndarray): `(H, W, 3)` RGB patch image.
        instances_px (List[dict]): instances with `contour` (K,2 patch-px) and `class_id`.
        class_names (List[str], optional): unused for drawing; kept for parity/legend hooks.
        saveto (str): output `.jpg` path.
        thickness (int, optional): contour line thickness. Defaults to 2.

    Returns:
        str: `saveto`.
    """
    canvas = np.ascontiguousarray(tile_rgb[..., ::-1])  # RGB -> BGR for cv2
    present = set()
    for inst in instances_px:
        contour = np.asarray(inst['contour'], dtype=np.int32)
        if contour.ndim != 2 or contour.shape[0] < 3:
            continue
        class_id = int(inst['class_id'])
        present.add(class_id)
        cv2.polylines(canvas, [contour], isClosed=True, color=cell_class_color(class_id),
                      thickness=thickness)

    def _name(cid):
        if class_names is not None and 0 <= cid < len(class_names):
            return str(class_names[cid])
        return f"class {cid}"
    entries = [(_name(cid), cell_class_color(cid)) for cid in sorted(present)]
    _draw_cell_legend(canvas, entries)

    os.makedirs(os.path.dirname(saveto), exist_ok=True)
    cv2.imwrite(saveto, canvas)
    return saveto


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

    if max_workers is not None and max_workers <= 0:
        return 0
    
    num_cores = os.cpu_count() or fallback
    num_workers = int(factor * num_cores)  # Use a fraction of available cores
    max_workers = max_workers or (2 * batch_size)  # Optional cap
    num_workers = np.clip(num_workers, 1, max_workers)
    return int(num_workers)
