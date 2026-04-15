"""
Example usage:

```
python run_batch_of_hancock_tmas.py --task all --job_dir output --patch_encoder uni_v1 --mag 20 --patch_size 256
```

This script processes TMA slides for case_ids containing any of the following numbers as a substring:
104,116,120,121,162,176,191,225,250,296,334,342,346,357,403,407,476,530,559,564,583,606,632,664,668,698,706,723,740,741

For each case_id and TMA marker, it:
1. Loads the binary mask from /data3/hancock/TMA_Visualizations/{MARKER}/TumorCenter_{MARKER}_block{N}_case{case_id:03d}_mask.png
2. Loads the corresponding SVS from /data3/hancock/TMA_TumorCenter/{MARKER}/TumorCenter_{MARKER}_block{N}.svs
3. Applies the binary mask (upscaled to WSI level 0) before running segmentation
4. Performs standard Trident processing (segmentation, coords, features)

The binary mask is downsampled from the original image, so it needs to be upscaled to match WSI level 0 resolution.
"""
import os
import argparse
import torch
os.environ["OPJ_QUIET"] = "true"  # Suppresses OpenJPEG (dicom/svs) warnings
from typing import Dict, List
from collections import defaultdict
import re
import glob
import tempfile
import pandas as pd
import numpy as np
from PIL import Image
import geopandas as gpd
from trident import Processor, load_wsi
from trident.patch_encoder_models import encoder_registry as patch_encoder_registry
from trident.slide_encoder_models import encoder_registry as slide_encoder_registry
from trident.IO import mask_to_gdf
os.environ['VIPS_WARNING'] = '0' 
os.environ['OPJ_QUIET'] = 'true'

import warnings
warnings.filterwarnings("ignore")


# Hancock case numbers to filter
HANCOCK_NUMBERS = [
    '104', '116', '120', '121', '162', '176', '191', '225', '250', '296',
    '334', '342', '346', '357', '403', '407', '476', '530', '559', '564',
    '583', '606', '632', '664', '668', '698', '706', '723', '740', '741'
]

# Base directories for TMA data
TMA_MASKS_DIR = '/data3/hancock/TMA_Visualizations'
TMA_SLIDES_DIR = '/data3/hancock/TMA_TumorCenter'


def build_parser() -> argparse.ArgumentParser:
    """
    Parse command-line arguments for the Trident TMA processing script.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all Trident processing options.
    """
    parser = argparse.ArgumentParser(description='Run Trident on Hancock TMA slides only')

    # Generic arguments 
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use for processing tasks.')
    parser.add_argument('--task', type=str, default='seg', 
                        choices=['seg', 'coords', 'feat', 'all'], 
                        help='Task to run: seg (segmentation), coords (save tissue coordinates), feat (extract features).')
    parser.add_argument('--job_dir', type=str, required=True, help='Directory to store outputs.')
    parser.add_argument('--skip_errors', action='store_true', default=False, 
                        help='Skip errored slides and continue processing.')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of workers. Set to 0 to use main process.')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help="Batch size used for segmentation and feature extraction. Will be override by"
                        "`seg_batch_size` and `feat_batch_size` if you want to use different ones. Defaults to 64.")

    # Caching argument for fast WSI processing
    parser.add_argument(
        '--wsi_cache', type=str, default=None,
        help='Path to a local cache (e.g., SSD) used to speed up access to WSIs stored on slower drives (e.g., HDD).'
    )
    parser.add_argument(
        '--cache_batch_size', type=int, default=32,
        help='Maximum number of slides to cache locally at once. Helps control disk usage.'
    )

    # Slide-related arguments
    parser.add_argument(
        '--default_mpp',
        type=float,
        default=None,
        help='Fallback microns-per-pixel to use when a slide lacks metadata.'
    )
    parser.add_argument('--custom_mpp_keys', type=str, nargs='+', default=None,
                    help='Custom keys used to store the resolution as MPP (micron per pixel) in your list of whole-slide image.')
    parser.add_argument('--reader_type', type=str, choices=['openslide', 'image', 'cucim', 'sdpc'], default=None,
                    help='Force the use of a specific WSI image reader. Options are ["openslide", "image", "cucim", "sdpc"]. Defaults to None (auto-determine which reader to use).')
    
    # Segmentation arguments 
    parser.add_argument('--segmenter', type=str, default='hest', 
                        choices=['hest', 'grandqc'], 
                        help='Type of tissue vs background segmenter. Options are HEST or GrandQC.')
    parser.add_argument('--seg_conf_thresh', type=float, default=0.5, 
                    help='Confidence threshold to apply to binarize segmentation predictions. Lower this threhsold to retain more tissue. Defaults to 0.5. Try 0.4 as 2nd option.')
    parser.add_argument('--remove_holes', action='store_true', default=False, 
                        help='Do you want to remove holes?')
    parser.add_argument('--remove_artifacts', action='store_true', default=False, 
                        help='Do you want to run an additional model to remove artifacts (including penmarks, blurs, stains, etc.)?')
    parser.add_argument('--remove_penmarks', action='store_true', default=False, 
                        help='Do you want to run an additional model to remove penmarks?')
    parser.add_argument('--seg_batch_size', type=int, default=None, 
                        help='Batch size for segmentation. Defaults to None (use `batch_size` argument instead).')
    
    # Patching arguments
    parser.add_argument('--mag', type=int, choices=[5, 10, 20, 40, 80], default=20, 
                        help='Magnification for coords/features extraction.')
    parser.add_argument('--patch_size', type=int, default=512, 
                        help='Patch size for coords/image extraction.')
    parser.add_argument('--overlap', type=int, default=0, 
                        help='Absolute overlap for patching in pixels. Defaults to 0.')
    parser.add_argument('--min_tissue_proportion', type=float, default=0., 
                        help='Minimum proportion of the patch under tissue to be kept. Between 0. and 1.0. Defaults to 0.')
    parser.add_argument('--coords_dir', type=str, default=None, 
                        help='Directory to save/restore tissue coordinates.')
    
    # Feature extraction arguments 
    parser.add_argument('--patch_encoder', type=str, default='conch_v15', 
                        choices=patch_encoder_registry.keys(),
                        help='Patch encoder to use')
    parser.add_argument(
        '--patch_encoder_ckpt_path', type=str, default=None,
        help=(
            "Optional local path to a patch encoder checkpoint (.pt, .pth, .bin, or .safetensors). "
            "This is only needed in offline environments (e.g., compute clusters without internet). "
            "If not provided, models are downloaded automatically from Hugging Face. "
            "You can also specify local paths via the model registry at "
            "`./trident/patch_encoder_models/local_ckpts.json`."
        )
    )
    parser.add_argument('--slide_encoder', type=str, default=None, 
                        choices=slide_encoder_registry.keys(), 
                        help='Slide encoder to use')
    parser.add_argument('--feat_batch_size', type=int, default=None, 
                        help='Batch size for feature extraction. Defaults to None (use `batch_size` argument instead).')
    return parser


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments and return the parsed namespace.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    return build_parser().parse_args()


def find_tma_masks_for_hancock_cases() -> List[Dict[str, str]]:
    """
    Find all TMA mask files matching Hancock case IDs.

    Returns
    -------
    List[Dict[str, str]]
        List of dictionaries containing mask_path, marker, block, case_id, and slide_path.
    """
    tma_cases = []
    
    if not os.path.exists(TMA_MASKS_DIR):
        print(f"[WARNING] TMA masks directory not found: {TMA_MASKS_DIR}")
        return tma_cases
    
    # Iterate through marker directories (e.g., CD3, PDL1, etc.)
    for marker_dir in os.listdir(TMA_MASKS_DIR):
        marker_path = os.path.join(TMA_MASKS_DIR, marker_dir)
        if not os.path.isdir(marker_path):
            continue
        
        # Find all mask files matching the pattern: TumorCenter_{marker}_block{N}_case{case_id:03d}_mask.png
        mask_pattern = os.path.join(marker_path, f'TumorCenter_{marker_dir}_block*_case*_mask.png')
        mask_files = glob.glob(mask_pattern)
        
        for mask_path in mask_files:
            # Extract case_id from filename
            basename = os.path.basename(mask_path)
            match = re.search(r'case(\d+)', basename)
            if not match:
                continue
            
            case_id_str = match.group(1)
            # Check if case_id contains any Hancock number
            if any(num in case_id_str for num in HANCOCK_NUMBERS):
                # Extract block number
                block_match = re.search(r'block(\d+)', basename)
                if not block_match:
                    continue
                block_num = block_match.group(1)
                
                # Construct slide path
                slide_path = os.path.join(TMA_SLIDES_DIR, marker_dir, f'TumorCenter_{marker_dir}_block{block_num}.svs')
                
                if not os.path.exists(slide_path):
                    print(f"[WARNING] Slide not found for mask {mask_path}: {slide_path}")
                    continue
                
                tma_cases.append({
                    'mask_path': mask_path,
                    'marker': marker_dir,
                    'block': block_num,
                    'case_id': case_id_str,
                    'slide_path': slide_path
                })
    
    return sorted(tma_cases, key=lambda x: (x['marker'], x['block'], x['case_id']))


def load_binary_mask(mask_path: str) -> np.ndarray:
    """
    Load a binary mask PNG file.

    Parameters
    ----------
    mask_path : str
        Path to the binary mask PNG file.

    Returns
    -------
    np.ndarray
        Binary mask as numpy array (uint8, 0=background, 255=foreground).
    """
    mask_img = Image.open(mask_path).convert('L')
    mask_array = np.array(mask_img, dtype=np.uint8)
    # Ensure binary: 0 or 255
    mask_array = (mask_array > 127).astype(np.uint8) * 255
    return mask_array


def convert_thumbnail_mask_to_geojson(
    mask: np.ndarray,
    mask_width: int,
    mask_height: int,
    wsi_width: int,
    wsi_height: int,
    wsi_mpp: float,
    output_path: str
) -> str:
    """
    Convert a thumbnail-resolution binary mask to GeoJSON at WSI level 0 coordinates.
    
    This function efficiently converts the mask by:
    1. Converting the thumbnail mask directly to GeoJSON coordinates
    2. Scaling the coordinates up to WSI level 0 resolution
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask at thumbnail resolution.
    mask_width : int
        Width of the mask at thumbnail resolution.
    mask_height : int
        Height of the mask at thumbnail resolution.
    wsi_width : int
        Width of WSI at level 0.
    wsi_height : int
        Height of WSI at level 0.
    wsi_mpp : float
        Microns per pixel of the WSI.
    output_path : str
        Path to save the GeoJSON file.

    Returns
    -------
    str
        Path to the saved GeoJSON file.
    """
    # Calculate scale factors
    scale_x = wsi_width / mask_width
    scale_y = wsi_height / mask_height
    
    # Convert thumbnail mask to GeoJSON at thumbnail resolution
    # Use a small pixel_size since we're working at thumbnail resolution
    # The actual MPP will be corrected when we scale the coordinates
    thumbnail_mpp = wsi_mpp * (mask_width / wsi_width)  # Approximate MPP at thumbnail
    
    # Convert mask to GeoDataFrame at thumbnail resolution
    # contour_scale=1.0 means coordinates are at thumbnail resolution
    gdf = mask_to_gdf(
        mask=mask,
        max_nb_holes=0,
        min_contour_area=100,  # Smaller threshold for thumbnail resolution
        pixel_size=thumbnail_mpp,
        contour_scale=1.0
    )
    
    if gdf.empty:
        print("    [WARNING] No contours detected in mask")
        # Create empty GeoJSON
        gdf = gpd.GeoDataFrame(columns=['tissue_id', 'geometry'])
    else:
        # Scale coordinates from thumbnail to WSI level 0
        # Use shapely's affine transformation to scale X and Y independently
        from shapely.affinity import scale as shapely_scale
        
        scaled_geoms = []
        for geom in gdf.geometry:
            # Scale geometry: xfact=scale_x, yfact=scale_y, origin=(0,0)
            scaled_geom = shapely_scale(geom, xfact=scale_x, yfact=scale_y, origin=(0, 0))
            scaled_geoms.append(scaled_geom)
        
        gdf = gpd.GeoDataFrame(gdf[['tissue_id']], geometry=scaled_geoms)
    
    # Save GeoJSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf.to_file(output_path, driver='GeoJSON')
    
    return output_path


def rename_contours_folders(job_dir: str) -> None:
    """
    Recursively rename all contours_geojson folders to contours in the job directory.
    
    Parameters
    ----------
    job_dir : str
        Root job directory to search for contours_geojson folders.
    """
    return
    renamed_count = 0
    for root, dirs, files in os.walk(job_dir):
        if 'contours_geojson' in dirs:
            old_path = os.path.join(root, 'contours_geojson')
            new_path = os.path.join(root, 'contours')
            
            if os.path.exists(new_path):
                # Both exist, merge contents
                import shutil
                for item in os.listdir(old_path):
                    src = os.path.join(old_path, item)
                    dst = os.path.join(new_path, item)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                print(f"  Merged {old_path} into {new_path}")
            else:
                # Rename the folder
                import shutil
                shutil.move(old_path, new_path)
                print(f"  Renamed {old_path} to {new_path}")
            renamed_count += 1
    
    if renamed_count > 0:
        print(f"[MAIN] Renamed {renamed_count} contours_geojson folder(s) to contours")
    else:
        print("[MAIN] No contours_geojson folders found to rename")


def sync_contours_directories(job_dir: str) -> None:
    """
    Sync files from contours_geojson to contours directory.
    This ensures files saved by segment_tissue (which writes to contours_geojson) 
    are also available in contours.
    
    Parameters
    ----------
    job_dir : str
        Job directory containing contours and contours_geojson folders.
    """
    old_contours_dir = os.path.join(job_dir, 'contours_geojson')
    contours_dir = os.path.join(job_dir, 'contours')
    
    # If contours_geojson exists as a directory (not symlink) and contours exists
    if (os.path.exists(old_contours_dir) and os.path.isdir(old_contours_dir) and 
        not os.path.islink(old_contours_dir) and os.path.exists(contours_dir)):
        import shutil
        synced_count = 0
        for item in os.listdir(old_contours_dir):
            src = os.path.join(old_contours_dir, item)
            dst = os.path.join(contours_dir, item)
            # Only copy if destination doesn't exist or source is newer
            if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                shutil.copy2(src, dst)
                synced_count += 1
        if synced_count > 0:
            print(f"    Synced {synced_count} file(s) from contours_geojson to contours")


def process_tma_case_with_mask(
    mask_path: str,
    slide_path: str,
    case_id: str,
    marker: str,
    block: str,
    args: argparse.Namespace,
    temp_geojson_dir: str
) -> None:
    """
    Process a single TMA case by applying binary mask before segmentation.

    Parameters
    ----------
    mask_path : str
        Path to the binary mask PNG file.
    slide_path : str
        Path to the WSI SVS file.
    case_id : str
        Case ID string.
    marker : str
        TMA marker name (e.g., 'CD3').
    block : str
        Block number string.
    args : argparse.Namespace
        Parsed command-line arguments.
    temp_geojson_dir : str
        Temporary directory to save GeoJSON files.
    """
    case_display_name = f"{marker}_block{block}_case{case_id}"
    print(f"\n[PROCESSING] {case_display_name}")
    print(f"  Mask: {mask_path}")
    print(f"  Slide: {slide_path}")
    
    # Load WSI to get dimensions and MPP
    try:
        wsi = load_wsi(slide_path, lazy_init=False, custom_mpp_keys=args.custom_mpp_keys, mpp=args.default_mpp)
        wsi_width = wsi.width
        wsi_height = wsi.height
        wsi_mpp = wsi.mpp if wsi.mpp is not None else 0.25  # Default MPP if not available
        print(f"  WSI dimensions: {wsi_width} x {wsi_height}, MPP: {wsi_mpp}")
    except Exception as e:
        print(f"  [ERROR] Failed to load WSI: {e}")
        if not args.skip_errors:
            raise
        return
    
    # Load binary mask
    try:
        mask_thumbnail = load_binary_mask(mask_path)
        mask_thumb_height, mask_thumb_width = mask_thumbnail.shape
        print(f"  Mask thumbnail dimensions: {mask_thumb_width} x {mask_thumb_height}")
    except Exception as e:
        print(f"  [ERROR] Failed to load mask: {e}")
        if not args.skip_errors:
            raise
        return
    
    # Convert thumbnail mask directly to GeoJSON and scale coordinates to WSI level 0
    geojson_filename = f"{case_display_name}_mask.geojson"
    geojson_path = os.path.join(temp_geojson_dir, geojson_filename)
    try:
        convert_thumbnail_mask_to_geojson(
            mask=mask_thumbnail,
            mask_width=mask_thumb_width,
            mask_height=mask_thumb_height,
            wsi_width=wsi_width,
            wsi_height=wsi_height,
            wsi_mpp=wsi_mpp,
            output_path=geojson_path
        )
        print(f"  Converted thumbnail mask to GeoJSON (scaled to WSI level 0): {geojson_path}")
    except Exception as e:
        print(f"  [ERROR] Failed to convert mask to GeoJSON: {e}")
        if not args.skip_errors:
            raise
        return
    
    # Create a temporary CSV with just this slide
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_csv:
        df = pd.DataFrame({'wsi': [os.path.basename(slide_path)]})
        df.to_csv(tmp_csv.name, index=False)
        temp_csv_path = tmp_csv.name
    
    # Create job directory for this case
    case_job_dir = os.path.join(args.job_dir, marker, f"block{block}_case{case_id}")
    os.makedirs(case_job_dir, exist_ok=True)
    
    # Rename existing contours_geojson to contours if it exists
    old_contours_dir = os.path.join(case_job_dir, 'contours_geojson')
    contours_dir = os.path.join(case_job_dir, 'contours')
    if os.path.exists(old_contours_dir) and not os.path.exists(contours_dir):
        import shutil
        shutil.move(old_contours_dir, contours_dir)
        print("  Renamed contours_geojson to contours")
    elif os.path.exists(old_contours_dir) and os.path.exists(contours_dir):
        # Both exist, merge contents and remove old
        import shutil
        for item in os.listdir(old_contours_dir):
            src = os.path.join(old_contours_dir, item)
            dst = os.path.join(contours_dir, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        print("  Merged contours_geojson into contours and removed old folder")
    
    # Create contours directory and save the mask GeoJSON there
    os.makedirs(contours_dir, exist_ok=True)
    slide_basename = os.path.splitext(os.path.basename(slide_path))[0]
    mask_geojson_path = os.path.join(contours_dir, f'{slide_basename}_mask.geojson')
    final_geojson_path = os.path.join(contours_dir, f'{slide_basename}.geojson')
    
    # Create symlink from contours_geojson to contours for Processor compatibility
    if not os.path.exists(old_contours_dir):
        try:
            os.symlink('contours', old_contours_dir)
            print("  Created symlink contours_geojson -> contours for Processor compatibility")
        except OSError:
            # Symlink creation failed (might be on Windows or permission issue)
            # Create the directory and copy files instead
            os.makedirs(old_contours_dir, exist_ok=True)
            print("  Created contours_geojson directory for Processor compatibility")
    
    # Copy the GeoJSON to the contours directory as the mask (we'll use this for intersection)
    import shutil
    shutil.copy2(geojson_path, mask_geojson_path)
    # Also copy to the final path so Processor can find it initially
    shutil.copy2(geojson_path, final_geojson_path)
    
    # If contours_geojson exists as a directory (not symlink), copy files there too for compatibility
    if os.path.exists(old_contours_dir) and os.path.isdir(old_contours_dir) and not os.path.islink(old_contours_dir):
        old_mask_path = os.path.join(old_contours_dir, f'{slide_basename}_mask.geojson')
        old_final_path = os.path.join(old_contours_dir, f'{slide_basename}.geojson')
        shutil.copy2(geojson_path, old_mask_path)
        shutil.copy2(geojson_path, old_final_path)
    
    print(f"  Saved mask GeoJSON: {mask_geojson_path}")
    print(f"  Initial GeoJSON (for Processor): {final_geojson_path}")
    
    try:
        # Initialize processor for this case
        from contextlib import contextmanager
        import sys
        @contextmanager
        def suppress_stderr():
            with open(os.devnull, 'w') as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    yield
                finally:
                    sys.stderr = old_stderr
        
        with suppress_stderr():
            processor = Processor(
                job_dir=case_job_dir,
                wsi_source=os.path.dirname(slide_path),
                wsi_ext=['.svs'],
                skip_errors=args.skip_errors,
                custom_mpp_keys=args.custom_mpp_keys,
                custom_list_of_wsis=temp_csv_path,
                default_mpp=args.default_mpp,
                max_workers=args.max_workers,
                reader_type=args.reader_type,
                search_nested=False,
            )
        
        # The Processor should automatically find the GeoJSON in contours_geojson directory
        # and use it as the tissue mask
        
        # Run tasks
        tasks = ['seg', 'coords', 'feat'] if args.task == 'all' else [args.task]
        
        for task_name in tasks:
            print(f"  Running task: {task_name}")
            args.task = task_name
            
            # For coords and feat tasks, reload WSIs to ensure GeoJSON paths are updated
            if task_name in ['coords', 'feat']:
                processor.release()
                with suppress_stderr():
                    processor = Processor(
                        job_dir=case_job_dir,
                        wsi_source=os.path.dirname(slide_path),
                        wsi_ext=['.svs'],
                        skip_errors=args.skip_errors,
                        custom_mpp_keys=args.custom_mpp_keys,
                        custom_list_of_wsis=temp_csv_path,
                        default_mpp=args.default_mpp,
                        max_workers=args.max_workers,
                        reader_type=args.reader_type,
                        search_nested=False,
                    )
            
            run_task(processor, args, case_display_name)
        
        # Clean up processor resources
        processor.release()
        
    except Exception as e:
        print(f"  [ERROR] Failed to process case: {e}")
        if not args.skip_errors:
            raise
    finally:
        # Clean up temporary CSV
        try:
            os.unlink(temp_csv_path)
        except OSError:
            pass


def run_task(processor: Processor, args: argparse.Namespace, case_name: str = '') -> None:
    """
    Execute the specified task using the Trident Processor.

    Parameters
    ----------
    processor : Processor
        Initialized Trident Processor instance.
    args : argparse.Namespace
        Parsed command-line arguments containing task configuration.
    case_name : str
        Case display name for logging.
    """
    if args.task == 'seg':
        from trident.segmentation_models.load import segmentation_model_factory

        # instantiate segmentation model and artifact remover if requested by user
        segmentation_model = segmentation_model_factory(
            args.segmenter,
            confidence_thresh=args.seg_conf_thresh,
        )
        if args.remove_artifacts or args.remove_penmarks:
            artifact_remover_model = segmentation_model_factory(
                'grandqc_artifact',
                remove_penmarks_only=args.remove_penmarks and not args.remove_artifacts
            )
        else:
            artifact_remover_model = None

        # run segmentation 
        # Note: The processor will use the pre-existing GeoJSON mask if available
        # After segmentation, we'll intersect the result with the original mask
        processor.run_segmentation_job(
            segmentation_model,
            seg_mag=segmentation_model.target_mag,
            holes_are_tissue=not args.remove_holes,
            artifact_remover_model=artifact_remover_model,
            batch_size=args.seg_batch_size if args.seg_batch_size is not None else args.batch_size,
            device=f'cuda:{args.gpu}',
        )
        
        # Sync files from contours_geojson to contours if they're separate directories
        # (segment_tissue saves to contours_geojson, but we want everything in contours)
        sync_contours_directories(processor.job_dir)
        
        # After segmentation, intersect with the original mask
        # This ensures we only keep tissue within the TMA core
        for spec in processor.slide_specs:
            tissue_seg_path = os.path.join(
                processor.job_dir, 'contours_geojson', f"{spec['stem']}.geojson"
            )
            if tissue_seg_path and os.path.exists(tissue_seg_path):
                try:
                    seg_gdf = gpd.read_file(tissue_seg_path)
                    # Load the original mask GeoJSON (saved with _mask suffix)
                    mask_geojson_path = tissue_seg_path.replace('.geojson', '_mask.geojson')
                    if os.path.exists(mask_geojson_path):
                        mask_gdf = gpd.read_file(mask_geojson_path)
                        if not mask_gdf.empty and not seg_gdf.empty:
                            # Union all mask geometries
                            mask_union = mask_gdf.geometry.unary_union
                            if mask_union.is_empty:
                                print(f"    [WARNING] Empty mask union for {case_name}")
                                continue
                            
                            # Intersect each segmentation polygon with the mask
                            intersected_geoms = []
                            for idx, seg_geom in seg_gdf.geometry.items():
                                if seg_geom.intersects(mask_union):
                                    intersection = seg_geom.intersection(mask_union)
                                    if not intersection.is_empty:
                                        # Handle MultiPolygon results
                                        if hasattr(intersection, 'geoms'):
                                            for geom in intersection.geoms:
                                                if not geom.is_empty:
                                                    intersected_geoms.append(geom)
                                        else:
                                            intersected_geoms.append(intersection)
                            
                            if intersected_geoms:
                                # Create new GeoDataFrame with intersected geometries
                                intersected_gdf = gpd.GeoDataFrame(
                                    {'tissue_id': range(len(intersected_geoms))},
                                    geometry=intersected_geoms
                                )
                                # Save intersected result
                                intersected_gdf.to_file(tissue_seg_path, driver='GeoJSON')
                                print(f"    Intersected segmentation with mask for {case_name}: {len(intersected_geoms)} regions")
                            else:
                                print(f"    [WARNING] No intersection found for {case_name}, keeping original segmentation")
                    else:
                        print(f"    [WARNING] Mask GeoJSON not found: {mask_geojson_path}")
                except Exception as e:
                    print(f"    [WARNING] Could not intersect segmentation with mask: {e}")
                    import traceback
                    traceback.print_exc()
                    
    elif args.task == 'coords':
        processor.run_patching_job(
            target_magnification=args.mag,
            patch_size=args.patch_size,
            overlap=args.overlap,
            saveto=args.coords_dir,
            min_tissue_proportion=args.min_tissue_proportion
        )
    elif args.task == 'feat':
        # Construct coords_dir path (relative to job_dir)
        coords_dir = args.coords_dir or f'{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap'
        
        # Ensure coords_dir exists (Processor needs it for config/log files)
        coords_dir_full = os.path.join(processor.job_dir, coords_dir)
        os.makedirs(coords_dir_full, exist_ok=True)
        
        if args.slide_encoder is None: 
            from trident.patch_encoder_models.load import encoder_factory
            encoder = encoder_factory(args.patch_encoder, weights_path=args.patch_encoder_ckpt_path)
            # saveto will default to os.path.join(coords_dir, f'features_{encoder.enc_name}') if None
            processor.run_patch_feature_extraction_job(
                coords_dir=coords_dir,
                patch_encoder=encoder,
                device=f'cuda:{args.gpu}',
                saveas='h5',
                batch_limit=args.feat_batch_size if args.feat_batch_size is not None else args.batch_size,
                saveto=None,  # Let Processor use default path
            )
        else:
            from trident.slide_encoder_models.load import encoder_factory
            encoder = encoder_factory(args.slide_encoder)
            # saveto will default to os.path.join(coords_dir, f'slide_features_{encoder.enc_name}') if None
            processor.run_slide_feature_extraction_job(
                slide_encoder=encoder,
                coords_dir=coords_dir,
                device=f'cuda:{args.gpu}',
                saveas='h5',
                batch_limit=args.feat_batch_size if args.feat_batch_size is not None else args.batch_size,
                saveto=None,  # Let Processor use default path
            )
    else:
        raise ValueError(f'Invalid task: {args.task}')


def main() -> None:
    """
    Main entry point for the Trident batch processing script for Hancock TMA slides.
    
    Finds all TMA masks matching Hancock case IDs, applies binary masks before segmentation,
    and processes each TMA case separately.
    """
    args = parse_arguments()
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    # Find all TMA masks for Hancock cases
    print(f"[MAIN] Searching for TMA masks in {TMA_MASKS_DIR}...")
    tma_cases = find_tma_masks_for_hancock_cases()
    print(f"[MAIN] Found {len(tma_cases)} TMA cases matching Hancock case IDs.")
    
    if len(tma_cases) == 0:
        print("[MAIN] No TMA cases found. Exiting.")
        return
    
    # Group by marker and block for better organization
    cases_by_marker_block = defaultdict(list)
    for case in tma_cases:
        key = (case['marker'], case['block'])
        cases_by_marker_block[key].append(case)
    
    print(f"[MAIN] Grouped into {len(cases_by_marker_block)} marker/block combinations.")
    
    # Rename any existing contours_geojson folders to contours in the job directory
    print("[MAIN] Renaming existing contours_geojson folders to contours...")
    rename_contours_folders(args.job_dir)
    
    # Create temporary directory for GeoJSON files
    temp_geojson_dir = os.path.join(args.job_dir, '_temp_geojsons')
    os.makedirs(temp_geojson_dir, exist_ok=True)
    
    # Process each TMA case
    processed_count = 0
    for (marker, block), cases in sorted(cases_by_marker_block.items()):
        print(f"\n[MAIN] Processing {marker} block {block}: {len(cases)} cases")
        
        for case in cases:
            try:
                process_tma_case_with_mask(
                    mask_path=case['mask_path'],
                    slide_path=case['slide_path'],
                    case_id=case['case_id'],
                    marker=case['marker'],
                    block=case['block'],
                    args=args,
                    temp_geojson_dir=temp_geojson_dir
                )
                processed_count += 1
            except Exception as e:
                print(f"[MAIN] Error processing case {case['case_id']}: {e}")
                if not args.skip_errors:
                    raise
    
    # Clean up temporary directory
    try:
        import shutil
        shutil.rmtree(temp_geojson_dir)
    except Exception:
        pass
    
    print(f"\n[MAIN] Processing complete. Processed {processed_count}/{len(tma_cases)} TMA cases.")


if __name__ == "__main__":
    main()

