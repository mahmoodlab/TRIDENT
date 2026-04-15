"""
Example usage:

```
python run_batch_of_hancock_slides.py --task all --wsi_dir output/wsis --job_dir output --patch_encoder uni_v1 --mag 20 --patch_size 256
```

This script processes only slides whose names contain any of the following numbers as a substring:
104,116,120,121,162,176,191,225,250,296,334,342,346,357,403,407,476,530,559,564,583,606,632,664,668,698,706,723,740,741

It also maintains the subdirectory structure from the input directory in the output directory.
"""
import os
import argparse
import torch
os.environ["OPJ_QUIET"] = "true"  # Suppresses OpenJPEG (dicom/svs) warnings
from typing import Dict, List, Tuple
from collections import defaultdict
import pdb
from trident import Processor 
from trident.patch_encoder_models import encoder_registry as patch_encoder_registry
from trident.slide_encoder_models import encoder_registry as slide_encoder_registry
from trident.IO import collect_valid_slides
os.environ['VIPS_WARNING'] = '0' 
os.environ['OPJ_QUIET'] = 'true'

import warnings

warnings.filterwarnings("ignore")


# Hancock slide numbers to filter
HANCOCK_NUMBERS = [
    '104', '116', '120', '121', '162', '176', '191', '225', '250', '296',
    '334', '342', '346', '357', '403', '407', '476', '530', '559', '564',
    '583', '606', '632', '664', '668', '698', '706', '723', '740', '741'
]


def build_parser() -> argparse.ArgumentParser:
    """
    Parse command-line arguments for the Trident processing script.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all Trident processing options.
    """
    parser = argparse.ArgumentParser(description='Run Trident on Hancock slides only')

    # Generic arguments 
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use for processing tasks.')
    parser.add_argument('--task', type=str, default='seg', 
                        choices=['seg', 'coords', 'feat', 'all'], 
                        help='Task to run: seg (segmentation), coords (save tissue coordinates), img (save tissue images), feat (extract features).')
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
    parser.add_argument('--wsi_dir', type=str, required=True, 
                        help='Directory containing WSI files (nested subdirectories are supported).')
    parser.add_argument('--wsi_ext', type=str, nargs='+', default=None, 
                        help='List of allowed file extensions for WSI files (e.g., .svs .ndpi .tif). Extensions will be normalized to start with a dot.')
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


def filter_hancock_slides(full_paths: List[str], rel_paths: List[str]) -> Tuple[List[str], List[str]]:
    """
    Filter slides to only include those whose filename contains any of the Hancock numbers.

    Parameters
    ----------
    full_paths : List[str]
        List of full paths to WSI files.
    rel_paths : List[str]
        List of relative paths to WSI files.

    Returns
    -------
    Tuple[List[str], List[str]]
        Filtered (full_paths, rel_paths) containing only Hancock slides.
    """
    filtered_full = []
    filtered_rel = []
    
    for full_path, rel_path in zip(full_paths, rel_paths):
        filename = os.path.basename(full_path)
        # Check if filename contains any of the Hancock numbers
        if any(num in filename for num in HANCOCK_NUMBERS):
            filtered_full.append(full_path)
            filtered_rel.append(rel_path)
    
    return filtered_full, filtered_rel


def group_slides_by_subdir(full_paths: List[str], rel_paths: List[str], wsi_dir: str) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Group slides by their subdirectory structure.

    Parameters
    ----------
    full_paths : List[str]
        List of full paths to WSI files.
    rel_paths : List[str]
        List of relative paths to WSI files.
    wsi_dir : str
        Base directory containing WSIs.

    Returns
    -------
    Dict[str, Tuple[List[str], List[str]]]
        Dictionary mapping subdirectory paths to (full_paths, rel_paths) tuples.
        Empty string '' represents the root directory.
    """
    groups: Dict[str, Tuple[List[str], List[str]]] = defaultdict(lambda: ([], []))
    
    for full_path, rel_path in zip(full_paths, rel_paths):
        # Get the directory part of the relative path
        rel_dir = os.path.dirname(rel_path)
        groups[rel_dir][0].append(full_path)
        groups[rel_dir][1].append(rel_path)
    
    return dict(groups)


def create_custom_list_csv(rel_paths_in_group: List[str], temp_csv_path: str) -> None:
    """
    Create a temporary CSV file with the list of slides for a specific group.

    Parameters
    ----------
    rel_paths_in_group : List[str]
        Relative paths to slides in this group (relative to wsi_dir).
    temp_csv_path : str
        Path to save the temporary CSV file.
    """
    import pandas as pd
    
    # Create DataFrame with relative paths
    df = pd.DataFrame({'wsi': rel_paths_in_group})
    df.to_csv(temp_csv_path, index=False)


from contextlib import contextmanager
import sys
import os 
@contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr
def initialize_processor_for_group(
    args: argparse.Namespace,
    subdir: str,
    wsi_dir: str,
    temp_csv_path: str
) -> Processor:
    """
    Initialize a Processor for a specific subdirectory group.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    subdir : str
        Subdirectory path (relative to wsi_dir).
    wsi_dir : str
        Base WSI directory.
    temp_csv_path : str
        Path to temporary CSV file with slide list.

    Returns
    -------
    Processor
        Initialized Processor for this group.
    """
    # Create job_dir that maintains subdirectory structure
    if subdir:
        job_dir = os.path.join(args.job_dir, subdir)
    else:
        job_dir = args.job_dir
    
    # Create a custom args namespace for this group
    group_args = argparse.Namespace(**vars(args))
    group_args.job_dir = job_dir
    group_args.wsi_dir = wsi_dir
    group_args.custom_list_of_wsis = temp_csv_path
    group_args.search_nested = False  # We're using custom_list, so no need for .
    with suppress_stderr():
        return Processor(
            job_dir=group_args.job_dir,
            wsi_source=group_args.wsi_dir,
            wsi_ext=group_args.wsi_ext,
            wsi_cache=group_args.wsi_cache,
            skip_errors=group_args.skip_errors,
            custom_mpp_keys=group_args.custom_mpp_keys,
            custom_list_of_wsis=group_args.custom_list_of_wsis,
            default_mpp=group_args.default_mpp,
            max_workers=group_args.max_workers,
            reader_type=group_args.reader_type,
            search_nested=False,  # Using custom_list, so nested search not needed
    )


def run_task(processor: Processor, args: argparse.Namespace, subdir: str = '') -> None:
    """
    Execute the specified task using the Trident Processor.

    Parameters
    ----------
    processor : Processor
        Initialized Trident Processor instance.
    args : argparse.Namespace
        Parsed command-line arguments containing task configuration.
    subdir : str
        Subdirectory path (relative to wsi_dir) to use for organizing feature outputs.
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
        processor.run_segmentation_job(
            segmentation_model,
            seg_mag=segmentation_model.target_mag,
            holes_are_tissue= not args.remove_holes,
            artifact_remover_model=artifact_remover_model,
            batch_size=args.seg_batch_size if args.seg_batch_size is not None else args.batch_size,
            device=f'cuda:{args.gpu}',
        )
    elif args.task == 'coords':
        processor.run_patching_job(
            target_magnification=args.mag,
            patch_size=args.patch_size,
            overlap=args.overlap,
            saveto=args.coords_dir,
            min_tissue_proportion=args.min_tissue_proportion
        )
    elif args.task == 'feat':
        # Construct saveto path to maintain subdirectory structure in trident_feats
        # Since processor.job_dir already includes subdir (e.g., output/Ax),
        # we need to save to output/trident_feats/Ax instead of output/Ax/trident_feats
        if subdir:
            # Get the base job_dir (one level up from processor.job_dir)
            base_job_dir = os.path.dirname(processor.job_dir)
            # Construct absolute path: output/trident_feats/Ax
            saveto = os.path.join(base_job_dir, 'trident_feats', subdir)
        else:
            # For root directory, save to trident_feats relative to job_dir
            saveto = 'trident_feats'
        
        if args.slide_encoder is None: 
            from trident.patch_encoder_models.load import encoder_factory
            encoder = encoder_factory(args.patch_encoder, weights_path=args.patch_encoder_ckpt_path)
            processor.run_patch_feature_extraction_job(
                coords_dir=args.coords_dir or f'{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap',
                patch_encoder=encoder,
                device=f'cuda:{args.gpu}',
                saveas='h5',
                batch_limit=args.feat_batch_size if args.feat_batch_size is not None else args.batch_size,
                saveto=saveto,
            )
        else:
            from trident.slide_encoder_models.load import encoder_factory
            encoder = encoder_factory(args.slide_encoder)
            processor.run_slide_feature_extraction_job(
                slide_encoder=encoder,
                coords_dir=args.coords_dir or f'{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap',
                device=f'cuda:{args.gpu}',
                saveas='h5',
                batch_limit=args.feat_batch_size if args.feat_batch_size is not None else args.batch_size,
                saveto=saveto,
            )
    else:
        raise ValueError(f'Invalid task: {args.task}')


def normalize_extensions(extensions: List[str]) -> List[str]:
    """
    Normalize file extensions to ensure they start with a dot and are lowercase.
    
    Parameters
    ----------
    extensions : List[str]
        List of file extensions (with or without leading dots).
    
    Returns
    -------
    List[str]
        Normalized extensions (all starting with a dot, lowercase).
    """
    normalized = []
    for ext in extensions:
        ext = ext.lower().strip()
        if not ext.startswith('.'):
            ext = '.' + ext
        normalized.append(ext)
    return normalized


def main() -> None:
    """
    Main entry point for the Trident batch processing script for Hancock slides.
    
    Filters slides to only process those containing Hancock numbers, maintains
    subdirectory structure in output, and processes each subdirectory group separately.
    """

    args = parse_arguments()
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    # Normalize extensions if provided
    if args.wsi_ext is not None:
        args.wsi_ext = normalize_extensions(args.wsi_ext)
        print(f"[MAIN] Filtering slides by extensions: {', '.join(args.wsi_ext)}")
    else:
        print(f"[MAIN] No extension filter specified - will use default WSI extensions.")

    # Collect all valid slides with nested search enabled
    print(f"[MAIN] Collecting slides from {args.wsi_dir} (searching nested directories)...")
    full_paths, rel_paths = collect_valid_slides(
        wsi_dir=args.wsi_dir,
        custom_list_path=None,
        wsi_ext=args.wsi_ext,
        search_nested=True,  # Always search nested for Hancock slides
        max_workers=args.max_workers,
        return_relative_paths=True
    )
    
    print(f"[MAIN] Found {len(full_paths)} total valid slides in {args.wsi_dir}.")
    
    # Filter to only Hancock slides
    hancock_full_paths, hancock_rel_paths = filter_hancock_slides(full_paths, rel_paths)
    print(f"[MAIN] Filtered to {len(hancock_full_paths)} Hancock slides (containing numbers: {', '.join(HANCOCK_NUMBERS[:5])}...)")
    
    if len(hancock_full_paths) == 0:
        print("[MAIN] No Hancock slides found. Exiting.")
        return
    
    # Group slides by subdirectory
    slide_groups = group_slides_by_subdir(hancock_full_paths, hancock_rel_paths, args.wsi_dir)
    print(f"[MAIN] Grouped slides into {len(slide_groups)} subdirectory groups.")
    
    # Process each group separately
    import tempfile
    
    tasks = ['seg', 'coords', 'feat'] if args.task == 'all' else [args.task]
    
    for subdir, (group_full_paths, group_rel_paths) in sorted(slide_groups.items()):
        subdir_display = subdir if subdir else '(root)'
        print(f"\n[MAIN] Processing subdirectory: {subdir_display} ({len(group_full_paths)} slides)")
        
        # Create temporary CSV file for this group
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            temp_csv_path = tmp_file.name
            create_custom_list_csv(group_rel_paths, temp_csv_path)
        
        try:
            # Initialize processor for this group
            with suppress_stderr():
                processor = initialize_processor_for_group(args, subdir, args.wsi_dir, temp_csv_path)
            
            # Run tasks for this group
            for task_name in tasks:
                print(f"[MAIN] Running task '{task_name}' for subdirectory: {subdir_display}")
                args.task = task_name
                
                # For coords and feat tasks, reload WSIs to ensure GeoJSON paths are updated
                # after segmentation has run
                if task_name in ['coords', 'feat']:
                    # Re-initialize processor to reload WSIs with updated GeoJSON paths
                    processor.release()
                    with suppress_stderr():
                        processor = initialize_processor_for_group(args, subdir, args.wsi_dir, temp_csv_path)
                
                run_task(processor, args, subdir=subdir)
            
            # Clean up processor resources
            processor.release()
            
        except Exception as e:
            print(f"[MAIN] Error processing subdirectory {subdir_display}: {e}")
            if not args.skip_errors:
                raise
        finally:
            # Clean up temporary CSV file
            try:
                os.unlink(temp_csv_path)
            except OSError:
                pass
    
    print("\n[MAIN] All Hancock slides processed.")


if __name__ == "__main__":
    main()

