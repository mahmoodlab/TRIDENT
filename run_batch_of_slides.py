"""
Example usage:

```
python run_batch_of_slides.py --task all --wsi_dir output/wsis --job_dir output --patch_encoder uni_v1 --mag 20 --patch_size 256
```

"""
import os
import argparse
import torch
import multiprocessing as mp
import shutil
import sys
from queue import Queue
from threading import Thread
from typing import List
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from trident import Processor
from trident.patch_encoder_models import encoder_registry as patch_encoder_registry
from trident.patch_segmentation_models import patch_segmenter_registry
from trident.slide_encoder_models import encoder_registry as slide_encoder_registry
from trident.Concurrency import batch_producer, batch_consumer
from trident.IO import collect_valid_slides
from trident.Summary import start_run, finalize_run


def _pick_mp_context() -> mp.context.BaseContext:
    """
    Pick a multiprocessing context that is portable across OSes.

    - Windows: only spawn is supported reliably.
    - CUDA: prefer spawn (recommended by PyTorch).
    - POSIX CPU-only: prefer forkserver when available to avoid fork() hazards
      with multi-threaded / native-library-heavy parents; fall back to spawn.
    """
    available = set(mp.get_all_start_methods())

    if sys.platform.startswith("win"):
        return mp.get_context("spawn")

    if torch.cuda.is_available():
        return mp.get_context("spawn")

    if "forkserver" in available:
        return mp.get_context("forkserver")
    return mp.get_context("spawn")


def build_parser() -> argparse.ArgumentParser:
    """
    Parse command-line arguments for the Trident processing script.

    Returns:
        argparse.ArgumentParser: Configured argument parser with all Trident processing options.
    """
    parser = argparse.ArgumentParser(description='Run Trident')

    # Generic arguments 
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='[DEPRECATED] Single GPU index. Use `--gpus <id>` instead.',
    )
    parser.add_argument('--gpus', type=int, nargs='+', default=None,
                        help='Optional space-separated list of GPU indices to enable multi-GPU execution.')
    parser.add_argument('--task', type=str, default='seg',
                        choices=['seg', 'coords', 'feat', 'patch_seg', 'all'],
                        help='Task to run: seg (tissue vs background segmentation), coords (save tissue coordinates), '
                             'feat (extract patch/slide features), patch_seg (run a dense patch segmentation model such '
                             'as HistoPlus/SAM over tissue patches), or all (run the full seg->coords->feat pipeline).')
    parser.add_argument('--job_dir', type=str, required=True, help='Directory to store outputs.')
    parser.add_argument('--skip_errors', action='store_true', default=False, 
                        help='Skip errored slides and continue processing.')
    parser.add_argument(
        '--clear_dead_locks',
        action='store_true',
        default=False,
        help='If set, remove stale `.lock` files under `--job_dir` (safe heuristics) before running.',
    )
    parser.add_argument(
        '--dead_lock_max_age_hours',
        type=float,
        default=24.0,
        help='Max age (hours) before a `.lock` file is considered stale (when its target output is missing). Defaults to 24.',
    )
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
                        help='Directory containing WSI files. Top-level files only by default; pass --search_nested to also include slides in subdirectories.')
    parser.add_argument('--wsi_ext', type=str, nargs='+', default=None, 
                        help='List of allowed file extensions for WSI files.')
    parser.add_argument('--custom_mpp_keys', type=str, nargs='+', default=None,
                    help='Custom keys used to store the resolution as MPP (micron per pixel) in your list of whole-slide image.')
    parser.add_argument('--custom_list_of_wsis', type=str, default=None,
                    help='Custom list of WSIs specified in a csv file.')
    parser.add_argument('--reader_type', type=str, choices=['openslide', 'image', 'cucim', 'sdpc', 'omezarr', 'czi'], default=None,
                    help='Force the use of a specific WSI image reader. Options are ["openslide", "image", "cucim", "sdpc", "omezarr", "czi"]. Defaults to None (auto-determine which reader to use).')
    parser.add_argument("--search_nested", action="store_true",
                        help=("If set, recursively search for whole-slide images (WSIs) within all subdirectories of "
                              "`wsi_source`. Uses `os.walk` to include slides from nested folders. "
                              "This allows processing of datasets organized in hierarchical structures. "
                              "Defaults to False (only top-level slides are included)."))
    # Segmentation arguments 
    parser.add_argument('--segmenter', type=str, default='hest', 
                        choices=['hest', 'grandqc', 'otsu'],
                        help='Type of tissue vs background segmenter. Options are HEST, GrandQC, or Otsu.')
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
    parser.add_argument('--mag', type=float, default=20.0,
                        help='Magnification for coords/features extraction. Supports fractional values (e.g., 1.25x, 2.5x, 5x, etc.).')
    parser.add_argument('--patch_size', type=int, default=512, 
                        help='Patch size for coords/image extraction.')
    parser.add_argument('--overlap', type=int, default=0, 
                        help='Absolute overlap for patching in pixels. Defaults to 0.')
    parser.add_argument('--min_tissue_proportion', type=float, default=0., 
                        help='Minimum proportion of the patch under tissue to be kept. Between 0. and 1.0. Defaults to 0.')
    parser.add_argument('--coords_dir', type=str, default=None, 
                        help='Directory to save/restore tissue coordinates.')
    parser.add_argument(
        '--dump_patches', action='store_true', default=False,
        help='During the coords task, also dump patch images (PNGs) to disk.'
    )
    parser.add_argument(
        '--dump_patches_max', type=int, default=0,
        help='Max number of patch images to dump per slide (0 = no limit).'
    )
    parser.add_argument(
        '--dump_patches_format', type=str, default="png", choices=["png", "jpg"],
        help='Patch image format to dump (png or jpg). Defaults to png.'
    )
    parser.add_argument(
        '--dump_patches_jpeg_quality', type=int, default=90,
        help='JPEG quality (1-100) when --dump_patches_format=jpg. Defaults to 90.'
    )
    
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
    parser.add_argument(
        '--patch_encoder_img_size', type=int, default=None,
        help=(
            "Optional custom input resolution (in pixels) for the patch encoder. When set, the "
            "encoder resizes patches to this size and interpolates its positional embeddings "
            "(timm `dynamic_img_size`). Must be a multiple of the model's patch size. Only "
            "supported for ViT-based encoders; see `RESIZE_SUPPORTED_PATCH_ENCODERS`. Defaults to "
            "the model's native resolution."
        )
    )
    parser.add_argument('--slide_encoder', type=str, default=None, 
                        choices=slide_encoder_registry.keys(), 
                        help='Slide encoder to use')
    parser.add_argument('--feat_batch_size', type=int, default=None,
                        help='Batch size for feature extraction. Defaults to None (use `batch_size` argument instead).')

    # Patch segmentation arguments (task=patch_seg)
    parser.add_argument('--patch_segmenter', type=str, default='histoplus',
                        choices=patch_segmenter_registry.keys(),
                        help='Cell/object segmentation model to run over tissue patches (task=patch_seg). '
                             'histoplus expects --patch_size 784 at --mag 20 (mpp 0.5) or 40 (mpp 0.25); '
                             'if you hit a silent crash on newer torch, set --feat_batch_size 1.')
    parser.add_argument('--patch_segmenter_ckpt_path', type=str, default=None,
                        help='Optional local path to patch-segmenter weights (offline environments).')
    parser.add_argument('--seg_viz', action='store_true', default=False,
                        help='For task=patch_seg, also save a debug overlay JPEG of the predicted cells per slide.')
    return parser


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments and return the parsed namespace.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    args = build_parser().parse_args()

    # Normalize to always have `args.gpus` so downstream code only needs one path.
    if args.gpus is None:
        args.gpus = [args.gpu]
    elif "--gpu" in sys.argv:
        print("[MAIN] Warning: `--gpu` is deprecated and ignored when `--gpus` is provided.")

    # Normalize a default device for single-process paths.
    # Multi-process workers override this per worker via `worker_args.device`.
    primary_gpu = (args.gpus or [-1])[0]
    args.device = f"cuda:{primary_gpu}" if primary_gpu >= 0 else "cpu"

    return args


def generate_help_text() -> str:
    """
    Generate the command-line help text for documentation purposes.
    
    Returns:
        str: The full help message string from the argument parser.
    """
    parser = build_parser()
    return parser.format_help()


def initialize_processor(args: argparse.Namespace) -> Processor:
    """
    Initialize the Trident Processor with arguments set in `run_batch_of_slides`.

    Parameters:
        args (argparse.Namespace):
            Parsed command-line arguments containing processor configuration.

    Returns:
        Processor: Initialized Trident Processor instance.
    """
    return Processor(
        job_dir=args.job_dir,
        wsi_source=args.wsi_dir,
        wsi_ext=args.wsi_ext,
        wsi_cache=args.wsi_cache,
        skip_errors=args.skip_errors,
        custom_mpp_keys=args.custom_mpp_keys,
        custom_list_of_wsis=args.custom_list_of_wsis,
        max_workers=args.max_workers,
        reader_type=args.reader_type,
        search_nested=args.search_nested,
        selected_wsi_paths=getattr(args, 'selected_wsi_paths', None),
    )


def run_task(processor: Processor, args: argparse.Namespace) -> None:
    """
    Execute the specified task using the Trident Processor.

    Parameters:
        processor (Processor):
            Initialized Trident Processor instance.
        args (argparse.Namespace):
            Parsed command-line arguments containing task configuration.
    """

    # Prefer normalized `args.device` (set by `parse_arguments()` and overridden per worker),
    # but keep backward compatibility for callers constructing `args` manually.
    device = getattr(args, "device", None)
    if device is None:
        if getattr(args, "gpus", None):
            primary_gpu = args.gpus[0]
            device = f"cuda:{primary_gpu}" if primary_gpu >= 0 else "cpu"
        else:
            primary_gpu = getattr(args, "gpu", -1)
            device = f"cuda:{primary_gpu}" if primary_gpu >= 0 else "cpu"

    if args.task == 'seg':
        from trident.segmentation_models.load import segmentation_model_factory

        seg_device = "cpu" if args.segmenter == "otsu" else device

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
            device=seg_device,
        )
    elif args.task == 'coords':
        processor.run_patching_job(
            target_magnification=args.mag,
            patch_size=args.patch_size,
            overlap=args.overlap,
            saveto=args.coords_dir,
            min_tissue_proportion=args.min_tissue_proportion,
            dump_patches=args.dump_patches,
            dump_patches_max=args.dump_patches_max,
            dump_patches_format=args.dump_patches_format,
            dump_patches_jpeg_quality=args.dump_patches_jpeg_quality,
        )
    elif args.task == 'feat':
        if args.slide_encoder is None: 
            from trident.patch_encoder_models.load import encoder_factory, RESIZE_SUPPORTED_PATCH_ENCODERS
            encoder_kwargs = {}
            patch_encoder_img_size = getattr(args, "patch_encoder_img_size", None)
            if patch_encoder_img_size is not None:
                if args.patch_encoder not in RESIZE_SUPPORTED_PATCH_ENCODERS:
                    raise ValueError(
                        f"--patch_encoder_img_size is not supported for '{args.patch_encoder}'. "
                        f"It is only available for ViT-based encoders: {sorted(RESIZE_SUPPORTED_PATCH_ENCODERS)}."
                    )
                encoder_kwargs['target_img_size'] = patch_encoder_img_size
            encoder = encoder_factory(args.patch_encoder, weights_path=args.patch_encoder_ckpt_path, **encoder_kwargs)
            mag_str = f"{float(args.mag):g}"
            processor.run_patch_feature_extraction_job(
                coords_dir=args.coords_dir or f'{mag_str}x_{args.patch_size}px_{args.overlap}px_overlap',
                patch_encoder=encoder,
                device=device,
                saveas='h5',
                batch_limit=args.feat_batch_size if args.feat_batch_size is not None else args.batch_size,
            )
        else:
            from trident.slide_encoder_models.load import encoder_factory
            encoder = encoder_factory(args.slide_encoder)
            mag_str = f"{float(args.mag):g}"
            processor.run_slide_feature_extraction_job(
                slide_encoder=encoder,
                coords_dir=args.coords_dir or f'{mag_str}x_{args.patch_size}px_{args.overlap}px_overlap',
                device=device,
                saveas='h5',
                batch_limit=args.feat_batch_size if args.feat_batch_size is not None else args.batch_size,
            )
    elif args.task == 'patch_seg':
        from trident.patch_segmentation_models import patch_segmenter_factory
        patch_segmenter = patch_segmenter_factory(
            args.patch_segmenter,
            weights_path=args.patch_segmenter_ckpt_path,
        )
        mag_str = f"{float(args.mag):g}"
        processor.run_patch_segmentation_job(
            coords_dir=args.coords_dir or f'{mag_str}x_{args.patch_size}px_{args.overlap}px_overlap',
            patch_segmenter=patch_segmenter,
            device=device,
            batch_limit=args.feat_batch_size if args.feat_batch_size is not None else args.batch_size,
            visualize=getattr(args, 'seg_viz', False),
        )
    else:
        raise ValueError(f'Invalid task: {args.task}')


def remove_dead_locks(job_dir: str, *, max_age_hours: float = 24.0) -> dict[str, int]:
    """
    Backward-compatible wrapper. Prefer `trident.IO.clear_dead_locks`.
    """
    from trident.IO import clear_dead_locks

    stats = clear_dead_locks(job_dir, legacy_max_age_seconds=float(max_age_hours) * 3600.0)
    return {"scanned": int(stats["scanned"]), "removed": int(stats["removed"]), "kept": int(stats["kept"])}


def cleanup_cache(cache_dir: str | None) -> None:
    if not cache_dir:
        return
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
    elif os.path.exists(cache_dir):
        try:
            os.remove(cache_dir)
        except OSError:
            pass
    os.makedirs(cache_dir, exist_ok=True)


def get_pending_slides(args: argparse.Namespace) -> List[str]:
    all_slides = collect_valid_slides(
        wsi_dir=args.wsi_dir,
        custom_list_path=args.custom_list_of_wsis,
        wsi_ext=args.wsi_ext,
        search_nested=args.search_nested,
        max_workers=args.max_workers,
    )

    tasks = ['seg', 'coords', 'feat'] if args.task == 'all' else [args.task]
    mag_str = f"{float(args.mag):g}"
    coords_dir = args.coords_dir or f'{mag_str}x_{args.patch_size}px_{args.overlap}px_overlap'

    def safe_listdir(path: str) -> List[str]:
        try:
            return os.listdir(path)
        except (FileNotFoundError, NotADirectoryError):
            return []

    seg_done = set()
    coords_done = set()
    feat_done = set()
    patch_seg_done = set()

    if 'seg' in tasks:
        contour_dir = os.path.join(args.job_dir, 'contours')
        seg_done = {
            os.path.splitext(filename)[0]
            for filename in safe_listdir(contour_dir)
            if filename.lower().endswith('.jpg')
        }

    if 'coords' in tasks:
        patches_dir = os.path.join(args.job_dir, coords_dir, 'patches')
        coords_done = {
            stem[:-len('_patches')]
            for stem in (
                os.path.splitext(filename)[0]
                for filename in safe_listdir(patches_dir)
                if filename.endswith('_patches.h5')
            )
        }

    if 'feat' in tasks:
        if args.slide_encoder:
            feat_subdirs = [f'slide_features_{args.slide_encoder}']
        else:
            feat_subdirs = [
                f'features_{args.patch_encoder}',
                f'patch_features_{args.patch_encoder}',
            ]

        for feat_sub in feat_subdirs:
            feat_dir = os.path.join(args.job_dir, coords_dir, feat_sub)
            feat_done.update(
                os.path.splitext(filename)[0]
                for filename in safe_listdir(feat_dir)
                if os.path.splitext(filename)[1] in {'.h5', '.pt'}
            )

    if 'patch_seg' in tasks:
        seg_dir = os.path.join(args.job_dir, coords_dir, f'seg_{args.patch_segmenter}')
        patch_seg_done = {
            os.path.splitext(filename)[0]
            for filename in safe_listdir(seg_dir)
            if filename.endswith('.geojson')
        }

    pending = []
    for slide_path in all_slides:
        stem = os.path.splitext(os.path.basename(slide_path))[0]
        is_done = True

        for task_name in tasks:
            if task_name == 'seg' and stem not in seg_done:
                is_done = False
            elif task_name == 'coords' and stem not in coords_done:
                is_done = False
            elif task_name == 'feat' and stem not in feat_done:
                is_done = False
            elif task_name == 'patch_seg' and stem not in patch_seg_done:
                is_done = False

            if not is_done:
                break

        if not is_done:
            pending.append(slide_path)

    print(
        f"[MAIN] Found {len(all_slides)} slides. "
        f"Processing {len(pending)} pending slides ({len(all_slides) - len(pending)} skipped)."
    )
    return pending


def worker_entrypoint(args: argparse.Namespace) -> None:
    """
    Entry point for each worker.

    Supports both cache mode (threaded producer/consumer pipeline) and
    non-cache mode (direct sequential processing).
    """
    if args.wsi_cache:
        gpu_cache_dir = os.path.join(args.wsi_cache, f"gpu_{args.gpu}")
        os.makedirs(gpu_cache_dir, exist_ok=True)

        assigned_slides = list(getattr(args, 'selected_wsi_paths', None) or [])
        if not assigned_slides:
            print(f"[WORKER {args.gpu}] No slides assigned. Skipping cached pipeline.")
            return

        batch_size = max(1, args.cache_batch_size or len(assigned_slides))
        queue = Queue(maxsize=1)

        # Caching flattens slides to their basename, which drops the original
        # `--custom_list_of_wsis` CSV (and therefore any per-slide MPP values).
        # Build a basename -> mpp map so we can re-attach MPPs to cached batches.
        basename_to_mpp: dict[str, object] = {}
        if args.custom_list_of_wsis is not None:
            import pandas as pd
            wsi_df = pd.read_csv(args.custom_list_of_wsis)
            if 'wsi' in wsi_df.columns and 'mpp' in wsi_df.columns:
                for wsi_name, mpp in zip(wsi_df['wsi'].astype(str), wsi_df['mpp']):
                    basename_to_mpp[os.path.basename(wsi_name)] = mpp

        def processor_factory(wsi_dir: str) -> Processor:
            local_args = argparse.Namespace(**vars(args))
            local_args.wsi_dir = wsi_dir
            local_args.wsi_cache = None
            local_args.custom_list_of_wsis = None
            local_args.search_nested = False
            local_args.selected_wsi_paths = None

            # Re-attach per-slide MPPs (if provided) by writing a per-batch CSV
            # that maps the cached (basename) slides to their MPP values.
            if basename_to_mpp:
                import pandas as pd
                cached_files = [
                    f for f in os.listdir(wsi_dir)
                    if os.path.isfile(os.path.join(wsi_dir, f))
                ]
                rows = [
                    {'wsi': f, 'mpp': basename_to_mpp.get(f)}
                    for f in cached_files
                    if f in basename_to_mpp
                ]
                if rows:
                    batch_csv = os.path.join(wsi_dir, '_trident_cached_batch_list.csv')
                    pd.DataFrame(rows).to_csv(batch_csv, index=False)
                    local_args.custom_list_of_wsis = batch_csv

            return initialize_processor(local_args)

        def run_task_fn(processor: Processor, task_name: str) -> None:
            local_args = argparse.Namespace(**vars(args))
            local_args.task = task_name
            local_args.selected_wsi_paths = None
            run_task(processor, local_args)

        producer = Thread(
            target=batch_producer,
            args=(queue, assigned_slides, 0, batch_size, gpu_cache_dir),
        )
        consumer = Thread(
            target=batch_consumer,
            args=(queue, args.task, gpu_cache_dir, processor_factory, run_task_fn),
        )

        producer.start()
        consumer.start()
        producer.join()
        consumer.join()
        return

    processor = initialize_processor(args)
    tasks = ['seg', 'coords', 'feat'] if args.task == 'all' else [args.task]

    try:
        for task_name in tasks:
            local_args = argparse.Namespace(**vars(args))
            local_args.task = task_name
            run_task(processor, local_args)
    finally:
        if hasattr(processor, 'release'):
            processor.release()


def main() -> None:
    """
    Main entry point for the Trident batch processing script.
    
    Handles both sequential and parallel processing modes based on whether
    WSI caching is enabled. Supports segmentation, coordinate extraction,
    and feature extraction tasks.
    """

    args = parse_arguments()
    cleanup_cache(args.wsi_cache)
    if getattr(args, "clear_dead_locks", False):
        stats = remove_dead_locks(args.job_dir, max_age_hours=float(args.dead_lock_max_age_hours))
        print(
            f"[MAIN] Dead lock cleanup under {args.job_dir}: "
            f"removed={stats['removed']} scanned={stats['scanned']} kept={stats['kept']}"
        )

    # Deduplicate positive GPU IDs (running two workers on the same CUDA device
    # is wasteful), but keep duplicate `-1` entries so users can request multiple
    # parallel CPU workers via e.g. `--gpus -1 -1`.
    seen_gpus: set[int] = set()
    gpu_ids: List[int] = []
    for gid in args.gpus or []:
        if gid >= 0 and gid in seen_gpus:
            continue
        seen_gpus.add(gid)
        gpu_ids.append(gid)

    if not torch.cuda.is_available() and any(gpu_id >= 0 for gpu_id in gpu_ids):
        print('[MAIN] Warning: CUDA not available, using CPU.')
        gpu_ids = [-1]

    run_id = start_run(args.job_dir, tool="run_batch_of_slides", args=vars(args))
    run_status = "completed"
    run_error = None

    try:
        pending_slides = get_pending_slides(args)
        if not pending_slides:
            return

        shard_count = len(gpu_ids)
        shards = [[] for _ in range(shard_count)]
        for idx, slide_path in enumerate(pending_slides):
            shards[idx % shard_count].append(slide_path)

        if len(gpu_ids) == 1:
            worker_args = argparse.Namespace(**vars(args))
            worker_args.gpu = gpu_ids[0]
            worker_args.device = f"cuda:{gpu_ids[0]}" if gpu_ids[0] >= 0 else "cpu"
            worker_args.selected_wsi_paths = shards[0]
            worker_entrypoint(worker_args)
            return

        ctx = _pick_mp_context()
        processes: List[tuple[int, mp.Process]] = []

        for shard_idx, gpu_id in enumerate(gpu_ids):
            if not shards[shard_idx]:
                continue

            worker_args = argparse.Namespace(**vars(args))
            worker_args.gpu = gpu_id
            worker_args.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu"
            worker_args.selected_wsi_paths = shards[shard_idx]

            process = ctx.Process(target=worker_entrypoint, args=(worker_args,))
            process.start()
            processes.append((gpu_id, process))

        failed_workers = []
        for gpu_id, process in processes:
            process.join()
            if process.exitcode != 0:
                failed_workers.append((gpu_id, process.exitcode))

        if failed_workers:
            failures = ", ".join(f"gpu={gpu_id}, exit={exit_code}" for gpu_id, exit_code in failed_workers)
            raise RuntimeError(f"One or more workers failed: {failures}")
    except Exception as e:
        run_status = "error"
        run_error = str(e)
        raise
    finally:
        try:
            finalize_run(args.job_dir, run_id, status=run_status, error=run_error)
        except Exception:
            pass


if __name__ == "__main__":
    main()
