"""
Example usage:

Standard mode:
```
python run_batch_of_slides.py --task all --wsi_dir output/wsis --job_dir output --patch_encoder uni_v1 --mag 20 --patch_size 256
```

Nested directory mode (preserves subdirectory structure):
```
python run_batch_of_slides.py --task feat --nested_dir /data/rag/data/WSIs/histai --output_dir /data/rag/data/WSIs --patch_encoder uni_v1 --mag 20 --patch_size 256
```
This will process WSIs from /data/rag/data/WSIs/histai/breast/case_xxx/ and save features to /data/rag/data/WSIs/breast/case_xxx/

"""

import os
import argparse
import random
import torch
import pandas as pd
from typing import Any

from trident import Processor, load_wsi
from trident.patch_encoder_models import encoder_registry as patch_encoder_registry
from trident.slide_encoder_models import encoder_registry as slide_encoder_registry


def build_parser() -> argparse.ArgumentParser:
    """
    Parse command-line arguments for the Trident processing script.

    Returns:
        argparse.ArgumentParser: Configured argument parser with all Trident processing options.
    """
    parser = argparse.ArgumentParser(description="Run Trident")

    # Generic arguments
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU index to use for processing tasks."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="seg",
        choices=["seg", "coords", "feat", "all"],
        help="Task to run: seg (segmentation), coords (save tissue coordinates), img (save tissue images), feat (extract features).",
    )
    parser.add_argument(
        "--job_dir", type=str, default=None, help="Directory to store outputs."
    )
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        default=False,
        help="Skip errored slides and continue processing.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of workers. Set to 0 to use main process.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size used for segmentation and feature extraction. Will be override by"
        "`seg_batch_size` and `feat_batch_size` if you want to use different ones. Defaults to 64.",
    )

    # Caching argument for fast WSI processing
    parser.add_argument(
        "--wsi_cache",
        type=str,
        default=None,
        help="Path to a local cache (e.g., SSD) used to speed up access to WSIs stored on slower drives (e.g., HDD).",
    )
    parser.add_argument(
        "--cache_batch_size",
        type=int,
        default=32,
        help="Maximum number of slides to cache locally at once. Helps control disk usage.",
    )

    # Slide-related arguments
    parser.add_argument(
        "--wsi_dir",
        type=str,
        required=False,
        help="Directory containing WSI files (no nesting allowed).",
    )
    parser.add_argument(
        "--nested_dir",
        type=str,
        default=None,
        help="Base directory containing nested WSI files. Use with --output_dir to process nested structures.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base output directory for nested processing. Subdirectory structure from --nested_dir will be preserved.",
    )
    parser.add_argument(
        "--wsi_ext",
        type=str,
        nargs="+",
        default=None,
        help="List of allowed file extensions for WSI files.",
    )
    parser.add_argument(
        "--custom_mpp_keys",
        type=str,
        nargs="+",
        default=None,
        help="Custom keys used to store the resolution as MPP (micron per pixel) in your list of whole-slide image.",
    )
    parser.add_argument(
        "--use_previous_mpp",
        action="store_true",
        default=False,
        help="If a slide is missing MPP metadata, use the MPP from the previously processed slide. Useful for batch processing slides with inconsistent metadata.",
    )
    parser.add_argument(
        "--hardcode_mag",
        type=int,
        default=None,
        choices=[5, 10, 20, 40, 80],
        help="Hardcode the magnification level instead of detecting from slide metadata. Converts to MPP as 10/mag.",
    )
    parser.add_argument(
        "--custom_list_of_wsis",
        type=str,
        default=None,
        help="Custom list of WSIs specified in a csv file.",
    )
    parser.add_argument(
        "--reader_type",
        type=str,
        choices=["openslide", "image", "cucim", "sdpc"],
        default=None,
        help='Force the use of a specific WSI image reader. Options are ["openslide", "image", "cucim", "sdpc"]. Defaults to None (auto-determine which reader to use).',
    )
    parser.add_argument(
        "--search_nested",
        action="store_true",
        help=(
            "If set, recursively search for whole-slide images (WSIs) within all subdirectories of "
            "`wsi_source`. Uses `os.walk` to include slides from nested folders. "
            "This allows processing of datasets organized in hierarchical structures. "
            "Defaults to False (only top-level slides are included)."
        ),
    )
    parser.add_argument(
        "--randomize-order",
        action="store_true",
        default=False,
        help="Shuffle the order slides are processed (nested subfolders, cache batches, or sequential).",
    )
    # Segmentation arguments
    parser.add_argument(
        "--segmenter",
        type=str,
        default="hest",
        choices=["hest", "grandqc"],
        help="Type of tissue vs background segmenter. Options are HEST or GrandQC.",
    )
    parser.add_argument(
        "--seg_conf_thresh",
        type=float,
        default=0.5,
        help="Confidence threshold to apply to binarize segmentation predictions. Lower this threhsold to retain more tissue. Defaults to 0.5. Try 0.4 as 2nd option.",
    )
    parser.add_argument(
        "--remove_holes",
        action="store_true",
        default=False,
        help="Do you want to remove holes?",
    )
    parser.add_argument(
        "--remove_artifacts",
        action="store_true",
        default=False,
        help="Do you want to run an additional model to remove artifacts (including penmarks, blurs, stains, etc.)?",
    )
    parser.add_argument(
        "--remove_penmarks",
        action="store_true",
        default=False,
        help="Do you want to run an additional model to remove penmarks?",
    )
    parser.add_argument(
        "--seg_batch_size",
        type=int,
        default=None,
        help="Batch size for segmentation. Defaults to None (use `batch_size` argument instead).",
    )

    # Patching arguments
    parser.add_argument(
        '--dump_patches', action='store_true', default=False,
        help='Dump patch images for each slide.'
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
        "--mag",
        type=int,
        choices=[5, 10, 20, 40, 80],
        default=20,
        help="Magnification for coords/features extraction.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=512,
        help="Patch size for coords/image extraction.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Absolute overlap for patching in pixels. Defaults to 0.",
    )
    parser.add_argument(
        "--min_tissue_proportion",
        type=float,
        default=0.0,
        help="Minimum proportion of the patch under tissue to be kept. Between 0. and 1.0. Defaults to 0.",
    )
    parser.add_argument(
        "--coords_dir",
        type=str,
        default=None,
        help="Directory to save/restore tissue coordinates.",
    )

    # Feature extraction arguments
    parser.add_argument(
        "--patch_encoder_ckpt_path",
        type=str,
        default=None,
        help=(
            "Optional local path to a patch encoder checkpoint (.pt, .pth, .bin, or .safetensors). "
            "This is only needed in offline environments (e.g., compute clusters without internet). "
            "If not provided, models are downloaded automatically from Hugging Face. "
            "You can also specify local paths via the model registry at "
            "`./trident/patch_encoder_models/local_ckpts.json`."
        ),
    )
    parser.add_argument(
        "--slide_encoder",
        type=str,
        default=None,
        choices=slide_encoder_registry.keys(),
        help="Slide encoder to use",
    )
    parser.add_argument(
        "--feat_batch_size",
        type=int,
        default=None,
        help="Batch size for feature extraction. Defaults to None (use `batch_size` argument instead).",
    )
    return parser


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments and return the parsed namespace.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    return build_parser().parse_args()


def generate_help_text() -> str:
    """
    Generate the command-line help text for documentation purposes.
    
    Returns:
        str: The full help message string from the argument parser.

    Returns
    -------
    str
        The full help message string from the argument parser.
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
    default_mpp = 10 / args.hardcode_mag if args.hardcode_mag is not None else None

    return Processor(
        job_dir=args.job_dir,
        wsi_source=args.wsi_dir,
        wsi_ext=args.wsi_ext,
        wsi_cache=args.wsi_cache,
        skip_errors=args.skip_errors,
        custom_mpp_keys=args.custom_mpp_keys,
        default_mpp=default_mpp,
        custom_list_of_wsis=args.custom_list_of_wsis,
        max_workers=args.max_workers,
        reader_type=args.reader_type,
        search_nested=args.search_nested,
        use_previous_mpp=args.use_previous_mpp,
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
    print(
        f"[run_batch_of_slides] Running task '{args.task}' "
        f"(WSI objects were already built by Processor(); task-specific work starts now)."
    )

    if args.task == "seg":
        from trident.segmentation_models.load import segmentation_model_factory

        seg_device = "cpu" if args.segmenter == "otsu" else f"cuda:{args.gpu}"

        # instantiate segmentation model and artifact remover if requested by user
        segmentation_model = segmentation_model_factory(
            args.segmenter,
            confidence_thresh=args.seg_conf_thresh,
        )
        if args.remove_artifacts or args.remove_penmarks:
            artifact_remover_model = segmentation_model_factory(
                "grandqc_artifact",
                remove_penmarks_only=args.remove_penmarks and not args.remove_artifacts,
            )
        else:
            artifact_remover_model = None

        # run segmentation
        processor.run_segmentation_job(
            segmentation_model,
            seg_mag=segmentation_model.target_mag,
            holes_are_tissue=not args.remove_holes,
            artifact_remover_model=artifact_remover_model,
            batch_size=args.seg_batch_size if args.seg_batch_size is not None else args.batch_size,
            device=seg_device,
        )
    elif args.task == "coords":
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
    elif args.task == "feat":
        if args.slide_encoder is None:
            from trident.patch_encoder_models.load import encoder_factory

            encoder = encoder_factory(
                args.patch_encoder, weights_path=args.patch_encoder_ckpt_path
            )
            processor.run_patch_feature_extraction_job(
                coords_dir=args.coords_dir
                or f"{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap",
                patch_encoder=encoder,
                device=f"cuda:{args.gpu}",
                saveas="h5",
                batch_limit=args.feat_batch_size
                if args.feat_batch_size is not None
                else args.batch_size,
            )
        else:
            from trident.slide_encoder_models.load import encoder_factory

            encoder = encoder_factory(args.slide_encoder)
            processor.run_slide_feature_extraction_job(
                slide_encoder=encoder,
                coords_dir=args.coords_dir
                or f"{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap",
                device=f"cuda:{args.gpu}",
                saveas="h5",
                batch_limit=args.feat_batch_size
                if args.feat_batch_size is not None
                else args.batch_size,
            )
    else:
        raise ValueError(f"Invalid task: {args.task}")


def run_nested_processing(args: argparse.Namespace) -> None:
    """
    Process WSIs from a nested directory structure, preserving subdirectory layout in output.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments. Must include nested_dir and output_dir.
    """
    from trident.IO import collect_valid_slides

    wsi_base = args.nested_dir
    output_base = args.output_dir

    full_paths, rel_paths = collect_valid_slides(
        wsi_dir=wsi_base,
        custom_list_path=args.custom_list_of_wsis,
        wsi_ext=args.wsi_ext,
        search_nested=True,
        max_workers=args.max_workers,
        return_relative_paths=True,
    )

    print(f"[NESTED] Found {len(full_paths)} valid slides in {wsi_base}.")

    grouped_slides = {}
    for full_path, rel_path in zip(full_paths, rel_paths):
        parent_rel = os.path.dirname(rel_path)
        if parent_rel not in grouped_slides:
            grouped_slides[parent_rel] = []
        grouped_slides[parent_rel].append((full_path, os.path.basename(full_path)))

    print(f"[NESTED] Processing {len(grouped_slides)} subdirectories.")

    nested_groups = list(grouped_slides.items())
    if args.randomize_order:
        random.shuffle(nested_groups)

    for subfolder, slides in nested_groups:
        if args.randomize_order:
            random.shuffle(slides)
        job_dir = os.path.join(output_base, subfolder)
        os.makedirs(job_dir, exist_ok=True)

        local_args = argparse.Namespace(**vars(args))
        local_args.wsi_dir = os.path.dirname(slides[0][0])
        local_args.job_dir = job_dir
        local_args.nested_dir = None
        local_args.output_dir = None
        local_args.search_nested = False
        local_args.custom_list_of_wsis = None

        if len(slides) > 1:
            temp_csv = os.path.join(job_dir, "_temp_slide_list.csv")
            pd.DataFrame({"wsi": [s[1] for s in slides]}).to_csv(temp_csv, index=False)
            local_args.custom_list_of_wsis = temp_csv

        print(f"\n[NESTED] Processing subfolder: {subfolder} ({len(slides)} slides)")
        print(f"[NESTED] Job directory: {job_dir}")

        processor = initialize_processor(local_args)

        tasks = (
            ["seg", "coords", "feat"] if local_args.task == "all" else [local_args.task]
        )
        for task_name in tasks:
            local_args.task = task_name
            run_task(processor, local_args)

        if (
            len(slides) > 1
            and local_args.custom_list_of_wsis
            and os.path.exists(local_args.custom_list_of_wsis)
        ):
            os.remove(local_args.custom_list_of_wsis)


def main() -> None:
    """
    Main entry point for the Trident batch processing script.

    Handles both sequential and parallel processing modes based on whether
    WSI caching is enabled. Supports segmentation, coordinate extraction,
    and feature extraction tasks.
    """

    args = parse_arguments()
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    if args.nested_dir is not None:
        if args.output_dir is None:
            raise ValueError("--output_dir is required when using --nested_dir")
        run_nested_processing(args)
        return

    if args.wsi_dir is None:
        raise ValueError("Either --wsi_dir or --nested_dir must be provided")

    if args.job_dir is None:
        raise ValueError("--job_dir is required when not using --nested_dir")

    if args.wsi_cache:
        # === Parallel pipeline with caching ===

        from queue import Queue
        from threading import Thread

        from trident.Concurrency import batch_producer, batch_consumer, cache_batch
        from trident.IO import collect_valid_slides

        queue = Queue(maxsize=1)
        valid_slides = collect_valid_slides(
            wsi_dir=args.wsi_dir,
            custom_list_path=args.custom_list_of_wsis,
            wsi_ext=args.wsi_ext,
            search_nested=args.search_nested,
            max_workers=args.max_workers,
        )
        if args.randomize_order:
            random.shuffle(valid_slides)
        print(f"[MAIN] Found {len(valid_slides)} valid slides in {args.wsi_dir}.")

        # Print metadata keys for the first slide
        if len(valid_slides) > 0:
            first_slide_path = valid_slides[0]
            print("\n" + "=" * 80)
            print(
                f"Metadata keys for first slide: {os.path.basename(first_slide_path)}"
            )
            print("=" * 80)
            try:
                first_slide = load_wsi(
                    slide_path=first_slide_path,
                    custom_mpp_keys=args.custom_mpp_keys,
                    reader_type=args.reader_type,
                    lazy_init=False,
                )
                if (
                    hasattr(first_slide, "properties")
                    and first_slide.properties is not None
                ):
                    for key, value in sorted(first_slide.properties.items()):
                        print(f"  {key}: {value}")
                else:
                    print("  No properties available")
            except Exception as e:
                print(f"  Error loading first slide: {e}")
            print("=" * 80 + "\n")

        warm = valid_slides[: args.cache_batch_size]
        warmup_dir = os.path.join(args.wsi_cache, "batch_0")
        print(f"[MAIN] Warmup caching batch: {warmup_dir}")
        cache_batch(warm, warmup_dir)
        queue.put(0)

        def processor_factory(wsi_dir: str) -> Processor:
            local_args = argparse.Namespace(**vars(args))
            local_args.wsi_dir = wsi_dir
            local_args.wsi_cache = None
            local_args.custom_list_of_wsis = None
            local_args.search_nested = False
            # use_previous_mpp is already in args, so it will be passed through
            return initialize_processor(local_args)

        def run_task_fn(processor: Processor, task_name: str) -> None:
            args.task = task_name
            run_task(processor, args)

        producer = Thread(
            target=batch_producer,
            args=(
                queue,
                valid_slides,
                args.cache_batch_size,
                args.cache_batch_size,
                args.wsi_cache,
            ),
        )

        consumer = Thread(
            target=batch_consumer,
            args=(queue, args.task, args.wsi_cache, processor_factory, run_task_fn),
        )

        print("[MAIN] Starting producer and consumer threads.")
        producer.start()
        consumer.start()
        producer.join()
        consumer.join()
    else:
        # === Sequential mode ===
        processor = initialize_processor(args)
        if args.randomize_order:
            random.shuffle(processor.slide_specs)

        # Print metadata keys for the first slide (open once; catalog does not keep WSIs in memory)
        if len(processor.slide_specs) > 0:
            from trident import load_wsi

            s0 = processor.slide_specs[0]
            first_slide = load_wsi(
                slide_path=s0["abs_path"],
                name=f"{s0['stem']}{s0['ext']}",
                tissue_seg_path=s0["tissue_seg_path"],
                custom_mpp_keys=args.custom_mpp_keys,
                mpp=s0["mpp"],
                max_workers=args.max_workers,
                reader_type=args.reader_type,
                lazy_init=False,
            )
            try:
                print("\n" + "=" * 80)
                print(f"Metadata keys for first slide: {first_slide.name}")
                print("=" * 80)
                if (
                    hasattr(first_slide, "properties")
                    and first_slide.properties is not None
                ):
                    for key, value in sorted(first_slide.properties.items()):
                        print(f"  {key}: {value}")
                else:
                    print("  No properties available (slide may not be initialized yet)")
                print("=" * 80 + "\n")
            finally:
                first_slide.release()

        tasks = ["seg", "coords", "feat"] if args.task == "all" else [args.task]
        for task_name in tasks:
            args.task = task_name
            run_task(processor, args)


if __name__ == "__main__":
    main()
