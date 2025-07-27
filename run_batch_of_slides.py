"""
Example usage:

```
python run_batch_of_slides.py --task all --wsi_dir output/wsis --job_dir output --patch_encoder uni_v1 --mag 20 --patch_size 256
```

"""
import os
import argparse
import torch

from trident import Processor 


def build_parser():
    """
    Parse command-line arguments for the Trident processing script.
    """
    parser = argparse.ArgumentParser(description='Run Trident')

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
                        help='Directory containing WSI files (no nesting allowed).')
    parser.add_argument('--wsi_ext', type=str, nargs='+', default=None, 
                        help='List of allowed file extensions for WSI files.')
    parser.add_argument('--custom_mpp_keys', type=str, nargs='+', default=None,
                    help='Custom keys used to store the resolution as MPP (micron per pixel) in your list of whole-slide image.')
    parser.add_argument('--custom_list_of_wsis', type=str, default=None,
                    help='Custom list of WSIs specified in a csv file.')
    parser.add_argument('--reader_type', type=str, choices=['openslide', 'image', 'cucim'], default=None,
                    help='Force the use of a specific WSI image reader. Options are ["openslide", "image", "cucim"]. Defaults to None (auto-determine which reader to use).')
    parser.add_argument("--search_nested", action="store_true",
                        help=("If set, recursively search for whole-slide images (WSIs) within all subdirectories of "
                              "`wsi_source`. Uses `os.walk` to include slides from nested folders. "
                              "This allows processing of datasets organized in hierarchical structures. "
                              "Defaults to False (only top-level slides are included)."))
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
                        choices=['conch_v1', 'uni_v1', 'uni_v2', 'ctranspath', 'phikon', 
                                 'resnet50', 'gigapath', 'virchow', 'virchow2', 
                                 'hoptimus0', 'hoptimus1', 'phikon_v2', 'conch_v15', 'musk', 'hibou_l',
                                 'kaiko-vits8', 'kaiko-vits16', 'kaiko-vitb8', 'kaiko-vitb16',
                                 'kaiko-vitl14', 'lunit-vits8', 'midnight12k'],
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
                        choices=['threads', 'titan', 'prism', 'gigapath', 'chief', 'madeleine', 'feather',
                                 'mean-virchow', 'mean-virchow2', 'mean-conch_v1', 'mean-conch_v15', 'mean-ctranspath',
                                 'mean-gigapath', 'mean-resnet50', 'mean-hoptimus0', 'mean-phikon', 'mean-phikon_v2',
                                 'mean-musk', 'mean-uni_v1', 'mean-uni_v2', 
                                 ], 
                        help='Slide encoder to use')
    parser.add_argument('--feat_batch_size', type=int, default=None, 
                        help='Batch size for feature extraction. Defaults to None (use `batch_size` argument instead).')
    return parser


def parse_arguments():
    return build_parser().parse_args()


def generate_help_text() -> str:
    """
    Generate the command-line help text for documentation purposes.
    
    Returns:
        str: The full help message string from the argument parser.
    """
    parser = build_parser()
    return parser.format_help()


def initialize_processor(args):
    """
    Initialize the Trident Processor with arguments set in `run_batch_of_slides`.
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
    )


def run_task(processor, args):
    """
    Execute the specified task using the Trident Processor.
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
        if args.slide_encoder is None: 
            from trident.patch_encoder_models.load import encoder_factory
            encoder = encoder_factory(args.patch_encoder, weights_path=args.patch_encoder_ckpt_path)
            processor.run_patch_feature_extraction_job(
                coords_dir=args.coords_dir or f'{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap',
                patch_encoder=encoder,
                device=f'cuda:{args.gpu}',
                saveas='h5',
                batch_limit=args.feat_batch_size if args.feat_batch_size is not None else args.batch_size,
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
            )
    else:
        raise ValueError(f'Invalid task: {args.task}')


def main():

    args = parse_arguments()
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

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
            max_workers=args.max_workers
        )
        print(f"[MAIN] Found {len(valid_slides)} valid slides in {args.wsi_dir}.")

        warm = valid_slides[:args.cache_batch_size]
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
            return initialize_processor(local_args)

        def run_task_fn(processor: Processor, task_name: str):
            args.task = task_name
            run_task(processor, args)

        producer = Thread(target=batch_producer, args=(
            queue, valid_slides, args.cache_batch_size, args.cache_batch_size, args.wsi_cache
        ))

        consumer = Thread(target=batch_consumer, args=(
            queue, args.task, args.wsi_cache, processor_factory, run_task_fn
        ))

        print("[MAIN] Starting producer and consumer threads.")
        producer.start()
        consumer.start()
        producer.join()
        consumer.join()
    else:
        # === Sequential mode ===
        processor = initialize_processor(args)
        tasks = ['seg', 'coords', 'feat'] if args.task == 'all' else [args.task]
        for task_name in tasks:
            args.task = task_name
            run_task(processor, args)


if __name__ == "__main__":
    main()
