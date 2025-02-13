"""
Example usage:

```
python run_batch_of_slides.py --task all --wsi_dir output/wsis --job_dir output --patch_encoder uni_v1 --mag 20 --patch_size 256
```

"""

import argparse
from trident.Processor import Processor


def parse_arguments():
    """
    Parse command-line arguments for the Trident processing script.
    """
    parser = argparse.ArgumentParser(description='Run Trident')
    # Generic arguments 
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use for processing tasks')
    parser.add_argument('--task', type=str, default='seg', 
                        choices=['cache', 'seg', 'coords', 'feat', 'all'], 
                        help='Task to run: cache, seg (segmentation), coords (save tissue coordinates), img (save tissue images), feat (extract features)')
    parser.add_argument('--job_dir', type=str, required=True, help='Directory to store outputs')
    parser.add_argument('--wsi_dir', type=str, required=True, 
                        help='Directory containing WSI files (no nesting allowed)')
    parser.add_argument('--wsi_ext', type=str, nargs='+', default=None, 
                        help='List of allowed file extensions for WSI files')
    parser.add_argument('--custom_mpp_keys', type=str, nargs='+', default=None,
                    help='Custom keys used to store the resolution as MPP (micron per pixel) in your list of whole-slide image.')
    parser.add_argument('--custom_list_of_wsis', type=str, default=None,
                    help='Custom list of WSIs specified in a csv file.')
    parser.add_argument('--wsi_cache', type=str, default=None, 
                        help='Directory to copy slides to for local processing')
    parser.add_argument('--clear_cache', action='store_true', default=False, 
                        help='Delete slides from cache after processing')
    parser.add_argument('--skip_errors', action='store_true', default=False, 
                        help='Skip errored slides and continue processing')
    # Segmentation arguments 
    parser.add_argument('--fast_seg', action='store_true', default=False, 
                        help='Run tissue segmentation at 5x (faster) or 10x (default)')
    parser.add_argument('--remove_holes', action='store_true', default=False, 
                        help='Do you want to remove holes?')
    # Patching arguments
    parser.add_argument('--mag', type=int, choices=[5, 10, 20, 40, 80], default=20, 
                        help='Magnification for coords/features extraction')
    parser.add_argument('--patch_size', type=int, default=512, 
                        help='Patch size for coords/image extraction')
    parser.add_argument('--overlap', type=int, default=0, 
                        help='Absolute overlap for patching in pixels. Defaults to 0. ')
    parser.add_argument('--min_tissue_proportion', type=float, default=0., 
                        help='Minimum proportion of the patch under tissue to be kept. Between 0. and 1.0. Defaults to 0. ')
    parser.add_argument('--coords_dir', type=str, default=None, 
                        help='Directory to save/restore tissue coordinates')
    # Feature extraction arguments 
    parser.add_argument('--patch_encoder', type=str, default='conch_v15', 
                        choices=['conch_v1', 'uni_v1', 'uni_v2', 'ctranspath', 'phikon', 
                                 'resnet50', 'gigapath', 'virchow', 'virchow2', 
                                 'hoptimus0', 'phikon_v2', 'conch_v15', 'musk', 'hibou_vitl',
                                 'kaiko-vits8', 'kaiko-vits16', 'kaiko-vitb8', 'kaiko-vitb16',
                                 'kaiko-vitl14', 'lunit-vits8'],
                        help='Patch encoder to use')
    parser.add_argument('--slide_encoder', type=str, default=None, 
                        choices=['threads', 'titan', 'prism', 'gigapath', 'chief', 'madeleine',
                                 'mean-virchow', 'mean-virchow2', 'mean-conch_v1', 'mean-conch_v15', 'mean-ctranspath',
                                 'mean-gigapath', 'mean-resnet50', 'mean-hoptimus0', 'mean-phikon', 'mean-phikon_v2',
                                 'mean-musk', 'mean-uni_v1', 'mean-uni_v2',  
                                 ], 
                        help='Slide encoder to use')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for feature extraction. Defaults to 32.')
    return parser.parse_args()

def initialize_processor(args):
    """
    Initialize the Trident Processor with the given arguments.
    """
    return Processor(
        job_dir=args.job_dir,
        wsi_source=args.wsi_dir,
        wsi_ext=args.wsi_ext,
        wsi_cache=args.wsi_cache,
        clear_cache=args.clear_cache,
        skip_errors=args.skip_errors,
        custom_mpp_keys=args.custom_mpp_keys,
        custom_list_of_wsis=args.custom_list_of_wsis
    )

def run_task(processor, args):
    """
    Execute the specified task using the Trident Processor.
    """
    if args.task == 'cache':
        processor.populate_cache()
    elif args.task == 'seg':
        # Minimal example for tissue segmentation:
        # python run_batch_of_slides.py --task seg --wsi_dir wsis --job_dir trident_processed --gpu 0
        from trident.segmentation_models.load import segmentation_model_factory
        segmentation_model = segmentation_model_factory('hest', device=f'cuda:{args.gpu}')
        processor.run_segmentation_job(
            segmentation_model,
            seg_mag=5 if args.fast_seg else 10,
            holes_are_tissue= not args.remove_holes,
        )
    elif args.task == 'coords':
        # Minimal example for tissue patching:
        # python run_batch_of_slides.py --task coords --wsi_dir wsis --job_dir trident_processed --mag 20 --patch_size 256
        processor.run_patching_job(
            target_magnification=args.mag,
            patch_size=args.patch_size,
            overlap=args.overlap,
            saveto=args.coords_dir,
            min_tissue_proportion=args.min_tissue_proportion
        )
    elif args.task == 'feat':
        if args.slide_encoder is None: # Run patch encoder:
            # Minimal example for feature extraction:
            # python run_batch_of_slides.py --task feat --wsi_dir wsis --job_dir trident_processed --patch_encoder uni_v1 --mag 20 --patch_size 256
            from trident.patch_encoder_models.load import encoder_factory
            encoder = encoder_factory(args.patch_encoder)
            processor.run_patch_feature_extraction_job(
                coords_dir=args.coords_dir or f'{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap',
                patch_encoder=encoder,
                device=f'cuda:{args.gpu}',
                saveas='h5',
                batch_limit=args.batch_size,
            )
        else:
            # Minimal example for feature extraction:
            # python run_batch_of_slides.py --task feat --wsi_dir wsis --job_dir trident_processed --slide_encoder threads --mag 20 --patch_size 256
            from trident.slide_encoder_models.load import encoder_factory
            encoder = encoder_factory(args.slide_encoder)
            processor.run_slide_feature_extraction_job(
                slide_encoder=encoder,
                coords_dir=args.coords_dir or f'{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap',
                device=f'cuda:{args.gpu}',
                saveas='h5',
            )
    else:
        raise ValueError(f'Invalid task: {args.task}')

def main():
    args = parse_arguments()
    processor = initialize_processor(args)

    if args.task == 'all':
        args.task = 'seg'
        run_task(processor, args)
        args.task = 'coords'
        run_task(processor, args)
        args.task = 'feat'
        run_task(processor, args)
    else:
        run_task(processor, args)

if __name__ == "__main__":
    main()
