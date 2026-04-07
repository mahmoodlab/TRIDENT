"""
Example usage:

```
python run_single_slide.py --slide_path output/wsis/394140.svs --job_dir output/ --mag 20 --patch_size 256
```

"""
import argparse
import os

from trident import load_wsi
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory
from trident.patch_encoder_models import encoder_registry as patch_encoder_registry


def parse_arguments():
    """
    Parse command-line arguments for processing a single WSI.
    """
    parser = argparse.ArgumentParser(description="Process a WSI from A to Z.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use for processing tasks")
    parser.add_argument("--slide_path", type=str, required=True, help="Path to the WSI file to process")
    parser.add_argument("--job_dir", type=str, required=True, help="Directory to store outputs")
    parser.add_argument('--patch_encoder', type=str, default='conch_v15', 
                        choices=patch_encoder_registry.keys(),
                        help='Patch encoder to use')
    parser.add_argument("--mag", type=int, choices=[5, 10, 20, 40], default=20,
                        help="Magnification at which patches/features are extracted")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size at which coords/features are extracted")
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
    parser.add_argument('--custom_mpp_keys', type=str, nargs='+', default=None,
                    help='Custom keys used to store the resolution as MPP (micron per pixel) in your list of whole-slide image.')
    parser.add_argument('--overlap', type=int, default=0, 
                        help='Absolute overlap for patching in pixels. Defaults to 0. ')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for feature extraction. Defaults to 32.')
    parser.add_argument(
        '--dump_patches', action='store_true', default=False,
        help='During the patching step, also dump patch images (PNGs) to disk.'
    )
    parser.add_argument(
        '--dump_patches_max', type=int, default=0,
        help='Max number of patch images to dump (0 = no limit).'
    )
    parser.add_argument(
        '--dump_patches_format', type=str, default="png", choices=["png", "jpg"],
        help='Patch image format to dump (png or jpg). Defaults to png.'
    )
    parser.add_argument(
        '--dump_patches_jpeg_quality', type=int, default=90,
        help='JPEG quality (1-100) when --dump_patches_format=jpg. Defaults to 90.'
    )
    return parser.parse_args()


def process_slide(args):
    """
    Process a single WSI by performing segmentation, patch extraction, and feature extraction sequentially.
    """

    # Initialize the WSI
    print(f"Processing slide: {args.slide_path}")
    with load_wsi(slide_path=args.slide_path, lazy_init=False, custom_mpp_keys=args.custom_mpp_keys) as slide:
        seg_device = "cpu" if args.segmenter == "otsu" else f"cuda:{args.gpu}"
        # Step 1: Tissue Segmentation
        print("Running tissue segmentation...")
        segmentation_model = segmentation_model_factory(
            model_name=args.segmenter,
            confidence_thresh=args.seg_conf_thresh,
        )
        if args.remove_artifacts or args.remove_penmarks:
            artifact_remover_model = segmentation_model_factory(
                'grandqc_artifact',
                remove_penmarks_only=args.remove_penmarks and not args.remove_artifacts
            )
        else:
            artifact_remover_model = None

        slide.segment_tissue(
            segmentation_model=segmentation_model,
            target_mag=segmentation_model.target_mag,
            job_dir=args.job_dir,
            device=seg_device,
            holes_are_tissue=not args.remove_holes
        )
        # additionally remove artifacts for better segmentation.
        if artifact_remover_model is not None:
            slide.segment_tissue(
                segmentation_model=artifact_remover_model,
                target_mag=artifact_remover_model.target_mag,
                holes_are_tissue=False,
                job_dir=args.job_dir
            )
        print(f"Tissue segmentation completed. Results saved to {os.path.join(args.job_dir, 'contours_geojson')} and {os.path.join(args.job_dir, 'contours')}")

        # Step 2: Tissue Coordinate Extraction (Patching)
        print("Extracting tissue coordinates...")
        save_coords = os.path.join(args.job_dir, f'{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap')

        coords_path = slide.extract_tissue_coords(
            target_mag=args.mag,
            patch_size=args.patch_size,
            save_coords=save_coords
        )
        print(f"Tissue coordinates extracted and saved to {coords_path}.")

        if getattr(args, "dump_patches", False):
            dumped_dir = slide.dump_patches(
                coords_path=coords_path,
                save_patches_dir=os.path.join(save_coords, "patch_images"),
                max_patches=getattr(args, "dump_patches_max", 0),
                image_format=getattr(args, "dump_patches_format", "png"),
                jpeg_quality=getattr(args, "dump_patches_jpeg_quality", 90),
            )
            print(f"Dumped patch images to {dumped_dir}.")

        # Step 3: Visualize patching
        viz_coords_path = slide.visualize_coords(
            coords_path=coords_path,
            save_patch_viz=os.path.join(save_coords, 'visualization'),
        )
        print(f"Tissue coordinates extracted and saved to {viz_coords_path}.")

        # Step 4: Feature Extraction
        print("Extracting features from patches...")
        encoder = encoder_factory(args.patch_encoder)
        encoder.eval()
        encoder.to(f"cuda:{args.gpu}")
        features_path = features_dir = os.path.join(save_coords, "features_{}".format(args.patch_encoder))
        slide.extract_patch_features(
            patch_encoder=encoder,
            coords_path=os.path.join(save_coords, 'patches', f'{slide.name}_patches.h5'),
            save_features=features_dir,
            device=f"cuda:{args.gpu}",
            batch_limit=args.batch_size
        )
        print(f"Feature extraction completed. Results saved to {features_path}")


def main():
    args = parse_arguments()
    process_slide(args)


if __name__ == "__main__":
    main()
