import os
import unittest
import torch  # Check for CUDA availability

import sys; sys.path.append('../')
from trident import load_wsi
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory

from huggingface_hub import snapshot_download


"""
Test the methods of the OpenSlideWSI object, i.e. bypassing the Processor class.
This is useful if you want to use the OpenSlideWSI class in a custom pipeline.
"""

class TestOpenSlideWSI(unittest.TestCase):
    HF_REPO = "MahmoodLab/unit-testing"
    TEST_SLIDE_FILENAMES = [
        "394140.svs",
        "TCGA-AN-A0XW-01Z-00-DX1.811E11E7-FA67-46BB-9BC6-1FD0106B789D.svs",
        "TCGA-B6-A0IJ-01Z-00-DX1.BF2E062F-06DA-4CA8-86C4-36674C035CAA.svs"
    ]
    TEST_OUTPUT_DIR = "test_single_slide_processing/"
    TEST_PATCH_ENCODER = "uni_v1"
    TEST_MAG = 20
    TEST_PATCH_SIZE = 256

    # Dynamically determine device
    TEST_DEVICE = f"cuda:0" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by downloading slides and creating directories.
        """
        os.makedirs(cls.TEST_OUTPUT_DIR, exist_ok=True)
        cls.local_wsi_dir = snapshot_download(
            repo_id=cls.HF_REPO,
            repo_type='dataset',
            local_dir=os.path.join(cls.TEST_OUTPUT_DIR, 'wsis'),
            allow_patterns=['*']
        )

    def test_integration(self):
        """
        Test all processing methods of OpenSlideWSI end-to-end with real slides.
        """
        for slide_filename in self.TEST_SLIDE_FILENAMES:
            with self.subTest(slide=slide_filename):
                slide_path = os.path.join(self.local_wsi_dir, slide_filename)
                slide = load_wsi(slide_path=slide_path, lazy_init=False)

                # Step 1: Tissue segmentation
                segmentation_model = segmentation_model_factory("hest")
                slide.segment_tissue(segmentation_model=segmentation_model, target_mag=10, job_dir=self.TEST_OUTPUT_DIR, device=self.TEST_DEVICE)

                # Step 2: Tissue coordinate extraction
                coords_path = slide.extract_tissue_coords(
                    target_mag=self.TEST_MAG,
                    patch_size=self.TEST_PATCH_SIZE,
                    save_coords=self.TEST_OUTPUT_DIR
                )

                # Step 3: Visualization
                viz_coords_path = slide.visualize_coords(
                    coords_path=coords_path,
                    save_patch_viz=os.path.join(self.TEST_OUTPUT_DIR, "visualization")
                )

                # Step 4: Feature extraction
                encoder = encoder_factory(self.TEST_PATCH_ENCODER)
                encoder.eval()
                encoder.to(self.TEST_DEVICE)
                features_dir = os.path.join(self.TEST_OUTPUT_DIR, f"features_{self.TEST_PATCH_ENCODER}")
                slide.extract_patch_features(
                    patch_encoder=encoder,
                    coords_path=coords_path,
                    save_features=features_dir,
                    device=self.TEST_DEVICE
                )

                # Verify outputs
                self.assertTrue(os.path.exists(os.path.join(self.TEST_OUTPUT_DIR, "contours_geojson")), "GDF contours were not saved.")
                self.assertTrue(os.path.exists(os.path.join(self.TEST_OUTPUT_DIR, "contours")), "Contours were not saved.")
                self.assertTrue(os.path.exists(coords_path), "Tissue coordinates file was not saved.")
                self.assertTrue(os.path.exists(viz_coords_path), "Visualization file was not saved.")
                self.assertTrue(os.path.exists(features_dir), "Feature extraction results were not saved.")

        expected_file_count = len(self.TEST_SLIDE_FILENAMES)
        output_dirs = [
            "visualization",
            "thumbnails",
            "patches",
            "contours",
            "contours_geojson",
            f"features_{self.TEST_PATCH_ENCODER}"
        ]

        for output_dir in output_dirs:
            dir_path = os.path.join(self.TEST_OUTPUT_DIR, output_dir)
            self.assertTrue(os.path.exists(dir_path), f"Directory '{output_dir}' does not exist.")
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            self.assertEqual(
                len(files), expected_file_count,
                f"Expected {expected_file_count} files in '{output_dir}', but found {len(files)}."
            )

if __name__ == "__main__":
    unittest.main()
