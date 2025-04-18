import os
import unittest
import sys
sys.path.append('../')
from trident import Processor
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory

from huggingface_hub import snapshot_download

"""
Unit tests for the Processor class, which is responsible for handling segmentation, patching, and feature extraction on whole slide images (WSIs).

Attributes:
    HF_REPO (str): The Hugging Face repository ID for downloading test data.
    TEST_OUTPUT_DIR (str): The directory where test outputs will be stored.
    TEST_WSI_EXT (list): List of file extensions for WSIs.
    TEST_GPU_INDEX (int): The index of the GPU to be used for testing.
    TEST_PATCH_ENCODER (str): The name of the patch encoder model to be used.
    TEST_MAG (int): The target magnification for patching.
    TEST_PATCH_SIZE (int): The size of the patches to be extracted.
    TEST_OVERLAP (int): The overlap between patches.

Methods:
    setUpClass(cls): Set up the test environment by creating directories and preparing mock data.
    setUp(self): Initialize the Processor instance for each test.
    test_tissue_processing(self): Test the processing tasks including segmentation, patching, and feature extraction.
    tearDown(self): Clean up after each test.
    tearDownClass(cls): Remove the test output directory after all tests are done.
"""

class TestProcessor(unittest.TestCase):
    HF_REPO = "MahmoodLab/unit-testing"
    TEST_OUTPUT_DIR = "test_processor_output/"
    TEST_WSI_EXT = [".svs", ".tif"]
    TEST_GPU_INDEX = 0
    TEST_PATCH_ENCODER = "uni_v1"
    TEST_MAG = 20
    TEST_PATCH_SIZE = 256
    TEST_OVERLAP = 0

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by creating directories and preparing mock data.
        """
        os.makedirs(cls.TEST_OUTPUT_DIR, exist_ok=True)
        cls.local_wsi_dir = snapshot_download(
            repo_id=cls.HF_REPO,
            repo_type='dataset',
            local_dir=os.path.join(cls.TEST_OUTPUT_DIR, 'wsis'),
            allow_patterns=['*svs']
        )

    def setUp(self):
        """
        Initialize the Processor instance for each test.
        """
        self.custom_list_of_wsis = snapshot_download(
            repo_id=self.HF_REPO,
            repo_type='dataset',
            local_dir=os.path.join(self.TEST_OUTPUT_DIR),
            allow_patterns=['*csv']
        )

    def test_processor_with_wsis(self):
        """
        Test the process constructor when processing a custom list of WSIs.
        """

        self.processor = Processor(
            job_dir=self.TEST_OUTPUT_DIR,
            wsi_source=os.path.join(TestProcessor.TEST_OUTPUT_DIR),
            wsi_ext=self.TEST_WSI_EXT,
            custom_list_of_wsis=os.path.join(self.custom_list_of_wsis, 'valid_list_of_wsis.csv')
        )

    def test_tissue_processing(self):
        """
        Test the processing tasks end-to-end including segmentation, patching, and feature extraction on a set of real WSIs.
        """

        self.processor = Processor(
            job_dir=self.TEST_OUTPUT_DIR,
            wsi_source=os.path.join(TestProcessor.TEST_OUTPUT_DIR, 'wsis'),
            wsi_ext=self.TEST_WSI_EXT
        )

        segmentation_model = segmentation_model_factory('hest')
        self.processor.run_segmentation_job(
            segmentation_model=segmentation_model,
            seg_mag=5,
            device=f'cuda:{self.TEST_GPU_INDEX}'
        )
        output_dirs = ["contours", "contours_geojson"]
        for dir_name in output_dirs:
            dir_path = os.path.join(self.TEST_OUTPUT_DIR, dir_name)
            self.assertTrue(os.path.exists(dir_path), f"Segmentation output directory '{dir_name}' does not exist.")

        self.processor.run_patching_job(
            target_magnification=self.TEST_MAG,
            patch_size=self.TEST_PATCH_SIZE
        )
        coords_dir = f"{self.TEST_MAG}x_{self.TEST_PATCH_SIZE}px_{self.TEST_OVERLAP}px_overlap"

        encoder = encoder_factory(self.TEST_PATCH_ENCODER)
        encoder.eval()
        encoder.to(f"cuda:{self.TEST_GPU_INDEX}")
        self.processor.run_patch_feature_extraction_job(
            coords_dir=coords_dir,
            patch_encoder=encoder,
            device=f'cuda:{self.TEST_GPU_INDEX}',
            saveas="h5"
        )
        features_dir = os.path.join(self.TEST_OUTPUT_DIR, coords_dir, f"features_{self.TEST_PATCH_ENCODER}")
        self.assertTrue(len(os.listdir(features_dir)) > 0, "Feature extraction results are missing.")

    def tearDown(self):
        pass

    @classmethod
    def tearDownClass(cls):
        """
        Remove the test output directory.
        """
        pass
        # if os.path.exists(cls.TEST_OUTPUT_DIR):
        #     import shutil
        #     shutil.rmtree(cls.TEST_OUTPUT_DIR)

if __name__ == "__main__":
    unittest.main()
