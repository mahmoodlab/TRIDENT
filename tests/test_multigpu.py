import unittest
import sys
import os
import time
import shutil
import tempfile
from unittest.mock import patch
import torch

import run_batch_of_slides as trident_run

# ==========================================
# USER CONFIGURATION
# ==========================================
# Path to a directory containing WSI files
WSI_DIR = "/path/to/wsi/files"

# GPU IDs to use (e.g., [0, 1])
GPU_IDS = [0, 1] if torch.cuda.is_available() else [-1]

# Trident task and model
TASK = "all"
PATCH_ENCODER = "conch_v15"

# Inference and patch settings
BATCH_SIZE = 32
MAGNIFICATION = 20
PATCH_SIZE = 512

# Cache settings
CACHE_BATCH_SIZE = 2

# Misc
SCRIPT_NAME = "trident_run.py"
# ==========================================


class TestTridentProfiling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(WSI_DIR):
            raise FileNotFoundError(
                f"Please update WSI_DIR in the test script. Path not found: {WSI_DIR}"
            )

        cls.task = TASK
        cls.encoder = PATCH_ENCODER

        print(f"\n=== Starting Profiling on {len(GPU_IDS)} GPU(s) ===")
        print(f"Source: {WSI_DIR}")

    def setUp(self):
        # Fresh temp directories for each test
        self.test_dir = tempfile.mkdtemp()
        self.job_dir = os.path.join(self.test_dir, "job_output")
        self.cache_dir = os.path.join(self.test_dir, "wsi_cache")
        os.makedirs(self.job_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def run_trident_scenario(self, run_name, gpus, use_cache=False):
        """
        Helper to run the trident main function with specific arguments.
        """
        print(f"\n[Running Scenario]: {run_name}")

        args = [
            SCRIPT_NAME,
            "--wsi_dir", WSI_DIR,
            "--job_dir", self.job_dir,
            "--task", self.task,
            "--patch_encoder", self.encoder,
            "--batch_size", str(BATCH_SIZE),
            "--mag", str(MAGNIFICATION),
            "--patch_size", str(PATCH_SIZE),
            "--gpus",
        ] + [str(g) for g in gpus]

        if use_cache:
            args.extend([
                "--wsi_cache", self.cache_dir,
                "--cache_batch_size", str(CACHE_BATCH_SIZE),
            ])

        with patch.object(sys, "argv", args):
            start_time = time.time()
            try:
                trident_run.main()
            except SystemExit as e:
                if e.code != 0:
                    self.fail(f"Trident exited with error code {e.code}")
            except Exception as e:
                self.fail(f"Trident crashed: {e}")
            end_time = time.time()

        # Remove job output directory after each run
        if os.path.exists(self.job_dir):
            shutil.rmtree(self.job_dir)
            os.makedirs(self.job_dir, exist_ok=True)

        duration = end_time - start_time
        print(f"[Completed]: {run_name} in {duration:.2f} seconds")
        return duration

    def test_benchmark_scenarios(self):
        """
        Run 4 scenarios (Single/Multi GPU × Cache/NoCache) and print a comparison.
        """
        results = {}

        # 1. Single GPU - No Cache
        results["1GPU_NoCache"] = self.run_trident_scenario(
            "Single GPU | Direct Read",
            gpus=[GPU_IDS[0]],
            use_cache=False,
        )

        # 2. Single GPU - With Cache
        results["1GPU_Cache"] = self.run_trident_scenario(
            "Single GPU | Cached Read",
            gpus=[GPU_IDS[0]],
            use_cache=True,
        )

        # 3–4. Multi-GPU only if >1 GPU configured
        if len(GPU_IDS) > 1:
            results["MultiGPU_NoCache"] = self.run_trident_scenario(
                f"{len(GPU_IDS)} GPUs | Direct Read",
                gpus=GPU_IDS,
                use_cache=False,
            )

            results["MultiGPU_Cache"] = self.run_trident_scenario(
                f"{len(GPU_IDS)} GPUs | Cached Read",
                gpus=GPU_IDS,
                use_cache=True,
            )

        # Report
        print("\n" + "=" * 40)
        print(f"{'SCENARIO':<25} | {'TIME (s)':<10}")
        print("-" * 40)
        for name, duration in results.items():
            print(f"{name:<25} | {duration:<10.2f}")
        print("=" * 40)

        self.assertTrue(all(t > 0 for t in results.values()))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    unittest.main()