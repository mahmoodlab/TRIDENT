import unittest
import os
import tempfile
from unittest.mock import patch

import run_batch_of_slides as batch_mod


class _DummySegmentationModel:
    def __init__(self, target_mag=1.25):
        self.target_mag = target_mag


class _DummyProcessor:
    def __init__(self):
        self.calls = []

    def run_segmentation_job(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class TestRunBatchOfSlides(unittest.TestCase):
    def _base_args(self, segmenter: str):
        class Args:
            pass

        args = Args()
        args.task = "seg"
        args.segmenter = segmenter
        args.seg_conf_thresh = 0.5
        args.remove_holes = False
        args.remove_artifacts = False
        args.remove_penmarks = False
        args.seg_batch_size = None
        args.batch_size = 8
        args.gpu = 0
        return args

    def test_run_task_seg_uses_cpu_for_otsu(self):
        processor = _DummyProcessor()
        args = self._base_args("otsu")

        with patch("trident.segmentation_models.load.segmentation_model_factory", return_value=_DummySegmentationModel()):
            batch_mod.run_task(processor, args)

        self.assertEqual(len(processor.calls), 1)
        kwargs = processor.calls[0][1]
        self.assertEqual(kwargs["device"], "cpu")

    def test_run_task_seg_uses_gpu_for_hest(self):
        processor = _DummyProcessor()
        args = self._base_args("hest")

        with patch("trident.segmentation_models.load.segmentation_model_factory", return_value=_DummySegmentationModel(target_mag=10)):
            batch_mod.run_task(processor, args)

        self.assertEqual(len(processor.calls), 1)
        kwargs = processor.calls[0][1]
        self.assertEqual(kwargs["device"], "cuda:0")

    def test_cleanup_cache_resets_cache_without_touching_job_locks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = os.path.join(tmpdir, "job")
            cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(os.path.join(job_dir, "nested"), exist_ok=True)
            os.makedirs(cache_dir, exist_ok=True)

            lock_fp = os.path.join(job_dir, "nested", "slide.lock")
            keep_fp = os.path.join(job_dir, "nested", "keep.txt")
            cache_fp = os.path.join(cache_dir, "old.bin")

            with open(lock_fp, "w", encoding="utf-8"):
                pass
            with open(keep_fp, "w", encoding="utf-8") as f:
                f.write("keep")
            with open(cache_fp, "w", encoding="utf-8") as f:
                f.write("cache")

            batch_mod.cleanup_cache(cache_dir)

            self.assertTrue(os.path.exists(lock_fp))
            self.assertTrue(os.path.exists(keep_fp))
            self.assertTrue(os.path.isdir(cache_dir))
            self.assertEqual(os.listdir(cache_dir), [])

    def test_remove_dead_locks_only_removes_stale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = os.path.join(tmpdir, "job")
            os.makedirs(job_dir, exist_ok=True)

            # Fresh lock without target should be kept.
            fresh_lock = os.path.join(job_dir, "fresh_output.h5.lock")
            with open(fresh_lock, "w", encoding="utf-8"):
                pass

            # Lock whose target exists should be removed.
            target = os.path.join(job_dir, "done_output.h5")
            stale_lock = target + ".lock"
            with open(target, "w", encoding="utf-8") as f:
                f.write("done")
            with open(stale_lock, "w", encoding="utf-8"):
                pass

            stats = batch_mod.remove_dead_locks(job_dir, max_age_hours=999.0)
            self.assertTrue(os.path.exists(fresh_lock))
            self.assertFalse(os.path.exists(stale_lock))
            self.assertEqual(stats["scanned"], 2)

    def test_get_pending_slides_skips_completed_feature_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = os.path.join(tmpdir, "job")
            os.makedirs(job_dir, exist_ok=True)

            coords_dir = "20x_256px_0px_overlap"
            feat_dir = os.path.join(job_dir, coords_dir, "features_conch_v15")
            os.makedirs(feat_dir, exist_ok=True)

            done_slide = os.path.join(tmpdir, "done.svs")
            pending_slide = os.path.join(tmpdir, "pending.svs")
            with open(os.path.join(feat_dir, "done.h5"), "w", encoding="utf-8"):
                pass

            class Args:
                pass

            args = Args()
            args.wsi_dir = os.path.join(tmpdir, "wsis")
            args.custom_list_of_wsis = None
            args.wsi_ext = [".svs"]
            args.search_nested = False
            args.max_workers = 1
            args.task = "feat"
            args.mag = 20
            args.patch_size = 256
            args.overlap = 0
            args.coords_dir = None
            args.job_dir = job_dir
            args.slide_encoder = None
            args.patch_encoder = "conch_v15"

            with patch("run_batch_of_slides.collect_valid_slides", return_value=[done_slide, pending_slide]):
                pending = batch_mod.get_pending_slides(args)

            self.assertEqual(pending, [pending_slide])

    def test_build_parser_accepts_gpus(self):
        parser = batch_mod.build_parser()
        args = parser.parse_args([
            "--job_dir", "job",
            "--wsi_dir", "wsis",
            "--gpus", "0", "1",
        ])
        self.assertEqual(args.gpus, [0, 1])

    def test_main_dedups_positive_gpus_but_keeps_repeated_cpu_workers(self):
        """
        `--gpus 0 0 1` should run 2 workers (one per unique GPU).
        `--gpus -1 -1` should still run 2 CPU workers (CPU `-1` entries are
        independent and must not be deduplicated).
        """
        captured: dict[str, list[int]] = {}

        def _fake_worker_entrypoint(args):
            captured.setdefault("single_gpu_ids", []).append(args.gpu)

        with tempfile.TemporaryDirectory() as tmp:
            wsi_dir = os.path.join(tmp, "wsis")
            job_dir = os.path.join(tmp, "job")
            os.makedirs(wsi_dir, exist_ok=True)
            os.makedirs(job_dir, exist_ok=True)

            shard_counts: list[int] = []

            class _CollectShardSizesProcess:
                def __init__(self, target, args=()):
                    self._args = args
                    self.exitcode = 0

                def start(self):
                    shard_counts.append(len(self._args[0].selected_wsi_paths))

                def join(self):
                    return

            class _CollectShardSizesContext:
                def Process(self, target, args=()):
                    return _CollectShardSizesProcess(target=target, args=args)

            def _make_args(gpus_argv: list[str]) -> list[str]:
                return [
                    "run_batch_of_slides.py",
                    "--task", "seg",
                    "--job_dir", job_dir,
                    "--wsi_dir", wsi_dir,
                    "--max_workers", "1",
                    "--gpus", *gpus_argv,
                ]

            slide_paths = [os.path.join(wsi_dir, f"s{i}.svs") for i in range(4)]
            for fp in slide_paths:
                with open(fp, "w", encoding="utf-8"):
                    pass

            with patch("run_batch_of_slides.collect_valid_slides", return_value=slide_paths), \
                 patch("run_batch_of_slides.start_run", return_value="rid"), \
                 patch("run_batch_of_slides.finalize_run"), \
                 patch.object(batch_mod, "_pick_mp_context", return_value=_CollectShardSizesContext()), \
                 patch.object(batch_mod, "torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False

                # Case 1: `--gpus -1 -1` should produce 2 CPU shards.
                shard_counts.clear()
                with patch.object(batch_mod.sys, "argv", _make_args(["-1", "-1"])):
                    batch_mod.main()
                self.assertEqual(len(shard_counts), 2, "Expected 2 CPU workers for `--gpus -1 -1`")
                self.assertEqual(sum(shard_counts), len(slide_paths))

                # Case 2: `--gpus -1` is a single-process path; no Process should be spawned.
                shard_counts.clear()
                with patch.object(batch_mod.sys, "argv", _make_args(["-1"])), \
                     patch.object(batch_mod, "worker_entrypoint") as wmock:
                    batch_mod.main()
                    self.assertEqual(wmock.call_count, 1)
                self.assertEqual(shard_counts, [])

                # Case 3: `--gpus 0 0 1` should dedup positives and produce 2 GPU shards.
                # Force CUDA-available so positive GPU IDs aren't downgraded to `-1`.
                mock_torch.cuda.is_available.return_value = True
                shard_counts.clear()
                with patch.object(batch_mod.sys, "argv", _make_args(["0", "0", "1"])):
                    batch_mod.main()
                self.assertEqual(len(shard_counts), 2, "Expected 2 GPU workers after dedup of `0 0 1`")
                self.assertEqual(sum(shard_counts), len(slide_paths))


if __name__ == "__main__":
    unittest.main()
