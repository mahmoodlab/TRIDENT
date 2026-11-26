import io
import unittest
import os
import tempfile
from unittest.mock import patch

import run_batch_of_slides as batch_mod
from trident.patch_encoder_models.load import PREFERRED_MAG_AND_PATCH_SIZE, encoder_registry


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

    def test_run_task_seg_tissueqc_removes_artifacts_itself(self):
        processor = _DummyProcessor()
        args = self._base_args("tissueqc")
        args.remove_artifacts = True
        model = _DummySegmentationModel(target_mag=5)
        model.detects_artifacts, model.batch_size = True, 4

        with patch("trident.segmentation_models.load.segmentation_model_factory", return_value=model) as factory:
            batch_mod.run_task(processor, args)

        factory.assert_called_once_with("tissueqc", confidence_thresh=0.5)
        self.assertTrue(model.remove_artifacts)
        kwargs = processor.calls[0][1]
        self.assertIsNone(kwargs["artifact_remover_model"])
        self.assertEqual(kwargs["batch_size"], 4)

    def test_run_task_seg_hest_keeps_grandqc_artifact_remover(self):
        processor = _DummyProcessor()
        args = self._base_args("hest")
        args.remove_artifacts = True

        with patch("trident.segmentation_models.load.segmentation_model_factory",
                   return_value=_DummySegmentationModel(target_mag=10)) as factory:
            batch_mod.run_task(processor, args)

        factory.assert_any_call("grandqc_artifact", remove_penmarks_only=False)
        self.assertEqual(processor.calls[0][1]["batch_size"], 8)

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

    def test_parse_arguments_records_patch_encoder_of_slide_encoder(self):
        # Regression for #239: run manifests must report the patch encoder the slide encoder uses.
        # Pass patching flags explicitly so every encoder reaches the manifest check.
        base = ["run_batch_of_slides.py", "--job_dir", "job", "--wsi_dir", "wsis",
                "--task", "feat", "--mag", "20", "--patch_size", "256"]
        cases = {
            "feather_uni_v2": "uni_v2",
            "titan": "conch_v15",
            "mean-virchow2": "virchow2",
        }
        for slide_encoder, expected in cases.items():
            with self.subTest(slide_encoder=slide_encoder):
                with patch("sys.argv", base + ["--slide_encoder", slide_encoder]):
                    args = batch_mod.parse_arguments()
                self.assertEqual(args.patch_encoder, expected)

        with patch("sys.argv", base + ["--patch_encoder", "uni_v1"]):
            self.assertEqual(batch_mod.parse_arguments().patch_encoder, "uni_v1")

    def test_preferred_patching_covers_patch_encoder_registry(self):
        self.assertEqual(PREFERRED_MAG_AND_PATCH_SIZE.keys(), encoder_registry.keys())

    def test_build_parser_keeps_numeric_patching_defaults(self):
        args = batch_mod.build_parser().parse_args(
            ["--job_dir", "job", "--wsi_dir", "wsis", "--task", "feat", "--patch_encoder", "uni_v1"]
        )
        self.assertEqual((args.mag, args.patch_size), (20.0, 512))
        self.assertIs(type(args.mag), float)
        self.assertIs(type(args.patch_size), int)

    def test_parse_arguments_accepts_matching_defaults_and_explicit_patching(self):
        base = ["run_batch_of_slides.py", "--job_dir", "job", "--wsi_dir", "wsis"]
        uni = ["--patch_encoder", "uni_v1"]
        cases = [
            (["--task", "feat", "--patch_encoder", "conch_v15"], (20.0, 512)),
            (["--task", "feat", *uni, "--patch_size", "512"], (20.0, 512)),
            (["--task", "all", *uni, "--mag", "1.25", "--patch_size", "384"], (1.25, 384)),
            (["--task", "coords", *uni], (20.0, 512)),
            (["--task", "feat", *uni, "--coords_dir", "stored"], (20.0, 512)),
            (["--task", "feat", "--slide_encoder", "mean-kaiko-vit8s"], (20.0, 512)),
            (["--task", "all", "--slide_encoder", "madeleine",
              "--mag", "20", "--patch_size", "512"], (20.0, 512)),
        ]
        for flags, expected in cases:
            with self.subTest(flags=flags), patch("sys.argv", base + flags):
                args = batch_mod.parse_arguments()
                self.assertEqual((args.mag, args.patch_size), expected)

    def test_parse_arguments_errors_when_omitted_default_is_not_preferred(self):
        base = ["run_batch_of_slides.py", "--job_dir", "job", "--wsi_dir", "wsis"]
        uni = ["--patch_encoder", "uni_v1"]
        cases = [
            (["--task", "feat", *uni],
             "uni_v1 prefers --patch_size 256 (default is --patch_size 512). Pass one explicitly."),
            (["--task", "feat", *uni, "--mag", "10"],
             "uni_v1 prefers --patch_size 256 (default is --patch_size 512). Pass one explicitly."),
            # all can generate a grid even with a coords directory.
            (["--task", "all", *uni, "--coords_dir", "stored"],
             "uni_v1 prefers --patch_size 256 (default is --patch_size 512). Pass one explicitly."),
            (["--task", "feat", "--slide_encoder", "feather_uni_v2"],
             "feather_uni_v2 prefers --patch_size 256 (default is --patch_size 512). Pass one explicitly."),
            (["--task", "feat", "--slide_encoder", "madeleine"],
             "madeleine prefers --mag 10 --patch_size 256 (default is --mag 20 --patch_size 512). Pass one explicitly."),
            (["--task", "feat", "--slide_encoder", "madeleine", "--patch_size", "256"],
             "madeleine prefers --mag 10 (default is --mag 20). Pass one explicitly."),
        ]
        for flags, message in cases:
            with self.subTest(flags=flags), patch("sys.argv", base + flags), \
                 patch("sys.stderr", new_callable=io.StringIO) as stderr:
                with self.assertRaises(SystemExit) as error:
                    batch_mod.parse_arguments()
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, stderr.getvalue())

    def test_cli_slide_encoders_exclude_untrained_abmil(self):
        # ABMIL needs hyperparameters the CLI cannot pass, so it is Python API only.
        parser = batch_mod.build_parser()
        base = ["--job_dir", "job", "--wsi_dir", "wsis", "--slide_encoder"]
        self.assertEqual(parser.parse_args(base + ["feather"]).slide_encoder, "feather")
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parser.parse_args(base + ["abmil"])


if __name__ == "__main__":
    unittest.main()
