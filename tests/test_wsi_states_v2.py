import json
import os
import tempfile
import time
import unittest


class TestWSIStatesV2(unittest.TestCase):
    def _mk(self, job_dir: str):
        from trident.State import make_slide_ref

        return make_slide_ref(
            name="slideA",
            ext=".svs",
            slide_path="/abs/path/slideA.svs",
            rel_path="slideA.svs",
            reader_type="OpenSlideWSI",
        )

    def test_attempts_merge_started_finished_into_single_attempt(self):
        from trident.State import make_attempt, update_task_state, load_all_states

        with tempfile.TemporaryDirectory() as job_dir:
            ref = self._mk(job_dir)
            update_task_state(job_dir, ref, "segmentation", "running", attempt=make_attempt("started"))
            time.sleep(0.01)
            update_task_state(job_dir, ref, "segmentation", "completed", attempt=make_attempt("finished"))

            st = load_all_states(job_dir)[ref["id"]]
            attempts = st["tasks"]["segmentation"]["attempts"]
            self.assertEqual(len(attempts), 1)
            a = attempts[0]
            self.assertEqual(a["result"], "completed")
            self.assertIsNotNone(a["started_at"])
            self.assertIsNotNone(a["finished_at"])
            self.assertIsNotNone(a["duration_s"])
            self.assertGreaterEqual(a["duration_s"], 0.0)

    def test_attempts_merge_error_without_started_creates_synthetic_attempt(self):
        from trident.State import make_attempt, update_task_state, load_all_states

        with tempfile.TemporaryDirectory() as job_dir:
            ref = self._mk(job_dir)
            update_task_state(
                job_dir,
                ref,
                "segmentation",
                "error",
                reason="exception",
                message="boom",
                attempt=make_attempt("error", error="boom"),
            )
            st = load_all_states(job_dir)[ref["id"]]
            attempts = st["tasks"]["segmentation"]["attempts"]
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["result"], "error")
            self.assertEqual(attempts[0]["error"], "boom")

    def test_reason_used_for_skipped_and_summary_includes_reason(self):
        from trident.State import update_task_state, load_all_states

        with tempfile.TemporaryDirectory() as job_dir:
            ref = self._mk(job_dir)
            update_task_state(job_dir, ref, "coords", "skipped", reason="already_generated")
            st = load_all_states(job_dir)[ref["id"]]
            self.assertEqual(st["tasks"]["coords"]["status"], "skipped")
            self.assertEqual(st["tasks"]["coords"]["reason"], "already_generated")
            self.assertIn("coords", st["summary"])
            self.assertEqual(st["summary"]["coords"], "skipped (already_generated)")

    def test_summary_groups_patch_and_slide_features_by_encoder(self):
        from trident.State import update_task_state, load_all_states

        with tempfile.TemporaryDirectory() as job_dir:
            ref = self._mk(job_dir)
            update_task_state(job_dir, ref, "patch_features:uni_v1", "completed")
            update_task_state(job_dir, ref, "patch_features:conch_v1", "completed")
            update_task_state(job_dir, ref, "slide_features:titan", "skipped", reason="patch_features_not_found")
            st = load_all_states(job_dir)[ref["id"]]

            self.assertIn("patch_features", st["summary"])
            self.assertEqual(st["summary"]["patch_features"]["uni_v1"], "completed")
            self.assertEqual(st["summary"]["patch_features"]["conch_v1"], "completed")
            self.assertIn("slide_features", st["summary"])
            self.assertEqual(st["summary"]["slide_features"]["titan"], "skipped (patch_features_not_found)")

    def test_resume_last_error_is_recorded(self):
        from trident.State import make_attempt, update_task_state, load_all_states

        with tempfile.TemporaryDirectory() as job_dir:
            ref = self._mk(job_dir)
            update_task_state(
                job_dir,
                ref,
                "patch_features:uni_v1",
                "error",
                reason="exception",
                message="cuda oom",
                attempt=make_attempt("error", error="cuda oom"),
            )
            st = load_all_states(job_dir)[ref["id"]]
            self.assertEqual(st["resume"]["last_task"], "patch_features:uni_v1")
            self.assertEqual(st["resume"]["last_status"], "error")
            self.assertIn("last_error", st["resume"])
            self.assertEqual(st["resume"]["last_error"]["task"], "patch_features:uni_v1")
            self.assertIn("cuda oom", st["resume"]["last_error"]["error"])

    def test_resume_last_error_cleared_when_same_task_recovers(self):
        """A later successful run of a task that previously errored must drop the stale
        resume.last_error, so a recovered slide stops reporting an error it no longer has."""
        from trident.State import make_attempt, update_task_state, load_all_states

        with tempfile.TemporaryDirectory() as job_dir:
            ref = self._mk(job_dir)
            task = "patch_segmentation:histoplus"
            # 1) First attempt errors.
            update_task_state(job_dir, ref, task, "error", reason="exception",
                              message="viz boom", attempt=make_attempt("error", error="viz boom"))
            st = load_all_states(job_dir)[ref["id"]]
            self.assertIn("last_error", st["resume"])

            # 2) Re-run succeeds for the SAME task.
            update_task_state(job_dir, ref, task, "completed",
                              attempt=make_attempt("finished"))
            st = load_all_states(job_dir)[ref["id"]]
            self.assertNotIn("last_error", st["resume"])              # stale error cleared
            self.assertEqual(st["tasks"][task]["status"], "completed")
            self.assertIsNone(st["tasks"][task]["reason"])
            self.assertEqual(st["summary"]["patch_segmentation"]["histoplus"], "completed")

    def test_resume_last_error_kept_when_a_different_task_succeeds(self):
        """Recovery only clears the error for the SAME task; an unrelated task's success
        must not hide a still-unresolved error elsewhere."""
        from trident.State import make_attempt, update_task_state, load_all_states

        with tempfile.TemporaryDirectory() as job_dir:
            ref = self._mk(job_dir)
            update_task_state(job_dir, ref, "patch_segmentation:histoplus", "error",
                              reason="exception", message="boom",
                              attempt=make_attempt("error", error="boom"))
            update_task_state(job_dir, ref, "coords:20x_256px_0px_overlap", "completed")
            st = load_all_states(job_dir)[ref["id"]]
            self.assertIn("last_error", st["resume"])
            self.assertEqual(st["resume"]["last_error"]["task"], "patch_segmentation:histoplus")

    def test_output_stats_added_for_paths(self):
        from trident.State import update_task_state, load_all_states

        with tempfile.TemporaryDirectory() as job_dir:
            ref = self._mk(job_dir)
            out_dir = os.path.join(job_dir, "out")
            os.makedirs(out_dir, exist_ok=True)
            fp = os.path.join(out_dir, "file.txt")
            with open(fp, "w", encoding="utf-8") as f:
                f.write("hello")

            update_task_state(job_dir, ref, "coords", "completed", outputs={"coords_h5": fp})
            st = load_all_states(job_dir)[ref["id"]]
            out = st["tasks"]["coords"]["outputs"]["coords_h5"]
            self.assertIsInstance(out, dict)
            self.assertEqual(out["path"], fp)
            self.assertTrue(out["exists"])
            self.assertGreater(out["bytes"], 0)

    def test_legacy_v1_state_is_upgraded_on_update(self):
        """
        Simulate a v1 state file (schema_version=1, event-style attempts)
        and ensure a v2 write adds new fields and does not crash.
        """
        from trident.State import make_slide_ref, state_path, update_task_state, load_state

        with tempfile.TemporaryDirectory() as job_dir:
            ref = make_slide_ref(
                name="slideA",
                ext=".svs",
                slide_path="/abs/path/slideA.svs",
                rel_path=None,
                reader_type="OpenSlideWSI",
            )
            fp = state_path(job_dir, ref["name"], ref["slide_path"])
            legacy = {
                "schema_version": 1,
                "trident_version": "0.0.0",
                "slide": {"id": ref["id"], "name": ref["name"], "ext": ref["ext"], "abs_path": ref["slide_path"]},
                "meta": {},
                "tasks": {"segmentation": {"status": "running", "attempts": [{"event": "started", "ts": "x", "ts_epoch": 1.0}]}},
                "updated_at": "x",
            }
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(legacy, f)

            # Updating should enrich to v2 markers and keep JSON valid.
            update_task_state(job_dir, ref, "segmentation", "completed")
            st = load_state(fp)
            self.assertGreaterEqual(st.get("schema_version", 0), 2)
            self.assertIn("summary", st)
            self.assertIn("resume", st)
            self.assertIn("segmentation", st["tasks"])

    def test_meta_does_not_overwrite_existing_values_with_none(self):
        from trident.State import make_slide_ref, update_task_state, load_all_states

        with tempfile.TemporaryDirectory() as job_dir:
            ref = make_slide_ref(
                name="slideA",
                ext=".svs",
                slide_path="/abs/path/slideA.svs",
                rel_path="slideA.svs",
                reader_type="OpenSlideWSI",
            )

            update_task_state(job_dir, ref, "segmentation", "completed", wsi_meta={"dimensions": [100, 200]})
            update_task_state(job_dir, ref, "coords", "completed", wsi_meta={"dimensions": None, "mpp": None})

            st = load_all_states(job_dir)[ref["id"]]
            self.assertEqual(st["meta"]["dimensions"], [100, 200])
            self.assertNotIn("mpp", st["meta"])

    def test_stale_started_message_is_cleared_when_task_becomes_skipped(self):
        from trident.State import make_slide_ref, update_task_state, load_all_states

        with tempfile.TemporaryDirectory() as job_dir:
            ref = make_slide_ref(
                name="slideA",
                ext=".svs",
                slide_path="/abs/path/slideA.svs",
                rel_path="slideA.svs",
                reader_type="OpenSlideWSI",
            )

            update_task_state(job_dir, ref, "coords", "running", message="started")
            # Later, a different run decides it's already generated and skips without message.
            update_task_state(job_dir, ref, "coords", "skipped", reason="already_generated")

            st = load_all_states(job_dir)[ref["id"]]
            self.assertIsNone(st["tasks"]["coords"].get("message"))


    def test_same_model_different_coords_do_not_collide(self):
        """A later run of the same segmenter/encoder/vlm on a DIFFERENT coords set must not
        overwrite the earlier config's state (keys are scoped per coords config)."""
        from trident.State import update_task_state, load_all_states
        with tempfile.TemporaryDirectory() as job_dir:
            ref = self._mk(job_dir)
            # HistoPlus completes on the full 20x/784 coords...
            update_task_state(job_dir, ref, "patch_segmentation:histoplus:20x_784px_0px_overlap",
                              "completed", outputs={"segmentation_h5": "/x/20x_784px_0px_overlap/seg_histoplus/slideA.h5"})
            # ...then is skipped on a restricted cluster_coords set (no coords for this slide).
            update_task_state(job_dir, ref, "patch_segmentation:histoplus:cluster_coords",
                              "skipped", reason="coords_not_found")

            st = load_all_states(job_dir)[ref["id"]]
            tasks = st["tasks"]
            # Both records coexist, each with its own truthful status.
            self.assertEqual(tasks["patch_segmentation:histoplus:20x_784px_0px_overlap"]["status"], "completed")
            self.assertIsNone(tasks["patch_segmentation:histoplus:20x_784px_0px_overlap"]["reason"])
            self.assertEqual(tasks["patch_segmentation:histoplus:cluster_coords"]["status"], "skipped")
            # The completed run's outputs are NOT polluted by the skip.
            outs = tasks["patch_segmentation:histoplus:20x_784px_0px_overlap"]["outputs"]
            self.assertIn("segmentation_h5", outs)
            # Summary groups both under patch_segmentation, keyed by model:config — no clobber.
            ps = st["summary"]["patch_segmentation"]
            self.assertEqual(ps["histoplus:20x_784px_0px_overlap"], "completed")
            self.assertEqual(ps["histoplus:cluster_coords"], "skipped (coords_not_found)")

    def test_vlm_query_grouped_in_summary(self):
        """vlm_query:<vlm>:<coords> is grouped under summary.vlm_query (not a flat key)."""
        from trident.State import update_task_state, load_all_states
        with tempfile.TemporaryDirectory() as job_dir:
            ref = self._mk(job_dir)
            update_task_state(job_dir, ref, "vlm_query:patho_r1_7b:vlm_rois_a", "completed")
            update_task_state(job_dir, ref, "vlm_query:patho_r1_7b:vlm_rois_b", "completed")
            st = load_all_states(job_dir)[ref["id"]]
            vq = st["summary"]["vlm_query"]
            self.assertEqual(vq["patho_r1_7b:vlm_rois_a"], "completed")
            self.assertEqual(vq["patho_r1_7b:vlm_rois_b"], "completed")


if __name__ == "__main__":
    unittest.main()

