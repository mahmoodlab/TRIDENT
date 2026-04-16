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


if __name__ == "__main__":
    unittest.main()

