import os
import tempfile
import unittest


class TestSummaryMD(unittest.TestCase):
    def test_finalize_run_creates_manifest_and_appends_summary(self):
        from trident.State import make_slide_ref, make_attempt, update_task_state
        from trident.Summary import start_run, finalize_run

        with tempfile.TemporaryDirectory() as job_dir:
            # Create a couple slide states
            s1 = make_slide_ref(
                name="slide1",
                ext=".svs",
                slide_path="/abs/slide1.svs",
                rel_path="slide1.svs",
                reader_type="OpenSlideWSI",
            )
            s2 = make_slide_ref(
                name="slide2",
                ext=".svs",
                slide_path="/abs/slide2.svs",
                rel_path="slide2.svs",
                reader_type="OpenSlideWSI",
            )

            update_task_state(
                job_dir,
                s1,
                "patch_features:uni_v1",
                "completed",
                outputs={"features_path": os.path.join(job_dir, "dummy1.h5")},
                attempt=make_attempt("finished"),
            )
            update_task_state(
                job_dir,
                s2,
                "patch_features:uni_v1",
                "completed",
                outputs={"features_path": os.path.join(job_dir, "dummy2.h5")},
                attempt=make_attempt("finished"),
            )
            update_task_state(
                job_dir,
                s2,
                "slide_features:titan",
                "error",
                reason="exception",
                message="boom",
                attempt=make_attempt("error", error="boom"),
            )

            run_id = start_run(job_dir, tool="unit_test", args={"task": "feat"})
            finalize_run(job_dir, run_id, status="completed")

            manifest_path = os.path.join(job_dir, "runs", f"{run_id}.json")
            self.assertTrue(os.path.exists(manifest_path))

            summary_path = os.path.join(job_dir, "summary.md")
            self.assertTrue(os.path.exists(summary_path))

            with open(summary_path, "r", encoding="utf-8") as f:
                md = f.read()

            self.assertIn(f"run_id={run_id}", md)
            self.assertIn("Patch features:", md)
            self.assertIn("uni_v1: completed: 2", md)
            self.assertIn("Slide features:", md)
            self.assertIn("titan: error (exception): 1", md)
            self.assertIn("Errors (1):", md)
            self.assertIn("slide2: boom", md)


if __name__ == "__main__":
    unittest.main()

