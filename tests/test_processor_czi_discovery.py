import os
import unittest


class TestProcessorCZIDiscovery(unittest.TestCase):
    def test_processor_discovers_czi_by_default_extensions(self):
        try:
            import pylibCZIrw  # noqa: F401
        except Exception:
            self.skipTest("pylibCZIrw not installed")

        czi_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "czis"))
        czi_fp = os.path.join(czi_dir, "239_GS_PD_P1.czi")
        if not os.path.exists(czi_fp):
            self.skipTest(f"Missing test asset: {czi_fp}")

        from trident import Processor

        p = Processor(job_dir="/tmp/trident_test_jobdir", wsi_source=czi_dir, skip_errors=True)
        try:
            self.assertGreaterEqual(len(p.wsis), 1)
            self.assertTrue(any(w.ext == ".czi" for w in p.wsis))
        finally:
            p.release()


if __name__ == "__main__":
    unittest.main()

