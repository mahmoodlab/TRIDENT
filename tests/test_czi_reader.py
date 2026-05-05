import os
import unittest


class TestCZIReader(unittest.TestCase):
    def test_czi_reader_can_open_and_read_region(self):
        try:
            import pylibCZIrw  # noqa: F401
        except Exception:
            self.skipTest("pylibCZIrw not installed")

        from trident import load_wsi

        czi_path = os.path.join(os.path.dirname(__file__), "..", "czis", "239_GS_PD_P1.czi")
        czi_path = os.path.abspath(czi_path)
        if not os.path.exists(czi_path):
            self.skipTest(f"Missing test asset: {czi_path}")

        with load_wsi(slide_path=czi_path, reader_type="czi", lazy_init=False) as wsi:
            w, h = wsi.get_dimensions()
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)

            tile = wsi.read_region((0, 0), level=0, size=(256, 256), read_as="numpy")
            self.assertEqual(tile.shape, (256, 256, 3))
            self.assertEqual(str(tile.dtype), "uint8")

            thumb = wsi.get_thumbnail((128, 96))
            self.assertEqual(thumb.size, (128, 96))

    def test_czi_autodetect_by_extension(self):
        try:
            import pylibCZIrw  # noqa: F401
        except Exception:
            self.skipTest("pylibCZIrw not installed")

        from trident import load_wsi
        from trident.wsi_objects.CZIWSI import CZIWSI

        czi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "czis", "239_GS_PD_P1.czi"))
        if not os.path.exists(czi_path):
            self.skipTest(f"Missing test asset: {czi_path}")

        wsi = load_wsi(slide_path=czi_path, reader_type=None, lazy_init=False)
        try:
            self.assertIsInstance(wsi, CZIWSI)
        finally:
            wsi.release()


if __name__ == "__main__":
    unittest.main()

