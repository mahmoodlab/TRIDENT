import tempfile
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

import numpy as np

from trident.Converter import AnyToTiffConverter


class TestAnyToTiffConverter(unittest.TestCase):
    def test_process_file_uses_compound_extension_stem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = AnyToTiffConverter(job_dir=tmpdir)
            with patch.object(converter, "_try_pyvips_convert", return_value=False), \
                 patch.object(converter, "_read_image", return_value=np.zeros((4, 4, 3), dtype=np.uint8)), \
                 patch.object(converter, "_save_tiff") as save_mock:
                converter.process_file("slide.ome.tif", mpp=0.25, zoom=1.0)

            # The stem must be "slide" and not "slide.ome".
            self.assertEqual(save_mock.call_args[0][1], "slide")

    def test_process_file_fallback_used_when_pyvips_path_not_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = AnyToTiffConverter(job_dir=tmpdir)
            with patch.object(converter, "_try_pyvips_convert", return_value=False), \
                 patch.object(converter, "_read_image", return_value=np.zeros((2, 2, 3), dtype=np.uint8)) as read_mock, \
                 patch.object(converter, "_save_tiff") as save_mock:
                converter.process_file("slide.svs", mpp=0.25, zoom=1.0)

            read_mock.assert_called_once()
            save_mock.assert_called_once()

    def test_process_file_skips_fallback_when_pyvips_path_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = AnyToTiffConverter(job_dir=tmpdir)
            with patch.object(converter, "_try_pyvips_convert", return_value=True), \
                 patch.object(converter, "_read_image") as read_mock, \
                 patch.object(converter, "_save_tiff") as save_mock:
                converter.process_file("slide.svs", mpp=0.25, zoom=1.0)

            read_mock.assert_not_called()
            save_mock.assert_not_called()

    def test_process_all_rejects_invalid_downscale(self):
        converter = AnyToTiffConverter(job_dir=".")
        with self.assertRaises(ValueError):
            converter.process_all(input_dir=".", mpp_csv="dummy.csv", downscale_by=0)

    def test_process_all_rejects_negative_workers(self):
        converter = AnyToTiffConverter(job_dir=".")
        with self.assertRaises(ValueError):
            converter.process_all(input_dir=".", mpp_csv="dummy.csv", num_workers=-1)

    def test_process_all_requires_wsi_and_mpp_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = AnyToTiffConverter(job_dir=tmpdir)
            csv_path = f"{tmpdir}/bad.csv"
            pd.DataFrame({"wsi": ["a.svs"]}).to_csv(csv_path, index=False)
            with self.assertRaises(ValueError):
                converter.process_all(input_dir=tmpdir, mpp_csv=csv_path)

    def test_process_all_rejects_missing_csv_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = AnyToTiffConverter(job_dir=tmpdir)
            with self.assertRaises(ValueError):
                converter.process_all(input_dir=tmpdir, mpp_csv=f"{tmpdir}/does_not_exist.csv")

    def test_process_all_raises_when_no_valid_tasks_from_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = AnyToTiffConverter(job_dir=tmpdir)
            csv_path = f"{tmpdir}/to_process.csv"
            pd.DataFrame({"wsi": ["missing_slide.svs"], "mpp": [0.25]}).to_csv(csv_path, index=False)
            with self.assertRaises(ValueError):
                converter.process_all(input_dir=tmpdir, mpp_csv=csv_path)

    def test_process_file_warns_when_embedded_mpp_differs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = AnyToTiffConverter(job_dir=tmpdir)
            with patch.object(converter, "_detect_embedded_mpp", return_value=0.5), \
                 patch.object(converter, "_try_pyvips_convert", return_value=True), \
                 patch("builtins.print") as print_mock:
                converter.process_file("slide.svs", mpp=0.25, zoom=1.0)
            calls = [str(c) for c in print_mock.call_args_list]
            self.assertTrue(any("MPP mismatch" in c for c in calls))


if __name__ == "__main__":
    unittest.main()
