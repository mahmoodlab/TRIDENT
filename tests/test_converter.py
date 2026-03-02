import tempfile
import unittest
from unittest.mock import MagicMock, patch

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


if __name__ == "__main__":
    unittest.main()
