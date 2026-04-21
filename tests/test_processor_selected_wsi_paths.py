import os
import tempfile
import unittest
import importlib
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from trident import Processor


processor_module = importlib.import_module("trident.Processor")


class TestProcessorSelectedWSIPaths(unittest.TestCase):
    def test_selected_wsi_paths_bypass_discovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wsi_source = os.path.join(tmpdir, "wsis")
            os.makedirs(wsi_source, exist_ok=True)
            selected = [
                os.path.join(wsi_source, "a.svs"),
                os.path.join(wsi_source, "b.svs"),
            ]

            def fake_load_wsi(**kwargs):
                slide = SimpleNamespace(name=kwargs["name"], ext=os.path.splitext(kwargs["name"])[1])
                return nullcontext(slide)

            with patch.object(processor_module, "collect_valid_slides") as mock_collect, patch.object(
                processor_module, "load_wsi", side_effect=fake_load_wsi
            ):
                processor = Processor(
                    job_dir=tmpdir,
                    wsi_source=wsi_source,
                    wsi_ext=[".svs"],
                    selected_wsi_paths=selected,
                )

            self.assertEqual(len(processor.wsis), 2)
            mock_collect.assert_not_called()
            processor.release()

    def test_selected_wsi_paths_custom_list_mpp_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wsi_source = os.path.join(tmpdir, "wsis")
            os.makedirs(wsi_source, exist_ok=True)

            selected = [
                os.path.join(wsi_source, "a.svs"),
                os.path.join(wsi_source, "b.svs"),
            ]
            for fp in selected:
                with open(fp, "w", encoding="utf-8"):
                    pass

            csv_path = os.path.join(tmpdir, "slides.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("wsi,mpp\n")
                f.write("b.svs,0.5\n")
                f.write("a.svs,0.25\n")

            seen_mpps = []

            def fake_load_wsi(**kwargs):
                seen_mpps.append(kwargs.get("mpp"))
                slide = SimpleNamespace(name=kwargs["name"], ext=os.path.splitext(kwargs["name"])[1])
                return nullcontext(slide)

            with patch.object(processor_module, "collect_valid_slides") as mock_collect, patch.object(
                processor_module, "load_wsi", side_effect=fake_load_wsi
            ):
                processor = Processor(
                    job_dir=tmpdir,
                    wsi_source=wsi_source,
                    wsi_ext=[".svs"],
                    custom_list_of_wsis=csv_path,
                    selected_wsi_paths=selected,
                )

            mock_collect.assert_not_called()
            self.assertEqual(seen_mpps, [0.25, 0.5])
            processor.release()


if __name__ == "__main__":
    unittest.main()
