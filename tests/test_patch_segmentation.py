"""
Unit tests for the patch segmentation framework (task=patch_seg).

These tests are dependency-light: they exercise the model-agnostic plumbing
(instance extraction, HDF5 storage, visualization, the semantic default path, and the
registry/factory) without requiring HistoPlus or CellViT++ to be installed.
"""

import json
import os
import tempfile
import unittest

import h5py
import numpy as np
import torch
from torchvision import transforms

from trident.IO import (
    mask_to_instances,
    save_cell_segmentation_h5,
    overlay_instances_on_thumbnail,
)
from trident.patch_segmentation_models import (
    patch_segmenter_registry,
    patch_segmenter_factory,
    BasePatchSegmenter,
    CustomInferenceSegmenter,
)


class _DummyLogitsModel(torch.nn.Module):
    """Returns fixed per-class logits: center square -> class 1, a corner -> class 2."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        b, _, h, w = x.shape
        logits = torch.zeros((b, self.num_classes, h, w))
        logits[:, 0] = 1.0
        logits[:, 1, h // 4:3 * h // 4, w // 4:3 * w // 4] = 5.0
        logits[:, 2, :h // 8, :w // 8] = 5.0
        return logits


def _semantic_segmenter(num_classes=3):
    return CustomInferenceSegmenter(
        seg_name="dummy",
        model=_DummyLogitsModel(num_classes),
        transforms=transforms.Compose([transforms.ToTensor()]),
        precision=torch.float32,
        class_names=["background", "regionA", "regionB"],
    )


class TestMaskToInstances(unittest.TestCase):
    def test_extracts_one_instance_per_component(self):
        cm = np.zeros((40, 40), dtype=np.int32)
        cm[5:15, 5:15] = 1          # one class-1 blob
        cm[25:35, 25:35] = 2        # one class-2 blob
        insts = mask_to_instances(cm, class_names=["bg", "a", "b"], min_contour_area=1)
        self.assertEqual(len(insts), 2)
        by_class = {i["class_id"]: i for i in insts}
        self.assertEqual(set(by_class), {1, 2})
        self.assertEqual(by_class[1]["class_name"], "a")
        self.assertEqual(by_class[1]["confidence"], 1.0)
        # contour is (K, 2), centroid ~ blob center (10, 10)
        self.assertEqual(by_class[1]["contour"].shape[1], 2)
        self.assertTrue(np.allclose(by_class[1]["centroid"], [10, 10], atol=2))

    def test_background_excluded(self):
        cm = np.zeros((20, 20), dtype=np.int32)  # all background
        self.assertEqual(mask_to_instances(cm), [])

    def test_min_area_filters_noise(self):
        cm = np.zeros((40, 40), dtype=np.int32)
        cm[0, 0] = 1                # 1px speck
        cm[10:25, 10:25] = 1        # real blob
        insts = mask_to_instances(cm, min_contour_area=10)
        self.assertEqual(len(insts), 1)

    def test_two_components_same_class(self):
        cm = np.zeros((40, 40), dtype=np.int32)
        cm[2:10, 2:10] = 1
        cm[20:30, 20:30] = 1
        insts = mask_to_instances(cm, min_contour_area=1)
        self.assertEqual(len(insts), 2)
        self.assertTrue(all(i["class_id"] == 1 for i in insts))


class TestSemanticPredictPatches(unittest.TestCase):
    def test_base_forward_returns_class_map(self):
        seg = _semantic_segmenter()
        x = torch.zeros((2, 3, 32, 32))
        out = seg(x)
        self.assertEqual(tuple(out.shape), (2, 32, 32))
        self.assertEqual(out.dtype, torch.uint8)
        self.assertEqual(set(np.unique(out.numpy()).tolist()), {0, 1, 2})

    def test_predict_patches_returns_per_image_instances(self):
        seg = _semantic_segmenter()
        x = torch.zeros((3, 3, 64, 64))
        per_image = seg.predict_patches(x)
        self.assertEqual(len(per_image), 3)
        for instances in per_image:
            class_ids = sorted({i["class_id"] for i in instances})
            self.assertEqual(class_ids, [1, 2])  # never background


class TestRegistryFactory(unittest.TestCase):
    def test_registry_has_both_models(self):
        self.assertIn("histoplus", patch_segmenter_registry)
        self.assertIn("cellvit_plus_plus", patch_segmenter_registry)

    def test_factory_unknown_raises(self):
        with self.assertRaises(ValueError):
            patch_segmenter_factory("does_not_exist")

    def test_factory_maps_to_registered_class(self):
        # Do not instantiate (needs upstream packages); just verify the mapping.
        for name, cls in patch_segmenter_registry.items():
            self.assertTrue(issubclass(cls, BasePatchSegmenter), name)


class TestCellSegmentationH5(unittest.TestCase):
    def _make_instances(self):
        return [
            {
                "contour": np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=np.float64),
                "centroid": np.array([5.0, 5.0]),
                "class_id": 1,
                "class_name": "a",
                "confidence": 0.9,
            },
            {
                "contour": np.array([[20, 20], [20, 25], [25, 25]], dtype=np.float64),
                "centroid": np.array([23.0, 23.0]),
                "class_id": 2,
                "class_name": "b",
                "confidence": 0.5,
            },
        ]

    def test_roundtrip(self):
        insts = self._make_instances()
        attrs = {"model": "dummy", "class_names": json.dumps(["bg", "a", "b"])}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.h5")
            save_cell_segmentation_h5(path, insts, attrs)
            with h5py.File(path, "r") as f:
                g = f["cells"]
                offsets = g["contour_offsets"][:]
                contours = g["contours"][:]
                class_ids = g["class_ids"][:]
                confidences = g["confidences"][:]
                centroids = g["centroids"][:]
                self.assertEqual(g.attrs["model"], "dummy")
            # N+1 offsets, ragged reconstruction
            self.assertEqual(len(offsets), len(insts) + 1)
            self.assertEqual(class_ids.tolist(), [1, 2])
            self.assertTrue(np.allclose(confidences, [0.9, 0.5]))
            self.assertTrue(np.allclose(centroids[0], [5.0, 5.0]))
            first = contours[offsets[0]:offsets[1]]
            self.assertEqual(len(first), 4)  # first contour had 4 points
            second = contours[offsets[1]:offsets[2]]
            self.assertEqual(len(second), 3)

    def test_empty(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "empty.h5")
            save_cell_segmentation_h5(path, [], {"model": "dummy"})
            with h5py.File(path, "r") as f:
                g = f["cells"]
                self.assertEqual(g["class_ids"].shape[0], 0)
                self.assertEqual(g["contour_offsets"][:].tolist(), [0])


class TestOverlayViz(unittest.TestCase):
    def test_overlay_writes_file(self):
        import geopandas as gpd
        from shapely import Polygon
        gdf = gpd.GeoDataFrame(
            [{"class": 1, "geometry": Polygon([(10, 10), (10, 50), (50, 50), (50, 10)])},
             {"class": 2, "geometry": Polygon([(60, 60), (60, 90), (90, 90)])}],
            columns=["class", "geometry"], geometry="geometry",
        )
        thumb = np.full((100, 100, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "viz.jpg")
            overlay_instances_on_thumbnail(gdf, thumb, out, scale=1.0)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 0)


if __name__ == "__main__":
    unittest.main()
