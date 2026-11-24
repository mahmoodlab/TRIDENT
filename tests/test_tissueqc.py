import unittest
from unittest.mock import patch

import numpy as np

from trident.segmentation_models import load as seg_load
from trident.segmentation_models.load import TissueQCSegmenter
from trident.segmentation_models.model_zoo.tissueqc import (
    ARTIFACT, BACKGROUND, TISSUE, mask_to_polygons, tissue_masks,
)

"""
TissueQC slide-level mask rules, on synthetic class maps (no weights needed).
"""


def _class_map():
    """800 x 800 px at 2 um/px: a tissue square with a fold, a thin crack and a small hole, plus ink on glass."""
    m = np.full((800, 800), BACKGROUND, np.uint8)
    m[100:700, 100:700] = TISSUE
    m[300:400, 200:600] = ARTIFACT          # fold, 200 um wide
    m[500:506, 200:600] = ARTIFACT          # crack, 12 um wide
    m[600:620, 600:620] = BACKGROUND        # small hole inside tissue
    m[20:60, 20:300] = ARTIFACT             # ink lying on bare glass
    return m


def _segmenter(remove_artifacts):
    """TissueQC post-processing without loading weights."""
    seg = TissueQCSegmenter.__new__(TissueQCSegmenter)
    seg.remove_artifacts = remove_artifacts
    seg.target_mag = 5
    return seg


class TestTissueQCMasks(unittest.TestCase):
    def setUp(self):
        self.default, self.clean = tissue_masks(_class_map())   # masks at 8 um/px: 4 x 4 blocks

    def test_fold_is_tissue_by_default_and_cut_when_removing(self):
        self.assertTrue(self.default[80:95, 55:145].all())
        self.assertFalse(self.clean[80:95, 55:145].any())

    def test_thin_crack_is_never_cut(self):
        self.assertTrue(self.clean[125:127, 55:145].all())

    def test_ink_on_glass_is_background(self):
        self.assertFalse(self.default[5:15, 5:75].any())
        self.assertFalse(self.clean[5:15, 5:75].any())

    def test_holes_are_filled_only_when_asked(self):
        self.assertTrue(self.default[151:154, 151:154].all())
        default_with_holes, _ = tissue_masks(_class_map(), holes_are_tissue=False)
        self.assertFalse(default_with_holes[151:154, 151:154].any())


class TestTissueQCContours(unittest.TestCase):
    def test_removed_artifact_becomes_a_hole_in_level0_pixels(self):
        gdf = _segmenter(remove_artifacts=True).to_contours(_class_map(), level0_mpp=0.25)
        self.assertEqual(len(gdf), 1)
        polygon = gdf.geometry.iloc[0]
        self.assertEqual(len(polygon.interiors), 1)                       # the fold only
        x0, _, x1, _ = polygon.bounds                                     # tissue spans 200-1400 um
        self.assertAlmostEqual(x0, 200 / 0.25, delta=40)
        self.assertAlmostEqual(x1, 1400 / 0.25, delta=40)

    def test_default_mask_has_no_holes(self):
        gdf = _segmenter(remove_artifacts=False).to_contours(_class_map(), level0_mpp=0.25)
        self.assertEqual(sum(len(p.interiors) for p in gdf.geometry), 0)

    def test_small_regions_are_dropped(self):
        m = np.zeros((400, 400), bool)
        m[10:20, 10:20] = True                                            # 0.0064 mm2 at 8 um/px
        m[100:300, 100:300] = True
        self.assertEqual(len(mask_to_polygons(m, mask_mpp=8.0, level0_mpp=0.5)), 1)


class TestTissueQCFactory(unittest.TestCase):
    def test_factory_builds_tissueqc(self):
        class Dummy:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
        with patch.object(seg_load, "TissueQCSegmenter", Dummy):
            model = seg_load.segmentation_model_factory('tissueqc', confidence_thresh=0.3, remove_artifacts=True)
        self.assertEqual(model.kwargs, {'freeze': True, 'confidence_thresh': 0.3, 'remove_artifacts': True})
        self.assertEqual(model.model_name, 'tissueqc')


if __name__ == '__main__':
    unittest.main()
