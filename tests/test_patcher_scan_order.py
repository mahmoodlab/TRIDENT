import os
import sys
import unittest
import importlib
import numpy as np

# Ensure we import the local `trident/` package from the repo.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def _import_local_wsipatcher():
    """
    Ensure `trident` is imported from this repo (not an installed package).
    Some test environments may already have an installed `trident` on sys.path.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # If an installed `trident` was imported earlier in the test session, evict it.
    mod = sys.modules.get("trident")
    if mod is not None:
        mod_file = getattr(mod, "__file__", "") or ""
        if not os.path.abspath(mod_file).startswith(repo_root):
            for k in list(sys.modules.keys()):
                if k == "trident" or k.startswith("trident."):
                    del sys.modules[k]

    import trident.wsi_objects.WSIPatcher as wsipatcher_mod
    importlib.reload(wsipatcher_mod)
    return wsipatcher_mod.WSIPatcher


WSIPatcher = _import_local_wsipatcher()


class _DummyWSI:
    """
    Minimal WSI stub for testing WSIPatcher coordinate generation.
    """

    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height
        self.level_downsamples = [1]

    def get_dimensions(self):
        return self._width, self._height

    def get_best_level_and_custom_downsample(self, downsample, tolerance=0.1):
        return 0, 1.0


def _seek_distance(coords: np.ndarray, *, weight_x: int = 1, weight_y: int = 1) -> int:
    """
    Deterministic proxy for HDD seek cost.

    We model moves in Y as more expensive than moves in X to reflect that many
    pyramidal/tiled WSI layouts have better locality when scanning left-to-right
    within a row than jumping vertically.
    """
    if coords.shape[0] <= 1:
        return 0
    d = np.abs(np.diff(coords, axis=0))
    return int((weight_x * d[:, 0] + weight_y * d[:, 1]).sum())


class TestWSIPatcherScanOrder(unittest.TestCase):
    def test_row_major_has_better_locality_than_col_major(self):
        patch_size = 256
        overlap = 0

        # Choose dimensions that yield a non-trivial grid.
        cols = 20
        rows = 10
        width = cols * patch_size
        height = rows * patch_size
        wsi = _DummyWSI(width=width, height=height)

        row_major = WSIPatcher(
            wsi=wsi,
            patch_size=patch_size,
            src_mag=20,
            dst_mag=20,
            overlap=overlap,
            coords_only=True,
            scan_order="row-major",
        )
        col_major = WSIPatcher(
            wsi=wsi,
            patch_size=patch_size,
            src_mag=20,
            dst_mag=20,
            overlap=overlap,
            coords_only=True,
            scan_order="col-major",
        )

        self.assertEqual(len(row_major), len(col_major))

        coords_row = np.asarray([xy for xy in row_major], dtype=np.int64)
        coords_col = np.asarray([xy for xy in col_major], dtype=np.int64)

        # Same set of coords, different order.
        self.assertEqual(set(map(tuple, coords_row)), set(map(tuple, coords_col)))

        # Locality proxy: row-major should reduce travel distance vs column-major.
        dist_row = _seek_distance(coords_row, weight_x=1, weight_y=10)
        dist_col = _seek_distance(coords_col, weight_x=1, weight_y=10)
        self.assertLess(dist_row, dist_col)

    def test_default_scan_order_is_row_major(self):
        patch_size = 256
        wsi = _DummyWSI(width=3 * patch_size, height=2 * patch_size)

        patcher = WSIPatcher(
            wsi=wsi,
            patch_size=patch_size,
            src_mag=20,
            dst_mag=20,
            coords_only=True,
        )
        coords = [xy for xy in patcher]

        # Expected order for a 3x2 grid in row-major:
        # row 0: (0,0), (256,0), (512,0)
        # row 1: (0,256), (256,256), (512,256)
        self.assertEqual(
            coords,
            [
                (0, 0),
                (256, 0),
                (512, 0),
                (0, 256),
                (256, 256),
                (512, 256),
            ],
        )


if __name__ == "__main__":
    unittest.main()

