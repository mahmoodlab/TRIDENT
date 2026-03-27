import os
import sys


def _ensure_local_trident_first_on_syspath() -> None:
    """
    Ensure tests import *this* repo's `trident` package, not an installed one
    or another checkout that happens to be on PYTHONPATH.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if sys.path[0] != repo_root:
        sys.path.insert(0, repo_root)

    mod = sys.modules.get("trident")
    if mod is None:
        return

    mod_file = getattr(mod, "__file__", "") or ""
    if not os.path.abspath(mod_file).startswith(repo_root):
        # Evict foreign `trident` so subsequent imports use the local one.
        for k in list(sys.modules.keys()):
            if k == "trident" or k.startswith("trident."):
                del sys.modules[k]


_ensure_local_trident_first_on_syspath()

