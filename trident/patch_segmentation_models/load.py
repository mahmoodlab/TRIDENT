"""
Cell / nuclei segmentation models for TRIDENT's ``patch_seg`` task.

Instance contract — ``predict_patches(imgs)`` returns, for each image in the batch, a list of:
    {
        "contour":    np.ndarray (K, 2) float, polygon vertices in *input-patch* pixel coords,
        "class_id":   int,    # 0 is background and is never emitted as an instance,
        "class_name": str | None,
        "confidence": float,  # in [0, 1]
        "centroid":   np.ndarray (2,) float, (x, y) in input-patch pixel coords,
    }

To add a model: subclass ``BasePatchSegmenter``, implement ``_build()`` (return
``(model, eval_transforms, precision)``), then either rely on the default semantic
``predict_patches`` (for models whose ``forward`` returns ``(B, C, H, W)`` logits) or
override ``predict_patches`` to delegate to the model's own post-processor. Register the
class in ``patch_segmenter_registry`` at the bottom of this file.

----------------------------
* **HistoPlus** — Adjadj, Bannier, Horent et al., *"Towards Comprehensive Cellular
  Characterisation of H&E Slides"*, arXiv:2508.09926 (Owkin/Bioptimus). Repo:
  https://github.com/owkin/histoplus · Weights (gated, CC-BY-NC-ND 4.0):
  https://huggingface.co/Owkin-Bioptimus/histoplus
* **CellViT++** — Hörst et al., *"CellViT++: Energy-Efficient and Adaptive Cell Segmentation
  and Classification Using Foundation Models"*, arXiv:2501.05269, building on **CellViT**
  (Hörst et al., *Medical Image Analysis* 2024, arXiv:2306.15350). Repo:
  https://github.com/TIO-IKIM/CellViT-Plus-Plus · License: Apache-2.0 + Commons Clause.
"""

import os
from typing import Dict, Any, Tuple, Callable, Optional, List
import numpy as np
import torch
from torch import nn
from abc import abstractmethod

from trident.IO import mask_to_instances


def patch_segmenter_factory(model_name: str, **kwargs) -> "BasePatchSegmenter":
    """
    Instantiate a patch segmentation model by name.

    Parameters:
        model_name (str): One of the keys in `patch_segmenter_registry`.
        **kwargs: Forwarded to the model constructor (e.g. `weights_path`, `mpp`).

    Returns:
        BasePatchSegmenter: The instantiated model wrapper.

    Raises:
        ValueError: If `model_name` is not registered.
    """
    if model_name in patch_segmenter_registry:
        return patch_segmenter_registry[model_name](**kwargs)
    raise ValueError(
        f"Unknown patch segmenter '{model_name}'. "
        f"Available: {sorted(patch_segmenter_registry.keys())}"
    )


class BasePatchSegmenter(torch.nn.Module):
    """
    Base wrapper for a dense / instance patch segmentation model.

    Attributes:
        seg_name (Optional[str]): Unique model identifier (used for output dir names).
        class_names (Optional[List[str]]): Name per class index (index 0 = background).
        eval_transforms (Callable): Preprocessing applied to each patch (PIL -> tensor),
            normally taken verbatim from the upstream package.
        precision (torch.dtype): Autocast precision used during inference.
        target_mpp (Optional[float]): MPP the model expects, when applicable. Informational.
    """

    def __init__(self, weights_path: Optional[str] = None, **build_kwargs: Dict[str, Any]):
        super().__init__()
        self.seg_name: Optional[str] = None
        self.class_names: Optional[List[str]] = None
        self.target_mpp: Optional[float] = None
        self.weights_path: Optional[str] = weights_path
        self.model, self.eval_transforms, self.precision = self._build(**build_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Default forward for semantic models: run the model and argmax over the channel
        dimension to obtain a `(B, H, W)` uint8 class-index map. Instance models override
        `predict_patches` instead and need not use this.
        """
        logits = self.model(x)
        if isinstance(logits, dict):
            logits = logits['out']  # torchvision segmentation convention
        return logits.argmax(dim=1).to(torch.uint8)

    def predict_patches(self, imgs: torch.Tensor) -> List[List[dict]]:
        """
        Default (semantic) implementation: produce one class-index map per image and turn
        each connected component of every non-background class into an instance. Instance
        models (HistoPlus, CellViT++) override this to delegate to their post-processor.

        Precision is owned here (not by the caller): autocast is applied around the forward
        pass when ``self.precision`` is a reduced-precision dtype.
        """
        device_type = imgs.device.type
        precision = getattr(self, 'precision', torch.float32) or torch.float32
        with torch.autocast(device_type=device_type, dtype=precision,
                            enabled=(precision != torch.float32 and device_type == 'cuda')):
            class_maps = self(imgs).detach().cpu().numpy().astype(np.int32)
        return [mask_to_instances(cm, class_names=self.class_names) for cm in class_maps]

    @abstractmethod
    def _build(self, **build_kwargs: Dict[str, Any]) -> Tuple[nn.Module, Callable, torch.dtype]:
        """Return `(model, eval_transforms, precision)`."""
        pass


class CustomInferenceSegmenter(BasePatchSegmenter):
    """
    Wrap an already-instantiated semantic model + transforms, for quick experimentation
    without registering a new class. Uses the default (semantic) `predict_patches`.
    """

    def __init__(
        self,
        seg_name: str,
        model: torch.nn.Module,
        transforms: Callable,
        precision: torch.dtype = torch.float32,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.seg_name = seg_name
        self.model = model
        self.eval_transforms = transforms
        self.precision = precision
        self.class_names = class_names

    def _build(self) -> Tuple[None, None, None]:
        return None, None, None


def _instances_from_contours(
    contours: List[np.ndarray],
    class_ids: List[int],
    class_names_per_cell: List[Optional[str]],
    confidences: List[float],
    centroids: Optional[List] = None,
) -> List[dict]:
    """
    Assemble TRIDENT instance dicts from aligned per-cell lists. Shared by the instance
    model wrappers. Drops degenerate contours (<3 points) and background (class_id 0).
    """
    instances: List[dict] = []
    for i, contour in enumerate(contours):
        contour = np.asarray(contour, dtype=np.float64)
        if contour.ndim != 2 or contour.shape[0] < 3:
            continue
        class_id = int(class_ids[i])
        if class_id == 0:
            continue
        if centroids is not None and centroids[i] is not None:
            centroid = np.asarray(centroids[i], dtype=np.float64).reshape(2)
        else:
            centroid = contour.mean(axis=0)
        instances.append({
            "contour": contour,
            "class_id": class_id,
            "class_name": class_names_per_cell[i],
            "confidence": float(confidences[i]),
            "centroid": centroid,
        })
    return instances


class HistoPlusSegmenter(BasePatchSegmenter):
    """
    Wrapper around **HistoPlus** (Owkin/Bioptimus): a CellViT model with an H0-mini
    foundation-model encoder for pan-cancer cell segmentation + classification (14 cell
    types). 

    Attribution
    -----------
    Adjadj, Bannier, Horent et al., *"Towards Comprehensive Cellular Characterisation of
    H&E Slides"*, arXiv:2508.09926 (2025).
    Code: https://github.com/owkin/histoplus ·
    Weights: https://huggingface.co/Owkin-Bioptimus/histoplus (**gated**, CC-BY-NC-ND 4.0).
    HistoPlus and its weights are for **non-commercial research use** under that license.

    Upstream API used (all loaded lazily from the ``histoplus`` package):
    ``histoplus.helpers.segmentor.CellViTSegmentor.from_histoplus`` to build the model;
    ``.transform`` for normalization; ``.forward`` for the HoVerNet-style maps;
    ``.get_postprocess_fn()`` for instance extraction; ``.class_mapping`` for cell types.

    Requirements
    ------------
    * Install HistoPlus **from source** (not yet on PyPI):
      ``pip install git+https://github.com/owkin/histoplus.git``. It pulls ``timm==1.0.8``
      and ``xformers`` (so use an isolated env — these conflict with TRIDENT's ``timm`` pin).
    * An accepted HF license + ``HF_TOKEN`` to download the gated weights (cached after first use).
    * Inputs at one of the trained resolutions: ``mpp=0.5`` (20x) or ``mpp=0.25`` (40x), with
      ``inference_image_size=784``. Extract coords to match, e.g. ``--mag 20 --patch_size 784``.

    Known issue (batch size): HistoPlus's attention runs through xformers, whose batched
    kernel can hard-crash (segfault) on recent torch (>=2.10). If inference dies silently,
    run with batch size 1 (``--feat_batch_size 1``); single-patch inference is stable.

    Args:
        mpp (float): Target microns-per-pixel; must be 0.5 (20x) or 0.25 (40x). Defaults to 0.5.
        inference_image_size (int): Model input size; HistoPlus uses 784. Defaults to 784.
        mixed_precision (bool): Enable the upstream AMP path. Defaults to False.
    """

    def __init__(self, mpp: float = 0.5, inference_image_size: int = 784,
                 mixed_precision: bool = False, **build_kwargs):
        self._mpp = mpp
        self._inference_image_size = inference_image_size
        self._mixed_precision = mixed_precision
        super().__init__(**build_kwargs)

    def _build(self) -> Tuple[nn.Module, Callable, torch.dtype]:
        # HistoPlus imports xformers, whose flash-attn version check rejects TRIDENT's pinned
        # flash-attn on torch >= 2.10. HistoPlus only uses xformers for SwiGLU (not attention),
        # so skipping the check is safe. Set before the import that pulls xformers.
        os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")
        try:
            from histoplus.helpers.segmentor import CellViTSegmentor
        except ImportError as e:
            raise ImportError(
                "HistoPlus is not installed. It is not yet on PyPI; install from source: "
                "`pip install --no-deps git+https://github.com/owkin/histoplus.git` then "
                "`pip install timm==1.0.8`, and accept the gated license at "
                "https://huggingface.co/Owkin-Bioptimus/histoplus. Original error: " + str(e)
            )

        segmentor = CellViTSegmentor.from_histoplus(
            mixed_precision=self._mixed_precision,
            mpp=self._mpp,
            inference_image_size=self._inference_image_size,
        )
        self._segmentor = segmentor
        # HistoPlus wraps its network in nn.DataParallel over ALL visible GPUs, which deadlocks
        # at batch sizes > 1 when more than one GPU is visible. TRIDENT runs one GPU per worker,
        # so unwrap it to a plain single-device module.
        import torch.nn as _nn
        _inf = getattr(segmentor, 'model', None)
        if isinstance(getattr(_inf, 'module', None), _nn.DataParallel):
            _inf.module = _inf.module.module
        self.seg_name = 'histoplus'
        self.target_mpp = segmentor.target_mpp
        # class_mapping: {0: 'Background', 1: ...}. Build an index-aligned name list.
        mapping = segmentor.class_mapping
        self.class_names = [mapping.get(i, str(i)) for i in range(max(mapping) + 1)]
        self._name_to_id = {name: idx for idx, name in mapping.items()}
        return segmentor.model, segmentor.transform, torch.float16

    @torch.inference_mode()
    def predict_patches(self, imgs: torch.Tensor) -> List[List[dict]]:
        """Run the upstream HistoPlus forward + post-processor and adapt the result.

        Delegates entirely to ``CellViTSegmentor``: ``.forward`` produces the HoVerNet-style
        maps and ``.get_postprocess_fn()`` turns them into per-tile ``TilePrediction`` objects
        (aligned ``contours`` / ``cell_types`` / ``cell_type_probabilities`` / ``centroids``).
        """
        raw = self._segmentor.forward(imgs)
        raw_np = {k: v.detach().cpu().numpy() for k, v in raw.items()}
        tile_predictions = self._segmentor.get_postprocess_fn()(raw_np)
        results: List[List[dict]] = []
        for tp in tile_predictions:
            class_ids = [self._name_to_id.get(name, 0) for name in tp.cell_types]
            results.append(_instances_from_contours(
                contours=tp.contours,
                class_ids=class_ids,
                class_names_per_cell=list(tp.cell_types),
                confidences=list(tp.cell_type_probabilities),
                centroids=list(tp.centroids),
            ))
        return results


class CellViTPlusPlusSegmenter(BasePatchSegmenter):
    """
    Wrapper around **CellViT++** (TIO-IKIM, University Medicine Essen): a ViT/SAM-based
    cell segmentation + classification model. With the default checkpoint it predicts the
    5 PanNuke cell types.

    Attribution
    -----------
    Hörst, Rempe, Becker, Heine, Keyl, Kleesiek, *"CellViT++: Energy-Efficient and Adaptive
    Cell Segmentation and Classification Using Foundation Models"*, arXiv:2501.05269 (2025),
    building on **CellViT** (Hörst et al., *Medical Image Analysis* 2024, arXiv:2306.15350).
    Code: https://github.com/TIO-IKIM/CellViT-Plus-Plus ·
    License: Apache-2.0 + Commons Clause (the Commons Clause restricts commercial selling).
    Default checkpoint ``CellViT-SAM-H-x40-AMP.pth`` is auto-downloaded from Zenodo
    (record 15094831) by the upstream ``cache_cellvit_sam_h`` helper.

    Upstream API used (all loaded lazily from the ``cellvit`` package): the ``CellViTSAM`` /
    ``CellViT256`` model classes; ``cache_cellvit_sam_h`` to fetch weights; ``unflatten_dict``
    to read the checkpoint config; the model's ``forward`` and ``calculate_instance_map``
    post-processor; cell-type names follow the PanNuke taxonomy.

    Requirements
    ------------
    * ``pip install cellvit`` (on PyPI). Use Python **3.10/3.11** (TRIDENT's supported
      versions); on 3.13 its pinned Shapely fails to build, so install with ``--no-deps``
      and add the small missing deps (``colorama colour geojson natsort opt-einsum pyaml``).
    * Inputs at the checkpoint's resolution, e.g. ``--mag 40 --patch_size 1024`` for
      ``CellViT-SAM-H-x40``.

    Args:
        weights_path (str | None): Local checkpoint path. If None, the default CellViT-SAM-H
            checkpoint is auto-downloaded from Zenodo. Defaults to None.
        magnification (int): Magnification passed to the upstream post-processor (20 or 40);
            controls its size thresholds. Defaults to 40.
    """

    def __init__(self, weights_path: Optional[str] = None, magnification: int = 40, **build_kwargs):
        self._magnification = magnification
        super().__init__(weights_path=weights_path, **build_kwargs)

    # PanNuke nuclei taxonomy (Gamper et al., 2019/2020), the labels of the default
    # CellViT++ checkpoints, index-aligned (class 0 = background).
    _PANNUKE_CLASSES = [
        "Background", "Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial",
    ]

    def _build(self) -> Tuple[nn.Module, Callable, torch.dtype]:
        try:
            from cellvit.utils.cache_models import cache_cellvit_sam_h
            from cellvit.utils.tools import unflatten_dict
            from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
            from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
        except ImportError as e:
            raise ImportError(
                "CellViT++ is not installed. Install it with `pip install cellvit` "
                "(see https://github.com/TIO-IKIM/CellViT-Plus-Plus). Original error: " + str(e)
            )

        from torchvision import transforms

        # Resolve the checkpoint: user-provided path, else upstream auto-download (Zenodo).
        weights_path = self.weights_path or str(cache_cellvit_sam_h())

        # --- adapted from cellvit.inference.inference.CellViTInference._load_model / _get_model ---
        # CellViT++ exposes no public checkpoint loader, so we reproduce its short recipe:
        # read the checkpoint, unflatten its config, branch on the stored arch, build the
        # matching upstream model class, and load the state dict. (Apache-2.0 + Commons Clause.)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        run_conf = unflatten_dict(checkpoint["config"], ".")
        arch = checkpoint["arch"]  # "CellViTSAM" or "CellViT256"
        num_types = run_conf["data"]["num_nuclei_classes"]
        num_tissue = run_conf["data"]["num_tissue_classes"]
        if arch == "CellViTSAM":
            model = CellViTSAM(
                model_path=None,
                num_nuclei_classes=num_types,
                num_tissue_classes=num_tissue,
                vit_structure=run_conf["model"]["backbone"],
                regression_loss=run_conf["model"].get("regression_loss", False),
            )
        elif arch == "CellViT256":
            model = CellViT256(
                model256_path=None,
                num_nuclei_classes=num_types,
                num_tissue_classes=num_tissue,
                regression_loss=run_conf["model"].get("regression_loss", False),
            )
        else:
            raise ValueError(f"Unsupported CellViT++ arch '{arch}'.")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        # --- end adapted block ---
        self._model = model

        self.seg_name = 'cellvit_plus_plus'
        self.class_names = (
            self._PANNUKE_CLASSES if num_types == len(self._PANNUKE_CLASSES)
            else [str(i) for i in range(num_types)]
        )

        # Reuse the upstream normalization (run_conf transformations, default 0.5/0.5).
        norm = run_conf.get("transformations", {}).get("normalize", {})
        mean = tuple(norm.get("mean", (0.5, 0.5, 0.5)))
        std = tuple(norm.get("std", (0.5, 0.5, 0.5)))
        eval_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return model, eval_transforms, torch.float16

    @torch.inference_mode()
    def predict_patches(self, imgs: torch.Tensor) -> List[List[dict]]:
        """Run the upstream CellViT forward + ``calculate_instance_map`` and adapt the result."""
        out = self._model(imgs)
        # Softmax the binary/type maps before the instance map, matching the upstream
        # inference (cellvit ...CellViTInference.apply_softmax_reorder); the model's own
        # `calculate_instance_map` does the HoVerNet watershed and returns per-cell dicts.
        out["nuclei_binary_map"] = torch.softmax(out["nuclei_binary_map"], dim=1)
        out["nuclei_type_map"] = torch.softmax(out["nuclei_type_map"], dim=1)
        _, type_preds = self._model.calculate_instance_map(out, magnification=self._magnification)
        results: List[List[dict]] = []
        for tile in type_preds:  # one dict per image: {nucleus_id: {contour, centroid, type, type_prob, ...}}
            cells = list(tile.values())
            class_ids = [int(c["type"]) for c in cells]
            names = [self.class_names[i] if 0 <= i < len(self.class_names) else str(i) for i in class_ids]
            results.append(_instances_from_contours(
                contours=[np.asarray(c["contour"]) for c in cells],
                class_ids=class_ids,
                class_names_per_cell=names,
                confidences=[float(c.get("type_prob", 1.0)) for c in cells],
                centroids=[c.get("centroid") for c in cells],
            ))
        return results


patch_segmenter_registry = {
    'histoplus': HistoPlusSegmenter,
    'cellvit_plus_plus': CellViTPlusPlusSegmenter,
}
