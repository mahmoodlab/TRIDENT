import traceback
from abc import abstractmethod
from typing import Literal, Optional, Any, Dict, Tuple, Callable
import torch
import os 

from trident.patch_encoder_models.utils.constants import get_constants
from trident.patch_encoder_models.utils.transform_utils import get_eval_transforms
from trident.IO import get_weights_path, has_internet_connection

"""
This file contains 20+ pretrained patch encoders, all loadable via the encoder_factory() function.
"""

# Patch encoders that support a user-configurable input resolution via positional-embedding
# interpolation (timm `dynamic_img_size`). For these models, a `target_img_size` keyword can be
# forwarded through `encoder_factory(...)` into the encoder's `_build(...)`. See
# `_resolve_target_img_size` for the validation rules.
RESIZE_SUPPORTED_PATCH_ENCODERS = frozenset({
    # Category A: dynamic_img_size already enabled on the timm backbone.
    "uni_v1", "uni_v2", "virchow", "virchow2",
    "kaiko-vitb8", "kaiko-vitb16", "kaiko-vits8", "kaiko-vits16", "kaiko-vitl14",
    # Category B: dynamic_img_size enabled as part of this feature.
    "gigapath", "hoptimus0", "hoptimus1", "gpfm", "lunit-vits8", "h0-mini",
})


def _resolve_target_img_size(enc_name: str, target_img_size: Optional[int], default_img_size: int, patch_size: int) -> int:
    """
    Validate and resolve a user-requested patch-encoder input resolution.

    Parameters:
        enc_name (str): Encoder name, used in error messages.
        target_img_size (Optional[int]): Requested square input size in pixels. If None, the
            model's native `default_img_size` is used.
        default_img_size (int): Native training resolution of the encoder.
        patch_size (int): ViT patch size. `target_img_size` must be an exact multiple of this so
            positional embeddings can be interpolated without padding.

    Returns:
        int: The resolution to build the model and eval transform with.
    """
    if target_img_size is None:
        return default_img_size
    assert isinstance(target_img_size, int) and target_img_size > 0, (
        f"{enc_name}: target_img_size must be a positive integer, got {target_img_size!r}."
    )
    assert target_img_size % patch_size == 0, (
        f"{enc_name}: target_img_size={target_img_size} must be a multiple of the model patch size "
        f"({patch_size}) so positional embeddings can be interpolated cleanly. "
        f"Nearest valid sizes: {patch_size * (target_img_size // patch_size)} or "
        f"{patch_size * (target_img_size // patch_size + 1)}."
    )
    return target_img_size


def encoder_factory(model_name: str, **kwargs) -> torch.nn.Module:
    """
    Instantiate a patch encoder model by name.

    This factory function returns a pre-configured encoder model class based on the provided
    `model_name`. Each encoder is designed for extracting representations from image patches
    using specific backbones or pretraining strategies.

    Parameters:
        model_name (str):
            Name of the encoder to instantiate. Must be one of the following:
        - "conch_v1"
        - "conch_v15"
        - "uni_v1"
        - "uni_v2"
        - "ctranspath"
        - "phikon"
        - "phikon_v2"
        - "resnet50"
        - "keep"
        - "gigapath"
        - "virchow"
        - "virchow2"
        - "hoptimus0"
        - "hoptimus1"
        - "h0-mini"
        - "musk"
        - "openmidnight"
        - "gpfm"
        - "hibou_l"
        - "kaiko-vitb8"
        - "kaiko-vitb16"
        - "kaiko-vits8"
        - "kaiko-vits16"
        - "kaiko-vitl14"
        - "lunit-vits8"
        - "genbio-pathfm"
        - "gemma4-e4b"
        - "gemma4-26b"

        **kwargs (dict):
            Optional keyword arguments passed directly to the encoder constructor. These may include parameters such as:
        - weights_path (str): Path to a local checkpoint (optional)
        - normalize (bool): Whether to normalize output embeddings (default: False)
        - with_proj (bool): Whether to apply the projection head (default: True)
        - any model-specific configuration parameters

    Returns:
        torch.nn.Module: An instance of the specified encoder model.

    Raises:
        ValueError:
            If `model_name` is not among the recognized encoder names.

    Example
    -------
    >>> # Load a high-performance vision transformer
    >>> encoder = encoder_factory("conch_v15")
    >>> 
    >>> # Load with custom weights
    >>> encoder = encoder_factory("uni_v2", weights_path="custom_weights.pth")
    >>> 
    >>> # Load a fast CNN model
    >>> encoder = encoder_factory("ctranspath")
    """
    if model_name in encoder_registry:
        return encoder_registry[model_name](**kwargs)
    else:
        raise ValueError(f"Unknown encoder name {model_name}")


class BasePatchEncoder(torch.nn.Module):

    _has_internet = has_internet_connection()
    
    def __init__(self, weights_path: Optional[str] = None, **build_kwargs: Dict[str, Any]):
        """
        Initialize BasePatchEncoder.

        Parameters:
            weights_path (Optional[str]):
                Optional path to local model weights. If None, the model is loaded from the model registry or
                downloaded from Hugging Face Hub.
            **build_kwargs (dict):
                Additional keyword arguments passed to the `_build()` method to customize model creation.

        Attributes:
            enc_name (Optional[str]):
                Name of the encoder architecture (set during `_build()`).
            weights_path (Optional[str]):
                Path to local model weights (if provided).
            model (nn.Module):
                The instantiated encoder model.
            eval_transforms (Callable):
                Evaluation-time preprocessing transforms.
            precision (torch.dtype):
                Precision used for inference.
        """

        super().__init__()
        self.enc_name: Optional[str] = None
        self.weights_path: Optional[str] = weights_path
        self.model, self.eval_transforms, self.precision = self._build(**build_kwargs)

    def ensure_valid_weights_path(self, weights_path: str) -> None:
        if not weights_path:
            return
        if os.path.isfile(weights_path) or os.path.isdir(weights_path):
            return
        raise FileNotFoundError(
            f"Expected checkpoint file or model directory at '{weights_path}', but it was not found."
        )
    
    def ensure_has_internet(self, enc_name: str) -> None:
        if not BasePatchEncoder._has_internet:
            raise FileNotFoundError(
                f"Internet connection does seem not available. Auto checkpoint download is disabled."
                f"To proceed, please manually download: {enc_name},\n"
                f"and place it in the model registry in:\n`trident/patch_encoder_models/local_ckpts.json`"
            )
        
    def _get_weights_path(self) -> str:
        """
        If self.weights_path is provided, use it. 
        If not provided, check the model registry. 
            If path in model registry is empty, auto-download from huggingface
            else, use the path from the registry.
        """
        if self.weights_path:
            self.ensure_valid_weights_path(self.weights_path)
            return self.weights_path
        else:
            weights_path = get_weights_path('patch', self.enc_name)
            self.ensure_valid_weights_path(weights_path)
            return weights_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Can be overwritten if model requires special forward pass.
        """
        z = self.model(x)
        return z
        
    @abstractmethod
    def _build(self, **build_kwargs: Dict[str, Any]) -> Tuple[torch.nn.Module, Callable, torch.dtype]:
        pass


class CustomInferenceEncoder(BasePatchEncoder):

    def __init__(self, enc_name: str, model: torch.nn.Module, transforms: Callable, precision: torch.dtype):
        """
        Initialize a CustomInferenceEncoder from user-defined components.

        This class is used when the model, transforms, and precision are pre-instantiated externally 
        and should be injected directly into the encoder wrapper.

        Parameters:
            enc_name (str):
                A unique name or identifier for the encoder (used for registry or logging).
            model (torch.nn.Module):
                A PyTorch model instance to use for inference.
            transforms (Callable):
                A callable (e.g., torchvision or timm transform) to preprocess input images for evaluation.
            precision (torch.dtype):
                The precision to use for inference (e.g., torch.float32, torch.float16).
        """
        super().__init__()
        self.enc_name = enc_name
        self.model = model
        self.eval_transforms = transforms
        self.precision = precision
        
    def _build(self) -> Tuple[None, None, None]:
        return None, None, None


class KeepInferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        KEEP initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        from transformers import AutoModel, AutoTokenizer
        from torchvision import transforms

        self.enc_name = 'keep'
        weights_path = self._get_weights_path()

        if weights_path:
            model_source = weights_path
            if os.path.isfile(model_source):
                model_source = os.path.dirname(model_source)
            try:
                model = AutoModel.from_pretrained(model_source, trust_remote_code=True)
                _ = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
            except Exception:
                traceback.print_exc()
                raise Exception(
                    "Failed to load KEEP from local checkpoint path. "
                    "Set `keep` in `trident/patch_encoder_models/local_ckpts.json` to a local Hugging Face model directory."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = AutoModel.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True)
                _ = AutoTokenizer.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True)
            except Exception:
                traceback.print_exc()
                raise Exception(
                    "Failed to download KEEP model from Hugging Face. "
                    "Set `keep` in `trident/patch_encoder_models/local_ckpts.json` for offline use."
                )

        eval_transform = transforms.Compose([
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        precision = torch.float16
        return model, eval_transform, precision

    def forward(self, x):
        if hasattr(self.model, 'encode_image'):
            return self.model.encode_image(x)
        return self.model(x)


class MuskInferenceEncoder(BasePatchEncoder):
    
    def __init__(self, **build_kwargs):
        """
        MUSK initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, inference_aug=False, with_proj=False, out_norm=False, return_global=True):
        """
        Args:
            inference_aug (bool): Whether to use test-time multiscale augmentation. Default is False to allow for fair comparison with other models.
        """
        import timm
        
        self.enc_name = 'musk'
        self.inference_aug = inference_aug
        self.with_proj = with_proj
        self.out_norm = out_norm
        self.return_global = return_global
    
        try:
            from musk import utils, modeling
        except:
            traceback.print_exc()
            raise Exception("Please install MUSK `pip install fairscale git+https://github.com/lilab-stanford/MUSK`")

        weights_path = self._get_weights_path()

        if weights_path:
            raise NotImplementedError("MUSK doesn't support local model loading. PR welcome!")
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("musk_large_patch16_384")
                utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '')
            except:
                traceback.print_exc()
                raise Exception("Failed to download MUSK model, make sure that you were granted access and that you correctly registered your token")
        
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        from torchvision.transforms import InterpolationMode
        eval_transform = get_eval_transforms(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, target_img_size = 384, center_crop = True, interpolation=InterpolationMode.BICUBIC, antialias=True)
        precision = torch.float16
        
        return model, eval_transform, precision
    
    def forward(self, x):
        return self.model(
                image=x,
                with_head=self.with_proj,
                out_norm=self.out_norm,
                ms_aug=self.inference_aug,
                return_global=self.return_global  
                )[0]  # Forward pass yields (vision_cls, text_cls). We only need vision_cls.


class Conchv1InferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        CONCH initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, with_proj=False, normalize=False):
        self.enc_name = 'conch_v1'
        self.with_proj = with_proj
        self.normalize = normalize

        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except:
            traceback.print_exc()
            raise Exception("Please install CONCH `pip install git+https://github.com/Mahmoodlab/CONCH.git`")
        
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                model, eval_transform = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path=weights_path)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create CONCH v1 model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/MahmoodLab/CONCH."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model, eval_transform = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path="hf_hub:MahmoodLab/conch")
            except:
                traceback.print_exc()
                raise Exception("Failed to download CONCH v1 model, make sure that you were granted access and that you correctly registered your token")
    
        precision = torch.float32
        
        return model, eval_transform, precision
    
    def forward(self, x):
        return self.model.encode_image(x, proj_contrast=self.with_proj, normalize=self.normalize)
    

class CTransPathInferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        CTransPath initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        from torchvision.transforms import InterpolationMode
        from torch import nn

        try:
            from .model_zoo.ctranspath.ctran import ctranspath
        except:
            traceback.print_exc()
            raise Exception("Failed to import CTransPath model, make sure timm_ctp is installed. `pip install timm_ctp`")
        
        self.enc_name = 'ctranspath'
        weights_path = self._get_weights_path()

        model = ctranspath(img_size=224)
        model.head = nn.Identity()

        if not weights_path:
            self.ensure_has_internet(self.enc_name)
            try:
                from huggingface_hub import hf_hub_download   
                weights_path = hf_hub_download(
                    repo_id="MahmoodLab/hest-bench",
                    repo_type="dataset",
                    filename="CHIEF_CTransPath.pth",
                    subfolder="fm_v1/ctranspath",
                )
            except:
                traceback.print_exc()
                raise Exception("Failed to download CTransPath model, make sure that you were granted access and that you correctly registered your token")

        try:
            state_dict = torch.load(weights_path, weights_only=True)['model']
        except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create CTransPath model from local checkpoint at '{weights_path}'. "
                    "You can download the required `CHIEF_CTransPath.pth` from: https://huggingface.co/datasets/MahmoodLab/hest-bench/tree/main/fm_v1/ctranspath."
                )
        state_dict = {key: val for key, val in state_dict.items() if 'attn_mask' not in key}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        assert len(unexpected) == 0, f"Unexpected keys found in state dict: {unexpected}"
        assert missing == ['layers.0.blocks.1.attn_mask', 'layers.1.blocks.1.attn_mask', 'layers.2.blocks.1.attn_mask', 'layers.2.blocks.3.attn_mask', 'layers.2.blocks.5.attn_mask'], f"Unexpected missing keys: {missing}"

        mean, std = get_constants('imagenet')
        eval_transform = get_eval_transforms(mean, std, target_img_size=224, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)

        precision = torch.float32
        
        return model, eval_transform, precision


class PhikonInferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        Phikon initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        from transformers import ViTModel
        from torchvision.transforms import InterpolationMode

        self.enc_name = 'phikon'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                model_dir = os.path.dirname(weights_path)
                model = ViTModel.from_pretrained(model_dir, add_pooling_layer=False, local_files_only=True)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create Phikon model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/owkin/phikon."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Phikon model, make sure that you were granted access and that you correctly registered your token")

        mean, std = get_constants('imagenet')
        eval_transform = get_eval_transforms(mean, std, target_img_size=224, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)
        precision = torch.float32
        return model, eval_transform, precision
    
    def forward(self, x):
        out = self.forward_features(x)
        out = out.last_hidden_state[:, 0, :]
        return out
    
    def forward_features(self, x):
        out = self.model(pixel_values=x)
        return out
    

class HibouLInferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        Hibou initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        from transformers import AutoModel
        from torchvision.transforms import InterpolationMode

        self.enc_name = 'hibou_l'
        weights_path = self._get_weights_path()

        if weights_path:
            raise NotImplementedError("Hibou-Large doesn't support local model loading. PR welcome!")
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Hibou-L model, make sure that you were granted access and that you correctly registered your token")
        
        mean, std = get_constants('hibou')
        eval_transform = get_eval_transforms(mean, std, target_img_size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True)
        precision = torch.float32

        return model, eval_transform, precision
    
    def forward(self, x):
        out = self.forward_features(x)
        out = out.pooler_output
        return out
    
    def forward_features(self, x):
        out = self.model(pixel_values=x)
        return out


class KaikoInferenceEncoder(BasePatchEncoder):
    MODEL_NAME = None  # set in subclasses
    HF_HUB_ID = None # set in subclasses
    IMG_SIZE = None
    PATCH_SIZE = None  # set in subclasses

    def __init__(self, **build_kwargs):
        """
        Kaiko initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, target_img_size=None):
        import timm
        from torchvision.transforms import InterpolationMode
        self.enc_name = f"kaiko-{self.MODEL_NAME}"
        weights_path = self._get_weights_path()

        # Default input resolution for Kaiko encoders is 224 (the eval transform size), even
        # for the L14 variant whose backbone is built at 518. `model_img_size` preserves that
        # original behavior when no custom size is requested.
        transform_img_size = _resolve_target_img_size(self.enc_name, target_img_size, 224, self.PATCH_SIZE)
        model_img_size = self.IMG_SIZE if target_img_size is None else transform_img_size

        if weights_path:
            try:
                model = timm.create_model(
                    f"{self.HF_HUB_ID}",
                    num_classes=0,
                    checkpoint_path=weights_path,
                    img_size=model_img_size,
                    dynamic_img_size=True
                )
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create Kaiko model from local checkpoint at '{weights_path}'. "
                    "You can download the required `model.safetensors` and `config.yaml` from: https://huggingface.co/collections/1aurent/kaikoai-models-66636c99d8e1e34bc6dcf795."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model(
                    model_name=f"hf-hub:1aurent/{self.HF_HUB_ID}.kaiko_ai_towards_large_pathology_fms",
                    dynamic_img_size=True,
                    pretrained=True,
                    num_classes=0,
                    img_size=model_img_size,
                )
            except:
                traceback.print_exc()
                raise Exception("Failed to download Kaiko model.")

        mean, std = get_constants("kaiko")
        eval_transform = get_eval_transforms(mean, std, target_img_size=transform_img_size, center_crop=True, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)
        precision = torch.float32

        return model, eval_transform, precision

    def forward(self, x):
        return self.model(x)


class KaikoS16InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vits16"
    HF_HUB_ID = "vit_small_patch16_224"
    IMG_SIZE = 224
    PATCH_SIZE = 16

    def __init__(self, **build_kwargs):
        """
        Kaiko Small 16 initialization.
        """
        super().__init__(**build_kwargs)
    

class KaikoS8InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vits8"
    HF_HUB_ID = "vit_small_patch8_224"
    IMG_SIZE = 224
    PATCH_SIZE = 8

    def __init__(self, **build_kwargs):
        """
        Kaiko Small 8 initialization.
        """
        super().__init__(**build_kwargs)
    

class KaikoB16InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vitb16"
    HF_HUB_ID = "vit_base_patch16_224"
    IMG_SIZE = 224
    PATCH_SIZE = 16

    def __init__(self, **build_kwargs):
        """
        Kaiko Base 16 initialization.
        """
        super().__init__(**build_kwargs)
    

class KaikoB8InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vitb8"
    HF_HUB_ID = "vit_base_patch8_224"
    IMG_SIZE = 224
    PATCH_SIZE = 8

    def __init__(self, **build_kwargs):
        """
        Kaiko Base 8 initialization.
        """
        super().__init__(**build_kwargs)
    

class KaikoL14InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vitl14"
    HF_HUB_ID = "vit_large_patch14_reg4_dinov2"
    IMG_SIZE = 518
    PATCH_SIZE = 14

    def __init__(self, **build_kwargs):
        """
        Kaiko Large 14 initialization.
        """
        super().__init__(**build_kwargs)
    

class ResNet50InferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        ResNet50-ImageNet initialization.
        """
        super().__init__(**build_kwargs)

    def _build(
        self, 
        pretrained=True, 
        timm_kwargs={"features_only": True, "out_indices": [3], "num_classes": 0},
        img_size=224,
        pool=True
    ):
        import timm
        from torchvision.transforms import InterpolationMode

        self.enc_name = 'resnet50'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                model = timm.create_model("resnet50", pretrained=False, **timm_kwargs)
                if weights_path.suffix == ".safetensors":
                    from safetensors.torch import load_file
                    state_dict = load_file(weights_path)
                else:
                    state_dict = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(state_dict, strict=False)

            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create ResNet50 model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` or ` model.safetensors` from: https://huggingface.co/timm/resnet50.tv_in1k."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("resnet50.tv_in1k", pretrained=pretrained, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download ResNet50 model.")

        mean, std = get_constants('imagenet')
        eval_transform = get_eval_transforms(mean, std, target_img_size=img_size, center_crop=True, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)

        precision = torch.float32
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
        
        return model, eval_transform, precision
    
    def forward(self, x):
        out = self.forward_features(x)
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out
    
    def forward_features(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        return out


class LunitS8InferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        Lunit initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, target_img_size=None):
        import timm
        from timm.data import resolve_model_data_config
        from timm.data.transforms_factory import create_transform

        self.enc_name = 'lunit-vits8'
        weights_path = self._get_weights_path()
        img_size = _resolve_target_img_size(self.enc_name, target_img_size, 224, 8)

        if weights_path:
            try:
                timm_kwargs = {"img_size": img_size, "dynamic_img_size": True}
                model = timm.create_model("vit_small_patch8_224", checkpoint_path=weights_path, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create Lunit-Small model from local checkpoint at '{weights_path}'. "
                    "You can download the required `model.safetensors` and `config.yaml` from: https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:1aurent/vit_small_patch8_224.lunit_dino", pretrained=True, img_size=img_size, dynamic_img_size=True)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Lunit S8 model, make sure that you were granted access and that you correctly registered your token.")

        data_config = resolve_model_data_config(model)
        if target_img_size is not None:
            data_config["input_size"] = (3, img_size, img_size)
            data_config["crop_pct"] = 1.0
        eval_transform = create_transform(**data_config, is_training=False)
        precision = torch.float32

        return model, eval_transform, precision
    

class UNIInferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        UNI initialization.
        """
        super().__init__(**build_kwargs)

    def _build(
        self, 
        target_img_size=None,
    ):
        import timm
        from torchvision import transforms

        self.enc_name = 'uni_v1'
        weights_path = self._get_weights_path()
        img_size = _resolve_target_img_size(self.enc_name, target_img_size, 224, 16)

        timm_kwargs = {
            'img_size': img_size,
            'patch_size': 16,
            'init_values': 1e-5,
            'num_classes': 0,
            'dynamic_img_size': True,
        }

        if weights_path:
            try:
                model = timm.create_model("vit_large_patch16_224", **timm_kwargs)
                model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create UNI model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/MahmoodLab/UNI."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download UNI model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        precision = torch.float16
        return model, eval_transform, precision
    

class UNIv2InferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        UNIv2 initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, target_img_size=None):
        import timm
        from torchvision import transforms

        self.enc_name = 'uni_v2'
        weights_path = self._get_weights_path()
        img_size = _resolve_target_img_size(self.enc_name, target_img_size, 224, 14)

        timm_kwargs = {
            'img_size': img_size,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }

        if weights_path:
            try:
                model = timm.create_model(model_name='vit_giant_patch14_224', pretrained=False, **timm_kwargs)
                model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create UNI2-h model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/MahmoodLab/UNI2-h."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download UNI v2 model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        precision = torch.bfloat16
        return model, eval_transform, precision
    

class GigaPathInferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        GigaPath initialization.
        """
        super().__init__(**build_kwargs)

    def _build(
        self, 
        target_img_size=None,
    ):
        import timm
        # 0.9.16 and 1.0.8 give bit-identical features here; 1.0.8 lets HistoPlus share the env.
        assert timm.__version__ in ('0.9.16', '1.0.8'), f"Gigapath requires timm 0.9.16 or 1.0.8, but found {timm.__version__}."
        from torchvision import transforms

        self.enc_name = 'gigapath'
        weights_path = self._get_weights_path()
        # GigaPath uses a 16px patch size (overriding the patch14 timm backbone name).
        img_size = _resolve_target_img_size(self.enc_name, target_img_size, 224, 16)

        if weights_path:
            try:
                timm_kwargs = {
                    "img_size": img_size,
                    "in_chans": 3,
                    "patch_size": 16,
                    "embed_dim": 1536,
                    "depth": 40,
                    "num_heads": 24,
                    "mlp_ratio": 5.33334,
                    "num_classes": 0,
                    "dynamic_img_size": True,
                }
                model = timm.create_model("vit_giant_patch14_dinov2", pretrained=False, **timm_kwargs)
                model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create GigaPath model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/prov-gigapath/prov-gigapath."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, img_size=img_size, dynamic_img_size=True)
            except:
                traceback.print_exc()
                raise Exception("Failed to download GigaPath model, make sure that you were granted access and that you correctly registered your token")

        mean, std = get_constants('imagenet')
        if target_img_size is None:
            eval_transform = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            eval_transform = transforms.Compose(
                [
                    transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        precision = torch.float16
        return model, eval_transform, precision

    
class VirchowInferenceEncoder(BasePatchEncoder):
    import timm
    
    def __init__(self, **build_kwargs):
        """
        Virchow initialization.
        """
        super().__init__(**build_kwargs)

    def _build(
        self,
        return_cls=False,
        target_img_size=None,
    ):
        import timm
        import torchvision
        from torchvision import transforms

        self.enc_name = 'virchow'
        weights_path = self._get_weights_path()
        img_size = _resolve_target_img_size(self.enc_name, target_img_size, 224, 14)

        timm_kwargs = {
            "img_size": img_size,
            "init_values": 1e-5,
            "num_classes": 0,
            "mlp_ratio": 5.3375,
            "global_pool": "",
            "dynamic_img_size": True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
        }

        if weights_path:
            try:
                model = timm.create_model("vit_huge_patch14_224", **timm_kwargs)
                model.load_state_dict(state_dict=torch.load(weights_path, map_location="cpu"), strict=True)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create Virchow model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/paige-ai/Virchow."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Virchow model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = transforms.Compose(
            [
                transforms.Resize(img_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        precision = torch.float16
        self.return_cls = return_cls
        
        return model, eval_transform, precision

    def forward(self, x):
        output = self.model(x)
        class_token = output[:, 0]

        if self.return_cls:
            return class_token
        else:
            patch_tokens = output[:, 1:]
            embeddings = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
            return embeddings


class Virchow2InferenceEncoder(BasePatchEncoder):
    import timm
    
    def __init__(self, **build_kwargs):
        """
        Virchow 2 initialization.
        """
        super().__init__(**build_kwargs)

    def _build(
        self,
        return_cls=False,
        target_img_size=None,
    ):
        import timm
        import torchvision
        from torchvision import transforms

        self.enc_name = 'virchow2'
        weights_path = self._get_weights_path()
        img_size = _resolve_target_img_size(self.enc_name, target_img_size, 224, 14)

        timm_kwargs = {
            "img_size": img_size,
            "init_values": 1e-5,
            "num_classes": 0,
            "reg_tokens": 4,
            "mlp_ratio": 5.3375,
            "global_pool": "",
            "dynamic_img_size": True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
        }

        if weights_path:
            try:
                model = timm.create_model("vit_huge_patch14_224", **timm_kwargs)
                model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create Virchow2 model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/paige-ai/Virchow2."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Virchow-2 model, make sure that you were granted access and that you correctly registered your token")
        
        eval_transform = transforms.Compose(
            [
                transforms.Resize(img_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        precision = torch.float16
        self.return_cls = return_cls
        
        return model, eval_transform, precision

    def forward(self, x):
        output = self.model(x)
    
        class_token = output[:, 0]
        if self.return_cls:
            return class_token
        
        patch_tokens = output[:, 5:]
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        return embedding


class HOptimus0InferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        H-Optimus0 initialization.
        """
        super().__init__(**build_kwargs)

    def _build(
        self,
        target_img_size=None,
    ):
        import timm
        # 0.9.16 and 1.0.8 give bit-identical features here; 1.0.8 lets HistoPlus share the env.
        assert timm.__version__ in ('0.9.16', '1.0.8'), f"H-Optimus requires timm 0.9.16 or 1.0.8, but found {timm.__version__}."
        from torchvision import transforms

        self.enc_name = 'hoptimus0'
        weights_path = self._get_weights_path()
        img_size = _resolve_target_img_size(self.enc_name, target_img_size, 224, 14)

        timm_kwargs = {
            "num_classes": 0,
            "img_size": img_size,
            "global_pool": "token",
            'init_values': 1e-5,
            'dynamic_img_size': True,
        }

        if weights_path:
            try:
                model = timm.create_model("vit_giant_patch14_reg4_dinov2", **timm_kwargs)
                model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create H-Optimus-0 model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/bioptimus/H-optimus-0."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download HOptimus-0 model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = transforms.Compose([
            transforms.Resize(img_size),  
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])
        
        precision = torch.float16
        return model, eval_transform, precision


class HOptimus1InferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        H-Optimus1 initialization.
        """
        super().__init__(**build_kwargs)

    def _build(
        self,
        target_img_size=None,
        **kwargs
    ):
        import timm
        # 0.9.16 and 1.0.8 give bit-identical features here; 1.0.8 lets HistoPlus share the env.
        assert timm.__version__ in ('0.9.16', '1.0.8'), f"H-Optimus requires timm 0.9.16 or 1.0.8, but found {timm.__version__}."
        from torchvision import transforms

        self.enc_name = 'hoptimus1'
        weights_path = self._get_weights_path()
        img_size = _resolve_target_img_size(self.enc_name, target_img_size, 224, 14)

        timm_kwargs = {
            "num_classes": 0,
            "img_size": img_size,
            "global_pool": "token",
            'init_values': 1e-5,
            'dynamic_img_size': True,
        }

        if weights_path:
            try:
                model = timm.create_model("vit_giant_patch14_reg4_dinov2", **timm_kwargs)
                model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create H-Optimus-1 model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/bioptimus/H-optimus-1."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download HOptimus-1 model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = transforms.Compose([
            transforms.Resize(img_size),  
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])
        
        precision = torch.float16
        return model, eval_transform, precision


class Phikonv2InferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        Phikonv2 initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        from transformers import AutoModel
        import torchvision.transforms as T
        from .utils.constants import IMAGENET_MEAN, IMAGENET_STD

        self.enc_name = 'phikon_v2'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                model_dir = os.path.dirname(weights_path)
                model = AutoModel.from_pretrained(model_dir)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create Phikonv2 model from local checkpoint at '{weights_path}'. "
                    "You can download the required `model.safetensors` and `config.json` from: https://huggingface.co/owkin/phikon-v2."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = AutoModel.from_pretrained("owkin/phikon-v2")
            except:
                traceback.print_exc()
                raise Exception("Failed to download Phikon v2 model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = T.Compose([
            T.Resize(224),  
            T.CenterCrop(224),  
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # Normalize with specified mean and std
        ])

        precision = torch.float32
        return model, eval_transform, precision
    
    def forward(self, x):
        out = self.model(x)
        out = out.last_hidden_state[:, 0, :]
        return out


class Conchv15InferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        CONCHv1.5 initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, img_size=448):
        from trident.patch_encoder_models.model_zoo.conchv1_5.conchv1_5 import create_model_from_pretrained

        self.enc_name = 'conch_v15'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                model, eval_transform = create_model_from_pretrained(checkpoint_path=weights_path, img_size=img_size)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create CONCH v1.5 model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model_vision.bin` and `config.json` from: https://huggingface.co/MahmoodLab/conchv1_5."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model, eval_transform = create_model_from_pretrained(checkpoint_path="hf_hub:MahmoodLab/conchv1_5", img_size=img_size)
            except:
                traceback.print_exc()
                raise Exception("Failed to download CONCH v1.5 model, make sure that you were granted access and that you correctly registered your token")

        precision = torch.float16
        return model, eval_transform, precision


class Midnight12kInferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        Midnight 12-k initialization by Kaiko.
        """
        super().__init__(**build_kwargs)

    def _build(self, return_type: Literal["cls_token", "cls+mean"] = "cls_token"):
        from transformers import AutoModel
        from .utils.constants import KAIKO_MEAN, KAIKO_STD
        from torchvision import transforms

        self.enc_name = "midnight12k"
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                model_dir = os.path.dirname(weights_path)
                model = AutoModel.from_pretrained(model_dir)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create Midnight-12k model from local checkpoint at '{weights_path}'. "
                    "You can download the required `model.safetensors` and `config.json` from: https://huggingface.co/kaiko-ai/midnight."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = AutoModel.from_pretrained("kaiko-ai/midnight")
            except:
                traceback.print_exc()
                raise Exception("Failed to download Midnight-12k model")

        eval_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=KAIKO_MEAN, std=KAIKO_STD),
            ]
        )

        precision = torch.float16
        self.return_type = return_type
        return model, eval_transform, precision

    def forward(self, x):
        out = self.model(x).last_hidden_state
        cls_token = out[:, 0, :]
        if self.return_type == "cls_token":
            return cls_token
        elif self.return_type == "cls+mean":
            patch_embeddings = out[:, 1:, :]
            return torch.cat([cls_token, patch_embeddings.mean(1)], dim=-1)
        else:
            raise ValueError(
                f"expected return_type to be one of 'cls_token' or 'cls+mean', but got '{self.return_type}'"
            )


class H0MiniInferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        H0-mini initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, return_type: Literal["cls_token", "cls+mean"] = "cls_token", target_img_size=None):
        import timm
        from timm.data import resolve_model_data_config
        from timm.data.transforms_factory import create_transform

        self.enc_name = "h0-mini"
        weights_path = self._get_weights_path()
        self.return_type = return_type
        img_size = _resolve_target_img_size(self.enc_name, target_img_size, 224, 14)

        if weights_path:
            raise NotImplementedError(
                "H0-mini currently supports loading from Hugging Face only. "
                "Please leave `weights_path` unset."
            )

        self.ensure_has_internet(self.enc_name)
        try:
            model = timm.create_model(
                "hf-hub:bioptimus/H0-mini",
                pretrained=True,
                mlp_layer=timm.layers.SwiGLUPacked,
                act_layer=torch.nn.SiLU,
                img_size=img_size,
                dynamic_img_size=True,
            )
        except Exception:
            traceback.print_exc()
            raise Exception(
                "Failed to download H0-mini model, make sure that you were granted access "
                "and that you correctly registered your token"
            )

        # timm>=0.9 expects the model instance directly here.
        data_config = resolve_model_data_config(model)
        if target_img_size is not None:
            data_config["input_size"] = (3, img_size, img_size)
            data_config["crop_pct"] = 1.0
        eval_transform = create_transform(**data_config, is_training=False)
        precision = torch.float16

        return model, eval_transform, precision

    def forward(self, x):
        out = self.model(x)
        if out.ndim != 3:
            return out
        cls_token = out[:, 0, :]
        if self.return_type == "cls_token":
            return cls_token
        elif self.return_type == "cls+mean":
            patch_embeddings = out[:, self.model.num_prefix_tokens:, :]
            return torch.cat([cls_token, patch_embeddings.mean(1)], dim=-1)
        else:
            raise ValueError(
                f"expected return_type to be one of 'cls_token' or 'cls+mean', but got '{self.return_type}'"
            )


class OpenMidnightInferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        OpenMidnight initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        from huggingface_hub import hf_hub_download
        from torchvision import transforms

        self.enc_name = "openmidnight"
        weights_path = self._get_weights_path()

        try:
            model = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vitg14_reg",
                pretrained=False,
            )
        except Exception:
            traceback.print_exc()
            raise Exception("Failed to initialize DINOv2 ViT-G/14 backbone for OpenMidnight.")

        if not weights_path:
            self.ensure_has_internet(self.enc_name)
            try:
                weights_path = hf_hub_download(
                    repo_id="SophontAI/OpenMidnight",
                    filename="teacher_checkpoint_load.pt",
                )
            except Exception:
                traceback.print_exc()
                raise Exception(
                    "Failed to download OpenMidnight model, make sure that you were granted access "
                    "and that you correctly registered your token"
                )

        try:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            pos_embed = checkpoint["pos_embed"]
            model.pos_embed = torch.nn.parameter.Parameter(pos_embed)
            model.load_state_dict(checkpoint, strict=True)
        except Exception:
            traceback.print_exc()
            raise Exception(
                f"Failed to create OpenMidnight model from checkpoint at '{weights_path}'. "
                "You can download the required `teacher_checkpoint_load.pt` from: "
                "https://huggingface.co/SophontAI/OpenMidnight."
            )

        mean, std = get_constants("imagenet")
        eval_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        precision = torch.float16
        return model, eval_transform, precision

    def forward(self, x):
        return self.model(x)


class GPFMInferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        GPFM initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, target_img_size=None):
        import timm
        from huggingface_hub import hf_hub_download
        from torchvision.transforms import InterpolationMode

        self.enc_name = "gpfm"
        weights_path = self._get_weights_path()
        img_size = _resolve_target_img_size(self.enc_name, target_img_size, 224, 14)

        if not weights_path:
            self.ensure_has_internet(self.enc_name)
            try:
                weights_path = hf_hub_download(repo_id="majiabo/GPFM", filename="GPFM.pth")
            except Exception:
                traceback.print_exc()
                raise Exception("Failed to download GPFM model.")

        try:
            model = timm.create_model(
                "vit_large_patch14_dinov2.lvd142m",
                pretrained=False,
                img_size=img_size,
                init_values=1e-5,
                dynamic_img_size=True,
            )
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            traceback.print_exc()
            raise Exception(
                f"Failed to create GPFM model from checkpoint at '{weights_path}'. "
                "You can download the required `GPFM.pth` from: https://huggingface.co/majiabo/GPFM."
            )

        mean, std = get_constants("imagenet")
        eval_transform = get_eval_transforms(
            mean,
            std,
            target_img_size=img_size,
            interpolation=InterpolationMode.BICUBIC,
            max_size=None,
            antialias=True,
        )
        precision = torch.float16
        return model, eval_transform, precision


class GenBioPathFMInferenceEncoder(BasePatchEncoder):

    def __init__(self, **build_kwargs):
        """
        GenBio-PathFM initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        from huggingface_hub import hf_hub_download
        from torchvision.transforms import InterpolationMode
        from trident.patch_encoder_models.model_zoo.genbio_pathfm.genbio_pathfm import GenBioPathFMInference

        self.enc_name = "genbio-pathfm"
        weights_path = self._get_weights_path()

        if not weights_path:
            self.ensure_has_internet(self.enc_name)
            try:
                weights_path = hf_hub_download(
                    repo_id="genbio-ai/genbio-pathfm",
                    filename="model.pth",
                )
            except:
                traceback.print_exc()
                raise Exception(
                    "Failed to download GenBio-PathFM model. "
                    "You can manually download 'model.pth' from: "
                    "https://huggingface.co/genbio-ai/genbio-pathfm and set the path in local_ckpts.json."
                )

        try:
            model = GenBioPathFMInference(weights_path, device="cpu")
        except:
            traceback.print_exc()
            raise Exception(
                f"Failed to create GenBio-PathFM model from local checkpoint at '{weights_path}'. "
                "You can download the required 'model.pth' from: "
                "https://huggingface.co/genbio-ai/genbio-pathfm."
            )

        mean, std = get_constants("genbio_pathfm")
        eval_transform = get_eval_transforms(
            mean,
            std,
            target_img_size=224,
            interpolation=InterpolationMode.BILINEAR,
            max_size=None,
            antialias=True,
        )

        precision = torch.bfloat16
        return model, eval_transform, precision

    def forward(self, x):
        return self.model(x)


class Gemma4InferenceEncoder(BasePatchEncoder):
    """Gemma 4 vision tower (base class). Subclassed per variant (see below)."""
    VARIANT = None    # "e4b" or "26b", set in subclasses
    HF_REPO = None    # HuggingFace repo id, set in subclasses

    def __init__(self, **build_kwargs):
        super().__init__(**build_kwargs)

    def _build(self):
        from PIL import Image

        try:
            from transformers import (
                Gemma4Config,
                Gemma4VisionModel,
                Gemma4ImageProcessor,
            )
        except ImportError:
            raise ImportError(
                "Gemma 4 requires transformers>=5.0. "
                "Install with: pip install 'transformers>=5.0'"
            )

        self.enc_name = f"gemma4-{self.VARIANT}"
        weights_path = self._get_weights_path()

        if not weights_path:
            self.ensure_has_internet(self.enc_name)
            try:
                from huggingface_hub import snapshot_download
                weights_path = snapshot_download(
                    repo_id=self.HF_REPO,
                    allow_patterns=[
                        "config.json",
                        "processor_config.json",
                        # Gemma checkpoints may be single-file `model.safetensors` or sharded
                        # `model-00001-of-000XX.safetensors` + `model.safetensors.index.json`.
                        "model.safetensors",
                        "model.safetensors.index.json",
                        "model-*.safetensors*",
                    ],
                )
            except Exception:
                traceback.print_exc()
                raise Exception(
                    f"Failed to download Gemma 4 ({self.VARIANT}). Provide a "
                    "local directory with config.json + safetensors via weights_path, "
                    "or set the path in local_ckpts.json."
                )

        cfg = Gemma4Config.from_pretrained(weights_path)
        vision = Gemma4VisionModel(cfg.vision_config)

        state = self._load_gemma4_vision_state(weights_path)
        missing, _ = vision.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(
                f"Gemma 4 vision tower load: {len(missing)} missing keys "
                f"(first: {missing[:3]})"
            )

        processor = Gemma4ImageProcessor.from_pretrained(weights_path)
        image_position_ids = processor(images=Image.new("RGB", (224, 224)), return_tensors="pt").get("image_position_ids")

        model = self._GemmaWrapper(vision, image_position_ids)
        eval_transform = self._GemmaTransform(processor)
        precision = torch.bfloat16
        return model, eval_transform, precision

    def forward(self, x):
        return self.model(x)

    def _load_gemma4_vision_state(self, model_path: str):
        """Read tensors with prefix `model.vision_tower.` from a Gemma 4 checkpoint
        without materializing the LLM weights. The full multimodal checkpoint is large
        (the 26B's vision tower lives inside a ~50 GB shard), so instead of loading the
        whole shard we parse the safetensors header and seek+read only the vision tensor
        byte ranges. Works for single-shard (E4B) and multi-shard (26B) layouts. If a
        pre-extracted `vision_tower_only.safetensors` exists in the model dir, it is used
        directly.
        """
        import json
        import struct
        import numpy as np

        cached = os.path.join(model_path, "vision_tower_only.safetensors")
        if os.path.exists(cached):
            from safetensors.torch import load_file
            return load_file(cached)

        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            shards = sorted({os.path.join(model_path, v) for v in weight_map.values()})
        else:
            single = os.path.join(model_path, "model.safetensors")
            if not os.path.exists(single):
                raise FileNotFoundError(f"No safetensors found in {model_path}")
            shards = [single]

        dtype_map = {
            "F16": (torch.float16, np.float16),
            "BF16": (torch.bfloat16, None),
            "F32": (torch.float32, np.float32),
            "F64": (torch.float64, np.float64),
            "I8": (torch.int8, np.int8),
            "I16": (torch.int16, np.int16),
            "I32": (torch.int32, np.int32),
            "I64": (torch.int64, np.int64),
            "U8": (torch.uint8, np.uint8),
            "BOOL": (torch.bool, np.bool_),
        }
        prefix = "model.vision_tower."
        state = {}
        for shard in shards:
            with open(shard, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))
                data_start = 8 + header_len
                items = [
                    (k, m) for k, m in header.items()
                    if k != "__metadata__" and k.startswith(prefix)
                ]
                items.sort(key=lambda kv: kv[1]["data_offsets"][0])
                for k, meta in items:
                    torch_dtype, np_dtype = dtype_map[meta["dtype"]]
                    shape = meta["shape"]
                    start, end = meta["data_offsets"]
                    f.seek(data_start + start)
                    buf = f.read(end - start)
                    if torch_dtype == torch.bfloat16:
                        arr = np.frombuffer(buf, dtype=np.uint16).copy()
                        t = torch.from_numpy(arr).view(torch.bfloat16).reshape(shape)
                    else:
                        arr = np.frombuffer(buf, dtype=np_dtype).copy()
                        t = torch.from_numpy(arr).reshape(shape)
                    state[k[len(prefix):]] = t
        return state

    class _GemmaTransform:
        def __init__(self, processor):
            self.processor = processor
        def __call__(self, img):
            return self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

    class _GemmaWrapper(torch.nn.Module):
        def __init__(self, vision_model, image_position_ids):
            super().__init__()
            self.vision_model = vision_model
            self.register_buffer("image_position_ids", image_position_ids, persistent=False)
        def forward(self, x):
            x = x.unsqueeze(0) if x.dim() == 2 else x
            bs = x.shape[0]
            pid = self.image_position_ids
            if pid.shape[0] == 1 and bs > 1:
                pid = pid.expand(bs, -1, -1)
            out = self.vision_model(pixel_values=x, pixel_position_ids=pid)
            lhs = out.last_hidden_state            # (bs * tokens_per_image, hidden)
            # Gemma 4 vision flattens the batch into the token axis and emits a fixed
            # 256 tokens per 224px image (same for E4B and 26B, the pinned checkpoints).
            tokens_per_image = 256
            assert lhs.shape[0] == bs * tokens_per_image, (
                f"Gemma4 vision: expected {bs * tokens_per_image} tokens, got {lhs.shape[0]}"
            )
            return lhs.reshape(bs, tokens_per_image, -1).mean(dim=1)   # (bs, hidden)


class Gemma4E4BInferenceEncoder(Gemma4InferenceEncoder):
    """Gemma 4 E4B vision tower (hidden=768)."""
    VARIANT = "e4b"
    HF_REPO = "google/gemma-4-E4B"

    def __init__(self, **build_kwargs):
        super().__init__(**build_kwargs)


class Gemma426BInferenceEncoder(Gemma4InferenceEncoder):
    """Gemma 4 26B-A4B vision tower (hidden=1152)."""
    VARIANT = "26b"
    HF_REPO = "google/gemma-4-26B-A4B"

    def __init__(self, **build_kwargs):
        super().__init__(**build_kwargs)


encoder_registry = {
    "conch_v1": Conchv1InferenceEncoder,
    "conch_v15": Conchv15InferenceEncoder,
    "uni_v1": UNIInferenceEncoder,
    "uni_v2": UNIv2InferenceEncoder,
    "ctranspath": CTransPathInferenceEncoder,
    "phikon": PhikonInferenceEncoder,
    "phikon_v2": Phikonv2InferenceEncoder,
    "resnet50": ResNet50InferenceEncoder,
    "keep": KeepInferenceEncoder,
    "gigapath": GigaPathInferenceEncoder,
    "virchow": VirchowInferenceEncoder,
    "virchow2": Virchow2InferenceEncoder,
    "hoptimus0": HOptimus0InferenceEncoder,
    "hoptimus1": HOptimus1InferenceEncoder,
    "h0-mini": H0MiniInferenceEncoder,
    "musk": MuskInferenceEncoder,
    "openmidnight": OpenMidnightInferenceEncoder,
    "gpfm": GPFMInferenceEncoder,
    "hibou_l": HibouLInferenceEncoder,
    "kaiko-vitb8": KaikoB8InferenceEncoder,
    "kaiko-vitb16": KaikoB16InferenceEncoder,
    "kaiko-vits8": KaikoS8InferenceEncoder,
    "kaiko-vits16": KaikoS16InferenceEncoder,
    "kaiko-vitl14": KaikoL14InferenceEncoder,
    "lunit-vits8": LunitS8InferenceEncoder,
    "midnight12k": Midnight12kInferenceEncoder,
    "genbio-pathfm": GenBioPathFMInferenceEncoder,
    "gemma4-e4b": Gemma4E4BInferenceEncoder,
    "gemma4-26b": Gemma426BInferenceEncoder,
}
