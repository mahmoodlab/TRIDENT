import traceback
from abc import abstractmethod
from typing import Literal, Optional
import torch
import os 

from trident.patch_encoder_models.utils.constants import get_constants
from trident.patch_encoder_models.utils.transform_utils import get_eval_transforms
from trident.IO import get_weights_path, has_internet_connection

"""
This file contains an assortment of pretrained patch encoders, all loadable via the encoder_factory() function.
"""

def encoder_factory(model_name: str, **kwargs):
    """
    Instantiate a patch encoder model by name.

    This factory function returns a pre-configured encoder model class based on the provided
    `model_name`. Each encoder is designed for extracting representations from image patches
    using specific backbones or pretraining strategies.

    Args:
        model_name (str): Name of the encoder to instantiate. Must be one of the following:
            - "conch_v1"
            - "conch_v15"
            - "uni_v1"
            - "uni_v2"
            - "ctranspath"
            - "phikon"
            - "phikon_v2"
            - "resnet50"
            - "gigapath"
            - "virchow"
            - "virchow2"
            - "hoptimus0"
            - "hoptimus1"
            - "musk"
            - "hibou_l"
            - "kaiko-vitb8"
            - "kaiko-vitb16"
            - "kaiko-vits8"
            - "kaiko-vits16"
            - "kaiko-vitl14"
            - "lunit-vits8"

        **kwargs: Optional keyword arguments passed directly to the encoder constructor. These
            may include parameters such as:
            - weights_path (str): Path to a local checkpoint (optional)
            - normalize (bool): Whether to normalize output embeddings (default: False)
            - with_proj (bool): Whether to apply the projection head (default: True)
            - any model-specific configuration parameters

    Returns:
        torch.nn.Module: An instance of the specified encoder model.

    Raises:
        ValueError: If `model_name` is not among the recognized encoder names.
    """
    if model_name == 'conch_v1':
        enc = Conchv1InferenceEncoder
    elif model_name == 'conch_v15':
        enc = Conchv15InferenceEncoder
    elif model_name == 'uni_v1':
        enc = UNIInferenceEncoder
    elif model_name == 'uni_v2':
        enc = UNIv2InferenceEncoder
    elif model_name == 'ctranspath':
        enc = CTransPathInferenceEncoder
    elif model_name == 'phikon':
        enc = PhikonInferenceEncoder
    elif model_name == 'resnet50':
        enc = ResNet50InferenceEncoder
    elif model_name == 'gigapath':
        enc = GigaPathInferenceEncoder
    elif model_name == 'virchow':
        enc = VirchowInferenceEncoder
    elif model_name == 'virchow2':
        enc = Virchow2InferenceEncoder
    elif model_name == 'hoptimus0':
        enc = HOptimus0InferenceEncoder
    elif model_name == 'hoptimus1':
        enc = HOptimus1InferenceEncoder
    elif model_name == 'phikon_v2':
        enc = Phikonv2InferenceEncoder
    elif model_name == 'musk':
        enc = MuskInferenceEncoder
    elif model_name == 'hibou_l':
        enc = HibouLInferenceEncoder
    elif model_name == 'kaiko-vitb8':
        enc = KaikoB8InferenceEncoder
    elif model_name == 'kaiko-vitb16':
        enc = KaikoB16InferenceEncoder
    elif model_name == 'kaiko-vits8':
        enc = KaikoS8InferenceEncoder
    elif model_name == 'kaiko-vits16':
        enc = KaikoS16InferenceEncoder
    elif model_name == 'kaiko-vitl14':
        enc = KaikoL14InferenceEncoder
    elif model_name == 'lunit-vits8':
        enc = LunitS8InferenceEncoder
    elif model_name == 'midnight12k':
        enc = Midnight12kInferenceEncoder
    else:
        raise ValueError(f"Unknown encoder name {model_name}")

    return enc(**kwargs)


class BasePatchEncoder(torch.nn.Module):

    _has_internet = has_internet_connection()
    
    def __init__(self, weights_path: Optional[str] = None, **build_kwargs):
        """
        Initialize BasePatchEncoder.

        Args:
            weights_path (Optional[str]): 
                Optional path to local model weights. If None, the model is loaded from the model registry or downloaded from Hugging Face Hub.
            **build_kwargs: 
                Additional keyword arguments passed to the `_build()` method to customize model creation.

        Attributes:
            enc_name (Optional[str]): Name of the encoder architecture (set during `_build()`).
            weights_path (Optional[str]): Path to local model weights (if provided).
            model (nn.Module): The instantiated encoder model.
            eval_transforms (Callable): Evaluation-time preprocessing transforms.
            precision (torch.dtype): Precision used for inference.
        """

        super().__init__()
        self.enc_name: Optional[str] = None
        self.weights_path: Optional[str] = weights_path
        self.model, self.eval_transforms, self.precision = self._build(**build_kwargs)

    def ensure_valid_weights_path(self, weights_path):
        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Expected checkpoint at '{weights_path}', but the file was not found.")
    
    def ensure_has_internet(self, enc_name):
        if not BasePatchEncoder._has_internet:
            raise FileNotFoundError(
                f"Internet connection does seem not available. Auto checkpoint download is disabled."
                f"To proceed, please manually download: {enc_name},\n"
                f"and place it in the model registry in:\n`trident/patch_encoder_models/local_ckpts.json`"
            )
        
    def _get_weights_path(self):
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

    def forward(self, x):
        """
        Can be overwritten if model requires special forward pass.
        """
        z = self.model(x)
        return z
        
    @abstractmethod
    def _build(self, **build_kwargs):
        pass


class CustomInferenceEncoder(BasePatchEncoder):

    def __init__(self, enc_name, model, transforms, precision):
        """
        Initialize a CustomInferenceEncoder from user-defined components.

        This class is used when the model, transforms, and precision are pre-instantiated externally 
        and should be injected directly into the encoder wrapper.

        Args:
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
        
    def _build(self):
        return None, None, None


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

    def __init__(self, **build_kwargs):
        """
        Kaiko initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        import timm
        from torchvision.transforms import InterpolationMode
        self.enc_name = f"kaiko-{self.MODEL_NAME}"
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                model = timm.create_model(
                    f"{self.HF_HUB_ID}",
                    num_classes=0,
                    checkpoint_path=weights_path,
                    img_size=self.IMG_SIZE,
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
                    img_size=self.IMG_SIZE,
                )
            except:
                traceback.print_exc()
                raise Exception("Failed to download Kaiko model.")

        mean, std = get_constants("kaiko")
        eval_transform = get_eval_transforms(mean, std, target_img_size=224, center_crop=True, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)
        precision = torch.float32

        return model, eval_transform, precision

    def forward(self, x):
        return self.model(x)


class KaikoS16InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vits16"
    HF_HUB_ID = "vit_small_patch16_224"
    IMG_SIZE = 224

    def __init__(self, **build_kwargs):
        """
        Kaiko Small 16 initialization.
        """
        super().__init__(**build_kwargs)
    

class KaikoS8InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vits8"
    HF_HUB_ID = "vit_small_patch8_224"
    IMG_SIZE = 224

    def __init__(self, **build_kwargs):
        """
        Kaiko Small 8 initialization.
        """
        super().__init__(**build_kwargs)
    

class KaikoB16InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vitb16"
    HF_HUB_ID = "vit_base_patch16_224"
    IMG_SIZE = 224

    def __init__(self, **build_kwargs):
        """
        Kaiko Base 16 initialization.
        """
        super().__init__(**build_kwargs)
    

class KaikoB8InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vitb8"
    HF_HUB_ID = "vit_base_patch8_224"
    IMG_SIZE = 224

    def __init__(self, **build_kwargs):
        """
        Kaiko Base 8 initialization.
        """
        super().__init__(**build_kwargs)
    

class KaikoL14InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vitl14"
    HF_HUB_ID = "vit_large_patch14_reg4_dinov2"
    IMG_SIZE = 518

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
                model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create ResNet50 model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/timm/resnet50.tv_in1k."
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

    def _build(self):
        import timm
        from timm.data import resolve_model_data_config
        from timm.data.transforms_factory import create_transform

        self.enc_name = 'lunit-vits8'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                timm_kwargs = {"img_size": 224}
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
                model = timm.create_model("hf-hub:1aurent/vit_small_patch8_224.lunit_dino", pretrained=True)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Lunit S8 model, make sure that you were granted access and that you correctly registered your token.")

        data_config = resolve_model_data_config(model)
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
        timm_kwargs={"dynamic_img_size": True, "num_classes": 0, "init_values": 1e-5}
    ):
        import timm
        from torchvision import transforms

        self.enc_name = 'uni_v1'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                timm_kwargs = {
                    'img_size': 224,
                    'patch_size': 16,
                    'init_values': 1e-5,
                    'num_classes': 0,
                    'dynamic_img_size': True,
                }
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
            transforms.Resize(224),
            transforms.CenterCrop(224),
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

    def _build(self):
        import timm
        from torchvision import transforms

        self.enc_name = 'uni_v2'
        weights_path = self._get_weights_path()

        timm_kwargs = {
            'img_size': 224,
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
            transforms.Resize(224),
            transforms.CenterCrop(224),
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
    ):
        import timm
        assert timm.__version__ == '0.9.16', f"Gigapath requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
        from torchvision import transforms

        self.enc_name = 'gigapath'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                timm_kwargs = {
                    "img_size": 224,
                    "in_chans": 3,
                    "patch_size": 16,
                    "embed_dim": 1536,
                    "depth": 40,
                    "num_heads": 24,
                    "mlp_ratio": 5.33334,
                    "num_classes": 0
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
                model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
            except:
                traceback.print_exc()
                raise Exception("Failed to download GigaPath model, make sure that you were granted access and that you correctly registered your token")

        mean, std = get_constants('imagenet')
        eval_transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        precision = torch.float32
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
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
    ):
        import timm
        import torchvision
        from torchvision import transforms

        self.enc_name = 'virchow'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                timm_kwargs = {
                    "img_size": 224,
                    "init_values": 1e-5,
                    "num_classes": 0,
                    "mlp_ratio": 5.3375,
                    "global_pool": "",
                    "dynamic_img_size": True,
                    'mlp_layer': timm.layers.SwiGLUPacked,
                    'act_layer': torch.nn.SiLU,
                }
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
                transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
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
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
    ):
        import timm
        import torchvision
        from torchvision import transforms

        self.enc_name = 'virchow2'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                timm_kwargs = {
                    "img_size": 224,
                    "init_values": 1e-5,
                    "num_classes": 0,
                    "reg_tokens": 4,
                    "mlp_ratio": 5.3375,
                    "global_pool": "",
                    "dynamic_img_size": True,
                    'mlp_layer': timm.layers.SwiGLUPacked,
                    'act_layer': torch.nn.SiLU,
                }
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
                transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
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
        timm_kwargs={'init_values': 1e-5, 'dynamic_img_size': False}
    ):
        import timm
        assert timm.__version__ == '0.9.16', f"H-Optimus requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
        from torchvision import transforms

        self.enc_name = 'hoptimus0'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                timm_kwargs = {
                    "num_classes": 0,
                    "img_size": 224,
                    "global_pool": "token",
                    'init_values': 1e-5,
                    'dynamic_img_size': False
                }
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
            transforms.Resize(224),  
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
        timm_kwargs={'init_values': 1e-5, 'dynamic_img_size': False},
        **kwargs
    ):
        import timm
        assert timm.__version__ == '0.9.16', f"H-Optimus requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
        from torchvision import transforms

        self.enc_name = 'hoptimus1'
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                timm_kwargs = {
                    "num_classes": 0,
                    "img_size": 224,
                    "global_pool": "token",
                    'init_values': 1e-5,
                    'dynamic_img_size': False
                }
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
            transforms.Resize(224),  
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

        precision = torch.float32
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
