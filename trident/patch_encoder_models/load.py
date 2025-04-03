import traceback
from abc import abstractmethod
import torch
import os 

from trident.patch_encoder_models.utils.constants import get_constants
from trident.patch_encoder_models.utils.transform_utils import get_eval_transforms
from trident.IO import get_weights_path

"""
This file contains an assortment of pretrained patch encoders, all loadable via the encoder_factory() function.
"""

def encoder_factory(model_name, **kwargs):
    '''
    Build a patch encoder model.

    Args:   
        model_name (str): The name of the model to build.
        **kwargs: Additional arguments to pass to the encoder constructor.

    Returns:
        torch.nn.Module: The patch encoder model.
    '''

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
    else:
        raise ValueError(f"Unknown encoder name {model_name}")

    return enc(**kwargs)

####################################################################################################

class BasePatchEncoder(torch.nn.Module):
    
    def __init__(self, **build_kwargs):
        super().__init__()
        self.enc_name = None
        self.model, self.eval_transforms, self.precision = self._build(**build_kwargs)
        
    def forward(self, x):
        '''
        Can be overwritten if model requires special forward pass.
        '''
        z = self.model(x)
        return z
        
    @abstractmethod
    def _build(self, **build_kwargs):
        pass


class CustomInferenceEncoder(BasePatchEncoder):
    def __init__(self, enc_name, model, transforms, precision):
        super().__init__()
        self.enc_name = enc_name
        self.model = model
        self.eval_transforms = transforms
        self.precision = precision
        
    def _build(self):
        return None, None, None


class MuskInferenceEncoder(BasePatchEncoder):
    
    def _build(self, inference_aug = False, with_proj = False, out_norm = False, return_global = True):
        '''
        Args:
            inference_aug (bool): Whether to use test-time multiscale augmentation. Default is False to allow for fair comparison with other models.
        '''
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

        weights_path = get_weights_path('patch', self.enc_name)

        if weights_path != "":
            model = timm.create_model("musk_large_patch16_384", checkpoint_path=weights_path)
            utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '')

        else:
            print("Loading model weights from cache, download if not exists...")
            try:
                model = timm.create_model("musk_large_patch16_384")
                utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '')
            except:
                traceback.print_exc()
                raise Exception("Failed to download MUSK model, make sure that you were granted access and that you correctly registered your token")
        
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        from torchvision.transforms import Compose, Resize, InterpolationMode
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
    def _build(self, with_proj = False, normalize = False):
        self.enc_name = 'conch_v1'
        self.with_proj = with_proj
        self.normalize = normalize

        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except:
            traceback.print_exc()
            raise Exception("Please install CONCH `pip install git+https://github.com/Mahmoodlab/CONCH.git`")
        
        weights_path = get_weights_path('patch', self.enc_name)

        if weights_path != "":
            model, eval_transform = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path=weights_path)

        else:
            print("Loading model weights from cache, download if not exists...")
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
    def _build(self):
        from torchvision.transforms import Compose, Resize, InterpolationMode
        from torch import nn

        try:
            from .model_zoo.ctranspath.ctran import ctranspath
        except:
            traceback.print_exc()
            raise Exception("Failed to import CTransPath model, make sure timm_ctp is installed. `pip install timm_ctp`")
        
        self.enc_name = 'ctranspath'
        weights_path = get_weights_path('patch', self.enc_name)
        model = ctranspath(img_size=224)
        model.head = nn.Identity()

        if weights_path != "":
            pass

        else:
            print("Loading model weights from cache, download if not exists...")
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

        state_dict = torch.load(weights_path, weights_only = True)['model']
        state_dict = {key: val for key, val in state_dict.items() if 'attn_mask' not in key}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        assert len(unexpected) == 0, f"Unexpected keys found in state dict: {unexpected}"
        assert missing == ['layers.0.blocks.1.attn_mask', 'layers.1.blocks.1.attn_mask', 'layers.2.blocks.1.attn_mask', 'layers.2.blocks.3.attn_mask', 'layers.2.blocks.5.attn_mask'], f"Unexpected missing keys: {missing}"

        mean, std = get_constants('imagenet')
        eval_transform = get_eval_transforms(mean, std, target_img_size=224, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)

        precision = torch.float32
        
        return model, eval_transform, precision


class PhikonInferenceEncoder(BasePatchEncoder):
    def _build(self):
        from transformers import ViTModel
        from torchvision.transforms import Compose, Resize, InterpolationMode

        self.enc_name = 'phikon'
        weights_path = get_weights_path('patch', self.enc_name)
        
        if weights_path != "":
            model = ViTModel.from_pretrained(os.path.dirname(weights_path), add_pooling_layer=False, local_files_only=True)

        else:
            print("Loading model weights from cache, download if not exists...")
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
    def _build(self):
        from transformers import AutoModel
        from torchvision.transforms import InterpolationMode

        self.enc_name = 'hibou_l'
        weights_path = get_weights_path('patch', self.enc_name)

        if weights_path != "":
            model = AutoModel.from_pretrained(os.path.dirname(weights_path))

        else:
            print("Loading model weights from cache, download if not exists...")
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

    def _build(self):
        from torchvision.transforms import InterpolationMode
        self.enc_name = f"kaiko-{self.MODEL_NAME}"
        weights_path = get_weights_path("patch", self.enc_name)

        if weights_path != "":
            try:
                model = torch.hub.load(os.path.join(torch.hub.get_dir(), "kaiko-ai_towards_large_pathology_fms_main"), self.MODEL_NAME, trust_repo=True, source="local")
            except:
                traceback.print_exc()
                raise Exception(
                    f"Failed to locate Kaiko github repository {os.path.join(torch.hub.get_dir(), 'kaiko-ai_towards_large_pathology_fms_main')}"
                    f"in PyTorch cache, make sure that you renamed the repository with the correct name if you download manually using git clone"
                )
                
            model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))

        else:
            print("Loading model weights from cache, download if not exists...")
            try:
                model = torch.hub.load("kaiko-ai/towards_large_pathology_fms", self.MODEL_NAME, trust_repo=True)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Kaiko model, make sure that you were granted access and that you correctly registered your token")

        mean, std = get_constants("kaiko")
        eval_transform = get_eval_transforms(mean, std, target_img_size=224, center_crop=True, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)
        precision = torch.float32

        return model, eval_transform, precision

    def forward(self, x):
        return self.model(x)


class KaikoS16InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vits16"


class KaikoS8InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vits8"


class KaikoB16InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vitb16"


class KaikoB8InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vitb8"


class KaikoL14InferenceEncoder(KaikoInferenceEncoder):
    MODEL_NAME = "vitl14"


class ResNet50InferenceEncoder(BasePatchEncoder):
    def _build(
        self, 
        pretrained=True, 
        timm_kwargs={"features_only": True, "out_indices": [3], "num_classes": 0},
        img_size=224,
        pool=True
    ):
        import timm
        from torchvision.transforms import Compose, Resize, InterpolationMode

        self.enc_name = 'resnet50'
        weights_path = get_weights_path('patch', self.enc_name)

        if weights_path != "":
            model = timm.create_model("resnet50.tv_in1k", pretrained=pretrained, pretrained_cfg_overlay=dict(file=weights_path), **timm_kwargs)

        else:
            print("Loading model weights from cache, download if not exists...")
            try:
                model = timm.create_model("resnet50.tv_in1k", pretrained=pretrained, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download ResNet50 model, make sure that you were granted access and that you correctly registered your token")

        mean, std = get_constants('imagenet')
        eval_transform = get_eval_transforms(mean, std, target_img_size=img_size, center_crop = True, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True)

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
    def _build(self):
        import timm
        from timm.data import resolve_model_data_config
        from timm.data.transforms_factory import create_transform

        self.enc_name = 'lunit-vits8'
        weights_path = get_weights_path('patch', self.enc_name)
        
        if weights_path != "":
            model = timm.create_model("hf-hub:1aurent/vit_small_patch8_224.lunit_dino", pretrained=True, checkpoint_path=weights_path)

        else:
            print("Loading model weights from cache, download if not exists...")
            try:
                model = timm.create_model("hf-hub:1aurent/vit_small_patch8_224.lunit_dino", pretrained=True)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Lunit S8 model, make sure that you were granted access and that you correctly registered your token")

        data_config = resolve_model_data_config(model)
        eval_transform = create_transform(**data_config, is_training=False)
        precision = torch.float32

        return model, eval_transform, precision
    

class UNIInferenceEncoder(BasePatchEncoder):
    def _build(
        self, 
        timm_kwargs={"dynamic_img_size": True, "num_classes": 0, "init_values": 1.0}
    ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        self.enc_name = 'uni_v1'
        weights_path = get_weights_path('patch', self.enc_name)
        
        if weights_path != "":
            model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, checkpoint_path=weights_path, **timm_kwargs)
            
        else:
            print("Loading model weights from cache, download if not exists...")
            try:
                model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download UNI model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        precision = torch.float16
        return model, eval_transform, precision
    
    
class UNIv2InferenceEncoder(BasePatchEncoder):
    def _build(self):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        
        self.enc_name = 'uni_v2'
        weights_path = get_weights_path('patch', self.enc_name)
        
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        
        if weights_path != "":
            model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, checkpoint_path=weights_path, **timm_kwargs)
            
        else:
            print("Loading model weights from cache, download if not exists...")
            try:
                model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download UNI v2 model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        precision = torch.bfloat16
        return model, eval_transform, precision
    

class GigaPathInferenceEncoder(BasePatchEncoder):
    def _build(
        self, 
        timm_kwargs={}
    ):
        import timm
        assert timm.__version__ == '0.9.16', f"Gigapath requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
        from torchvision import transforms

        self.enc_name = 'gigapath'
        weights_path = get_weights_path('patch', self.enc_name)
        
        if weights_path != "":
            model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, checkpoint_path=weights_path, **timm_kwargs)

        else:
            print("Loading model weights from cache, download if not exists...")
            try:
                model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, **timm_kwargs)
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
    
    def _build(
        self,
        return_cls=False,
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
    ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        self.enc_name = 'virchow'
        weights_path = get_weights_path('patch', self.enc_name)
        
        if weights_path != "":
            model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, checkpoint_path=weights_path, **timm_kwargs)

        else:
            print("Loading model weights from cache, download if not exists...")
            try:
                model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Virchow model, make sure that you were granted access and that you correctly registered your token")

        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
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
    
    def _build(
        self,
        return_cls=False,
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
    ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        self.enc_name = 'virchow2'
        weights_path = get_weights_path('patch', self.enc_name)
        
        if weights_path != "":
            model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, checkpoint_path=weights_path, **timm_kwargs)

        else:
            print("Loading model weights from cache, download if not exists...")
            try:
                model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, **timm_kwargs)
            except:
                traceback.print_exc()
                raise Exception("Failed to download Virchow-2 model, make sure that you were granted access and that you correctly registered your token")
        
        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
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
    def _build(
        self,
        timm_kwargs={'init_values': 1e-5, 'dynamic_img_size': False}
    ):
        import timm
        assert timm.__version__ == '0.9.16', f"H-Optimus requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
        from torchvision import transforms

        self.enc_name = 'hoptimus0'
        weights_path = get_weights_path('patch', self.enc_name)
        
        if weights_path != "":
            model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, checkpoint_path=weights_path, **timm_kwargs)

        else:
            print("Loading model weights from cache, download if not exists...")
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
    def _build(
        self,
        timm_kwargs={'init_values': 1e-5, 'dynamic_img_size': False},
        **kwargs
    ):
        import timm
        assert timm.__version__ == '0.9.16', f"H-Optimus requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
        from torchvision import transforms

        self.enc_name = 'hoptimus1'
        weights_path = get_weights_path('patch', self.enc_name)
        
        if weights_path != "":
            model = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True, checkpoint_path=weights_path, **timm_kwargs)

        else:
            print("Loading model weights from cache, download if not exists...")
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
    def _build(self):
        from transformers import AutoModel
        import torchvision.transforms as T
        from .utils.constants import IMAGENET_MEAN, IMAGENET_STD

        self.enc_name = 'phikon_v2'
        weights_path = get_weights_path('patch', self.enc_name)
        
        if weights_path != "":
            model = AutoModel.from_pretrained(os.path.dirname(weights_path))

        else:
            print("Loading model weights from cache, download if not exists...")
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
    def _build(self, img_size = 448):
        from trident.patch_encoder_models.model_zoo.conchv1_5.conchv1_5 import create_model_from_pretrained

        self.enc_name = 'conch_v15'
        weights_path = get_weights_path('patch', self.enc_name)

        if weights_path != "":
            model, eval_transform = create_model_from_pretrained(checkpoint_path=weights_path, img_size=img_size)

        else:
            print("Loading model weights from cache, download if not exists...")
            try:
                model, eval_transform = create_model_from_pretrained(checkpoint_path="hf_hub:MahmoodLab/conchv1_5", img_size=img_size)
            except:
                traceback.print_exc()
                raise Exception("Failed to download CONCH v1.5 model, make sure that you were granted access and that you correctly registered your token")

        precision = torch.float16
        return model, eval_transform, precision
