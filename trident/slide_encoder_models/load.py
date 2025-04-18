import sys
import os
import torch
import traceback
from abc import abstractmethod
from einops import rearrange
from typing import Optional, Tuple

from trident.IO import get_weights_path

"""
This file contains an assortment of pretrained slide encoders, all loadable via the encoder_factory() function.
"""

def encoder_factory(model_name: str, pretrained: bool = True, freeze: bool = True, **kwargs) -> torch.nn.Module:
        """
        Build a slide encoder model.

        Args:
            model_name (str): Name of the model to build.
            pretrained (bool): Whether to load pretrained weights.
            freeze (bool): Whether to freeze the weights of the model.
            **kwargs: Additional arguments to pass to the model constructor.

        Returns:
            torch.nn.Module: The slide encoder model.
        """

        if model_name.startswith('mean-'):
            enc = MeanSlideEncoder
            return enc(model_name = model_name)
        elif 'threads' in model_name:
            enc = ThreadsSlideEncoder
        elif 'titan' in model_name:
            enc = TitanSlideEncoder
        elif 'prism' in model_name:
            enc = PRISMSlideEncoder
        elif 'chief' in model_name:
            enc = CHIEFSlideEncoder
        elif 'gigapath' in model_name:
            enc = GigaPathSlideEncoder
        elif 'madeleine' in model_name:
            enc = MadeleineSlideEncoder
        elif 'abmil' in model_name:
            enc = ABMILSlideEncoder
        else:
            raise ValueError(f"Model type {model_name} not supported")
        
        return enc(pretrained=pretrained, freeze=freeze, **kwargs)


# Map from slide encoder to required patch encoder
# Used in Processor.py to load the correct patch encoder for a given slide encoder
slide_to_patch_encoder_name = {
    'threads': 'conch_v15',
    'titan': 'conch_v15',
    'tcga': 'conch_v15',
    'prism': 'virchow',
    'chief': 'ctranspath',
    'gigapath': 'gigapath',
    'madeleine': 'conch_v1',
}



class BaseSlideEncoder(torch.nn.Module):
    
    def __init__(self, freeze: bool = True, **build_kwargs: dict) -> None:
        """
        Parent class for all pretrained slide encoders.
        """
        super().__init__()
        self.enc_name = None
        self.model, self.precision, self.embedding_dim = self._build(**build_kwargs)

        # Set all parameters to be non-trainable
        if freeze and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
    def forward(self, batch):
        """
        Can be overwritten if model requires special forward pass.
        """
        z = self.model(batch)
        return z
        
    @abstractmethod
    def _build(self, **build_kwargs):
        """
        Initialization method, must be defined in child class.
        """
        pass


class CustomSlideEncoder(BaseSlideEncoder):
    def __init__(
        self, 
        enc_name: str, 
        model: torch.nn.Module, 
        precision: torch.dtype = torch.float32, 
        embedding_dim: Optional[int] = None
    ):
        """
        CustomSlideEncoder initialization.

        This class is used when the model and precision are pre-instantiated externally 
        and should be injected directly into the encoder wrapper.

        Args:
            enc_name (str): 
                A unique name or identifier for the encoder.
            model (torch.nn.Module): 
                A PyTorch model instance to use for slide-level inference.
            precision (torch.dtype, optional): 
                The precision to use for inference (e.g., torch.float32, torch.float16).
            embedding_dim (int, optional): 
                The output embedding dimension. If not provided, will attempt to use 
                `model.embedding_dim` if it exists.
        """
        super().__init__(freeze=False)  # Freezing should be handled externally
        self.enc_name = enc_name
        self.model = model
        self.precision = precision
        self.embedding_dim = embedding_dim or getattr(model, 'embedding_dim', None)

    def _build(self, **build_kwargs):
        return None, None, None


class ABMILSlideEncoder(BaseSlideEncoder):

    def __init__(self, **build_kwargs):
        """
        ABMIL initialization.
        """
        super().__init__(**build_kwargs)
    
    def _build(
        self,
        input_feature_dim: int,
        n_heads: int,
        head_dim: int,
        dropout: float,
        gated: bool,
        pretrained: bool = False
    ) -> Tuple[torch.nn.ModuleDict, torch.dtype, int]:
        
        from trident.slide_encoder_models.model_zoo.reusable_blocks.ABMIL import ABMIL
        import torch.nn as nn

        self.enc_name = 'abmil'
        
        assert pretrained is False, "ABMILSlideEncoder has no corresponding pretrained models. Please load with pretrained=False."
                                
        pre_attention_layers = nn.Sequential(
            nn.Linear(input_feature_dim, input_feature_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        image_pooler = ABMIL(
            n_heads=n_heads,
            feature_dim=input_feature_dim,
            head_dim=head_dim,
            dropout=dropout,
            n_branches=1,
            gated=gated
        )
        
        post_attention_layers = nn.Sequential(
            nn.Linear(input_feature_dim, input_feature_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        model = nn.ModuleDict({
            'pre_attention_layers': pre_attention_layers,
            'image_pooler': image_pooler,
            'post_attention_layers': post_attention_layers
        })
        
        precision = torch.float32
        embedding_dim = input_feature_dim
        return model, precision, embedding_dim

    def forward(self, batch, device='cuda', return_raw_attention=False):
        image_features = self.model['pre_attention_layers'](batch['features'].to(device))
        image_features, attn = self.model['image_pooler'](image_features) # Features shape: (b n_branches f), where n_branches = 1. Branching is not used in this implementation.
        image_features = rearrange(image_features, 'b 1 f -> b f')
        image_features = self.model['post_attention_layers'](image_features)# Attention scores shape: (b r h n), where h is number of attention heads 
        if return_raw_attention:
            return image_features, attn
        return image_features


class PRISMSlideEncoder(BaseSlideEncoder):

    def __init__(self, **build_kwargs):
        """
        PRISM initialization.
        """
        super().__init__(**build_kwargs)
    
    def _build(self, pretrained=True):
        
        self.enc_name = 'prism'

        if sys.version_info < (3, 10):
            raise RuntimeError("PRISM requires Python 3.10 or above. Please update your Python interpreter.")

        try:
            import environs  # weird dependencies required by PRISM
            import sacremoses
            from transformers import AutoModel, AutoConfig
        except:
            traceback.print_exc()
            raise Exception(
                "Please run `pip install environs==11.0.0 transformers==4.42.4 sacremoses==0.1.1` "
                "and ensure Python version is 3.10 or above."
            )

        if pretrained:
            model = AutoModel.from_pretrained('paige-ai/Prism', trust_remote_code=True)
        else:
            model = AutoModel.from_config(AutoConfig.from_pretrained('paige-ai/Prism'))
        model.text_decoder = None
        precision = torch.float16
        embedding_dim = 1280
        return model, precision, embedding_dim
    
    def forward(self, batch, device='cuda'):
        # input should be of shape (batch_size, tile_seq_len, tile_embed_dim)
        x = batch['features'].to(device)
        z = self.model.slide_representations(x)
        z = z['image_embedding'] 
        return z
    

class CHIEFSlideEncoder(BaseSlideEncoder):

    def __init__(self, **build_kwargs):
        """
        CHIEF initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, pretrained=True):
        
        self.enc_name = 'chief'
        weights_path = get_weights_path('slide', self.enc_name)

        # Ensure model can be built.
        try:
            sys.path.append(weights_path)
            from models.CHIEF import CHIEF
        except Exception:
            traceback.print_exc()
            raise Exception(
                f"\nError: Unable to import the CHIEF repository from '{weights_path}'.\n\n"
                "To resolve this issue:\n"
                "1. Ensure you have cloned the CHIEF repository to a convenient location:\n"
                "   `git clone https://github.com/hms-dbmi/CHIEF/`\n"
                "2. Set the path to CHIEF repo in `trident/slide_encoder_models/load_ckpts.json`, e.g., `./CHIEF`.\n"
                "3. Verify that CHIEF dependencies are installed:\n"
                "   `pip install addict`\n\n"
            )

        # Ensure weights can be loaded.
        try:
            current_wd = os.getcwd()  # Get current working directory
            os.chdir(weights_path)  # Change to CHIEF repo directory
            os.makedirs(os.path.join(weights_path, "model_weight"), exist_ok=True)

            required_files = {
                "Text_emdding.pth": "https://drive.google.com/drive/folders/1uRv9A1HuTW5m_pJoyMzdN31bE1i-tDaV",
                "CHIEF_pretraining.pth": "https://drive.google.com/drive/folders/1uRv9A1HuTW5m_pJoyMzdN31bE1i-tDaV",
            }

            for file_name, download_link in required_files.items():
                file_path = os.path.join(weights_path, "model_weight", file_name)
                if not os.path.exists(file_path):
                    raise Exception(
                        f"\nError: Missing required file '{file_name}'.\n\n"
                        "To resolve this issue:\n"
                        f"1. Download the file from:\n   {download_link}\n"
                        f"2. Copy '{file_name}' to the following directory:\n   {file_path}\n\n"
                        "Ensure the file is correctly placed before retrying."
                    )

            print("All necessary files are present. CHIEF setup is complete!")

        except Exception as e:
            print("\nAn error occurred during CHIEF setup:")
            traceback.print_exc()
            raise e

        model = CHIEF(size_arg="small", dropout=True, n_classes=2)

        # Load pretrained weights
        if pretrained:
            td = torch.load(os.path.join('model_weight', 'CHIEF_pretraining.pth'), map_location='cpu', weights_only=True)
            model.load_state_dict(td, strict=True)
            
        # Return to original working directory
        os.chdir(current_wd)
        
        precision = torch.float32
        embedding_dim = 768
        return model, precision, embedding_dim
    
    def forward(self, batch, device='cuda'):
        x = batch['features'].squeeze(0).to(device)
        z = self.model(x, torch.tensor([0]))
        z = z['WSI_feature']  # Shape (1,768)
        return z
    

class GigaPathSlideEncoder(BaseSlideEncoder):

    def __init__(self, **build_kwargs):
        """
        GigaPath initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, pretrained=True):

        self.enc_name = 'gigapath'

        try:
            from gigapath.slide_encoder import create_model
        except:
            traceback.print_exc()
            raise Exception("Please install fairscale and gigapath using `pip install fairscale git+https://github.com/prov-gigapath/prov-gigapath.git`.")
        
        # Make sure flash_attn is correct version
        try:
            import flash_attn; assert flash_attn.__version__ == '2.5.8'
        except:
            traceback.print_exc()
            raise Exception("Please install flash_attn version 2.5.8 using `pip install flash_attn==2.5.8`.")
        
        if pretrained:
            model = create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536, global_pool=True)
        else:
            model = create_model("", "gigapath_slide_enc12l768d", 1536, global_pool=True)
        
        
        precision = torch.float16
        embedding_dim = 768
        return model, precision, embedding_dim

    def forward(self, batch, device='cuda'):
        self.model.tile_size = batch['attributes']['patch_size_level0']
        z = self.model(batch['features'].to(device), batch['coords'].to(device), all_layer_embed=True)[11]
        return z


class MadeleineSlideEncoder(BaseSlideEncoder):

    def __init__(self, **build_kwargs):
        """
        Madeleine initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, pretrained=True):

        assert pretrained, "MadeleineSlideEncoder has no non-pretrained models. Please load with pretrained=True."

        self.enc_name = 'madeleine'
        weights_path = get_weights_path('slide', self.enc_name)
        embedding_dim = 512

        try:
            from madeleine.models.factory import create_model_from_pretrained
        except:
            traceback.print_exc()
            raise Exception("Please install Madeleine using `pip install git+https://github.com/mahmoodlab/MADELEINE.git`")  
        
        model, precision = create_model_from_pretrained(weights_path)

        return model, precision, embedding_dim
    
    def forward(self, x, device='cuda'):
        z = self.model.encode_he(x['features'], device)
        return z


class ThreadsSlideEncoder(BaseSlideEncoder):

    def __init__(self, **build_kwargs):
        """
        Threads initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, pretrained=True):

        self.enc_name = 'threads'

        try:
            from threadsmodel.inference import create_model, create_model_from_pretrained
        except:
            traceback.print_exc()
            raise Exception("Coming Soon! Thanks for your patience.")
        
        return None, None, None

    def forward(self, batch, device='cuda', return_raw_attention=False):
        pass


class TitanSlideEncoder(BaseSlideEncoder):
    
    def __init__(self, **build_kwargs):
        """
        Titan initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, pretrained=True):
        self.enc_name = 'titan'
        assert pretrained, "TitanSlideEncoder has no non-pretrained models. Please load with pretrained=True."
        from transformers import AutoModel 
        model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        precision = torch.float16
        embedding_dim = 768
        return model, precision, embedding_dim

    def forward(self, batch, device='cuda'):
        z = self.model.encode_slide_from_patch_features(batch['features'].to(device), batch['coords'].to(device), batch['attributes']['patch_size_level0'])        
        return z


class MeanSlideEncoder(BaseSlideEncoder):

    def __init__(self, **build_kwargs):
        """
        Mean pooling initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, model_name = 'mean-default'):
        self.enc_name = model_name
        
        if model_name == 'mean-conch_v1':
            embedding_dim = 768
        elif model_name == 'mean-conch_v15':
            embedding_dim = 768
        elif model_name == 'mean-uni_v1':
            embedding_dim = 1024
        elif model_name == 'mean-uni_v2':
            embedding_dim = 1536
        elif model_name == 'mean-ctranspath':
            embedding_dim = 768
        elif model_name == 'mean-phikon':
            embedding_dim = 768
        elif model_name == 'mean-resnet50':
            embedding_dim = 1024
        elif model_name == 'mean-gigapath':
            embedding_dim = 1536
        elif model_name == 'mean-virchow':
            embedding_dim = 2560
        elif model_name == 'mean-virchow2':
            embedding_dim = 2560
        elif model_name == 'mean-hoptimus0':
            embedding_dim = 1536
        elif model_name == 'mean-phikon_v2':
            embedding_dim = 1024
        elif model_name == 'mean-musk':
            embedding_dim = 1024
        elif model_name == 'mean-hibou_l':
            embedding_dim = 1024
        elif model_name == 'mean-kaiko-vit8s':
            embedding_dim = 384
        elif model_name == 'mean-kaiko-vit16s':
            embedding_dim = 384
        elif model_name == 'mean-kaiko-vit8b':
            embedding_dim = 768
        elif model_name == 'mean-kaiko-vit16b':
            embedding_dim = 768
        elif model_name == 'mean-kaiko-vit14l':
            embedding_dim = 1024
        elif model_name == 'lunit-vits8':
            embedding_dim = 384
        else:
            print(f"\033[93mWARNING: Could not automatically infer embedding_dim for mean encoder {self.enc_name}. Setting to None.\033[0m")
            embedding_dim = None
            
        return None, None, embedding_dim

    def forward(self, batch, device='cuda'):
        z = batch['features'].to(device).mean(dim=1) # Just mean pooling
        return z
