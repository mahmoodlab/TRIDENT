"""
CTransPath / TransPath attribution

This file is adapted from the upstream TransPath implementation:
https://github.com/Xiyue-Wang/TransPath/blob/main/ctran.py

Upstream authors (citation):
@article{wang2022,
  title={Transformer-based Unsupervised Contrastive Learning for Histopathological Image Classification},
  author={Wang, Xiyue and Yang, Sen and Zhang, Jun and Wang, Minghui and Zhang, Jing and Yang, Wei and Huang, Junzhou and Han, Xiao},
  journal={Medical Image Analysis},
  year={2022},
  publisher={Elsevier}
}

@inproceedings{wang2021transpath,
  title={TransPath: Transformer-Based Self-supervised Learning for Histopathological Image Classification},
  author={Wang, Xiyue and Yang, Sen and Zhang, Jun and Wang, Minghui and Zhang, Jing and Huang, Junzhou and Yang, Wei and Han, Xiao},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={186--195},
  year={2021},
  organization={Springer}
}

"""

from timm_ctp.models.layers.helpers import to_2tuple
from timm_ctp import create_model as ctp_create_model
import torch.nn as nn
import pdb


class ConvStem(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

def ctranspath(img_size = 224, **kwargs):
    model = ctp_create_model('swin_tiny_patch4_window7_224', 
                                  embed_layer=ConvStem, 
                                  pretrained=False,
                                  img_size=img_size,
                                  **kwargs)
    return model
