# -----------------------------------------------------------------------------
# GenBio-PathFM
# -----------------------------------------------------------------------------
# This module is taken from the upstream GenBio-PathFM repository:
# https://github.com/genbio-ai/genbio-pathfm
#
# If you use this work, please consider citing:
#   @article{kapse2026genbiopathfm,
#     title={GenBio-PathFM: A State-of-the-Art Foundation Model for Histopathology},
#     author={Kapse, Saarthak and Aygun, Mehmet and Cole, Elijah and Lundberg, Emma and Song, Le and Xing, Eric P.},
#     journal={bioRxiv},
#     year={2026}
#   }
#
# License notice (required by the GenBio AI Community License):
#   This is licensed under the GenBio AI Community License Agreement,
#   Copyright © GENBIO.AI, INC. All Rights Reserved.
# -----------------------------------------------------------------------------

import math
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ──────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────

def _cat_keep_shapes(x_list: List[Tensor]) -> Tuple[Tensor, List[Tuple[int, ...]], List[int]]:
    shapes = [x.shape for x in x_list]
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    flattened = torch.cat([x.flatten(0, -2) for x in x_list])
    return flattened, shapes, num_tokens


def _uncat_with_shapes(
    flattened: Tensor,
    shapes: List[Tuple[int, ...]],
    num_tokens: List[int],
) -> List[Tensor]:
    outputs_splitted = torch.split_with_sizes(flattened, num_tokens, dim=0)
    shapes_adjusted = [shape[:-1] + torch.Size([flattened.shape[-1]]) for shape in shapes]
    return [o.reshape(s) for o, s in zip(outputs_splitted, shapes_adjusted)]


# ──────────────────────────────────────────────────────────────
# LayerScale
# ──────────────────────────────────────────────────────────────

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, device=device))
        self.init_values = init_values

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# ──────────────────────────────────────────────────────────────
# FFN layers  (Mlp  +  SwiGLUFFN)
# ──────────────────────────────────────────────────────────────

class _ListForwardMixin:
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def forward_list(self, x_list: List[Tensor]) -> List[Tensor]:
        x_flat, shapes, num_tokens = _cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return _uncat_with_shapes(x_flat, shapes, num_tokens)


class Mlp(nn.Module, _ListForwardMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module, _ListForwardMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
        align_to: int = 8,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        h = d + (-d % align_to)
        self.w1 = nn.Linear(in_features, h, bias=bias, device=device)
        self.w2 = nn.Linear(in_features, h, bias=bias, device=device)
        self.w3 = nn.Linear(h, out_features, bias=bias, device=device)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ──────────────────────────────────────────────────────────────
# PatchEmbed
# ──────────────────────────────────────────────────────────────

def _make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()
        image_HW = _make_2tuple(img_size)
        patch_HW = _make_2tuple(patch_size)
        patch_grid_size = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x

    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))


# ──────────────────────────────────────────────────────────────
# RoPE position encoding
# ──────────────────────────────────────────────────────────────

class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: Optional[float] = 100.0,
        min_period: Optional[float] = None,
        max_period: Optional[float] = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: Optional[float] = None,
        jitter_coords: Optional[float] = None,
        rescale_coords: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device=None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Provide either `base` or both `min_period`+`max_period`.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype

        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)
            periods = (base ** exponents) / base * self.max_period
        self.periods.data = periods

    def forward(self, *, H: int, W: int) -> Tuple[Tensor, Tensor]:
        device, dtype = self.periods.device, self.dtype
        dd = {"device": device, "dtype": dtype}

        if self.normalize_coords == "max":
            m = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / m
            coords_w = torch.arange(0.5, W, **dd) / m
        elif self.normalize_coords == "min":
            m = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / m
            coords_w = torch.arange(0.5, W, **dd) / m
        else:  # "separate"
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W

        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H,W,2]
        coords = coords.flatten(0, 1)        # [HW, 2]
        coords = 2.0 * coords - 1.0         # shift to [-1, +1]

        if self.training and self.shift_coords is not None:
            coords += torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
        if self.training and self.jitter_coords is not None:
            jmax = math.log(self.jitter_coords)
            coords *= torch.empty(2, **dd).uniform_(-jmax, jmax).exp()
        if self.training and self.rescale_coords is not None:
            rmax = math.log(self.rescale_coords)
            coords *= torch.empty(1, **dd).uniform_(-rmax, rmax).exp()

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW,2,D//4]
        angles = angles.flatten(1, 2).tile(2)                                     # [HW, D]
        return torch.sin(angles), torch.cos(angles)


# ──────────────────────────────────────────────────────────────
# Self-Attention
# ──────────────────────────────────────────────────────────────

def _rope_rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return (x * cos) + (_rope_rotate_half(x) * sin)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def _apply_rope(
        self, q: Tensor, k: Tensor, rope: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        q_dtype, k_dtype = q.dtype, k.dtype
        sin, cos = rope
        q = q.to(sin.dtype)
        k = k.to(sin.dtype)
        prefix = q.shape[-2] - sin.shape[-2]
        assert prefix >= 0
        q = torch.cat((q[:, :, :prefix], _rope_apply(q[:, :, prefix:], sin, cos)), dim=-2)
        k = torch.cat((k[:, :, :prefix], _rope_apply(k[:, :, prefix:], sin, cos)), dim=-2)
        return q.to(q_dtype), k.to(k_dtype)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]
        if rope is not None:
            q, k = self._apply_rope(q, k, rope)
        x = F.scaled_dot_product_attention(q, k, v)
        return x.transpose(1, 2).reshape(B, N, C)

    def forward(self, x: Tensor, attn_bias=None, rope=None) -> Tensor:
        x = self.proj(self.compute_attention(self.qkv(x), attn_bias=attn_bias, rope=rope))
        return self.proj_drop(x)

    def forward_list(self, x_list: List[Tensor], attn_bias=None, rope_list=None) -> List[Tensor]:
        x_flat, shapes, num_tokens = _cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = _uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = [
            self.compute_attention(qkv, attn_bias=attn_bias, rope=rope)
            for qkv, rope in zip(qkv_list, rope_list)
        ]
        x_flat, shapes, num_tokens = _cat_keep_shapes(att_out)
        return _uncat_with_shapes(self.proj(x_flat), shapes, num_tokens)


# ──────────────────────────────────────────────────────────────
# Transformer block
# ──────────────────────────────────────────────────────────────

torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.accumulated_cache_size_limit = 1024


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        device=None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
            attn_drop=attn_drop, proj_drop=drop, device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=int(dim * ffn_ratio),
            act_layer=act_layer, drop=drop, bias=ffn_bias, device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()
        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(rope, indices):
        if rope is None:
            return None
        sin, cos = rope
        if sin.ndim == 4:
            return sin[indices], cos[indices]
        return sin, cos

    def _forward_list(self, x_list: List[Tensor], rope_list=None) -> List[Tensor]:
        if self.training and self.sample_drop_ratio > 0.0:
            b_list = [x.shape[0] for x in x_list]
            ss = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
            rsf = [b / s for b, s in zip(b_list, ss)]

            idx1 = [(torch.randperm(b, device=x.device))[:s] for x, b, s in zip(x_list, b_list, ss)]
            sub1 = [x[i] for x, i in zip(x_list, idx1)]
            rope_sub = [self._maybe_index_rope(r, i) for r, i in zip(rope_list, idx1)] if rope_list else rope_list

            flat, shapes, nt = _cat_keep_shapes(sub1)
            norm1 = _uncat_with_shapes(self.norm1(flat), shapes, nt)
            res1 = self.attn.forward_list(norm1, rope_list=rope_sub)

            x_attn = [
                torch.index_add(x, 0, i, self.ls1(r), alpha=f)
                for x, r, i, f in zip(x_list, res1, idx1, rsf)
            ]
            idx2 = [(torch.randperm(b, device=x.device))[:s] for x, b, s in zip(x_attn, b_list, ss)]
            sub2 = [x[i] for x, i in zip(x_attn, idx2)]
            flat2, shapes2, nt2 = _cat_keep_shapes(sub2)
            res2 = self.mlp.forward_list(_uncat_with_shapes(self.norm2(flat2), shapes2, nt2))

            return [
                torch.index_add(xa, 0, i, self.ls2(r), alpha=f)
                for xa, r, i, f in zip(x_attn, res2, idx2, rsf)
            ]
        else:
            out = []
            for x, rope in zip(x_list, rope_list):
                x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x = x + self.ls2(self.mlp(self.norm2(x)))
                out.append(x)
            return out

    def forward(self, x_or_list, rope_or_list=None):
        if isinstance(x_or_list, Tensor):
            return self._forward_list([x_or_list], rope_list=[rope_or_list])[0]
        elif isinstance(x_or_list, list):
            if rope_or_list is None:
                rope_or_list = [None] * len(x_or_list)
            return self._forward_list(x_or_list, rope_list=rope_or_list)
        raise AssertionError


# ──────────────────────────────────────────────────────────────
# VisionTransformer
# ──────────────────────────────────────────────────────────────

_FFN_LAYERS = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}
_NORM_LAYERS = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
}
_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class VisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 1,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: Optional[float] = None,
        pos_embed_rope_max_period: Optional[float] = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: Optional[float] = None,
        pos_embed_rope_jitter_coords: Optional[float] = None,
        pos_embed_rope_rescale_coords: Optional[float] = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 3.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: Optional[float] = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "swiglu64",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 4,
        device=None,
        **ignored_kwargs,
    ):
        super().__init__()
        norm_layer_cls = _NORM_LAYERS[norm_layer]
        ffn_layer_cls = _FFN_LAYERS[ffn_layer]

        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.n_storage_tokens = n_storage_tokens

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim, flatten_embedding=False,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        if n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))

        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim, num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=_DTYPES[pos_embed_rope_dtype],
            device=device,
        )

        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                drop_path=drop_path_rate, norm_layer=norm_layer_cls,
                act_layer=nn.GELU, ffn_layer=ffn_layer_cls,
                init_values=layerscale_init, device=device,
            )
            for _ in range(depth)
        ])
        self.norm = norm_layer_cls(embed_dim)

    def prepare_tokens(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)
        ct = self.cls_token
        st = self.storage_tokens if self.n_storage_tokens > 0 else torch.empty(
            1, 0, ct.shape[-1], dtype=ct.dtype, device=ct.device
        )
        x = torch.cat([ct.expand(B, -1, -1), st.expand(B, -1, -1), x], dim=1)
        return x, (H, W)

    def forward_features(self, x: Tensor) -> Dict[str, Tensor]:
        tokens, (H, W) = self.prepare_tokens(x)
        rope = self.rope_embed(H=H, W=W)
        for blk in self.blocks:
            tokens = blk(tokens, rope)
        tokens = self.norm(tokens)
        n = self.n_storage_tokens
        return {
            "x_norm_clstoken":    tokens[:, 0],
            "x_storage_tokens":   tokens[:, 1:n + 1],
            "x_norm_patchtokens": tokens[:, n + 1:],
            "x_prenorm":          tokens,
        }

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self.forward_features(x)


# ──────────────────────────────────────────────────────────────
# GenBioPathFMInference
# ──────────────────────────────────────────────────────────────

class GenBioPathFMInference(nn.Module):
    """
    Loads a GenBio-PathFM checkpoint and runs RGB inference.

    ``forward`` returns a *single* tensor – the concatenated
    CLS token across R/G/B channels.  
    Use ``forward_with_patches`` when you also need patch tokens.

    Args:
        weights_path: Path to the .pth checkpoint.
        device:       "cuda" or "cpu".
    """

    def __init__(self, weights_path: str, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)


        self.model = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=1536,
            depth=40,
            num_heads=24,
            ffn_ratio=4,
            in_chans=1,
            n_storage_tokens=4,
            ffn_layer="swiglu64",
            layerscale_init=1.0e-5,
            qkv_bias=False,
            proj_bias=True,
            ffn_bias=True,
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_jitter_coords=True,
            pos_embed_rope_normalize_coords="separate",
        )

        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        self.to(self.device).eval()

    def _encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process a batch of single-channel [B,1,H,W] images."""
        tokens, (h, w) = self.model.prepare_tokens(x)
        rope = self.model.rope_embed(H=h, W=w)
        for blk in self.model.blocks:
            tokens = blk(tokens, rope)
        tokens = self.model.norm(tokens)
        n_storage_tokens = self.model.n_storage_tokens
        return {
            "x_norm_clstoken":    tokens[:, 0],
            "x_norm_patchtokens": tokens[:, n_storage_tokens + 1 :],
        }

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_rgb: ``[B, 3, H, W]`` – standard RGB batch delivered by the
                   dataloader (already transformed).

        Returns:
            ``[B, embed_dim * 3]`` – CLS features from R, G, B channels
            concatenated along the feature dimension.
        """
        if x_rgb.ndim != 4 or x_rgb.shape[1] != 3:
            raise ValueError(f"Expected input shape [B, 3, H, W], got {tuple(x_rgb.shape)}")

        b, _c, h, w = x_rgb.shape
        # Stack all channels into a single-channel batch → [B*3, 1, H, W]
        features = self._encode(x_rgb.reshape(b * 3, 1, h, w))

        cls = features["x_norm_clstoken"].view(b, 3, -1)            # [B, 3, D]
        return torch.cat([cls[:, 0], cls[:, 1], cls[:, 2]], dim=-1) # [B, 3*D]

    def forward_with_patches(
        self, x_rgb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extended forward that also returns patch-level features.

        Returns:
            cls_out:   ``[B, embed_dim * 3]``
            patch_out: ``[B, num_patches, embed_dim * 3]``
        """
        if x_rgb.ndim != 4 or x_rgb.shape[1] != 3:
            raise ValueError(f"Expected input shape [B, 3, H, W], got {tuple(x_rgb.shape)}")

        b, _c, h, w = x_rgb.shape
        features = self._encode(x_rgb.reshape(b * 3, 1, h, w))

        cls = features["x_norm_clstoken"].view(b, 3, -1)
        cls_out = torch.cat([cls[:, 0], cls[:, 1], cls[:, 2]], dim=-1)

        patches = features["x_norm_patchtokens"]                     # [B*3, N, D]
        n, d = patches.shape[1], patches.shape[2]
        patches = patches.view(b, 3, n, d)
        patch_out = torch.cat([patches[:, 0], patches[:, 1], patches[:, 2]], dim=-1)

        return cls_out, patch_out
