"""
Mix Transformer (MiT) 编码器：SegFormer 的分层 Transformer backbone。

"""

from __future__ import annotations

from functools import partial
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .layers import DropPath, init_weights, to_2tuple


class OverlapPatchEmbed(nn.Module):
    """重叠 Patch Embedding / Patch Merging（论文 Sec. 3.1）。"""

    def __init__(
        self,
        patch_size: int = 7,
        stride: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(init_weights)

    def forward(self, x: Tensor) -> Tuple[Tensor, int, int]:
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm(x)
        return x, h, w


class EfficientMultiHeadAttention(nn.Module):
    """带空间降采样（sequence reduction）的多头自注意力。"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} 必须能被 num_heads {num_heads} 整除"
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if sr_ratio > 1:
            # 等价于论文中用 reduction ratio R=sr_ratio^2 降低序列长度
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(init_weights)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_)
        else:
            kv = self.kv(x)

        kv = kv.reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj_drop(self.proj(x))
        return x


class DWConv(nn.Module):
    """3×3 深度可分离卷积，用于 Mix-FFN 泄露位置信息。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        b, _, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    """MLP + 3×3 DWConv + MLP（论文 Eq. 3）。"""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(init_weights)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, h, w)
        x = self.drop(self.act(x))
        x = self.drop(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio: int = 1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EfficientMultiHeadAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MixFFN(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.apply(init_weights)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
        return x


class MixVisionTransformer(nn.Module):
    """分层 Mix Transformer 编码器，输出 4 尺度特征图。"""

    def __init__(
        self,
        in_chans: int = 3,
        embed_dims: Sequence[int] = (64, 128, 256, 512),
        num_heads: Sequence[int] = (1, 2, 4, 8),
        mlp_ratios: Sequence[int] = (4, 4, 4, 4),
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        depths: Sequence[int] = (3, 4, 6, 3),
        sr_ratios: Sequence[int] = (8, 4, 2, 1),
    ):
        super().__init__()
        assert len(embed_dims) == len(num_heads) == len(depths) == 4
        self.depths = list(depths)
        self.embed_dims = list(embed_dims)
        self.feature_dims = list(embed_dims)

        # Stage1: K=7,S=4；其后 Overlap Patch Merging: K=3,S=2
        patch_sizes = (7, 3, 3, 3)
        strides = (4, 2, 2, 2)
        in_channels = [in_chans, *embed_dims[:-1]]

        self.patch_embeds = nn.ModuleList(
            [
                OverlapPatchEmbed(
                    patch_size=patch_sizes[i],
                    stride=strides[i],
                    in_chans=in_channels[i],
                    embed_dim=embed_dims[i],
                )
                for i in range(4)
            ]
        )

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(4):
            stage_blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur + j],
                        norm_layer=norm_layer,
                        sr_ratio=sr_ratios[i],
                    )
                    for j in range(depths[i])
                ]
            )
            self.blocks.append(stage_blocks)
            self.norms.append(norm_layer(embed_dims[i]))
            cur += depths[i]

        self.apply(init_weights)

    def forward_features(self, x: Tensor) -> List[Tensor]:
        b = x.shape[0]
        outs: List[Tensor] = []
        for i in range(4):
            x, h, w = self.patch_embeds[i](x)
            for blk in self.blocks[i]:
                x = blk(x, h, w)
            x = self.norms[i](x)
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs

    def forward(self, x: Tensor) -> List[Tensor]:
        return self.forward_features(x)


def _build_mit(
    embed_dims: Sequence[int],
    depths: Sequence[int],
    **kwargs,
) -> MixVisionTransformer:
    return MixVisionTransformer(
        embed_dims=embed_dims,
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(4, 4, 4, 4),
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=depths,
        sr_ratios=(8, 4, 2, 1),
        drop_rate=0.0,
        drop_path_rate=kwargs.pop("drop_path_rate", 0.1),
        **kwargs,
    )


def mit_b0(**kwargs) -> MixVisionTransformer:
    return _build_mit(embed_dims=(32, 64, 160, 256), depths=(2, 2, 2, 2), **kwargs)


def mit_b1(**kwargs) -> MixVisionTransformer:
    return _build_mit(embed_dims=(64, 128, 320, 512), depths=(2, 2, 2, 2), **kwargs)


def mit_b2(**kwargs) -> MixVisionTransformer:
    return _build_mit(embed_dims=(64, 128, 320, 512), depths=(3, 4, 6, 3), **kwargs)


def mit_b3(**kwargs) -> MixVisionTransformer:
    return _build_mit(embed_dims=(64, 128, 320, 512), depths=(3, 4, 18, 3), **kwargs)


def mit_b4(**kwargs) -> MixVisionTransformer:
    return _build_mit(embed_dims=(64, 128, 320, 512), depths=(3, 8, 27, 3), **kwargs)


def mit_b5(**kwargs) -> MixVisionTransformer:
    return _build_mit(embed_dims=(64, 128, 320, 512), depths=(3, 6, 40, 3), **kwargs)


MIT_VARIANTS = {
    "mit_b0": mit_b0,
    "mit_b1": mit_b1,
    "mit_b2": mit_b2,
    "mit_b3": mit_b3,
    "mit_b4": mit_b4,
    "mit_b5": mit_b5,
}


if __name__ == "__main__":
    enc = mit_b0(in_chans=3)
    y = enc(torch.randn(1, 3, 512, 512))
    for t in y:
        print(t.shape)
