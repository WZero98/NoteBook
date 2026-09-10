"""LiteMLA（多尺度线性注意力）与 EfficientViT Block。"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ops import (
    ConvLayer,
    IdentityLayer,
    MBConv,
    ResidualBlock,
    build_act,
    get_same_padding,
    val2tuple,
)


class LiteMLA(nn.Module):
    """Lightweight multi-scale linear attention（论文 Figure 2）。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        super().__init__()
        self.eps = eps
        heads = int(in_channels // dim * heads_ratio) if heads is None else heads
        total_dim = heads * dim
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        1,
                        groups=3 * heads,
                        bias=use_bias[0],
                    ),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)
        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def relu_linear_att(self, qkv: Tensor) -> Tensor:
        B, _, H, W = qkv.shape
        if qkv.dtype == torch.float16:
            qkv = qkv.float()
        qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W))
        q, k, v = qkv[:, :, : self.dim], qkv[:, :, self.dim : 2 * self.dim], qkv[:, :, 2 * self.dim :]
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk = torch.matmul(v, k.transpose(-1, -2))
        out = torch.matmul(vk, q)
        if out.dtype == torch.bfloat16:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
        return torch.reshape(out, (B, -1, H, W))

    def relu_quadratic_att(self, qkv: Tensor) -> Tensor:
        B, _, H, W = qkv.shape
        qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W))
        q, k, v = qkv[:, :, : self.dim], qkv[:, :, self.dim : 2 * self.dim], qkv[:, :, 2 * self.dim :]
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        att_map = torch.matmul(k.transpose(-1, -2), q)
        dtype = att_map.dtype
        if dtype in (torch.float16, torch.bfloat16):
            att_map = att_map.float()
        att_map = att_map / (torch.sum(att_map, dim=2, keepdim=True) + self.eps)
        att_map = att_map.to(dtype)
        out = torch.matmul(v, att_map)
        return torch.reshape(out, (B, -1, H, W))

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)
        h, w = qkv.shape[-2:]
        if h * w > self.dim:
            out = self.relu_linear_att(qkv).to(qkv.dtype)
        else:
            out = self.relu_quadratic_att(qkv)
        return self.proj(out)


class EfficientViTBlock(nn.Module):
    """Context (LiteMLA) + Local (MBConv / FFN+DW 语义）残差块。"""

    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        scales: tuple[int, ...] = (5,),
        norm: str = "bn2d",
        act_func: str = "hswish",
    ):
        super().__init__()
        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
                scales=scales,
            ),
            IdentityLayer(),
        )
        self.local_module = ResidualBlock(
            MBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False),
                norm=(None, None, norm),
                act_func=(act_func, act_func, None),
            ),
            IdentityLayer(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.local_module(self.context_module(x))
