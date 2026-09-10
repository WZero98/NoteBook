"""
SegFormer All-MLP Decoder（论文 Sec. 3.2，Eq. 4）。

"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvModule(nn.Module):
    """Conv → Norm → Act 捆绑块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple] = 1,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm = norm_layer(out_channels)
        self.act = act_layer(inplace=True) if act_layer is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class MLP(nn.Module):
    """通道投影： (B,C,H,W) → flatten → Linear → (B,HW,embed_dim)。"""

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        return self.proj(x)


class SegFormerHead(nn.Module):
    """
    All-MLP 解码器：
      Linear 统一通道 → 上采样到 1/4 → Concat → 1×1 融合 → 1×1 分类。
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: Sequence[int] = (64, 128, 320, 512),
        embedding_dim: int = 768,
        dropout_ratio: float = 0.1,
        upsample_output: bool = True,
    ):
        super().__init__()
        assert len(in_channels) == 4, "SegFormer decoder 需要四级特征"
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.upsample_output = upsample_output

        self.linear_c = nn.ModuleList(
            [MLP(c, embedding_dim) for c in in_channels]
        )
        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
        )
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    @staticmethod
    def _to_bchw(tokens: Tensor, h: int, w: int) -> Tensor:
        b, _, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, features: List[Tensor]) -> Tensor:
        c1, c2, c3, c4 = features
        n = c1.shape[0]
        target_size = c1.shape[2:]  # H/4, W/4

        fused_parts = []
        for feat, proj in zip(features, self.linear_c):
            _, _, h, w = feat.shape
            tokens = proj(feat)
            fmap = self._to_bchw(tokens, h, w)
            if fmap.shape[2:] != target_size:
                fmap = F.interpolate(
                    fmap, size=target_size, mode="bilinear", align_corners=False
                )
            fused_parts.append(fmap)

        x = self.linear_fuse(torch.cat(fused_parts, dim=1))
        x = self.dropout(x)
        mask = self.linear_pred(x)  # (B, Ncls, H/4, W/4)

        if self.upsample_output:
            # 回到输入分辨率（相对 c1 再 ×4）
            mask = F.interpolate(
                mask,
                size=(target_size[0] * 4, target_size[1] * 4),
                mode="bilinear",
                align_corners=False,
            )
        return mask
