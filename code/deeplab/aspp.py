"""
Atrous Spatial Pyramid Pooling (ASPP)。

对应 DeepLabv3 / DeepLabv3+ 编码器中的多尺度空洞卷积分支 + 图像级池化。
可选 depthwise separable 形式（论文中的 atrous separable convolution）。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ASPPConv(nn.Module):
    """标准 3×3 atrous conv 分支。"""

    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.block = _conv_bn_relu(
            in_channels,
            out_channels,
            3,
            padding=dilation,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class SeparableASPPConv(nn.Module):
    """Atrous depthwise 3×3 + pointwise 1×1（论文 Fig. 3）。"""

    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _conv_bn_relu(
                in_channels,
                in_channels,
                3,
                padding=dilation,
                dilation=dilation,
                groups=in_channels,
            ),
            _conv_bn_relu(in_channels, out_channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ASPPPooling(nn.Module):
    """Image-level pooling 分支：GAP → 1×1 → 双线性还原空间尺寸。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        y = self.conv(self.pool(x))
        # GAP 后空间为 1×1；batch=1 且 training 时 BN 无方差可估，改用 running stats
        if self.training and y.shape[0] == 1:
            y = F.batch_norm(
                y,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                training=False,
                momentum=self.bn.momentum,
                eps=self.bn.eps,
            )
        else:
            y = self.bn(y)
        y = self.relu(y)
        return F.interpolate(y, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """
    ASPP：1×1 + 多 rate 空洞卷积 + image pooling → concat → 1×1 投影。

    论文默认 rates（相对 output_stride=16）为 (6, 12, 18)；
    output_stride=8 时常取 (12, 24, 36)。
    """

    def __init__(
        self,
        in_channels: int,
        atrous_rates: Sequence[int],
        out_channels: int = 256,
        *,
        separable: bool = False,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        rates = tuple(atrous_rates)
        branch_cls = SeparableASPPConv if separable else ASPPConv

        modules: list[nn.Module] = [
            _conv_bn_relu(in_channels, out_channels, 1),
        ]
        for rate in rates:
            modules.append(branch_cls(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            _conv_bn_relu(len(modules) * out_channels, out_channels, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        feats = [conv(x) for conv in self.convs]
        return self.project(torch.cat(feats, dim=1))
