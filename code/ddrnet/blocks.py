"""
DDRNet 基础模块：ConvBNReLU、残差块、分割头。
"""

from __future__ import annotations

import math
from typing import Optional, Type

import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


def auto_padding(kernel_size: int, stride: int = 1, dilation: int = 1) -> int:
    """对称 padding，使可整除输入在 stride 下输出为 in // stride。"""
    effective_kernel = dilation * (kernel_size - 1) + 1
    return max(0, math.ceil((effective_kernel - stride) / 2))


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        *,
        bias: bool = False,
        no_relu: bool = False,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = auto_padding(kernel_size, stride, dilation)
        self.conv = nn.Conv2d(
            in_c, out_c, kernel_size, stride, padding, dilation, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.no_relu = no_relu

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn(self.conv(x))
        return x if self.no_relu else self.relu(x)


class BNReLUConv(nn.Module):
    """BN → ReLU → Conv（DAPPM / segmentation head 常用顺序）。"""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        *,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_c, out_c, kernel_size, stride, padding, dilation, bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        # 训练期 batch=1 且空间为 1×1（DAPPM 全局池化等）时 BN 无方差可估
        if self.training and x.numel() == x.shape[1]:
            x = F.batch_norm(
                x,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                training=False,
                momentum=self.bn.momentum,
                eps=self.bn.eps,
            )
        else:
            x = self.bn(x)
        return self.conv(self.relu(x))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_c: int,
        out_c: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        *,
        no_relu: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = ConvBNReLU(in_c, out_c, 3, stride=stride)
        self.conv2 = ConvBNReLU(out_c, out_c, 3, stride=1, no_relu=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv2(self.conv1(x)) + residual
        return out if self.no_relu else self.relu(out)


class Bottleneck(nn.Module):
    """DDRNet 风格 Bottleneck：expansion=2（非 ResNet 的 4）。"""

    expansion = 2

    def __init__(
        self,
        in_c: int,
        out_c: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        *,
        no_relu: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = ConvBNReLU(in_c, out_c, 1)
        self.conv2 = ConvBNReLU(out_c, out_c, 3, stride=stride)
        self.conv3 = ConvBNReLU(out_c, out_c * self.expansion, 1, no_relu=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv3(self.conv2(self.conv1(x))) + residual
        return out if self.no_relu else self.relu(out)


class SegmentHead(nn.Module):
    def __init__(
        self,
        in_c: int,
        inter_c: int,
        out_c: int,
        *,
        scale_factor: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.conv1 = BNReLUConv(in_c, inter_c, 3, padding=1)
        self.conv2 = BNReLUConv(inter_c, out_c, 1, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x: Tensor) -> Tensor:
        feats = self.conv1(x)
        out = self.conv2(feats)
        if self.scale_factor is not None:
            h = int(feats.shape[-2] * self.scale_factor)
            w = int(feats.shape[-1] * self.scale_factor)
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out


def make_layer(
    block: Type[BasicBlock] | Type[Bottleneck],
    in_c: int,
    out_c: int,
    blocks_count: int,
    stride: int = 1,
) -> nn.Sequential:
    downsample = None
    if stride != 1 or in_c != out_c * block.expansion:
        downsample = ConvBNReLU(
            in_c, out_c * block.expansion, 1, stride=stride, no_relu=True
        )

    layers: list[nn.Module] = [block(in_c, out_c, stride, downsample)]
    in_c = out_c * block.expansion
    for i in range(1, blocks_count):
        # 段末一块延迟 ReLU，便于与双边融合的「先 sum 再 ReLU」对齐
        layers.append(block(in_c, out_c, stride=1, no_relu=(i == blocks_count - 1)))
    return nn.Sequential(*layers)
