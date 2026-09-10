"""
DeepLabv3+ 完整模型。

参考：Chen et al., Encoder-Decoder with Atrous Separable Convolution for
Semantic Image Segmentation. arXiv:1802.02611

实现要点：
  - 输出尺寸随输入动态插值；
  - 支持 ResNet-50/101；可选 separable ASPP；
  - 预训练走 torchvision Weights API，亦可加载权重文件路径。
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .backbone import BackboneName, ResNetBackbone, build_resnet_backbone
from .decoder import DeepLabHeadV3Plus

OutputStride = Literal[8, 16]


def _default_aspp_rates(output_stride: int) -> tuple[int, ...]:
    # 与 DeepLabv3+ / torchvision 习惯一致
    if output_stride == 8:
        return (12, 24, 36)
    return (6, 12, 18)


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        num_classes: int = 21,
        in_channels: int = 3,
        *,
        backbone: BackboneName = "resnet101",
        output_stride: OutputStride = 16,
        pretrained_backbone: bool = True,
        backbone_weights_path: Optional[str] = None,
        aspp_dilate: Optional[Sequence[int]] = None,
        separable_aspp: bool = False,
        aspp_dropout: float = 0.5,
        low_level_project_channels: int = 48,
    ) -> None:
        super().__init__()
        if output_stride not in (8, 16):
            raise ValueError("output_stride 仅支持 8 或 16")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.output_stride = int(output_stride)

        # 非 1/3 通道输入时，用轻量 stem 映射到 3 通道以复用 ImageNet 骨干
        self.input_proj: Optional[nn.Module]
        if in_channels == 3:
            self.input_proj = None
        elif in_channels == 1:
            self.input_proj = None  # forward 里复制到 3 通道
        else:
            self.input_proj = nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True),
            )

        self.backbone: ResNetBackbone = build_resnet_backbone(
            backbone,
            output_stride=self.output_stride,
            pretrained=pretrained_backbone,
            weights_path=backbone_weights_path,
        )

        rates = (
            tuple(aspp_dilate)
            if aspp_dilate is not None
            else _default_aspp_rates(self.output_stride)
        )
        self.aspp_dilate = rates

        self.head = DeepLabHeadV3Plus(
            in_channels=self.backbone.out_channels,
            low_level_channels=self.backbone.low_level_channels,
            num_classes=num_classes,
            aspp_dilate=rates,
            separable_aspp=separable_aspp,
            aspp_dropout=aspp_dropout,
            low_level_project_channels=low_level_project_channels,
        )

    def _prepare_input(self, x: Tensor) -> Tensor:
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"期望输入通道 {self.in_channels}，实际得到 {x.shape[1]}"
            )
        if self.in_channels == 1:
            return x.repeat(1, 3, 1, 1)
        if self.input_proj is not None:
            return self.input_proj(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        input_size = x.shape[-2:]
        x = self._prepare_input(x)
        features = self.backbone(x)
        return self.head(features, output_size=input_size)

    def forward_features(self, x: Tensor) -> dict:
        x = self._prepare_input(x)
        return self.backbone(x)


def build_deeplabv3plus(
    num_classes: int = 21,
    backbone: BackboneName = "resnet50",
    **kwargs,
) -> DeepLabV3Plus:
    return DeepLabV3Plus(num_classes=num_classes, backbone=backbone, **kwargs)


if __name__ == "__main__":
    model = DeepLabV3Plus(
        num_classes=21,
        backbone="resnet50",
        output_stride=16,
        pretrained_backbone=False,
    )
    y = model(torch.randn(2, 3, 256, 256))
    print(y.shape)  # [2, 21, 256, 256]
