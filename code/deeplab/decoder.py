"""
DeepLabv3+ 解码器头。

流程（论文 Figure 2 / Sec. 3.1）：
  encoder(ASPP) 特征 ×4 上采样
  + low-level（ResNet Conv2 / layer1）经 1×1 压到 48 通道
  → concat → 两次 3×3(256) → 1×1 分类 → ×4 上采样到输入分辨率
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .aspp import ASPP, _conv_bn_relu


class DeepLabHeadV3Plus(nn.Module):
    def __init__(
        self,
        in_channels: int,
        low_level_channels: int,
        num_classes: int,
        *,
        aspp_dilate: Sequence[int] = (12, 24, 36),
        aspp_out_channels: int = 256,
        low_level_project_channels: int = 48,
        separable_aspp: bool = False,
        aspp_dropout: float = 0.5,
        classifier_channels: int = 256,
    ) -> None:
        super().__init__()
        self.project = _conv_bn_relu(
            low_level_channels, low_level_project_channels, 1
        )
        self.aspp = ASPP(
            in_channels,
            aspp_dilate,
            out_channels=aspp_out_channels,
            separable=separable_aspp,
            dropout=aspp_dropout,
        )

        fuse_in = aspp_out_channels + low_level_project_channels
        self.classifier = nn.Sequential(
            _conv_bn_relu(fuse_in, classifier_channels, 3, padding=1),
            _conv_bn_relu(classifier_channels, classifier_channels, 3, padding=1),
            nn.Conv2d(classifier_channels, num_classes, kernel_size=1),
        )
        self._init_weight()

    def forward(
        self,
        feature: Dict[str, Tensor],
        *,
        output_size: Optional[tuple[int, int]] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        feature : dict
            需含 ``'out'``（高层 / ASPP 输入）与 ``'low_level'``（浅层细节）。
        output_size : (H, W), optional
            最终插值目标尺寸；默认与 ``low_level`` 的 4 倍对齐
            （当 backbone output_stride=16 且 low_level 为 H/4 时即原图）。
        """
        low = feature["low_level"]
        low_proj = self.project(low)

        high = self.aspp(feature["out"])
        high = F.interpolate(
            high, size=low_proj.shape[-2:], mode="bilinear", align_corners=False
        )

        x = self.classifier(torch.cat([low_proj, high], dim=1))

        if output_size is None:
            # low_level 通常为 H/4；再 ×4 回到输入分辨率
            h, w = low.shape[-2:]
            output_size = (h * 4, w * 4)
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)

    def _init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
