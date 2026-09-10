"""
Deep Aggregation Pyramid Pooling Module (DAPPM)。

论文 Figure 5 / Eq. 2：多尺度池化 + 层级残差聚合，再 concat 压缩并加 shortcut。
在 1/64 特征上运行，几乎不影响推理延迟。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .blocks import BNReLUConv, auto_padding


class DAPPM(nn.Module):
    def __init__(self, in_c: int, branch_c: int, out_c: int) -> None:
        super().__init__()
        pad3 = auto_padding(3, 1)

        self.scale0 = BNReLUConv(in_c, branch_c, 1)
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=(5 - 1) // 2),
            BNReLUConv(in_c, branch_c, 1),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=(9 - 1) // 2),
            BNReLUConv(in_c, branch_c, 1),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=(17 - 1) // 2),
            BNReLUConv(in_c, branch_c, 1),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            BNReLUConv(in_c, branch_c, 1),
        )

        self.process1 = BNReLUConv(branch_c, branch_c, 3, padding=pad3)
        self.process2 = BNReLUConv(branch_c, branch_c, 3, padding=pad3)
        self.process3 = BNReLUConv(branch_c, branch_c, 3, padding=pad3)
        self.process4 = BNReLUConv(branch_c, branch_c, 3, padding=pad3)

        self.compression = BNReLUConv(branch_c * 5, out_c, 1)
        self.shortcut = BNReLUConv(in_c, out_c, 1)

    @staticmethod
    def _upsample(x: Tensor, size: tuple[int, int]) -> Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        size = (h, w)

        y0 = self.scale0(x)
        y1 = self.process1(self._upsample(self.scale1(x), size) + y0)
        y2 = self.process2(self._upsample(self.scale2(x), size) + y1)
        y3 = self.process3(self._upsample(self.scale3(x), size) + y2)
        y4 = self.process4(self._upsample(self.scale4(x), size) + y3)

        return self.compression(torch.cat([y0, y1, y2, y3, y4], dim=1)) + self.shortcut(x)
