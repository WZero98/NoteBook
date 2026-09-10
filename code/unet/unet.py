"""
经典 U-Net（同 padding 实用版）。

论文原版使用 valid padding（无填充）+ skip 时 crop；
工程中更常见 same padding（padding=1），本实现默认后者，并通过
`valid_padding=True` 可切换到更接近原论文的行为。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Type

import torch
import torch.nn as nn
from torch.nn import Module

from .unet_blocks import StackedConvLayers, Upsampling, center_crop_to_match


class UNet(Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    arXiv:1505.04597 / DOI: 10.48550/ARXIV.1505.04597
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 64,
        depth: int = 4,
        *,
        num_stacked: int = 2,
        bilinear: bool = False,
        valid_padding: bool = False,
        norm_ly: Optional[Type[nn.Module]] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        assert depth >= 1
        self.depth = depth
        self.valid_padding = valid_padding
        pad = 0 if valid_padding else 1
        conv_kwargs = {"kernel_size": 3, "stride": 1, "padding": pad}

        channels: List[int] = [base_channels * (2**i) for i in range(depth + 1)]

        # Encoder: e0 .. ed
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        c_in = in_channels
        for i, c_out in enumerate(channels):
            self.encoders.append(
                StackedConvLayers(
                    c_in,
                    c_out,
                    num_layers=num_stacked,
                    conv_kwargs=conv_kwargs,
                    norm_ly=norm_ly,
                )
            )
            if i < depth:
                self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            c_in = c_out

        # Decoder
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth, 0, -1):
            c_from, c_to = channels[i], channels[i - 1]
            if bilinear:
                self.ups.append(
                    Upsampling(
                        c_from,
                        c_to,
                        mode="interpolate",
                        interp_mode="bilinear",
                        conv_kernel_size=1 if not valid_padding else 3,
                    )
                )
            else:
                self.ups.append(
                    Upsampling(c_from, c_to, mode="transpose")
                )
            # skip concat → 通道变为 2 * c_to
            self.decoders.append(
                StackedConvLayers(
                    c_to * 2,
                    c_to,
                    num_layers=num_stacked,
                    conv_kwargs=conv_kwargs,
                    norm_ly=norm_ly,
                )
            )

        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < self.depth:
                skips.append(x)
                x = self.pools[i](x)

        # x 已是 bottleneck
        for i, (up, dec) in enumerate(zip(self.ups, self.decoders)):
            x = up(x)
            skip = skips[-(i + 1)]
            if self.valid_padding:
                skip = center_crop_to_match(skip, x)
            elif x.shape[-2:] != skip.shape[-2:]:
                # 插值对齐，防止奇数尺寸导致 1 像素偏差
                x = nn.functional.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = dec(torch.cat([skip, x], dim=1))

        return self.head(x)


def build_classic_unet(
    in_channels: int = 1,
    num_classes: int = 2,
    base_channels: Sequence[int] | None = None,
) -> UNet:
    """快捷构造：通道数接近原论文 64-128-256-512-1024。"""
    if base_channels is not None:
        # 若传入完整列表，取第一项作为 base
        base = int(base_channels[0])
    else:
        base = 64
    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base,
        depth=4,
        bilinear=False,
        valid_padding=False,
    )
