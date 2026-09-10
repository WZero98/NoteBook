"""
U-Net / UNet++ 共用基础模块。

"""

from __future__ import annotations

from collections import OrderedDict
from typing import Literal, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.nn import Module


class ConvNormAct(Module):
    """Conv → (Norm) → (Activation) → (Dropout) 的标准卷积单元。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        conv_ly: Type[nn.Module] = nn.Conv2d,
        conv_kwargs: Optional[dict] = None,
        norm_ly: Optional[Type[nn.Module]] = nn.BatchNorm2d,
        norm_kwargs: Optional[dict] = None,
        act_ly: Optional[Type[nn.Module]] = nn.ReLU,
        act_kwargs: Optional[dict] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        if conv_kwargs is None:
            conv_kwargs = {"kernel_size": 3, "stride": 1, "padding": 1}
        if norm_kwargs is None:
            norm_kwargs = {"eps": 1e-5, "momentum": 0.1}
        if act_kwargs is None and act_ly is nn.ReLU:
            act_kwargs = {"inplace": True}

        self.conv = conv_ly(in_channels, out_channels, **conv_kwargs)
        self.norm = norm_ly(out_channels, **norm_kwargs) if norm_ly is not None else None
        if act_ly is None:
            self.act = None
        elif act_kwargs is not None:
            self.act = act_ly(**act_kwargs)
        else:
            self.act = act_ly()
        self.drop = nn.Dropout2d(p=dropout) if dropout is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        if self.drop is not None:
            x = self.drop(x)
        return x


# 兼容旧命名
ConvNormNonlinearDropout = ConvNormAct


class StackedConvLayers(Module):
    """堆叠若干 ConvNormAct；第一层可改变通道数，后续层保持 out_channels。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        *,
        conv_ly: Type[nn.Module] = nn.Conv2d,
        conv_kwargs: Optional[dict] = None,
        norm_ly: Optional[Type[nn.Module]] = nn.BatchNorm2d,
        norm_kwargs: Optional[dict] = None,
        act_ly: Optional[Type[nn.Module]] = nn.ReLU,
        act_kwargs: Optional[dict] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        assert num_layers >= 1
        blocks: list[tuple[str, Module]] = []
        c_in = in_channels
        for i in range(num_layers):
            blocks.append(
                (
                    f"cnn_block{i + 1}",
                    ConvNormAct(
                        c_in,
                        out_channels,
                        conv_ly=conv_ly,
                        conv_kwargs=conv_kwargs,
                        norm_ly=norm_ly,
                        norm_kwargs=norm_kwargs,
                        act_ly=act_ly,
                        act_kwargs=act_kwargs,
                        dropout=dropout,
                    ),
                )
            )
            c_in = out_channels
        self.stack = nn.Sequential(OrderedDict(blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


class Upsampling(Module):
    """
    上采样模块。

    mode='interpolate'：bilinear/nearest 插值后再用卷积对齐通道（UNet++ 常用）。
    mode='transpose'  ：转置卷积一步完成上采样 + 通道变换（经典 U-Net 常用）。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        mode: Literal["interpolate", "transpose"] = "interpolate",
        interp_mode: Literal["nearest", "bilinear", "bicubic"] = "bilinear",
        scale_factor: float = 2.0,
        align_corners: Optional[bool] = False,
        conv_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.interp_mode = interp_mode
        self.scale_factor = scale_factor
        self.align_corners = None if interp_mode == "nearest" else align_corners

        if mode == "transpose":
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
            self.post = None
        else:
            self.up = None
            pad = conv_kernel_size // 2
            self.post = nn.Conv2d(
                in_channels, out_channels, kernel_size=conv_kernel_size, padding=pad
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "transpose":
            return self.up(x)
        x = nn.functional.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.interp_mode,
            align_corners=self.align_corners,
        )
        return self.post(x)


def center_crop_to_match(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """将 src 中心裁剪到与 ref 相同的空间尺寸（对应原论文 valid padding 的 crop）。"""
    _, _, h, w = src.shape
    _, _, th, tw = ref.shape
    if (h, w) == (th, tw):
        return src
    top = (h - th) // 2
    left = (w - tw) // 2
    return src[:, :, top : top + th, left : left + tw]
