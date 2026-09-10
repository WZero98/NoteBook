"""EfficientViT Backbone（B 系列）。"""

from __future__ import annotations

from typing import Dict

import torch.nn as nn
from torch import Tensor

from .attention import EfficientViTBlock
from .ops import (
    ConvLayer,
    DSConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResidualBlock,
    build_kwargs_from_config,
)


class EfficientViTBackbone(nn.Module):
    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        in_channels=3,
        dim=32,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
    ) -> None:
        super().__init__()
        self.width_list: list[int] = []

        stem: list[nn.Module | None] = [
            ConvLayer(in_channels, width_list[0], stride=2, norm=norm, act_func=act_func)
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                width_list[0], width_list[0], 1, 1, norm, act_func
            )
            stem.append(ResidualBlock(block, IdentityLayer()))
        in_c = width_list[0]
        self.input_stem = OpSequential(stem)
        self.width_list.append(in_c)

        stages: list[nn.Module] = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(in_c, w, stride, expand_ratio, norm, act_func)
                stage.append(ResidualBlock(block, IdentityLayer() if stride == 1 else None))
                in_c = w
            stages.append(OpSequential(stage))
            self.width_list.append(in_c)

        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = [
                ResidualBlock(
                    self.build_local_block(
                        in_c, w, 2, expand_ratio, norm, act_func, fewer_norm=True
                    ),
                    None,
                )
            ]
            in_c = w
            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_c, dim=dim, expand_ratio=expand_ratio, norm=norm, act_func=act_func
                    )
                )
            stages.append(OpSequential(stage))
            self.width_list.append(in_c)

        self.stages = nn.ModuleList(stages)

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if expand_ratio == 1:
            return DSConv(
                in_channels,
                out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        return MBConv(
            in_channels,
            out_channels,
            stride=stride,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False) if fewer_norm else False,
            norm=(None, None, norm) if fewer_norm else norm,
            act_func=(act_func, act_func, None),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {"input": x}
        out["stage0"] = x = self.input_stem(x)
        for i, stage in enumerate(self.stages, 1):
            out[f"stage{i}"] = x = stage(x)
        out["stage_final"] = x
        return out


def efficientvit_backbone_b0(**kwargs) -> EfficientViTBackbone:
    return EfficientViTBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )


def efficientvit_backbone_b1(**kwargs) -> EfficientViTBackbone:
    return EfficientViTBackbone(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[1, 2, 3, 3, 4],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )


def efficientvit_backbone_b2(**kwargs) -> EfficientViTBackbone:
    return EfficientViTBackbone(
        width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )


def efficientvit_backbone_b3(**kwargs) -> EfficientViTBackbone:
    return EfficientViTBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
