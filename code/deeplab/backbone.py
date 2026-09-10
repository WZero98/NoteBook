"""
ResNet 骨干（DeepLabv3+ 用）：输出高层 ``out`` 与浅层 ``low_level``。

通过 ``replace_stride_with_dilation`` 控制 output_stride：
  - OS=16: [False, False, True]
  - OS=8 : [False, True, True]
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import (
    ResNet50_Weights,
    ResNet101_Weights,
    resnet50,
    resnet101,
)

BackboneName = Literal["resnet50", "resnet101"]

_BUILDERS = {
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V2),
}

# torchvision ResNet: layer1 输出通道
LOW_LEVEL_CHANNELS = {
    "resnet50": 256,
    "resnet101": 256,
}

BACKBONE_OUT_CHANNELS = {
    "resnet50": 2048,
    "resnet101": 2048,
}


class ResNetBackbone(nn.Module):
    """
    返回::

        {
          'low_level': layer1 特征 (约 H/4),
          'out':       layer4 特征 (约 H/OS),
        }
    """

    def __init__(
        self,
        name: BackboneName = "resnet101",
        *,
        output_stride: int = 16,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        if name not in _BUILDERS:
            raise ValueError(f"未知骨干 {name}，可选: {list(_BUILDERS)}")
        if output_stride not in (8, 16):
            raise ValueError("output_stride 仅支持 8 或 16")

        if output_stride == 8:
            replace_stride_with_dilation: List[bool] = [False, True, True]
        else:
            replace_stride_with_dilation = [False, False, True]

        builder, default_weights = _BUILDERS[name]
        weights = None
        if pretrained and weights_path is None:
            weights = default_weights

        net = builder(
            weights=weights,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )

        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # 去掉可能的 module. 前缀
            cleaned = {
                (k[7:] if k.startswith("module.") else k): v for k, v in state.items()
            }
            net.load_state_dict(cleaned, strict=False)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self.name = name
        self.output_stride = output_stride
        self.low_level_channels = LOW_LEVEL_CHANNELS[name]
        self.out_channels = BACKBONE_OUT_CHANNELS[name]

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.stem(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        out = self.layer4(x)
        return {"low_level": low, "out": out}


def build_resnet_backbone(
    name: BackboneName = "resnet101",
    **kwargs,
) -> ResNetBackbone:
    return ResNetBackbone(name, **kwargs)
