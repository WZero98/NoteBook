"""
Deep Dual-resolution Network (DDRNet)。

参考：Hong et al., Deep Dual-resolution Networks for Real-time and Accurate
Semantic Segmentation of Road Scenes. arXiv:2101.06085

实现要点：
  - 类名 DualResolutionNet；保留 DualReslutionNet 别名以兼容常见拼写；
  - 插值统一 ``align_corners=False``；
  - 变体注册表 + ``build_ddrnet``；
  - ``augment`` 训练时返回 (main, aux)，推理仅用主头。
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .blocks import (
    BasicBlock,
    Bottleneck,
    ConvBNReLU,
    SegmentHead,
    auto_padding,
    make_layer,
)
from .dappm import DAPPM

VariantName = Literal["ddrnet23_slim", "ddrnet23", "ddrnet39", "ddrnet51"]

# layers / base_channels / spp_channels / head_channels
VARIANT_CFG: Dict[VariantName, Dict[str, object]] = {
    "ddrnet23_slim": {
        "layers": [2, 2, 2, 2],
        "planes": 32,
        "spp_planes": 128,
        "head_planes": 64,
    },
    "ddrnet23": {
        "layers": [2, 2, 2, 2],
        "planes": 64,
        "spp_planes": 128,
        "head_planes": 128,
    },
    "ddrnet39": {
        "layers": [3, 4, 6, 3],
        "planes": 64,
        "spp_planes": 128,
        "head_planes": 256,
    },
    "ddrnet51": {
        "layers": [4, 6, 8, 4],
        "planes": 64,
        "spp_planes": 128,
        "head_planes": 256,
    },
}


class DualResolutionNet(nn.Module):
    """
    双分辨率主干 + 多次双边融合 + DAPPM + SegHead。

    高分辨率支路保持约 1/8；低分辨率支路下采到 1/64 后经 DAPPM 回注。
    """

    def __init__(
        self,
        layers: Sequence[int],
        *,
        num_classes: int = 19,
        in_channels: int = 3,
        planes: int = 64,
        spp_planes: int = 128,
        head_planes: int = 128,
        augment: bool = False,
    ) -> None:
        super().__init__()
        if len(layers) != 4:
            raise ValueError("layers 应为长度 4 的列表，如 [2,2,2,2]")
        if layers[2] % 2 != 0:
            raise ValueError("layers[2] 需为偶数，以便拆成 layer3_1 / layer3_2")

        self.augment = augment
        self.num_classes = num_classes
        highres_planes = planes * 2
        pad_s2 = auto_padding(3, 2)

        self.conv1 = nn.Sequential(
            ConvBNReLU(in_channels, planes, 3, stride=2, padding=pad_s2),
            ConvBNReLU(planes, planes, 3, stride=2, padding=pad_s2),
        )
        self.relu = nn.ReLU(inplace=False)

        # 共享 trunk → 再分叉
        self.layer1 = make_layer(BasicBlock, planes, planes, layers[0])
        self.layer2 = make_layer(BasicBlock, planes, planes * 2, layers[1], stride=2)

        # 低分辨率支路
        self.layer3_1 = make_layer(
            BasicBlock, planes * 2, planes * 4, layers[2] // 2, stride=2
        )
        self.layer3_2 = make_layer(BasicBlock, planes * 4, planes * 4, layers[2] // 2)
        self.layer4 = make_layer(BasicBlock, planes * 4, planes * 8, layers[3], stride=2)
        self.layer5 = make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)

        # 高分辨率支路（始终约 1/8）
        self.layer3_1_ = make_layer(BasicBlock, planes * 2, highres_planes, layers[2] // 2)
        self.layer3_2_ = make_layer(
            BasicBlock, highres_planes, highres_planes, layers[2] // 2
        )
        self.layer4_ = make_layer(BasicBlock, highres_planes, highres_planes, layers[3])
        self.layer5_ = make_layer(Bottleneck, highres_planes, highres_planes, 1)

        # 双边融合：低→高 compression；高→低 down
        self.compression3_1 = ConvBNReLU(planes * 4, highres_planes, 1, no_relu=True)
        self.compression3_2 = ConvBNReLU(planes * 4, highres_planes, 1, no_relu=True)
        self.compression4 = ConvBNReLU(planes * 8, highres_planes, 1, no_relu=True)

        self.down3_1 = ConvBNReLU(
            highres_planes, planes * 4, 3, stride=2, padding=pad_s2, no_relu=True
        )
        self.down3_2 = ConvBNReLU(
            highres_planes, planes * 4, 3, stride=2, padding=pad_s2, no_relu=True
        )
        self.down4 = nn.Sequential(
            ConvBNReLU(highres_planes, planes * 4, 3, stride=2, padding=pad_s2),
            ConvBNReLU(
                planes * 4, planes * 8, 3, stride=2, padding=pad_s2, no_relu=True
            ),
        )

        # Bottleneck expansion=2 → layer5 输出 planes*16
        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        if self.augment:
            self.seghead_extra = SegmentHead(highres_planes, head_planes, num_classes)

        # DAPPM 输出 planes*4，与高分辨率 RBB（expansion=2）通道一致，按论文 Fig.4 做 sum
        self.final_layer = SegmentHead(planes * 4, head_planes, num_classes)
        self._init_weight()

    def forward(
        self, x: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        h, w = x.shape[-2:]
        out_h, out_w = h // 8, w // 8

        x = self.conv1(x)  # 1/4
        x = self.layer1(x)
        x = self.layer2(self.relu(x))  # 1/8
        x_high_in = x

        # --- stage 3-1 双边融合（compression 使用融合前的低分辨率特征）---
        x_low = self.layer3_1(self.relu(x))  # 1/16
        x_ = self.layer3_1_(self.relu(x_high_in))  # 1/8
        x = x_low + self.down3_1(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression3_1(self.relu(x_low)),
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        )

        # --- stage 3-2 ---
        x_low = self.layer3_2(self.relu(x))
        x_ = self.layer3_2_(self.relu(x_))
        x = x_low + self.down3_2(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression3_2(self.relu(x_low)),
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        )
        aux_feat = x_

        # --- stage 4 ---
        x_low = self.layer4(self.relu(x))  # 1/32
        x_ = self.layer4_(self.relu(x_))
        x = x_low + self.down4(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression4(self.relu(x_low)),
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        )

        # --- stage 5 + DAPPM；末次低→高：上采样后与高分辨率支路逐点相加（论文 Fig.4）---
        x_ = self.layer5_(self.relu(x_))
        x = F.interpolate(
            self.spp(self.layer5(self.relu(x))),  # 1/64 → DAPPM
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        )

        logits = self.final_layer(x + x_)
        main = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)

        if self.augment:
            aux = self.seghead_extra(aux_feat)
            aux = F.interpolate(aux, size=(h, w), mode="bilinear", align_corners=False)
            return main, aux
        return main

    def _init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# 兼容常见拼写
DualReslutionNet = DualResolutionNet


def build_ddrnet(
    variant: VariantName = "ddrnet23",
    *,
    num_classes: int = 19,
    in_channels: int = 3,
    augment: bool = False,
) -> DualResolutionNet:
    if variant not in VARIANT_CFG:
        raise ValueError(f"未知变体 {variant}，可选: {list(VARIANT_CFG)}")
    cfg = VARIANT_CFG[variant]
    return DualResolutionNet(
        layers=list(cfg["layers"]),  # type: ignore[arg-type]
        num_classes=num_classes,
        in_channels=in_channels,
        planes=int(cfg["planes"]),  # type: ignore[arg-type]
        spp_planes=int(cfg["spp_planes"]),  # type: ignore[arg-type]
        head_planes=int(cfg["head_planes"]),  # type: ignore[arg-type]
        augment=augment,
    )


def get_ddrnet_23_slim(num_classes: int, in_c: int = 3, augment: bool = False) -> DualResolutionNet:
    return build_ddrnet("ddrnet23_slim", num_classes=num_classes, in_channels=in_c, augment=augment)


def get_ddrnet_23(num_classes: int, in_c: int = 3, augment: bool = False) -> DualResolutionNet:
    return build_ddrnet("ddrnet23", num_classes=num_classes, in_channels=in_c, augment=augment)


def get_ddrnet_39(num_classes: int, in_c: int = 3, augment: bool = False) -> DualResolutionNet:
    return build_ddrnet("ddrnet39", num_classes=num_classes, in_channels=in_c, augment=augment)


def get_ddrnet_51(num_classes: int, in_c: int = 3, augment: bool = False) -> DualResolutionNet:
    return build_ddrnet("ddrnet51", num_classes=num_classes, in_channels=in_c, augment=augment)


if __name__ == "__main__":
    m = build_ddrnet("ddrnet23_slim", num_classes=19, in_channels=3)
    y = m(torch.randn(2, 3, 256, 256))
    print(y.shape)
