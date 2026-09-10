"""EfficientViT-Seg：轻量分割头 + 完整模型与变体工厂。"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .backbone import (
    EfficientViTBackbone,
    efficientvit_backbone_b0,
    efficientvit_backbone_b1,
    efficientvit_backbone_b2,
    efficientvit_backbone_b3,
)
from .ops import (
    ConvLayer,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResidualBlock,
    UpSampleLayer,
    build_kwargs_from_config,
    list_sum,
)

VariantName = Literal["b0", "b1", "b2", "b3"]


class DAGBlock(nn.Module):
    def __init__(
        self,
        inputs: dict[str, nn.Module],
        merge: str,
        post_input: Optional[nn.Module],
        middle: nn.Module,
        outputs: dict[str, nn.Module],
    ):
        super().__init__()
        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge = merge
        self.post_input = post_input
        self.middle = middle
        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        feat = [op(feature_dict[k]) for k, op in zip(self.input_keys, self.input_ops)]
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = torch_cat(feat)
        else:
            raise NotImplementedError(self.merge)
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict


def torch_cat(tensors: list[Tensor]) -> Tensor:
    import torch

    return torch.cat(tensors, dim=1)


class SegHead(DAGBlock):
    """融合 P2/P3/P4（stage2/3/4），经若干 MBConv 后输出类别图（约 1/8）。"""

    def __init__(
        self,
        fid_list: list[str],
        in_channel_list: list[int],
        stride_list: list[int],
        head_stride: int,
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        middle_op: str,
        final_expand: Optional[float],
        n_classes: int,
        dropout=0,
        norm="bn2d",
        act_func="hswish",
    ):
        inputs: dict[str, nn.Module] = {}
        for fid, in_channel, stride in zip(fid_list, in_channel_list, stride_list):
            factor = stride // head_stride
            if factor == 1:
                inputs[fid] = ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None)
            else:
                inputs[fid] = OpSequential(
                    [
                        ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None),
                        UpSampleLayer(factor=factor),
                    ]
                )

        middle_ops: list[Optional[nn.Module]] = []
        for _ in range(head_depth):
            if middle_op == "mbconv":
                block: nn.Module = MBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, act_func, None),
                )
            elif middle_op == "fmbconv":
                block = FusedMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                )
            else:
                raise NotImplementedError(middle_op)
            middle_ops.append(ResidualBlock(block, IdentityLayer()))
        middle = OpSequential(middle_ops)

        expand_w = head_width if final_expand is None else int(head_width * final_expand)
        pred = OpSequential(
            [
                None
                if final_expand is None
                else ConvLayer(head_width, expand_w, 1, norm=norm, act_func=act_func),
                ConvLayer(
                    expand_w,
                    n_classes,
                    1,
                    use_bias=True,
                    dropout=dropout,
                    norm=None,
                    act_func=None,
                ),
            ]
        )
        super().__init__(inputs, "add", None, middle=middle, outputs={"segout": pred})


class EfficientViTSeg(nn.Module):
    def __init__(self, backbone: EfficientViTBackbone, head: SegHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        feed = self.head(self.backbone(x))
        return F.interpolate(
            feed["segout"], size=x.shape[-2:], mode="bilinear", align_corners=False
        )


_HEAD_CFG = {
    "b0": dict(
        in_channel_list=[128, 64, 32],
        head_width=32,
        head_depth=1,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=4,
        builder=efficientvit_backbone_b0,
    ),
    "b1": dict(
        in_channel_list=[256, 128, 64],
        head_width=64,
        head_depth=3,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=4,
        builder=efficientvit_backbone_b1,
    ),
    "b2": dict(
        in_channel_list=[384, 192, 96],
        head_width=96,
        head_depth=3,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=4,
        builder=efficientvit_backbone_b2,
    ),
    "b3": dict(
        in_channel_list=[512, 256, 128],
        head_width=128,
        head_depth=3,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=4,
        builder=efficientvit_backbone_b3,
    ),
}


def build_efficientvit_seg(
    variant: VariantName = "b1",
    *,
    num_classes: int = 19,
    in_channels: int = 3,
    **kwargs,
) -> EfficientViTSeg:
    if variant not in _HEAD_CFG:
        raise ValueError(f"未知变体 {variant}，可选: {list(_HEAD_CFG)}")
    cfg = _HEAD_CFG[variant]
    backbone = cfg["builder"](in_channels=in_channels, **kwargs)
    head = SegHead(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=list(cfg["in_channel_list"]),
        stride_list=[32, 16, 8],
        head_stride=8,
        head_width=int(cfg["head_width"]),
        head_depth=int(cfg["head_depth"]),
        expand_ratio=float(cfg["expand_ratio"]),
        middle_op=str(cfg["middle_op"]),
        final_expand=cfg["final_expand"],
        n_classes=num_classes,
        **build_kwargs_from_config(kwargs, SegHead),
    )
    return EfficientViTSeg(backbone, head)


def efficientvit_seg_b0(in_channels: int, num_classes: int, **kwargs) -> EfficientViTSeg:
    return build_efficientvit_seg("b0", num_classes=num_classes, in_channels=in_channels, **kwargs)


def efficientvit_seg_b1(in_channels: int, num_classes: int, **kwargs) -> EfficientViTSeg:
    return build_efficientvit_seg("b1", num_classes=num_classes, in_channels=in_channels, **kwargs)


def efficientvit_seg_b2(in_channels: int, num_classes: int, **kwargs) -> EfficientViTSeg:
    return build_efficientvit_seg("b2", num_classes=num_classes, in_channels=in_channels, **kwargs)


def efficientvit_seg_b3(in_channels: int, num_classes: int, **kwargs) -> EfficientViTSeg:
    return build_efficientvit_seg("b3", num_classes=num_classes, in_channels=in_channels, **kwargs)


if __name__ == "__main__":
    import torch

    m = build_efficientvit_seg("b0", num_classes=19, in_channels=3)
    y = m(torch.randn(1, 3, 256, 256))
    print(y.shape)
