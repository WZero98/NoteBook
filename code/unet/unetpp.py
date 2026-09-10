"""
UNet++（嵌套稠密跳跃连接 + 可选深度监督）。
"""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.nn import Module

from .unet_blocks import ConvNormAct, StackedConvLayers, Upsampling


class UNetPlusPlus(Module):
    """
    UNet++: A Nested U-Net Architecture for Medical Image Segmentation
    arXiv:1807.10165 / DOI: 10.48550/ARXIV.1807.10165

    节点记为 X^{i,j}：
      i — 下采样深度（encoder 层）
      j — 同一深度上稠密跳跃路径的第 j 个卷积块
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        filters: Sequence[int] = (32, 64, 128, 256, 512),
        *,
        num_stacked: int = 2,
        deep_supervision: bool = True,
        prune_level: Optional[int] = None,
        upsample_mode: Literal["interpolate", "transpose"] = "interpolate",
        norm_ly: Optional[Type[nn.Module]] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.filters = list(filters)
        self.depth = len(self.filters) - 1  # 最大 i
        self.deep_supervision = deep_supervision
        # prune_level=L 表示只用到 X^{0,L}（论文 L1..L4）；None 表示满深度
        self.prune_level = self.depth if prune_level is None else int(prune_level)
        assert 1 <= self.prune_level <= self.depth

        # nodes[i][j] = 处理得到 X^{i,j} 的卷积堆叠
        # 输入通道：j==0 时为上一层下采样特征；j>0 时为
        #   cat(X^{i,0..j-1}, U(X^{i+1,j-1})) → 通道 = filters[i]*(j+1)
        self.nodes = nn.ModuleList()
        for i, c in enumerate(self.filters):
            row = nn.ModuleList()
            max_j = self.depth - i
            for j in range(max_j + 1):
                if j == 0:
                    c_in = in_channels if i == 0 else self.filters[i - 1]
                else:
                    c_in = c * (j + 1)  # j 个同层历史 + 1 个上采样
                row.append(
                    StackedConvLayers(
                        c_in,
                        c,
                        num_layers=num_stacked,
                        norm_ly=norm_ly,
                    )
                )
            self.nodes.append(row)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ups[i][j]: U(X^{i+1,j}) → 对齐到 filters[i]
        # 对应从深度 i+1 上采样到深度 i、稠密列 j
        self.ups = nn.ModuleList()
        for i in range(self.depth):
            row = nn.ModuleList()
            max_j = self.depth - i - 1
            for j in range(max_j + 1):
                row.append(
                    Upsampling(
                        self.filters[i + 1],
                        self.filters[i],
                        mode=upsample_mode,
                        interp_mode="bilinear",
                        conv_kernel_size=3,
                    )
                )
            self.ups.append(row)

        # 深度监督头：X^{0,1} .. X^{0,L}
        self.heads = nn.ModuleList(
            [
                ConvNormAct(
                    self.filters[0],
                    num_classes,
                    conv_kwargs={"kernel_size": 1, "stride": 1, "padding": 0},
                    norm_ly=None,
                    act_ly=None,
                )
                for _ in range(self.prune_level)
            ]
        )

    def _forward_nodes(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """返回特征网格 feats[i][j] = X^{i,j}（仅计算到 prune_level 所需）。"""
        L = self.prune_level
        # feats[i] 长度 = L - i + 1，下标 j 对应 X^{i,j}
        feats: List[List[torch.Tensor]] = [[] for _ in range(L + 1)]

        # encoder 主干 X^{i,0}
        feats[0].append(self.nodes[0][0](x))
        for i in range(1, L + 1):
            feats[i].append(self.nodes[i][0](self.pool(feats[i - 1][0])))

        # 嵌套稠密解码：按列 j=1..L，再按行 i=L-j .. 0
        for j in range(1, L + 1):
            for i in range(L - j, -1, -1):
                parts = list(feats[i][:j])  # X^{i,0} .. X^{i,j-1}
                up = self.ups[i][j - 1](feats[i + 1][j - 1])
                if up.shape[-2:] != parts[0].shape[-2:]:
                    up = nn.functional.interpolate(
                        up,
                        size=parts[0].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                parts.append(up)
                feats[i].append(self.nodes[i][j](torch.cat(parts, dim=1)))

        return feats

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        feats = self._forward_nodes(x)
        L = self.prune_level
        # X^{0,1} .. X^{0,L}
        logits = [self.heads[j - 1](feats[0][j]) for j in range(1, L + 1)]

        if not self.deep_supervision:
            return logits[-1]

        if self.training:
            # 训练：返回从深到浅，便于加权损失（最深在前）
            return tuple(reversed(logits))

        # 推理：多尺度平均
        stacked = torch.stack(logits, dim=0)
        return stacked.mean(dim=0)


class UNetPP(UNetPlusPlus):
    """别名，便于与文件名 UNetPP 对应。"""

    pass
