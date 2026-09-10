"""工具函数与基础算子：激活、归一化、卷积层、MBConv 等。"""

from __future__ import annotations

from functools import partial
from inspect import signature
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm


def list_sum(x: list) -> Any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def val2list(x: Union[list, tuple, Any], repeat_time: int = 1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: Union[list, tuple, Any], min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]
    return tuple(x)


def get_same_padding(kernel_size: Union[int, tuple[int, ...]]) -> Union[int, tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple(get_same_padding(ks) for ks in kernel_size)
    assert kernel_size % 2 == 1, "kernel size should be odd"
    return kernel_size // 2


def resize(
    x: Tensor,
    size: Optional[Any] = None,
    scale_factor: Optional[list[float]] = None,
    mode: str = "bicubic",
    align_corners: Optional[bool] = False,
) -> Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners
        )
    if mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    raise NotImplementedError(f"resize(mode={mode}) not implemented")


def build_kwargs_from_config(config: dict, target_func: Callable) -> dict[str, Any]:
    valid_keys = list(signature(target_func).parameters)
    return {k: config[k] for k in config if k in valid_keys}


REGISTERED_ACT_DICT: dict[str, type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
}


def build_act(name: Optional[str], **kwargs) -> Optional[nn.Module]:
    if name is None or name not in REGISTERED_ACT_DICT:
        return None
    return REGISTERED_ACT_DICT[name](**build_kwargs_from_config(kwargs, REGISTERED_ACT_DICT[name]))


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


REGISTERED_NORM_DICT: dict[str, type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
}


def build_norm(name: str = "bn2d", num_features=None, **kwargs) -> Optional[nn.Module]:
    if name in ("ln", "ln2d"):
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name not in REGISTERED_NORM_DICT:
        return None
    cls = REGISTERED_NORM_DICT[name]
    return cls(**build_kwargs_from_config(kwargs, cls))


def set_norm_eps(model: nn.Module, eps: Optional[float] = None) -> None:
    if eps is None:
        return
    for m in model.modules():
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm, _BatchNorm)):
            m.eps = eps


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super().__init__()
        padding = get_same_padding(kernel_size) * dilation
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels) if norm else None
        self.act = build_act(act_func)

    def forward(self, x: Tensor) -> Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode="bicubic",
        size: Optional[Union[int, tuple[int, int], list[int]]] = None,
        factor=2,
        align_corners=False,
    ):
        super().__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: Tensor) -> Tensor:
        if (self.size is not None and tuple(x.shape[-2:]) == tuple(self.size)) or self.factor == 1:
            return x
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class IdentityLayer(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.point_conv(self.depth_conv(x))


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        self.inverted_conv = ConvLayer(
            in_channels, mid_channels, 1, norm=norm[0], act_func=act_func[0], use_bias=use_bias[0]
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels, out_channels, 1, norm=norm[2], act_func=act_func[2], use_bias=use_bias[2]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.point_conv(self.depth_conv(self.inverted_conv(x)))


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            mid_channels, out_channels, 1, use_bias=use_bias[1], norm=norm[1], act_func=act_func[1]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.point_conv(self.spatial_conv(x))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward(self, x: Tensor) -> Tensor:
        if self.main is None:
            return x
        y = self.main(x if self.pre_norm is None else self.pre_norm(x))
        if self.shortcut is None:
            res = y
        else:
            res = y + self.shortcut(x)
            if self.post_act is not None:
                res = self.post_act(res)
        return res


class OpSequential(nn.Module):
    def __init__(self, op_list: list[Optional[nn.Module]]):
        super().__init__()
        self.op_list = nn.ModuleList([op for op in op_list if op is not None])

    def forward(self, x: Tensor) -> Tensor:
        for op in self.op_list:
            x = op(x)
        return x
