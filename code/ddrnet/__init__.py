"""DDRNet：双分辨率主干 + 双边融合 + DAPPM。"""

from .blocks import (
    BNReLUConv,
    BasicBlock,
    Bottleneck,
    ConvBNReLU,
    SegmentHead,
    auto_padding,
    make_layer,
)
from .dappm import DAPPM
from .ddrnet import (
    VARIANT_CFG,
    DualResolutionNet,
    DualReslutionNet,
    build_ddrnet,
    get_ddrnet_23,
    get_ddrnet_23_slim,
    get_ddrnet_39,
    get_ddrnet_51,
)

__all__ = [
    "auto_padding",
    "ConvBNReLU",
    "BNReLUConv",
    "BasicBlock",
    "Bottleneck",
    "SegmentHead",
    "make_layer",
    "DAPPM",
    "DualResolutionNet",
    "DualReslutionNet",
    "VARIANT_CFG",
    "build_ddrnet",
    "get_ddrnet_23_slim",
    "get_ddrnet_23",
    "get_ddrnet_39",
    "get_ddrnet_51",
]
