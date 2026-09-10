"""
SegFormer 完整模型：MiT 编码器 + All-MLP 解码器。

"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

from .mit_encoder import MIT_VARIANTS, MixVisionTransformer
from .mlp_decoder import SegFormerHead

EncoderName = Literal["mit_b0", "mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5"]

# 官方设置：B0/B1 的 decoder 通道为 256，其余为 768
DEFAULT_EMBEDDING_DIM = {
    "mit_b0": 256,
    "mit_b1": 256,
    "mit_b2": 768,
    "mit_b3": 768,
    "mit_b4": 768,
    "mit_b5": 768,
}


class SegFormer(nn.Module):
    def __init__(
        self,
        num_classes: int = 19,
        in_chans: int = 3,
        encoder: EncoderName = "mit_b2",
        embedding_dim: Optional[int] = None,
        dropout_ratio: float = 0.1,
        upsample_output: bool = True,
        drop_path_rate: float = 0.1,
        **encoder_kwargs,
    ):
        super().__init__()
        if encoder not in MIT_VARIANTS:
            raise ValueError(f"未知 encoder={encoder}，可选: {list(MIT_VARIANTS)}")

        self.num_classes = num_classes
        self.encoder_name = encoder
        self.encoder: MixVisionTransformer = MIT_VARIANTS[encoder](
            in_chans=in_chans,
            drop_path_rate=drop_path_rate,
            **encoder_kwargs,
        )

        if embedding_dim is None:
            embedding_dim = DEFAULT_EMBEDDING_DIM[encoder]

        self.decoder = SegFormerHead(
            num_classes=num_classes,
            in_channels=self.encoder.feature_dims,
            embedding_dim=embedding_dim,
            dropout_ratio=dropout_ratio,
            upsample_output=upsample_output,
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.encoder(x)
        return self.decoder(features)

    def forward_features(self, x: Tensor) -> List[Tensor]:
        return self.encoder(x)

    @torch.no_grad()
    def load_pretrained(
        self,
        path: str,
        strict: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Union[int, List[str]]]:
        """
        部分加载预训练权重：仅写入名称与 shape 均匹配的参数。
        返回统计信息，便于微调日志。
        """
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model", "module"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    ckpt = ckpt[key]
                    break

        # 去掉常见前缀
        cleaned = {}
        for k, v in ckpt.items():
            nk = k
            for prefix in ("module.", "model.", "backbone."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix) :]
            cleaned[nk] = v

        model_sd = self.state_dict()
        matched, skipped_shape, skipped_missing = {}, [], []
        for name, param in model_sd.items():
            if name not in cleaned:
                # 尝试 encoder. / decoder. 无前缀映射（兼容仅存 MiT 权重）
                alt = name
                if name.startswith("encoder."):
                    alt = name[len("encoder.") :]
                if alt not in cleaned:
                    skipped_missing.append(name)
                    continue
                src = cleaned[alt]
            else:
                src = cleaned[name]

            if src.shape != param.shape:
                skipped_shape.append(name)
                continue
            matched[name] = src

        model_sd.update(matched)
        self.load_state_dict(model_sd, strict=strict)

        if verbose:
            print(
                f"[SegFormer] loaded={len(matched)}, "
                f"missing={len(skipped_missing)}, shape_mismatch={len(skipped_shape)}"
            )
        return {
            "loaded": len(matched),
            "missing": skipped_missing,
            "shape_mismatch": skipped_shape,
        }


def build_segformer(
    encoder: EncoderName = "mit_b2",
    num_classes: int = 19,
    **kwargs,
) -> SegFormer:
    return SegFormer(encoder=encoder, num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    model = SegFormer(num_classes=10, encoder="mit_b0", in_chans=3)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print("logits:", y.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")
