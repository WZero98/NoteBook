# EfficientViT-Seg 说明笔记

> 个人学习笔记：Vision Transformer 发展脉络、EfficientViT（多尺度线性注意力）及其分割形态 EfficientViT-Seg 架构要点，以及可运行的 PyTorch 模块实现。

---

## 1. 发展脉络、效果与应用

### 1.1 从 CNN 密集预测到「高效 ViT」

Vision Transformer（**ViT**，Dosovitskiy et al., 2020）把图像切成 patch，用标准 Transformer 做分类，证明「大规模预训练 + 注意力」可与 CNN 分庭抗礼。随后密集预测（分割、超分、检测）迅速跟进，但面临两类矛盾：

| 需求 | Softmax 注意力 / 大核卷积的代价 |
|------|--------------------------------|
| **高分辨率输入**（路景 1024×2048、摄影超分） | Softmax 注意力对序列长度 **二次**，显存与延迟爆炸 |
| **全局感受野 + 多尺度** | SegFormer 等引入全局注意力；SegNeXt 用很大卷积核——硬件上往往不友好 |

于是出现多条「让 Transformer 跑得动」的路线：窗口注意力（Swin）、金字塔（PVT）、轻量混合（MobileViT / EfficientFormer），以及本文的 **EfficientViT**（Cai et al. / MIT-HAN Lab）：

> 用 **ReLU 线性注意力**拿全局感受野（复杂度近似线性），再用 **小核深度卷积聚合多尺度 token**，并在 FFN 中插入 DWConv 补局部信息——专为**高分辨率密集预测**设计，且延迟可在手机 CPU / 边缘 GPU / 云 GPU 上兑现。

> 命名提示：另有 Microsoft 等提出的同名「EfficientViT」（偏分类 / 不同结构）。本笔记专指 **arXiv:2205.14756 / mit-han-lab** 这一支及其分割头形态。

### 1.2 脉络年表（精选）

| 阶段 | 代表工作 | 核心改动 | 备注 |
|------|----------|----------|------|
| 2017 | Transformer (NLP) | Self-Attention | 后续视觉迁移的基础 |
| 2020 | **ViT** | 图像 patch + 纯 Transformer 分类 | [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) |
| 2020–21 | DeiT | 蒸馏与数据高效训练 | 降低 ViT 对超大数据依赖 |
| 2021 | **Swin** | 窗口注意力 + 移位 | 层次化，检测/分割骨干常用 |
| 2021 | PVT / **SETR** | 金字塔；ViT 做分割编码器 | 分割进入 Transformer 时代 |
| 2021 | **SegFormer** | 分层 MiT + All-MLP | 高效分割强基线 |
| 2021–22 | MobileViT、EfficientFormer | 移动端混合 CNN–Transformer | 偏分类 / 轻部署 |
| 2022 | SegNeXt | 多尺度卷积注意力 | 精度高，大核实现依赖硬件 |
| 2022–23 | **EfficientViT** | **多尺度 ReLU 线性注意力** | 本文重点；[doi:10.48550/ARXIV.2205.14756](https://doi.org/10.48550/arXiv.2205.14756) |
| 2023 | SAM / MobileSAM 等 | 提示分割；可用 EfficientViT 加速图像编码器 | 零样本实例分割生态 |
| 2023–25 | EfficientViT 后续发布、各类边缘 ViT-Seg | L 系列云端规格；超分 / SAM 适配 | 官方仓库持续更新 |

```text
ViT (2020) ── 分类
  ├─ DeiT / Swin / PVT ── 训练与层次化骨干
  ├─ SETR / SegFormer / SegNeXt ── 分割专用
  ├─ MobileViT / EfficientFormer ── 移动端混合
  └─ EfficientViT (2022) ── 线性注意力 + 多尺度，面向高分辨率密集预测
        ├─ EfficientViT-Seg（Cityscapes / ADE20K）
        ├─ 超分辨率头
        └─ SAM 图像编码器加速等
```

### 1.3 取得的效果（如何理解「强」）

论文在 **Cityscapes / ADE20K** 语义分割、超分、Segment Anything 与 ImageNet 上系统对比（单模型、单尺度）：

| 模型 | Cityscapes mIoU | 相对对照的延迟优势（论文口径） |
|------|-----------------|--------------------------------|
| EfficientViT-B0 | **75.7%** | 极轻量（约 0.7M / 4.4G MAC @1024×2048） |
| EfficientViT-B1 | **80.5%** | 相对 SegFormer-B1 / SegNeXt-T 同档精度，GPU 延迟显著更低 |
| EfficientViT-B2 | **82.1%** | 相对 SegFormer-B3 / SegNeXt-S 更快 |
| EfficientViT-B3 | **83.0%** | 相对 SegFormer-B5 / SegNeXt-B 更快 |
| EfficientViT-L2 | **83.2%** | 云端规格，与 SegNeXt-L 同档、延迟更优 |

要点解读：

1. **精度–延迟同时赢**：在相近 mIoU 分组里，EfficientViT 在 Jetson Orin / A100（TensorRT fp16）上 systematically 更快；论文称相对 SegFormer / SegNeXt 最高约 **13.9× / 6.2×** GPU 延迟下降（Cityscapes、无掉点设定下的对比叙述）。  
2. **线性注意力可落地**：相对 Softmax 注意力，同算量下移动 CPU 上可快约 **3.3–4.5×**（去掉 softmax 等不友好算子）。  
3. **不止分割**：超分相对 Restormer 最高约 **6.4×** 加速且 PSNR 略升；作 SAM 图像编码器相对 ViT-H 约 **48.9×** A100 吞吐；ImageNet 上 L2-r384 达 **86.0%** Top-1。  
4. **消融**：同 MAC 下，去掉「多尺度」或「全局注意力」任一，Cityscapes mIoU 从 74.5 掉到约 72.x（从头训）。

### 1.4 主要应用领域

| 领域 | 典型任务 | 为何常用 EfficientViT 系 |
|------|----------|---------------------------|
| **自动驾驶 / 街景分割** | Cityscapes、驾驶感知 | 原论文主战场；高分辨率下延迟优势明显 |
| **通用场景分割** | ADE20K 等 | B1–B3 覆盖不同算力档 |
| **边缘 / 移动端视觉** | 手机 CPU、Orin 等边缘 GPU | 算子硬件友好，FLOPs 易转化成实测 FPS |
| **计算摄影 / 超分辨率** | 轻量 SR、高分辨率 SR | 同一骨干思路迁移到像素级回归 |
| **提示分割加速** | SAM 类系统的图像编码器替换 | 大幅提高吞吐，便于交互式/批量分割 |
| **云端分类与骨干复用** | ImageNet、下游检测/分割预训练 | L 系列兼顾精度与吞吐 |

> 实务建议：边缘实时分割优先 **B0/B1**；服务器侧要精度用 **B2/B3 或 L 系列**；需要与 SegFormer 对照时，重点看 **同 mIoU 下的实测延迟**，而非只看参数量。

---

## 2. EfficientViT / EfficientViT-Seg 架构详解

### 2.1 文献

> **Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han.**  
> *EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction.*  
> DOI: [10.48550/ARXIV.2205.14756](https://doi.org/10.48550/arXiv.2205.14756) · [arXiv:2205.14756](https://arxiv.org/abs/2205.14756) · [官方代码 mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit)

### 2.2 宏观结构（对照图）

![EfficientViT 宏观结构（原论文 Figure 5）](assets/EfficientViT%20structure.png)

上图（`assets/EfficientViT structure.png`）即论文 **Figure 5**：标准 **Backbone + Head（Encoder–Decoder）**。

| 部分 | 内容 |
|------|------|
| **Input Stem** | Conv → DSConv，分辨率约 ×1/2 |
| **Stage 1–2** | 以 **MBConv** 为主，快速下采样并抽局部特征；**P2** 来自 Stage 2 |
| **Stage 3–4** | 下采样 MBConv + 堆叠 **EfficientViT Module**；输出 **P3、P4** |
| **Fusion** | P2 / P3↑×2 / P4↑×4 经 1×1 对齐通道后 **逐点相加** |
| **Head** | 若干 MBConv + 预测层；输出约 1/8，再插值回原图 |

设计取舍：骨干已有强上下文，故头极简（加法融合 + MBConv），避免重型 ASPP / 复杂解码器。

### 2.3 EfficientViT Module 与多尺度线性注意力

![EfficientViT 构建块与多尺度线性注意力（原论文 Figure 2）](assets/EfficientViT%20multiscale%20RELU%20attention.png)

上图（`assets/EfficientViT multiscale RELU attention.png`）即论文 **Figure 2**：

| 侧 | 内容 |
|----|------|
| **左：EfficientViT Module** | **Multi-Scale Linear Att**（上下文）→ **FFN+DWConv**（局部），均为残差式堆叠 |
| **右：Multi-Scale Linear Att** | Linear 得到 Q/K/V → 多尺度聚合（含 3×3 / 5×5 DWConv + 1×1 GConv）→ 各尺度 **ReLU Linear Attention** → Concat → Linear 融合 |

信息流可概括为：

```text
x → LiteMLA（多尺度 ReLU 线性注意力，残差）
  → MBConv / FFN+DWConv（局部，残差）
```

**（1）ReLU 线性注意力（全局、近似 O(N)）**

广义注意力：

\[
O_i=\sum_j\frac{\mathrm{Sim}(Q_i,K_j)}{\sum_{j'}\mathrm{Sim}(Q_i,K_{j'})}V_j
\]

取 \(\mathrm{Sim}(Q,K)=\mathrm{ReLU}(Q)\,\mathrm{ReLU}(K)^\top\)，利用乘法结合律先算 \(\sum_j \mathrm{ReLU}(K_j)^\top V_j\) 再与各 \(Q_i\) 相乘，复杂度相对 Softmax 的 \(O(N^2)\) 降为约 **\(O(N)\)**，且无 Softmax。

代价：注意力分布不够「尖」，局部刻画偏弱（论文 Figure 3）——故需卷积增强。

**（2）多尺度 token 聚合**

对 Q/K/V 用小核 **depthwise + 1×1 分组卷积**聚合邻域（实现上常融合成一次 DW + 分组 1×1），得到多尺度 token，再做线性注意力并 concat 投影。实验默认两支（含 **5×5** 聚合）以兼顾精度与速度。

**（3）局部支路**

MBConv（或 FFN 中插 DWConv）补局部纹理与边界，与上下文模块串成 EfficientViT Block。

### 2.4 EfficientViT-Seg 信息流

```text
Image
  → Stem → Stage0/1（局部卷积）
  → Stage2（P2, ~1/8）
  → Stage3（EfficientViT × L3 → P3, ~1/16）
  → Stage4（EfficientViT × L4 → P4, ~1/32）
  → 1×1 + upsample 对齐到 1/8 → add 融合
  → MBConv × L5 → 1×1 logits → upsample 至原图
```

### 2.5 B 系列规格（与实现一致）

| 变体 | width_list | depth_list | dim | 分割头 head_width / depth |
|------|------------|------------|-----|---------------------------|
| B0 | 8,16,32,64,128 | 1,2,2,2,2 | 16 | 32 / 1 |
| B1 | 16,32,64,128,256 | 1,2,3,3,4 | 16 | 64 / 3 |
| B2 | 24,48,96,192,384 | 1,3,4,4,6 | 32 | 96 / 3 |
| B3 | 32,64,128,256,512 | 1,4,6,6,9 | 32 | 128 / 3 |

头统一融合 `stage4/3/2`（stride 32/16/8），`head_stride=8`。云端还有 **L 系列**（更深更宽、局部块配方不同），官方仓库提供完整配置与权重。

---

## 3. Python 模块实现

实现目录：[`code/efficientvit/`](code/efficientvit/)：

| 文件 | 内容 |
|------|------|
| `ops.py` | 工具函数、`ConvLayer`、`MBConv`/`DSConv`、`ResidualBlock` 等 |
| `attention.py` | `LiteMLA`、`EfficientViTBlock` |
| `backbone.py` | `EfficientViTBackbone` 与 B0–B3 工厂 |
| `seg.py` | `SegHead`、`EfficientViTSeg`、`build_efficientvit_seg` |
| `__init__.py` | 包导出 |

实现要点：

- 结构对齐官方 **mit-han-lab/efficientvit** 的 B 系列分割配置；
- 插值统一 `align_corners=False`；分割输出动态上采样到输入尺寸；
- 变体注册 + `build_efficientvit_seg("b0"|"b1"|"b2"|"b3")`。

### 3.1 LiteMLA 核心（摘要）

```python
# code/efficientvit/attention.py（逻辑摘要）
qkv = Conv1x1(x)
ms = [qkv] + [DW_then_Group1x1(qkv) for scale in scales]
y = ReLU_LinearAttn(cat(ms))   # 或序列很短时用 quadratic 形式
y = Conv1x1_proj(y)
```

### 3.2 分割模型用法

```python
import sys
sys.path.insert(0, r".\code")

import torch
from efficientvit import build_efficientvit_seg, efficientvit_seg_b1

model = build_efficientvit_seg(
    "b1",                 # 或 b0 / b2 / b3
    num_classes=19,
    in_channels=3,
)

x = torch.randn(2, 3, 512, 512)
logits = model(x)         # (2, 19, 512, 512)

# 等价工厂
m0 = efficientvit_seg_b0(in_channels=3, num_classes=19)
```

### 3.3 快速自检

```bash
python -c "import sys; sys.path.insert(0,'code'); import torch; from efficientvit import build_efficientvit_seg; \
m=build_efficientvit_seg('b0', num_classes=10); y=m(torch.randn(1,3,256,256)); print(y.shape)"
```

期望输出形状：`[1, 10, 256, 256]`。

---

## 4. 小结与选用建议

1. **ViT 系列**从分类走向层次化与密集预测；高效路线的关键是在保留全局建模的同时压低高分辨率下的延迟。  
2. **EfficientViT** 用 **多尺度 ReLU 线性注意力 + 卷积局部增强**，在 Cityscapes / ADE20K 上达到与 SegFormer / SegNeXt 同档甚至更高的精度，同时在多类硬件上更快。  
3. **EfficientViT-Seg** 采用「Stage3/4 插 EfficientViT 模块 + P2/P3/P4 相加融合 + 轻量 MBConv 头」的简洁配方。  
4. 本目录 `code/efficientvit/` 提供 B0–B3 可运行分割实现，可直接接入训练脚本；更大 L 系列与预训练权重见官方仓库。

### 参考文献

1. Cai et al., 2022/2023. *EfficientViT.* [doi:10.48550/ARXIV.2205.14756](https://doi.org/10.48550/arXiv.2205.14756)  
2. Dosovitskiy et al., 2020. *ViT.*  
3. Xie et al., 2021. *SegFormer.*  
4. Guo et al., 2022. *SegNeXt.*  
5. Liu et al., 2021. *Swin Transformer.*  
6. Kirillov et al., 2023. *Segment Anything.*  

### 延伸阅读

- 官方实现与权重：[mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit)  
- 论文 Figure 2（Building Block / Multi-Scale Linear Attention）与 Figure 3–4（注意力分布与延迟对比）  
- 注意与其他同名 EfficientViT 工作区分开，以免对照错仓库与权重  
