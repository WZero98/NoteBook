# Mixture-of-Experts (MoE) 模块笔记

> 个人学习笔记：稀疏专家混合（Sparse Mixture-of-Experts）的定义、作用、均衡损失与最小可运行实现。

---

## 1. 定义与出处

### 1.1 什么是 MoE

**Mixture-of-Experts（专家混合）** 是一种条件计算（conditional computation）架构：网络中维护多个结构相同（或相近）的子网络——称为 **专家（experts）**，再由一个 **门控/路由网络（gating / router）** 根据输入决定「激活哪些专家、以多大权重组合它们的输出」。

稀疏 MoE 的核心思想是：

- **总参数量大**（专家数量 × 单专家容量），模型表达能力强；
- **每次前向只激活少数专家（top-\(k\)）**，计算量接近稠密小模型，而非与总参数成正比。

对 token / 样本 \(x\)，令专家为 \(\{E_1,\ldots,E_N\}\)，门控给出路由权重 \(g_i(x)\)，稀疏形式可写为：

\[
y = \sum_{i \in \mathrm{TopK}(g(x))} g_i(x)\, E_i(x)
\]

常见设定是 \(k=1\)（Switch）或 \(k=2\)（许多现代 LLM MoE）。

### 1.2 渊源：Switch Transformers

现代大规模稀疏 MoE 的里程碑文献是：

> **Fedus, Zoph, Shazeer.** *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.*  
> DOI: [10.48550/ARXIV.2101.03961](https://doi.org/10.48550/arXiv.2101.03961) · [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)

该工作在更早的稀疏门控 MoE（Shazeer et al., 2017）与 GShard 等基础上，做了工程上极有影响的简化：

- 用 **top-1 路由（Switch routing）** 替代复杂的 top-\(k\) / 多损失设计，降低通信与实现复杂度；
- 用可微的 **辅助负载均衡损失（auxiliary load balancing / switch loss）** 抑制「路由坍塌」（少数专家吃掉绝大多数 token）；
- 引入 **专家容量（expert capacity）** 等分布式训练细节，使万亿级稀疏模型可训。

因此业界常把「Transformer 中用 MoE 替换 FFN + 稀疏路由 + 均衡辅助损失」这一范式，追溯到 Switch Transformers。

更早脉络可简记为：

| 阶段 | 代表 | 要点 |
|------|------|------|
| 经典 MoE | Jacobs et al. 等 | 多个专家 + 门控，多为稠密混合 |
| 稀疏门控 | Shazeer et al., 2017 | top-\(k\) 稀疏激活，大规模语言模型 |
| GShard | Lepikhin et al. | 把 MoE 做到超大 Transformer |
| Switch | Fedus et al., 2021 | top-1 + 简化均衡损失，推向万亿参数 |

### 1.3 DeepSeek 对 MoE 的使用与推动

2024 年以来，**DeepSeek** 系列把 MoE 从「实验室稀疏技术」推到了学界与业界广泛讨论的中心：

1. **DeepSeekMoE**（[arXiv:2401.06066](https://arxiv.org/abs/2401.06066)）  
   - **细粒度专家切分（fine-grained expert segmentation）**：把传统大专家拆成更多小专家，激活更多个小专家，组合更灵活，促进专家特化。  
   - **共享专家隔离（shared expert isolation）**：部分专家对所有 token **始终激活**，吸收共性知识，减轻路由专家之间的冗余。

2. **DeepSeek-V2 / V3**  
   - 继续采用 DeepSeekMoE 作为 FFN 替代结构，并配合 MLA 等高效注意力。  
   - V3 进一步采用 **无辅助损失的负载均衡（auxiliary-loss-free load balancing）**：用可学习的专家偏置动态调节负载，减轻传统 switch 类辅助损失对主任务的干扰（见 [arXiv:2408.15664](https://arxiv.org/abs/2408.15664)、[DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)）。

DeepSeek 开源模型在「总参数很大、激活参数相对较小、推理成本可控」上的表现，使 MoE 再次成为开源 LLM、训练框架与硬件厂商的热点关键词。

---

## 2. 作用与应用场景

### 2.1 主要作用

| 作用 | 说明 |
|------|------|
| **扩容不线性涨算力** | 增加专家数 → 总参数↑；保持 top-\(k\) → 单次 FLOPs 大致不变 |
| **条件计算 / 特化** | 不同输入走不同专家，利于领域、语言、模式上的分工 |
| **训练–推理成本权衡** | 训练可利用海量参数；推理只跑激活专家，延迟更友好 |
| **替换稠密 FFN** | 在 Transformer 中最常见：把部分/全部 FFN 换成 MoE 层 |

代价与挑战同样明确：路由坍塌、专家负载不均、跨设备 all-to-all 通信、数值不稳定、实现复杂等——这也是第 3 节辅助损失存在的原因。

### 2.2 典型应用场景

1. **大语言模型（LLM）**  
   Switch、GLaM、Mixtral、DeepSeekMoE / V2 / V3 等：用稀疏 FFN 扩大容量。

2. **多语言 / 多任务**  
   不同语言或任务倾向不同专家，减少负迁移（参见各类 multilingual MoE）。

3. **视觉与稠密预测**  
   将卷积层换成稀疏 MoE（如 patch 级路由），用于语义分割等；参见  
   [arXiv:2604.13761](https://doi.org/10.48550/arXiv.2604.13761)（CNN 语义分割中的 sparse MoE 设计分析）。

4. **推荐、多模态、生成–理解统一架构**  
   需要大容量但算力预算有限的场景，常用 MoE 做条件子模块。

---

## 3. 促进专家路由均衡的损失函数

稀疏路由若无约束，门控容易把几乎所有 token 分给少数专家（**routing collapse**），其余专家得不到梯度、形同虚设。因此训练时常在主任务损失外加 **辅助损失**：

\[
\mathcal{L} = \mathcal{L}_{\mathrm{task}} + \alpha\, \mathcal{L}_{\mathrm{balance}} + \beta\, \mathcal{L}_{z} + \cdots
\]

下面介绍最常用的两类；并补充同主题文献中常见的变体。

### 3.1 Switch Loss（负载均衡损失 / Auxiliary Load Balancing Loss）

出处：**Switch Transformers** · [10.48550/ARXIV.2101.03961](https://doi.org/10.48550/arXiv.2101.03961)

对一批 \(T\) 个 token、\(N\) 个专家，定义：

- \(f_i\)：实际被路由到专家 \(i\) 的 token **占比**（由 hard top-\(k\) 决策统计，对 \(f\) 通常不可微）；
- \(P_i\)：门控对该专家的 **平均路由概率**（对 soft 概率可微）。

\[
f_i = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}\{\mathrm{argmax}\,g(x_t)=i\}
\qquad
P_i = \frac{1}{T}\sum_{t=1}^{T} g_i(x_t)
\]

Switch 辅助损失：

\[
\mathcal{L}_{\mathrm{switch}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i\, P_i
\]

直观理解：

- 希望 \(f_i \approx P_i \approx 1/N\)（均匀）；
- \(\sum_i f_i P_i\) 在均匀时等于 \(1/N\)，乘以 \(N\) 后量级与专家数无关；
- \(\alpha\) 常用 \(10^{-2}\) 量级：足够推动均衡，又不压过主损失。

[arXiv:2604.13761](https://doi.org/10.48550/arXiv.2604.13761) 在 CNN 稀疏 MoE 实验中也将 switch loss 描述为：通过让平均路由概率贴近均匀先验，促进专家激活均衡；并与 importance / entropy 等损失对比，发现 **switch 往往带来最均匀的专家利用率**，而 entropy 有时 mIoU 更高但坍塌更明显——说明「均衡」与「任务指标」不必完全同向。

### 3.2 Router Z-Loss

出处：ST-MoE（Zoph et al., [arXiv:2202.08906](https://arxiv.org/abs/2202.08906)）；与 Switch 一并成为现代 MoE 训练标配。  
（均衡主题综述与对比场景亦常见于后续 MoE 设计文，如 [2604.13761](https://doi.org/10.48550/arXiv.2604.13761) 对多种 balancing loss 的讨论。）

门控 logits 若无界增大，softmax 在低精度（尤其 bfloat16）下易数值不稳，训练出现尖峰甚至发散。**Z-loss** 惩罚 logits 的 log-sum-exp 幅度：

\[
\mathcal{L}_{z} = \frac{1}{T}\sum_{t=1}^{T}
\left(
\log \sum_{i=1}^{N} e^{z_{t,i}}
\right)^{2}
= \frac{1}{T}\sum_{t=1}^{T}
\big(\mathrm{logsumexp}(z_t)\big)^{2}
\]

其中 \(z_t\) 为 token \(t\) 上各专家的 **原始 logits**（softmax 之前）。系数 \(\beta\)（或 \(c_z\)）通常很小（如 \(10^{-3}\)～\(10^{-4}\) 量级，依实现而定）。

要点：

- 目标是 **稳定路由数值**，不是直接做负载均衡；
- 常与 switch loss **同时使用**：一个管「分得匀」，一个管「logit 别炸」。

### 3.3 其它常见均衡相关损失（简表）

| 名称 | 思路 | 常见出处 |
|------|------|----------|
| **Importance loss** | 惩罚专家重要性（平均门控概率）在 batch 上的方差 | Shazeer et al., 2017；[2604.13761](https://doi.org/10.48550/arXiv.2604.13761) |
| **Entropy loss** | 最大化路由分布熵，鼓励多样性 | 多篇视觉/路由工作；同上 |
| **Auxiliary-loss-free bias** | 用专家偏置在线调负载，弱化辅助损失 | DeepSeek 系列 |

笔记实现重点放在 **switch loss** 与 **z-loss**。

---

## 4. Python 代码示例

以下用 **线性层** 作为最小专家，实现：

1. Top-\(k\) MoE 模块；
2. Switch（load balancing）loss；
3. Router z-loss。

依赖：`torch`。无需完整训练循环。

```python
"""
最小 MoE：Linear 专家 + Top-k 路由 + Switch loss + Z-loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. 专家与 MoE 模块
# ---------------------------------------------------------------------------

class LinearExpert(nn.Module):
    """最小专家：单层 Linear（可按需换成 MLP）。"""

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        h = hidden or dim
        self.fc = nn.Linear(dim, h)
        self.out = nn.Linear(h, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.gelu(self.fc(x)))


class MoE(nn.Module):
    """
    稀疏 Mixture-of-Experts。

    输入:  (B, T, D) 或 (N, D)
    输出:  与输入同形状的 y，以及 router_logits（供辅助损失使用）
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_hidden: int | None = None,
    ):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [LinearExpert(dim, expert_hidden) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        # 展平为 token 维: (S, D)
        x_flat = x.reshape(-1, self.dim)
        s, d = x_flat.shape

        # 路由 logits / 概率: (S, E)
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        # top-k
        topk_probs, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)  # (S, K)
        # 在选中的专家上重新归一化权重（常见做法）
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # 聚合输出
        y = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_ids = topk_idx[:, k]          # (S,)
            weights = topk_probs[:, k]           # (S,)
            for e in range(self.num_experts):
                mask = expert_ids == e
                if not mask.any():
                    continue
                x_e = x_flat[mask]
                y_e = self.experts[e](x_e)
                y[mask] = y[mask] + weights[mask].unsqueeze(-1) * y_e

        y = y.reshape(orig_shape)
        return y, router_logits


# ---------------------------------------------------------------------------
# 2. 辅助损失
# ---------------------------------------------------------------------------

def switch_load_balancing_loss(
    router_logits: torch.Tensor,
    top_k: int = 1,
    alpha: float = 1e-2,
) -> torch.Tensor:
    """
    Switch Transformer 风格的负载均衡损失
    L = alpha * N * sum_i f_i * P_i

    Args:
        router_logits: (S, E) 或 (B, T, E)
        top_k: 与 MoE 路由一致；统计 f_i 时对 top-k 命中计数
        alpha: 辅助损失系数
    """
    logits = router_logits.reshape(-1, router_logits.size(-1))
    s, n = logits.shape
    probs = F.softmax(logits, dim=-1)  # P 的逐 token 形式

    # P_i: 平均路由概率
    P = probs.mean(dim=0)  # (E,)

    # f_i: top-k 硬分配占比（每个 token 可贡献到 k 个专家）
    topk_idx = torch.topk(probs, top_k, dim=-1).indices  # (S, K)
    # one-hot 累加后归一化到「每 token 贡献 1」
    mask = F.one_hot(topk_idx, num_classes=n).float().sum(dim=1)  # (S, E)
    mask = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    f = mask.mean(dim=0)  # (E,)

    loss = alpha * n * torch.sum(f * P)
    return loss


def router_z_loss(
    router_logits: torch.Tensor,
    beta: float = 1e-3,
) -> torch.Tensor:
    """
    ST-MoE router z-loss:
    L_z = mean( logsumexp(z)^2 )
    """
    logits = router_logits.reshape(-1, router_logits.size(-1))
    z = torch.logsumexp(logits, dim=-1)  # (S,)
    return beta * (z ** 2).mean()


def moe_auxiliary_losses(
    router_logits: torch.Tensor,
    top_k: int = 2,
    alpha: float = 1e-2,
    beta: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """打包常用辅助项，便于总损失: task + switch + z。"""
    sw = switch_load_balancing_loss(router_logits, top_k=top_k, alpha=alpha)
    zl = router_z_loss(router_logits, beta=beta)
    return {"switch_loss": sw, "z_loss": zl, "aux_total": sw + zl}


# ---------------------------------------------------------------------------
# 3. 用法示意（非训练循环）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D, E, K = 2, 8, 16, 4, 2

    moe = MoE(dim=D, num_experts=E, top_k=K)
    x = torch.randn(B, T, D)

    y, router_logits = moe(x)
    aux = moe_auxiliary_losses(router_logits, top_k=K)

    print("y:", tuple(y.shape))
    print("router_logits:", tuple(router_logits.shape))
    print("switch_loss:", float(aux["switch_loss"]))
    print("z_loss:", float(aux["z_loss"]))
    # 训练时示例:
    # total = task_loss + aux["aux_total"]
```

### 实现备注

- 上面的专家循环实现清晰但较慢；生产中常用 `einsum` / 分组 matmul / 专用 kernel（Megatron、DeepSpeed-MoE、FlashMoE 等）。
- Switch 原文以 **top-1** 定义 \(f_i\)；top-\(k\) 时常用「对选中专家做 soft 计数再归一」的推广，如上代码。
- DeepSeek-V3 等已可走 **无辅助损失偏置均衡**；笔记仍保留经典 switch / z-loss，便于对照文献与复现早期配方。

---

## 5. 参考文献（笔记引用）

1. Fedus, Zoph, Shazeer. *Switch Transformers.* [10.48550/ARXIV.2101.03961](https://doi.org/10.48550/arXiv.2101.03961)  
2. Pavlitska et al. *Design and Behavior of Sparse Mixture-of-Experts Layers in CNN-based Semantic Segmentation.* [10.48550/ARXIV.2604.13761](https://doi.org/10.48550/arXiv.2604.13761)  
3. Zoph et al. *ST-MoE: Designing Stable and Transferable Sparse Expert Models.* [arXiv:2202.08906](https://arxiv.org/abs/2202.08906)（z-loss 原始提出）  
4. Dai et al. *DeepSeekMoE.* [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)  
5. DeepSeek-AI. *DeepSeek-V3 Technical Report.* [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)  
6. Wang et al. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts.* [arXiv:2408.15664](https://arxiv.org/abs/2408.15664)

---

*文档用途：本地学习笔记，非论文复述；公式与实现以理解与可运行为优先。*
