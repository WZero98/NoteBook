# Kappa 一致性指标计算公式详解

## 一、应用背景

在**两个（或多个）评分者对同一批样本做类别判定**的研究中，Kappa 是最常用的一致性统计量之一。它衡量的是：**在扣除"纯属碰巧一致"之后，评分者之间真实一致的比例有多少**。

与简单一致率（percent agreement）不同，Kappa 会对**偶然一致**（chance agreement）进行校正，因此在类别极不平衡时更有意义。

需要特别注意的是：Kappa 也不是一个单一公式，而是一族指标。选择哪一个，取决于三个关键问题：

1. **评分者数量**：恰好 2 人，还是 3 人及以上？
2. **类别性质**：名义类别（无序，如疾病有/无）还是有序类别（如肿瘤分级 I–IV）？
3. **不一致的代价**：相邻类别不一致与相隔很远的不一致，是否应同等惩罚？

> 与 ICC 的分工：连续型分数、尺寸、量表总分 → 用 **ICC**；名义/有序**分类标签** → 用 **Kappa**（或加权 Kappa）。

---

## 二、数据设置

设有：

- $n$ = 样本数（被评对象）
- $k$ = 评分者数（双评分者时 $k=2$）
- $C$ = 类别数
- $x_{ij}$ = 第 $i$ 个样本被第 $j$ 个评分者给出的**类别标签**

典型的双评分者数据表：

| 样本 | 评分者A | 评分者B |
|------|---------|---------|
| 1 | 阳性 | 阳性 |
| 2 | 阴性 | 阳性 |
| ... | ... | ... |
| $n$ | ... | ... |

汇总后得到 $C \times C$ **列联表**（confusion / agreement table）。以二分类为例：

|  | B：阴性 | B：阳性 | 行合计 |
|--|---------|---------|--------|
| A：阴性 | $n_{00}$ | $n_{01}$ | $n_{0\cdot}$ |
| A：阳性 | $n_{10}$ | $n_{11}$ | $n_{1\cdot}$ |
| 列合计 | $n_{\cdot 0}$ | $n_{\cdot 1}$ | $n$ |

多分类时，行、列分别为评分者 A、B 的各类别，对角线表示完全一致。

---

## 三、核心思想：观测一致与偶然一致

Kappa 的核心思想是把"表面上的一致"拆成两部分：

```math
\kappa = \frac{p_o - p_e}{1 - p_e}
```

其中：

- $p_o$ = **观测一致率**（observed agreement）：实际落在对角线的比例
- $p_e$ = **偶然一致率**（expected agreement by chance）：若两评分者独立、仅按各自边缘分布乱猜，预期也会一致的比例
- 分子 $p_o - p_e$：超出偶然的一致
- 分母 $1 - p_e$：在完全一致之上，还能再"挤出"多少超出偶然的空间

### 1. 观测一致率

```math
p_o = \sum_{c=1}^{C} p_{cc} = \frac{1}{n}\sum_{c=1}^{C} n_{cc}
```

即列联表对角线元素之和除以 $n$。

### 2. 偶然一致率（Cohen 定义）

设评分者 A 将样本判为类别 $c$ 的边缘比例为 $p_{c\cdot}$，评分者 B 为 $p_{\cdot c}$，则：

```math
p_e = \sum_{c=1}^{C} p_{c\cdot}\, p_{\cdot c}
```

即假设两评分者独立时，各类别"碰巧都选 $c$"的概率之和。

### 3. 取值含义

| $\kappa$ | 含义 |
|----------|------|
| $1$ | 完全一致 |
| $0$ | 一致程度仅相当于偶然 |
| $<0$ | 一致程度还不如偶然（系统性对抗） |

---

## 四、Kappa 的主要形式

### Cohen's Kappa：两评分者、名义类别

适用：**恰好 2 个评分者**，类别为名义（无序）或虽有序但不想区分不一致程度。

```math
\kappa = \frac{p_o - p_e}{1 - p_e},\quad
p_o = \sum_{c} p_{cc},\quad
p_e = \sum_{c} p_{c\cdot} p_{\cdot c}
```

这是最经典、报告中最常见的形式（Cohen, 1960）。

### Weighted Kappa：两评分者、有序类别

当类别有序（如无/轻/中/重，或 TNM 分期）时，"差一级"与"差三级"不应同等惩罚，应使用**加权 Kappa**（Cohen, 1968）。

引入权重矩阵 $w_{ij}$（一致时 $w_{ii}=1$，不一致时 $0 \leq w_{ij} < 1$），则：

```math
\kappa_w = \frac{\sum_{i,j} w_{ij} p_{ij} - \sum_{i,j} w_{ij} p_{i\cdot} p_{\cdot j}}{1 - \sum_{i,j} w_{ij} p_{i\cdot} p_{\cdot j}}
```

等价的"不一致权重"写法（令不一致代价为 $W_{ij}$，对角为 0）为：

```math
\kappa_w = 1 - \frac{\sum_{i,j} W_{ij} O_{ij}}{\sum_{i,j} W_{ij} E_{ij}}
```

其中 $O_{ij}$、$E_{ij}$ 分别为观测与期望频数。

**两种常用权重（设类别编号为 $1,\ldots,C$）：**

| 权重方案 | 不一致代价 $W_{ij}$ | 特点 |
|----------|---------------------|------|
| **线性权重**（linear） | $\|i-j\|/(C-1)$ | 不一致代价与类别距离成正比 |
| **二次权重**（quadratic） | $(i-j)^2/(C-1)^2$ | 更宽容相邻不一致，对远距离不一致惩罚更重 |

> 同一数据上，二次加权 Kappa 通常高于线性加权 Kappa。权重方案须**事先选定并在报告中写明**，不能只写"weighted kappa = 0.71"。

### Fleiss' Kappa：多评分者、名义类别

适用：**每个样本有固定数量的评分**（如每例都由 3 人评），但评分者身份不必始终是同一批人（Fleiss, 1971）。

对每个样本 $i$，记 $n$ 个评分中判为类别 $c$ 的人数为 $n_{ic}$（$\sum_c n_{ic}=n$）。该样本的一致比例为：

```math
P_i = \frac{1}{n(n-1)}\sum_{c=1}^{C} n_{ic}(n_{ic}-1)
```

整体观测一致与偶然一致：

```math
\bar{P} = \frac{1}{N}\sum_{i=1}^{N} P_i,\quad
\bar{P}_e = \sum_{c=1}^{C} p_c^2,\quad
p_c = \frac{1}{Nn}\sum_{i=1}^{N} n_{ic}
```

```math
\kappa_{\text{Fleiss}} = \frac{\bar{P} - \bar{P}_e}{1 - \bar{P}_e}
```

> 注意：Fleiss' Kappa **不是**"三人版 Cohen's Kappa"。即使 $k=2$，Fleiss 用的是合并边缘分布估计 $p_e$（接近 Scott's π），与 Cohen 用各自边缘分布的定义不同。若是 3 个固定评分者的完全交叉设计，常同时报告两两 Cohen's Kappa 与总体 Fleiss' Kappa，以免单个离群评分者被掩盖。

---

## 五、针对"两个评分者"的推荐选择

在两个评分者的场景下：

### 首选：Cohen's Kappa —— 名义类别、二分类或多分类

**理由：**

- 两评分者、每例双方都评，是 Cohen Kappa 的标准设定
- 诊断有/无、包含/排除、主题标签等名义类别，不一致一律同等对待即可

**公式回顾：**

```math
\kappa = \frac{p_o - p_e}{1 - p_e}
```

### 若类别有序：Weighted Kappa

- 关心"差一级是否可接受" → 常用**线性权重**
- 更强调远距离不一致的严重性（或希望与 ICC 式信度更可比）→ 常用**二次权重**
- 报告时务必写明 linear / quadratic

### 若评分者 ≥ 3：Fleiss' Kappa（名义）或改用其他指标

- 名义多评分者 → Fleiss' Kappa
- 有序多评分者且要体现距离 → 可考虑 Krippendorff's α（ordinal）、或把等级当作数值用 ICC

---

## 六、一个直觉化理解

Kappa 本质上是：

```math
\kappa = \frac{\text{真实超出偶然的一致}}{\text{理想情况下还能超出偶然的最大空间}}
```

- $p_o$ 越高（对角线越满），Kappa 越高
- $p_e$ 越高（某类占比极大、"瞎猜也容易猜对"），即使 $p_o$ 看起来不低，Kappa 也会被压低 —— 这正是对**类别不平衡**的校正
- 因此会出现：$p_o$ 很高但 $\kappa$ 中等，即所谓 **Kappa 悖论**（kappa paradox）

---

## 七、结果解读标准

最常引用的是 Landis & Koch (1977)：

| Kappa 值 | 一致性程度 |
|----------|------------|
| $< 0$ | 差（poor） |
| 0.00–0.20 | 轻微（slight） |
| 0.21–0.40 | 一般（fair） |
| 0.41–0.60 | 中等（moderate） |
| 0.61–0.80 | 显著（substantial） |
| 0.81–1.00 | 几乎完美（almost perfect） |

Fleiss 的简化切点也常见：

| Kappa 值 | 一致性程度 |
|----------|------------|
| $< 0.40$ | 差（poor） |
| 0.40–0.75 | 中等至良好（fair to good） |
| $> 0.75$ | 优秀（excellent） |

> 注：这些切点带有主观性，宜结合研究领域、类别数、患病率与置信区间解读，不宜机械套用。报告时建议同时给出**原始一致率 $p_o$**、**类别分布**与 **95% CI**。

---

## 八、计算示例（Cohen's Kappa，$C=2$）

假设 100 个样本、2 个评分者，列联表为：

|  | B：阴性 | B：阳性 | 合计 |
|--|---------|---------|------|
| A：阴性 | 40 | 10 | 50 |
| A：阳性 | 20 | 30 | 50 |
| 合计 | 60 | 40 | 100 |

计算：

- $p_o = (40+30)/100 = 0.70$
- $p_e = (50/100)(60/100) + (50/100)(40/100) = 0.30 + 0.20 = 0.50$

```math
\kappa = \frac{0.70 - 0.50}{1 - 0.50} = \frac{0.20}{0.50} = 0.40
```

按 Landis & Koch，属"一般（fair）"——尽管表面一致率已有 70%，校正偶然后并不算高。

> 说明：此处仅演示流程；正式分析请用软件并报告置信区间。

---

## 九、常见应用情形

Kappa 主要用于**名义或有序分类**的一致性评估，常见场景如下：

| 应用类型 | 含义 | 典型例子 | 常用指标 |
|----------|------|----------|----------|
| **评分者间一致性**（inter-rater） | 不同评分者对同一批对象分类是否一致 | 两名影像医生判断有无病灶；病理分级 | 2 人名义 → Cohen；2 人有序 → Weighted；≥3 人名义 → Fleiss |
| **评分者内一致性**（intra-rater） | 同一评分者在不同时间对同一批对象重复分类是否稳定 | 同一医生隔周重读影像并再诊断 | 通常用 Cohen / Weighted Kappa（两次判定视为两"评分者"） |
| **重测 / 方法学一致** | 两种方法或两个时间点的分类结果是否一致 | 新旧诊断标准对照；AI 与专家标签对照 | 同上；有序分级优先 Weighted Kappa |

### 各领域中的具体例子

- **医学影像 / 诊断**：有无病灶、BI-RADS、肺结节良恶性、TNM 分期等；观察者间一致性研究中，分类指标最常用 Kappa。
- **病理与临床分级**：组织学分级、心功能分级、疼痛分级（有序）→ 加权 Kappa。
- **行为观察与质性编码**：主题标签、事件类型（名义）→ Cohen / Fleiss Kappa。
- **机器学习标注与评估**：多标注员类别标签一致性；分类模型与金标准的一致性（注意：与 accuracy 不同，Kappa 校正了类别不平衡下的偶然一致）。有序回归/分级任务常用 **Quadratic Weighted Kappa (QWK)** 作为评价指标。

### 选用时的对应关系（简表）

| 研究问题 | 更合适的方向 |
|----------|--------------|
| 恰好 2 个评分者，名义类别？ | **Cohen's Kappa** |
| 恰好 2 个评分者，有序类别？ | **Weighted Kappa**（写明 linear / quadratic） |
| ≥3 个评分者，名义类别，每例评分人数固定？ | **Fleiss' Kappa** |
| 连续分数 / 定量测量？ | 改用 **ICC**，不用 Kappa |
| 只要原始一致率、不做偶然校正？ | 报告 $p_o$（percent agreement），但通常应与 Kappa 一并给出 |

> 极度不平衡的二分类（如患病率极低）时，$p_e$ 可能很高，Kappa 会被压得很低（Kappa 悖论）。此时务必同时报告 $p_o$、各类别边际分布，必要时补充 PABAK、特异一致率等。

---

## 十、关键注意事项

1. **先判断数据与评分者结构**：2 人还是多人？名义还是有序？这决定用 Cohen / Weighted / Fleiss。
2. **加权方案必须预先声明**：linear 与 quadratic 数值不可直接横向对比。
3. **报告时需注明**：所用 Kappa 类型、权重（如有）、$p_o$、$p_e$（或可复算信息）、类别分布、95% CI。
4. **Kappa 与准确率 / 一致率的区别**：准确率不校正偶然；在类别不平衡时，高准确率仍可能对应中等甚至偏低的 Kappa。
5. **Kappa 与 ICC 的区别**：ICC 面向连续（或近似连续）测量；Kappa 面向分类。有序等级若当作数值，二次加权 Kappa 与某些 ICC 在数值上可接近，但模型假设不同，不宜混称。
6. **样本量与稀有类别**：类别很多或某些类别极少时，估计不稳定；尽量保证各类别有足够样本，并报告 CI。

---

## 参考资料

1. Cohen J. A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*. 1960;20(1):37-46. doi:[10.1177/001316446002000104](https://doi.org/10.1177/001316446002000104)
2. Cohen J. Weighted kappa: nominal scale agreement with provision for scaled disagreement or partial credit. *Psychological Bulletin*. 1968;70(4):213-220. doi:[10.1037/h0026256](https://doi.org/10.1037/h0026256)
3. Fleiss JL. Measuring nominal scale agreement among many raters. *Psychological Bulletin*. 1971;76(5):378-382. doi:[10.1037/h0031619](https://doi.org/10.1037/h0031619)
4. Landis JR, Koch GG. The measurement of observer agreement for categorical data. *Biometrics*. 1977;33(1):159-174. doi:[10.2307/2529310](https://doi.org/10.2307/2529310)
5. Fleiss JL, Levin B, Paik MC. *Statistical Methods for Rates and Proportions*. 3rd ed. Wiley; 2003.（含 Kappa 解读切点与多评分者扩展的系统讨论）
6. Li M, Gao Q, Yu T. Kappa statistic considerations in evaluating inter-rater reliability between two raters: which, when and context matters. *BMC Cancer*. 2023;23:799. doi:[10.1186/s12885-023-11325-z](https://doi.org/10.1186/s12885-023-11325-z)（[PMC 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10464133/)）
7. Banerjee M, Capozzoli M, McSweeney L, Sinha D. Beyond kappa: A review of interrater agreement measures. *Canadian Journal of Statistics*. 1999;27(1):3-23.（Kappa 族及局限的综述）
