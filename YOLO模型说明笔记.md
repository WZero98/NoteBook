# YOLO 系列模型说明笔记

> 个人学习笔记：梳理 YOLO（You Only Look Once）从 v1 到 YOLO26 的演进、提出者/论文来源、特色架构与应用举例。  
> 主要参考：  
> - Jegham et al., *YOLO Evolution* — [doi:10.48550/ARXIV.2411.00201](https://doi.org/10.48550/arXiv.2411.00201)  
> - Sapkota et al., *YOLO26: Key Architectural Enhancements* — [doi:10.48550/ARXIV.2509.25164](https://doi.org/10.48550/arXiv.2509.25164)  
> - [Ultralytics 官方文档 / 模型库](https://docs.ultralytics.com/models/)

---

## 0. 总览与命名说明

YOLO 把检测重写为**单次前向**的回归问题：一次推理同时输出边界框与类别概率，以实时性著称。自 2015/2016 年起，系列在骨干、多尺度融合、训练策略与任务扩展上持续迭代；约从 YOLOv5 起，**Ultralytics** 成为主流工程实现与生态入口。

**版本谱系示意（演进总图）**：[Figure 1 — YOLO Evolution](https://arxiv.org/html/2411.00201v4/yolo_evolution.png)（来源：[arXiv:2411.00201](https://arxiv.org/html/2411.00201v4)）

| 阶段 | 代表版本 | 大致时间 | 主导力量 |
|------|----------|----------|----------|
| 奠基期 | YOLOv1–v3 | 2015–2018 | Redmon / Farhadi（Darknet） |
| 社区爆发 | YOLOv4–v7 | 2020–2022 | Bochkovskiy、美团、王建耀等 |
| Ultralytics 主线 | YOLOv5、v8、YOLO11、YOLO26 | 2020–2026 | Ultralytics（PyTorch） |
| 并行研究线 | YOLOv9/v10/v12/v13 等 | 2024–2025 | 中研院、清华、社区论文等 |

> **关于「YOLO14–YOLO25」**：公开文献与 Ultralytics 模型列表中**并无**连续编号的官方 YOLO14…YOLO25 产品线。从 **YOLO11** 起 Ultralytics 采用偏年份的命名；下一代正式产品为 **YOLO26**（对应 2026）。中间的 **YOLOv13** 是独立学术工作（超图感知），**不是** Ultralytics 官方主线版本。

**典型应用领域（跨版本共性，综述归纳）**：ADAS / 自动驾驶、视频监控与安防、人脸与口罩检测、农业（作物/杂草/害虫）、医疗影像辅助、工业质检、遥感与海事、野生动物保护等（见 [2411.00201 §1](https://arxiv.org/html/2411.00201v4)）。

---

## 1. 各版本简介：提出者、论文与应用举例

### 1.1 YOLOv1（2015/2016）

| 项目 | 内容 |
|------|------|
| **提出者** | Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi |
| **论文** | *You Only Look Once: Unified, Real-Time Object Detection* — CVPR 2016；[arXiv:1506.02640](https://arxiv.org/abs/1506.02640) |
| **框架** | Darknet |
| **要点** | 首个统一单阶段检测器：整图一次前向预测框 + 类别 |
| **应用举例** | 早期实时视频分析、ADAS 原型、通用目标检测基线；奠定「速度优先」的工业落地范式 |

### 1.2 YOLOv2 / YOLO9000（2016/2017）

| 项目 | 内容 |
|------|------|
| **提出者** | Joseph Redmon, Ali Farhadi |
| **论文** | *YOLO9000: Better, Faster, Stronger* — CVPR 2017；[arXiv:1612.08242](https://arxiv.org/abs/1612.08242) |
| **框架** | Darknet-19 |
| **要点** | Anchor、BN、多尺度训练；YOLO9000 联合检测与分类扩展至约 9000 类 |
| **应用举例** | 更稳健的通用检测；面向大规模类别识别与实时监控场景 |

### 1.3 YOLOv3（2018）

| 项目 | 内容 |
|------|------|
| **提出者** | Joseph Redmon, Ali Farhadi |
| **论文** | *YOLOv3: An Incremental Improvement* — [arXiv:1804.02767](https://arxiv.org/abs/1804.02767) |
| **框架** | Darknet-53 |
| **要点** | 残差骨干 + 多尺度预测（类 FPN），小目标明显改善 |
| **应用举例** | 多年工业默认实时检测器；交通标志、安防、机器人视觉等；Ultralytics 另提供 **YOLOv3u**（无锚）变体 |

**架构图**：[YOLOv3 architecture](https://arxiv.org/html/2411.00201v4/YOLOv3_architecture.png)（综述 Fig.5）

### 1.4 YOLOv4（2020）

| 项目 | 内容 |
|------|------|
| **提出者** | Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao |
| **论文** | *YOLOv4: Optimal Speed and Accuracy of Object Detection* — [arXiv:2004.10934](https://arxiv.org/abs/2004.10934) |
| **框架** | Darknet / CSPDarknet-53 |
| **要点** | CSP、Mish、Bag-of-Freebies/Specials（Mosaic、CIoU 等）；SPP + PAN |
| **应用举例** | 口罩/防疫协议检测、通用高精度实时检测；Ultralytics **不提供**原生权重（仅作架构参考） |

### 1.5 YOLOv5（2020，Ultralytics）

| 项目 | 内容 |
|------|------|
| **提出者 / 维护** | Glenn Jocher / Ultralytics（工程发布，无传统 CVPR 式「正式论文」） |
| **来源** | [Ultralytics YOLOv5](https://docs.ultralytics.com/models/yolov5/)；GitHub `ultralytics/yolov5` |
| **框架** | PyTorch：CSPDarknet + PANet + SPPF |
| **要点** | 易用 API、n/s/m/l/x 缩放；Mosaic 等增强；后续 **YOLOv5u** 无锚头 |
| **应用举例** | UAV 作物/杂草分类、库存盘点、无人机番茄计数、广泛工业落地与教学基线 |

**架构图**：[YOLOv5 architecture](https://arxiv.org/html/2411.00201v4/yolov5_architecture.png)（综述 Fig.6）

### 1.6 YOLOv6（2022，美团）

| 项目 | 内容 |
|------|------|
| **提出者** | Chuyi Li 等（美团视觉） |
| **论文** | *YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications* — [arXiv:2209.02976](https://arxiv.org/abs/2209.02976) |
| **框架** | PyTorch：EfficientRep / RepVGG 风格、CSPStackRep；可无锚 |
| **要点** | 面向工业部署的速度–精度权衡 |
| **应用举例** | 工业产线检测、仓储物流视觉；Ultralytics 可训 YAML，**无官方预训练 `.pt`** |

### 1.7 YOLOv7（2022）

| 项目 | 内容 |
|------|------|
| **提出者** | Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao |
| **论文** | *YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors* — CVPR 2023；[arXiv:2207.02696](https://arxiv.org/abs/2207.02696) |
| **框架** | PyTorch：E-ELAN、重参数化 |
| **要点** | 「可训练 bag-of-freebies」刷新实时检测 SOTA |
| **应用举例** | 移动端多目标检测、跟踪相关扩展；Ultralytics 侧多为 **ONNX/TensorRT 推理**，无原生 checkpoint 训练流 |

### 1.8 YOLOv8（2023，Ultralytics）

| 项目 | 内容 |
|------|------|
| **提出者 / 维护** | Ultralytics |
| **来源** | [docs.ultralytics.com/models/yolov8](https://docs.ultralytics.com/models/yolov8/) |
| **框架** | C2f 骨干、解耦头、**无锚**；统一多任务 |
| **要点** | Detect / Segment / Classify / Pose / OBB 同一套 API |
| **应用举例** | 果园实例分割、智能工厂缺陷检测、姿态与旋转目标；生态最成熟之一 |

**架构图**：[YOLOv8 architecture](https://arxiv.org/html/2411.00201v4/yolov8_architecture.png)（综述 Fig.7）

### 1.9 YOLOv9（2024）

| 项目 | 内容 |
|------|------|
| **提出者** | Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao |
| **论文** | *YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information* — ECCV 2024；[arXiv:2402.13616](https://arxiv.org/abs/2402.13616) |
| **要点** | **PGI**（可编程梯度信息）+ **GELAN** |
| **应用举例** | 小样本骨折检测、蓝莓成熟度（密集遮挡）等；综述认为小数据集上表现突出、大目标场景稳健 |

**架构图**：[YOLOv9 / GELAN](https://arxiv.org/html/2411.00201v4/YOLOv9_architecture.png)（综述 Fig.8）

### 1.10 YOLOv10（2024，清华）

| 项目 | 内容 |
|------|------|
| **提出者** | Ao Wang, Hui Chen 等（清华大学等） |
| **论文** | *YOLOv10: Real-Time End-to-End Object Detection* — NeurIPS 2024；[arXiv:2405.14458](https://arxiv.org/abs/2405.14458) |
| **要点** | 双标签分配 + **NMS-free** 端到端；轻量分类头、空间–通道解耦下采样 |
| **应用举例** | 极致延迟敏感边缘检测；综述强调速度/效率突出，重叠目标场景需谨慎选型 |

**架构图**：[YOLOv10 dual assignment](https://arxiv.org/html/2411.00201v4/YOLOv10_architecture.png)（综述 Fig.9）

### 1.11 YOLO11（2024，Ultralytics）

| 项目 | 内容 |
|------|------|
| **提出者 / 维护** | Ultralytics（2024-09-10 发布；无独立正式研究论文，以文档/仓库为准） |
| **来源** | [docs.ultralytics.com/models/yolo11](https://docs.ultralytics.com/models/yolo11/) |
| **要点** | **C3k2** 替换 C2f；**C2PSA** 增强空间注意力；延续多任务（Detect/Seg/Cls/Pose/OBB） |
| **应用举例** | 智能交通车辆检测、SAR 飞机检测、UAV 多类车辆；综述认为 **精度–效率最均衡**，生产环境常与 YOLO26 并列推荐 |

**架构图**：[YOLO11 C3k2 + C2PSA](https://arxiv.org/html/2411.00201v4/YOLO11_Architecture.png)（综述 Fig.10）

### 1.12 YOLOv12（2025）

| 项目 | 内容 |
|------|------|
| **提出者** | Yunjie Tian, Qixiang Ye, David Doermann |
| **论文** | *YOLOv12: Attention-Centric Real-Time Object Detectors* — [arXiv:2502.12524](https://arxiv.org/abs/2502.12524) |
| **要点** | **Area Attention (A2)** + **R-ELAN**；FlashAttention；偏注意力中心设计 |
| **应用举例** | 检测 / 分割 / OBB 研究与基准；[2411.00201](https://doi.org/10.48550/arXiv.2411.00201) 基准中认为复杂度上升未必换来全面优势——选型宜实测 |
| **Ultralytics** | 支持；预训练权重以检测为主 |

**架构图**：[YOLOv12 A2 + R-ELAN](https://arxiv.org/html/2411.00201v4/YOLOv12_Architecture.drawio.png)（综述 Fig.11）

### 1.13 YOLOv13（2025，研究线）

| 项目 | 内容 |
|------|------|
| **提出者** | Mengqi Lei, Siqi Li, Yihong Wu, Han Hu, You Zhou, Xinhu Zheng, Guiguang Ding, Shaoyi Du, Zongze Wu, Yue Gao 等 |
| **论文** | *YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception* — [arXiv:2506.17733](https://arxiv.org/abs/2506.17733) |
| **要点** | **HyperACE**（超图高阶相关）+ **FullPAD**（全链路特征聚合–分发）+ DS-C3k2 轻量化 |
| **应用举例** | 复杂场景高阶关系建模的检测研究；代码：[iMoonLab/yolov13](https://github.com/iMoonLab/yolov13) |
| **说明** | **非** Ultralytics Featured 主线模型；[2509.25164](https://doi.org/10.48550/arXiv.2509.25164) 对比表中将其列为 YOLO26 之前的社区/研究版本 |

**架构图**：[YOLOv13 framework](https://arxiv.org/html/2506.17733v2/framework.png)

### 1.14 YOLO14–YOLO25（空缺说明）

Ultralytics 与主流综述**未发布**编号连续的 YOLO14–YOLO25 正式模型。命名从 YOLO11 跳至 **YOLO26**，与「2026」产品年一致。学习时可将 14–25 视为**未使用的版本号空洞**，避免与第三方非官方「YOLO14」等混淆。

### 1.15 YOLO26（2025/2026，Ultralytics 最新）

| 项目 | 内容 |
|------|------|
| **提出者 / 维护** | Ultralytics；官方技术描述见 Jocher et al., *Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models* — [arXiv:2606.03748](https://arxiv.org/abs/2606.03748)；架构解读综述：[arXiv:2509.25164](https://doi.org/10.48550/arXiv.2509.25164) |
| **来源** | [docs.ultralytics.com/models/yolo26](https://docs.ultralytics.com/models/yolo26/) |
| **要点** | 去除 **DFL**；原生可选 **NMS-free** 端到端；**ProgLoss + STAL**；**MuSGD**；七任务统一（Detect / Instance Seg / Semantic Seg / Depth / Classify / Pose / OBB）+ YOLOE-26 开放词汇 |
| **性能摘录（官方 COCO Detect）** | YOLO26n–x：mAP 40.9–57.5；T4 TensorRT 约 1.7–11.8 ms；相对 YOLO11n，CPU ONNX 最高约 **+43%** 加速（论文/文档口径） |
| **应用举例** | 机器人抓取与避障、产线缺陷检测、无人机/智能相机边缘部署、移动端 CoreML/TFLite、遥感 OBB、姿态与深度等；强调 **边缘优先、易导出、易量化** |

**架构相关图**：

- 四大增强示意：[architectures.png](https://arxiv.org/html/2509.25164v1/architectures.png)（[2509.25164](https://arxiv.org/html/2509.25164v1)）  
- 精度–延迟基准：[benchmarkingpic.png](https://arxiv.org/html/2509.25164v1/benchmarkingpic.png)  
- 官方说明：[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)

---

## 2. 各版本特色架构（要点 + 图链）

统一理解骨架：**Backbone（特征）→ Neck（多尺度融合）→ Head（框/类/掩码等）**。下表汇总「这一版最该记住的结构变化」。

| 版本 | 特色架构关键词 | 推荐结构图链接 |
|------|----------------|----------------|
| **v1** | 网格划分；单次回归框+类；无复杂 FPN | 原论文 Fig.：[arXiv:1506.02640](https://arxiv.org/abs/1506.02640)（PDF） |
| **v2** | Darknet-19；Anchor 聚类；多尺度训练 | [arXiv:1612.08242](https://arxiv.org/abs/1612.08242) |
| **v3** | Darknet-53 残差；三尺度预测 | [YOLOv3_architecture.png](https://arxiv.org/html/2411.00201v4/YOLOv3_architecture.png) |
| **v4** | CSPDarknet；SPP；PAN；Mish / BoF | [arXiv:2004.10934](https://arxiv.org/abs/2004.10934) |
| **v5** | CSPDarknet + **SPPF** + PANet；缩放家族 | [yolov5_architecture.png](https://arxiv.org/html/2411.00201v4/yolov5_architecture.png) |
| **v6** | EfficientRep / Rep 结构；工业部署导向 | [arXiv:2209.02976](https://arxiv.org/abs/2209.02976) |
| **v7** | **E-ELAN**；重参数化 | [arXiv:2207.02696](https://arxiv.org/abs/2207.02696) |
| **v8** | **C2f**；解耦无锚头；多任务头 | [yolov8_architecture.png](https://arxiv.org/html/2411.00201v4/yolov8_architecture.png) |
| **v9** | **PGI** + **GELAN**（信息瓶颈与可逆思想） | [YOLOv9_architecture.png](https://arxiv.org/html/2411.00201v4/YOLOv9_architecture.png) |
| **v10** | 双分配；**One-to-One Head** → NMS-free | [YOLOv10_architecture.png](https://arxiv.org/html/2411.00201v4/YOLOv10_architecture.png) |
| **YOLO11** | **C3k2** + **C2PSA** | [YOLO11_Architecture.png](https://arxiv.org/html/2411.00201v4/YOLO11_Architecture.png) |
| **v12** | **A2 Area Attention** + **R-ELAN** | [YOLOv12_Architecture.drawio.png](https://arxiv.org/html/2411.00201v4/YOLOv12_Architecture.drawio.png) |
| **v13** | **HyperACE** + **FullPAD** + DS-C3k2 | [framework.png](https://arxiv.org/html/2506.17733v2/framework.png) |
| **YOLO26** | 去 **DFL**；双头（1-to-many / 1-to-one）；**ProgLoss+STAL**；**MuSGD** | [architectures.png](https://arxiv.org/html/2509.25164v1/architectures.png) · [官方 Docs](https://docs.ultralytics.com/models/yolo26/) |

### 2.1 架构演进「一条线」记忆

```text
v1 单次回归
 → v2 Anchor / 多尺度训练
 → v3 多尺度头（类 FPN）
 → v4–v5 CSP + SPP(F) + PAN 工程化
 → v6–v7 Rep / E-ELAN 实时 SOTA
 → v8 无锚 + 多任务统一 API
 → v9 信息/梯度可编程（PGI、GELAN）
 → v10 训练双分配 → 推理可无 NMS
 → YOLO11 更高效 CSP 块 + 空间注意力
 → v12 区域注意力中心
 → v13 超图高阶相关（研究线）
 → YOLO26 部署向简化：去 DFL、可选端到端无 NMS、训练配方升级
```

### 2.2 YOLO26 架构要点（对照 Ultralytics）

1. **DFL-free 回归**：框回归更轻、导出（ONNX / TensorRT / CoreML / TFLite / OpenVINO）更友好。  
2. **双检测头**：默认 one-to-many（NMS，略准）；`nms=False` 走 one-to-one **端到端**。  
3. **ProgLoss**：训练中逐步把监督重心挪向推理所用头。  
4. **STAL**：小目标正样本分配更友好。  
5. **MuSGD**：SGD 与 Muon 思想混合，稳定收敛。  
6. **任务头扩展**：实例分割增强、姿态 RLE、OBB 角度损失等；另有 semantic / depth。

最小用法示例：

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model("image.jpg")                 # 默认含 NMS 路径
results_e2e = model.predict("image.jpg", nms=False)  # NMS-free
```

---

## 3. Ultralytics 支持一览（选型速查）

| 模型 | Ultralytics 支持概况 | 何时优先考虑 |
|------|----------------------|--------------|
| YOLO26 | 七任务最全；新项目默认 | 边缘部署、NMS-free、多任务 |
| YOLO11 | 五任务齐全、生态成熟 | 稳生产、要预训练全任务 |
| YOLO12 | 可训可推；检测预训练为主 | 注意力架构对比实验 |
| YOLOv10 | 检测；NMS-free 研究 | 延迟优先 |
| YOLOv9 | Detect / Seg | PGI/GELAN、小数据 |
| YOLOv8 | 五任务齐全 | 既有流水线兼容 |
| YOLOv5 / v3 | 检测（u 系列） | 遗留项目 |
| YOLOv6 | 仅 YAML，无官方 `.pt` | 复现美团结构 |
| YOLOv4 / v7 | 基本不原生训推 | 读论文 / 外部导出推理 |

官方入口：[Models Supported by Ultralytics](https://docs.ultralytics.com/models/)

---

## 4. 应用选型备忘（综合两篇综述与官方建议）

| 场景 | 更常推荐的方向 |
|------|----------------|
| 新项目 / 边缘 / 导出量化 | **YOLO26** |
| 生产稳定、多任务预训练齐全 | **YOLO11**（或 YOLO26） |
| 极致速度、可接受精度取舍 | **YOLOv10**、YOLO26n |
| 小数据集 / 重叠目标 | 综述偏 **YOLOv9**；另实测 YOLO11/26 |
| 旋转目标（遥感等） | YOLO11 / YOLO12 / **YOLO26-obb** |
| 开放词汇 | **YOLOE-26** / YOLO-World |
| 工业历史产线 | YOLOv5 / v8 存量多，迁移可评估 11/26 |

---

## 5. 主要参考文献

1. Jegham, Koh, Abdelatti, Hendawi. *YOLO Evolution: A Comprehensive Benchmark and Architectural Review of YOLOv12, YOLO11, and Their Previous Versions.* [arXiv:2411.00201](https://doi.org/10.48550/arXiv.2411.00201)  
2. Sapkota, Cheppally, Sharda, Karkee. *YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection.* [arXiv:2509.25164](https://doi.org/10.48550/arXiv.2509.25164)  
3. Jocher et al. *Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models.* [arXiv:2606.03748](https://arxiv.org/abs/2606.03748)  
4. Ultralytics Docs: [Models](https://docs.ultralytics.com/models/) · [YOLO26](https://docs.ultralytics.com/models/yolo26/) · [GitHub ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)  
5. 各代原始论文链接见上文各小节（v1–v4、v6–v7、v9–v10、v12–v13）。

---

*笔记整理日期：2026-09-09*
