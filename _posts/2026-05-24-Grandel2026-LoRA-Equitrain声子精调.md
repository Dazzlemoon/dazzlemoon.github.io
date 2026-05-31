---
layout:     post
title:      10 帧 DFT 就能修声子？LoRA 精调 MLIP 的新玩法
subtitle:   Grandel Equitrain（LoRA）与 Koker PFT 两条微调思路并排：少数据补丁 vs Hessian 直督；少数据就能把 MACE 声子拉回来的实践笔记。
date:       2026-05-24
author:     qqz
header-img: img/post-bg-desk.jpg
catalog:    true
tags:
    - MLIP
    - LoRA
    - Equitrain
    - 声子
    - 热物性
    - PFT
---

![一图总结](/img/posts/2026-05-24-grandel2026-lora-equitrain/cover.png)

> 基于原文：Grandel et al., *Parameter-Efficient Fine-Tuning of Machine-Learning Interatomic Potentials for Phonon and Thermal Property Prediction* (arXiv:2604.01017, 2026)

做材料计算的朋友大概都有这种体验：

**MACE、MACE-MP 这类通用势函数，跑结构优化、MD 已经挺准了；  
但一上声子——色散曲线飘、最高频偏低、虚频该有却没有——就开始“露馅”。**

热容这种“平均量”看着还行，  
可一旦你要算 **声子 DOS、热导、弹性常数，或者跟着虚频找相变路径**，  
基座模型的系统性软化就会放大成实打实的物理误差。

Grandel 等人 2026 年的工作（arXiv:2604.01017）给了一条很实用的路：  
**不用重训大模型，用 LoRA 式加性精调（Equitrain），每种材料加 ~10 帧 DFT，就能把声子拉回来。**

---

## 一、他们到底在比什么？

![精调策略总览：Transfer / Multihead / Equitrain（LoRA）](/img/posts/2026-05-24-grandel2026-lora-equitrain/fig01-workflow-finetuning-strategies.png)

在 **53 种**相变/硫族化物材料上，对 **MACE-MP-0b3** 逐材料精调，对比四条路线：

| 策略 | 一句话 |
|------|--------|
| **From scratch** | 小数据从头训——最差 |
| **Transfer** | 全参数精调——容易遗忘基座能力 |
| **Multihead** | replay 预训练数据防遗忘——贵，phonon 上略逊 |
| **Equitrain（LoRA）** | 冻结基座，只训 $\Delta W$ + 正则——本文最佳 |

训练数据怎么来？不是 phonon 位移超胞，而是 **rattled + 体积缩放 + MP-0b3 弛豫轨迹**，  
在平衡态附近按能量等间距采样——专门喂给“谐振 PES”。

---

## 二、LoRA 在这里是什么意思？（3 个公式够用）

LoRA 原本是大模型里的“小补丁”思路。Equitrain 把它搬到 MLIP 上：

**① 权重不直接改，只加增量**

$$
W = W_0 + \Delta W
$$

$W_0$ 是预训练 MACE，**冻住**；$\Delta W$ 是你要学的“补丁”。

**② 本文用 full-rank（满秩）LoRA**

不刻意压低秩——表达力和全量精调一样大，  
但优化时**只对 $\Delta W$ 做 weight decay**：

$$
\min_{\Delta W}\; \mathcal{L}(W_0 + \Delta W) + \lambda \|\Delta W\|^2
$$

直觉：**允许偏离基座，但别偏离太远。**  
这比 Transfer 更不容易 catastrophic forgetting，又比 Multihead 省 replay 成本。

**③ 任务损失还是老配方**

能量 + 力 + 应力 Huber，权重 $(10, 100, 1000)$。

---

## 另一条路：Koker 等人的「声子微调 PFT」

Grandel 的 **Equitrain（LoRA）** 核心是「**少动基座权重**」：冻结 $W_0$，只用 $\Delta W$ 在低数据上对齐 E/F/S，并用 $\lambda\lVert \Delta W\rVert^2$ 拉住预训练形状。同年 **[PFT（Phonon Fine-Tuning）](/2026/05/20/Koker2026-PFT声子微调/)**（Koker et al., 2026, arXiv:2601.07742）则换了一条轴——**直接在二阶力常数上监督**，让 MLIP 的 **解析能量 Hessian** 与有限位移算出的 $\Phi^{\mathrm{DFT}}$ 对齐。

简要对照如下（细节与指标不可混读）：

| 维度 | Equitrain（本文 Grandel） | PFT（Koker） |
|------|---------------------------|--------------|
| **核心改动** | 权重分解 $W=W_0+\Delta W$，只训增量 + 正则 | 在 E/F/S 外增加 **$\mathcal{L}_\Phi$**，对齐 $\Phi^{\mathrm{DFT}}$ |
| **“防遗忘”** | $\lambda\lVert\Delta W\rVert^2$ 锚住 $W_0$ | **MPtrj 共训练**（phonon 步与上游 EFS 交错，正文常用约一比四） |
| **训练数据形状** | rattled / 体积扫描 + MP-0b3 弛豫采样（刻意**不把** phonon 位移超胞当主训练集） | MDR Phonon **大规模**有限位移标签（数十万级位移 DFT） |
| **实现栈** | MACE / Equitrain | Nequix（JAX + NequIP 系）、**随机抽 Hessian 列 + HVP** 控算力 |
| **主报的声子精度** | 全 BZ 频率 **MAE（THz）**，如 **0.27 → 0.05 THz** | 300 K 下 $\omega_{\max},S,F,C_V$ **各物理量的 MAE**（不要把「THz MAE」和「$\omega_{\max}$ 的 K 偏差」横向硬比同一数字） |

PFT 总损失在四象限上多长一项「力常数项」：

$$
\mathcal{L}_{\mathrm{PFT}}=\lambda_E\mathcal{L}_E+\lambda_F\mathcal{L}_F+\lambda_\sigma\mathcal{L}_\sigma+\lambda_\Phi\mathcal{L}_\Phi
$$

作者在消融里反复强调：**如果只是拿 phonon 位移帧做普通 EFS 微调、却不加 $\mathcal{L}_\Phi$**，Hessian 误差可能**比基座还差**——这和 Equitrain「单靠好目标 + LoRA 正则也能大幅修声子」是两种不同假设条件下的结论，读起来要各归各的 benchmark。

**怎么选才不踩坑？**

- 手边只有 **几块 DFT / 每种材料十来帧**：优先读 Grandel → **LoRA（Equitrain）**路线，成本模型更亲切。  
- 已经能上 **大批量 phonon 位移库**、并希望 **系统化压 Hessian**：读 Koker → **PFT**；后面若要 Raman/IR 强度链路，可把 **Kim / SOG 式电响应** 与 $\mathcal{L}_\Phi$ **叠在同一总势能**上。

---

## 三、效果有多猛？几组数字记牢

### 3.1 数据效率：10 帧就够用

![训练数据量 vs 力 MAE：Equitrain 最低](/img/posts/2026-05-24-grandel2026-lora-equitrain/fig02-dataset-force-mae-data-efficiency.png)

- 训练构型从 2 增到 16，力 MAE 快速收敛；
- **~10 帧**已明显优于基座 MP-0b3；
- 排序：**Equitrain > Transfer > Multihead >> Scratch**。

### 3.2 声子：0.27 → 0.05 THz

![声子 MAE 对比（不同训练胞尺寸）](/img/posts/2026-05-24-grandel2026-lora-equitrain/table01-phonon-mae-structure-size.png)

大超胞（10–15 Å）训练时，全布里渊区声子 MAE：

- **MP-0b3 基座**：0.27 THz  
- **Equitrain**：**0.05 THz**（约 **5 倍**提升）

顺带修掉了基座**系统性低估最高声子频率**（$\omega_{\max}$ 偏差从 −7% 压到 −0.9%），  
声子 DOS 误差从 ~51% 降到 ~14%。

> 即使换成更大的 foundation model（MPA-0、OMAT-0），Equitrain 精调后的 0.05 THz 仍然更低——**换大模型不如给目标材料做精调。**

### 3.3 热导、弹性：多数材料 ±5% 以内

![热力学与弹性性质相对 DFT 偏差（300 K）](/img/posts/2026-05-24-grandel2026-lora-equitrain/fig03-thermal-elastic-deviation.png)

300 K 下看热容、熵、自由能、Slack 热导率、Bulk/Shear 模量：

- 基座对热导率**中位偏差约 −50%**（严重低估）；
- Equitrain 精调后，**绝大多数材料落在 ±5%**；
- scratch 小数据训出来的模型，弹性常数基本靠不住。

### 3.4 虚频与相变：Equitrain 最稳

![虚频稳定性混淆矩阵](/img/posts/2026-05-24-grandel2026-lora-equitrain/table03-imaginary-mode-confusion-matrix.png)

- Equitrain：**100% precision**（不误报不稳定），**89% recall**；
- 比“有没有虚频”更狠的测试是**沿虚频方向扫 PES、看 relax 到哪个相**。

**K₃Sb 案例**（Figure 4）：

![K₃Sb：Equitrain 抓到 K 点虚频并复现正确相变](/img/posts/2026-05-24-grandel2026-lora-equitrain/fig04-k3sb-phonon-pes-phase-transition.png)

- Transfer / 基座：**看不到** K 点虚频；
- Equitrain：双阱 PES 形状对，relax 到正确 **P6₃cm** 相（Multihead 会 relax 到错相）。

相变 F1：**Equitrain 0.66**，精调策略里最高。

---

## 四、算力账：DFT 时间也能省

![phonon DFT vs 精调数据生成 CPU 时间](/img/posts/2026-05-24-grandel2026-lora-equitrain/fig06-computational-cost.png)

- 传统 phonon：单材料平均 **27.3 h** DFT（大位移超胞）；
- 10 帧精调数据：**18.8 h** → 总体省 **~32%**；
- 低对称、需很多位移超胞的体系，省时可到 **54%–92%**；
- GPU 精调本身：**~7 min/材料**（A100）。

---

## 五、给做谱学/声子计算读者的 Takeaway

1. **基座 MLIP ≠ 声子可用**——软化、虚频、PES 形状是三道独立关卡。  
2. **精调数据要采“平衡态附近 PES”**，别只用 phonon 位移超胞当训练集。  
3. **大超胞训练**对声子至关重要（primitive 胞误差大很多）。  
4. **LoRA / Equitrain**：不是「只靠参数少」，而是 **$\Delta W$ + weight decay**，在精调的同时把参数更新**拴在预训练权重附近**。  
5. 与同年 **[PFT](https://arxiv.org/abs/2601.07742)**（力常数项 $\mathcal{L}_\Phi$）**并排读**：前者省事、十来帧可走通；后者要 phonon 位移库但更直抠 Hessian；**两篇论文主报的指标写法不同——THz 全谱 MAE ≠ 300 K $\omega_{\max}$ 的 K 偏差**。  
6. **Equitrain**（[BAMeScience/equitrain](https://github.com/BAMeScience/equitrain)），Zenodo 有配套数据。

---

## 六、一句话收束

> **每种材料约 10 帧 DFT + LoRA（Equitrain）能把 MACE 声子拉回可用尺度；若有大规模 phonon 位移库再上 PFT 直抠 Hessian。两条路子指标写法不同——别再用「一个数字盖住两篇论文」。**

---

## 延伸阅读（站内）

- [Koker2026：PFT 声子微调](/2026/05/20/Koker2026-PFT声子微调/)
- [红外拉曼声子谱：原理数学与电池应用](/2026/05/18/红外拉曼声子谱-原理数学与电池应用/)

## 参考

- Grandel et al. **Parameter-Efficient Fine-Tuning of Machine-Learning Interatomic Potentials for Phonon and Thermal Property Prediction**. arXiv:2604.01017, 2026.  
  https://arxiv.org/abs/2604.01017
- Koker T, Gangan A, Kotak M, et al. **PFT: Phonon Fine-tuning for Machine Learned Interatomic Potentials**. arXiv:2601.07742, 2026.  
  https://arxiv.org/abs/2601.07742
- Equitrain 代码库：https://github.com/BAMeScience/equitrain
