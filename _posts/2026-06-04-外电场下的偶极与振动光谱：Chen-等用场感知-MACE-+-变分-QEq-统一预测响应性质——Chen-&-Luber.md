---
layout:     post
title:      外电场下的偶极与振动光谱：Chen 等用场感知 MACE + 变分 QEq 统一预测响应性质——Chen & Luber 2026 解读
subtitle:   苏黎世大学 Chen & Luber（ChemRxiv 2026 预印本）：外电场等变嵌入 + 能量导数求 μ/α + 变分 QEq 长程；NMA 验证偶极 RMSE 0.007 Debye、α 0.063 a.u.；液态水 IR/Raman 与强场外推；MACE_LR 优于 4G-HDNNP/VQEq 基准。
date:       2026-06-04
author:     qqz
header-img: img/post-bg-desk.jpg
catalog:    true
tags:
    - MLIP
    - MACE
    - QEq
    - 外电场
    - 振动光谱
    - ChemRxiv
---

![一图总结](/img/posts/2026-06-04-chen-vqeq-field-aware-mace/cover.png)

# 外电场下的偶极与振动光谱：Chen 等用场感知 MACE + 变分 QEq 统一预测响应性质

> **论文**：Ke Chen & Sandra Luber, *Field-Aware and Charge-Informed Machine Learning for Predicting Molecular Responses and Vibrational Spectra*, **ChemRxiv**（2026-04-26 预印本，**尚未同行评议**）.  
> **预印本**：[10.26434/chemrxiv.15002402/v1](https://doi.org/10.26434/chemrxiv.15002402/v1)  
> **机构**：University of Zurich, Department of Chemistry.

---

## 一、背景：外电场很重要，MLIP 却常「看不见」

外电场在催化、电化学储能、材料表征与生物体系中无处不在，会改变 **电荷转移、几何、反应机理与光谱**。但用 **机器学习原子间势（MLIP）** 准确刻画 **场依赖的响应性质**（偶极 $\mu$、极化率 $\alpha$、IR/Raman）仍很难：

| 路线 | 代表 | 局限 |
|------|------|------|
| 核方法/核+场 | Christensen、早期核模型 | 难给高阶响应、缺多体场耦合 |
| 场嵌入原子网络 | FieldSchNet、FIREANN、PNNP MD | 多仍 **局域截断**，长程静电/极化不足 |
| 场+自洽场模块 | Gao & Reming 两模块网络 | 旋转等变性与场方向处理不严谨 |
| 直接预测 $\mu$ | MACE/NequIP 等 | 零场很强，**外加 $\varepsilon$ 时需改架构或重训** |

Chen & Luber 提出 **场感知 + 电荷知情** 的统一框架：在 **MACE** 等变消息传递网络上 **等变嵌入外电场 $\varepsilon$**，用 **能量对场的导数** 得到 $\mu$、$\alpha$，并嵌入 **变分 QEq（variant QEq，下文称 VQEq 方案）** 处理 **长程静电**；在 **分子与周期体系** 上验证响应性质与 **IR/Raman 光谱**。

![图 1：场感知 + 变分 QEq 框架（原文 Fig. 1）——$\boldsymbol\epsilon$ 嵌入节点特征（式 5）；MACE 消息传递（式 6–10）；读出 $E\_i^{(0)},E\_i^{(1)},E\_i^{(2)},\chi\_i$ 并代入式 (13)–(15) 求 $q\_i$ 与 $E\_{\mathrm{total}}$；$\mathbf F,\boldsymbol\mu,\boldsymbol\alpha$ 为 $E\_{\mathrm{total}}$ 对 $\mathbf R,\boldsymbol\epsilon$ 的导数（式 2–4）。](/img/posts/2026-06-04-chen-vqeq-field-aware-mace/fig01-field-aware-variant-qeq-framework.png)

---

## 二、理论框架与主要公式（与原文式 (1)–(15) 一致）

以下公式 **直接对应** Chen & Luber 预印本 [ChemRxiv](https://doi.org/10.26434/chemrxiv.15002402/v1) 中的编号；符号 $\epsilon=(\epsilon\_x,\epsilon\_y,\epsilon\_z)$ 与原文一致。式 (6)–(10) 中 MACE 各矩阵元定义见 Batatia et al. [42]（原文 §2.2 末句）。

### 2.1 偶极矩的三种途径（原文 §2.1）

**（a）电荷求和（MACE_LR_Q 等）**

$$
\boldsymbol\mu = \sum_{i=1}^{N} q_i \mathbf R_i
\tag{1}
$$

$q\_i$ 为原子部分电荷，$\mathbf R\_i$ 为原子位置；$N$ 为原子数。周期边界下位置算符问题见原文讨论 。

**（b）等变网络直接预测**  
NequIP、MACE 等直接映射几何 → 偶极矢量（零场表现好，外场下常需改架构或重训）。

**（c）能量导数（MACE_EF_dE / MACE_LR_EF_dE，本文主线）**

系统受均匀外电场 $\boldsymbol\epsilon$ 时，总能量 $E$ 为场的函数，偶极矩为

$$
\boldsymbol\mu = -\frac{\partial E}{\partial \boldsymbol\epsilon}
\tag{2}
$$

极化率（偶极–偶极极化率张量）为

$$
\boldsymbol\alpha = -\frac{\partial^2 E}{\partial \boldsymbol\epsilon\,\partial \boldsymbol\epsilon}
\tag{3}
$$

或等价地

$$
\boldsymbol\alpha = \frac{\partial \boldsymbol\mu}{\partial \boldsymbol\epsilon}
\tag{4}
$$

原文强调：**须在多种外场强度下训练** 能量，模型才学会场致响应；力由 $E$ 对坐标求导得到（保守势假设）。

### 2.2 场感知 MACE 短程能量（原文 §2.2，式 5–12）

**外场等变嵌入（式 5）**

$$
h_i^{(0)\prime}(z_i,\boldsymbol\epsilon) = \psi_{\mathrm{atom}}(z_i) \oplus \psi_{\mathrm{field}}(\boldsymbol\epsilon)
\tag{5}
$$

$z\_i$ 为原子序数，$\oplus$ 为拼接；$\psi\_{\mathrm{field}}$ 为 **O(3) 等变线性层**，将场矢量映射为多个矢量通道 [46]。

**第一层原子环境（式 6）**

$$
\mathbf A_i^{(1)}(\mathbf R,\boldsymbol\epsilon) = \sum_{j\in\mathcal N(i)} R_{k l_1}^{(1)}(r_{ji})\, Y_{l_1}^{m_1}(\hat{\mathbf r}_{ji})\, W_k^{(1)}\, h_j^{(0)}(z_j,\boldsymbol\epsilon)
\tag{6}
$$

**更高层消息传递（式 7）**

$$
A^{(t)}_{i,kl_3m_3}(\mathbf R,\boldsymbol\epsilon)
= \sum_{l_1m_1,l_2m_2}
C^{l_3m_3}_{l_1m_1,l_2m_2}
\sum_{j\in\mathcal N(i)}
R^{(t)}_{kl_1l_2l_3}(r_{ji})\,
Y^{m_1}_{l_1}(\hat{\mathbf r}_{ji})
\sum_{k}
W^{(t)}_{kkl_2}\,
h^{(t-1)}_{j,kl_2m_2}(\mathbf R,\boldsymbol\epsilon)
\tag{7}
$$

- $\mathcal N(i)$：原子 $i$ 的邻居；$t$：消息传递层指标。  
- $R^{(t)}\_{kl\_1l\_2l\_3}(r\_{ji})$：可学习径向基；$Y^{m\_1}\_{l\_1}$：球谐函数；$C^{l\_3m\_3}\_{l\_1m\_1,l\_2m\_2}$：标准 Clebsch–Gordan 系数。  
- $r\_{ji}$：原子间距；$\hat{\mathbf r}\_{ji}$：单位键向量；$k$ 为通道指标；$l\_2,m\_2$ 为 $h\_j$ 的角动量指标，$l\_3,m\_3$ 为输出角动量指标。

**张量积（式 8）**

$$
B^{(t)}_{i,\eta\nu kLM}(\mathbf R,\boldsymbol\epsilon)
= \sum_{\{lm\}}
C^{LM}_{\eta\nu,\{lm\}}
\prod_{\xi=1}^{\nu}
\sum_{k}
w^{(t)}_{kkl_\xi}\,
A^{(t)}_{i,kl_\xi m_\xi}(\mathbf R,\boldsymbol\epsilon)
\tag{8}
$$

- $C^{LM}\_{\eta\nu,\{lm\}}$：广义 Clebsch–Gordan 系数；$\nu$：**correlation order**（体相关阶）。  
- $\{lm\}$：多指标 $(l\_1m\_1,\ldots,l\_\nu m\_\nu)$；$w^{(t)}\_{kkl\_\xi}$：混合 $A^{(t)}\_i$ 各通道 $k$ 的权重。  
- $L$：输出对称阶（irrep 阶）；$M$：输出磁量子数。

**消息聚合（式 9）**

$$
m^{(t)}_{i,kLM}(\mathbf R,\boldsymbol\epsilon)
= \sum_{\nu}\sum_{\eta\nu}
W^{(t)}_{zikL,\eta\nu}\,
B^{(t)}_{i,\eta\nu kLM}(\mathbf R,\boldsymbol\epsilon)
\tag{9}
$$

$W^{(t)}\_{zikL,\eta\nu}$：由 **化学元素** $z\_i$ 与消息对称性 $L$ 决定的可学习权重矩阵。

**节点更新（式 10）**

$$
h^{(t+1)}_{i,kLM}(\mathbf R,\boldsymbol\epsilon)
= \sum_{k} W^{(t)}_{kL,k}\, m^{(t)}_{i,kLM}(\mathbf R,\boldsymbol\epsilon)
+ \sum_{k} W^{(t)}_{zikL,k}\, h^{(t)}_{i,kLM}(\mathbf R,\boldsymbol\epsilon)
\tag{10}
$$

式 (6)–(10) 中凡含 $h\_j^{(t)}$、$A^{(t)}\_i$、$B^{(t)}\_i$、$m^{(t)}\_i$ 的量均写为 $(\mathbf R,\boldsymbol\epsilon)$ 的函数；与无场 MACE [42] 形式相同，区别在 **初值** $h\_j^{(0)}(z\_j,\boldsymbol\epsilon)$ 来自式 (5)。原文说明：式 (6)–(12) 各参数的更完整定义见 Ref. [42]。

**无长程 QEq 时的总能量（式 11–12）**（MACE、MACE_EF、MACE_EF_dE 等）：

$$
E_{\mathrm{total}}(\mathbf R,\boldsymbol\epsilon) = \sum_{i=1}^{N} E_i(\mathbf R,\boldsymbol\epsilon)
\tag{11}
$$

站点能量 $E\_i = \sum\_{t=0}^{T} e\_i^{(t)}$，末层读出为（式 12）：

$$
e_i^{(t)}(\mathbf R,\boldsymbol\epsilon) =
\begin{cases}
\sum_k W_{\mathrm{readout},k}^{(t)}\, h_{i,k00}^{(t)}(\mathbf R,\boldsymbol\epsilon), & t < T \\[4pt]
\mathrm{MLP}_{\mathrm{readout}}^{(t)}\bigl(h_{i,k00}^{(t)}(\mathbf R,\boldsymbol\epsilon)\bigr)_k, & t = T
\end{cases}
\tag{12}
$$

$T$ 为消息传递层数。此时 $\boldsymbol\mu$、$\boldsymbol\alpha$ 由式 (2)–(4) 对 **式 (11)** 的 $E\_{\mathrm{total}}$ 求导。

### 2.3 变分 QEq 长程：式 (13)–(15)（原文 §2.3）

经典 ML+QEq 常由网络预测 $\chi\_i$、$J\_i$ 再解 QEq。**Shaidu 等 [51]** 在 $q=0$ 邻域对短程能做二阶 Taylor 展开，由网络预测 $E\_i^{(0)},E\_i^{(1)},E\_i^{(2)}$，且 **$\chi\_i$ 与 $J\_i$ 取固定值**，通过 **一次线性方程组** 求电荷。Chen 等将其扩展为 **场感知**，总能量写为（**MACE_LR** 族）：

$$
\begin{aligned}
E_{\mathrm{total}}(\mathbf R,\boldsymbol\epsilon) ={}& \sum_{i=1}^{N} \Bigl( E_i^{(0)}(\mathbf R,\boldsymbol\epsilon)
+ \bigl(\chi_i(\mathbf R,\boldsymbol\epsilon) + E_i^{(1)}(\mathbf R,\boldsymbol\epsilon)\bigr) q_i \\
&+ \frac{1}{2}\Bigl(J_i + E_i^{(2)}(\mathbf R,\boldsymbol\epsilon) + \frac{1}{\sigma_i\sqrt{\pi}}\Bigr) q_i^2 \Bigr) \\
&+ \frac{1}{2}\sum_{i\neq j} \frac{q_i q_j}{r_{ij}}\,
\operatorname{erf}\!\left(\frac{r_{ij}}{\sqrt{2(\sigma_i^2+\sigma_j^2)}}\right)
\end{aligned}
\tag{13}
$$

- $E\_i^{(0)}, E\_i^{(1)}, E\_i^{(2)}, \chi\_i$：**均由同一 MACE 读出模块预测**，且可依赖 $(\mathbf R,\boldsymbol\epsilon)$。  
- $J\_i$：**取自实验原子硬度**（原文：*obtained from experiments for the simplicity*），**不由网络学习**。  
- $\sigma\_i$：原子半径；$r\_{ij}$：原子间距；$\operatorname{erf}$ 项为 Gaussian 展宽电荷的库仑相互作用。  
- 周期体系：构造矩阵 $\mathbf A$ 时用 **Ewald 求和**（原文 §2.3 末句）。

对 QEq 能量关于 $q\_i$ 求导并令其为零，在总电荷 $Q=\sum\_i q\_i$ 约束下得到平衡电荷（式 14）：

$$
\sum_{j=1}^{N} A_{ij}(\mathbf R,\boldsymbol\epsilon)\, q_j + \bigl(\chi_i(\mathbf R,\boldsymbol\epsilon) + E_i^{(1)}(\mathbf R,\boldsymbol\epsilon)\bigr) + \lambda = 0
\tag{14}
$$

$\lambda$ 为 Lagrange 乘子（固定总电荷 $Q$）。矩阵元（式 15）：

$$
A_{ij}(\mathbf R,\boldsymbol\epsilon) =
\begin{cases}
E_i^{(2)}(\mathbf R,\boldsymbol\epsilon) + J_i + \dfrac{1}{\sigma_i\sqrt{\pi}}, & i=j \\[8pt]
\dfrac{1}{r_{ij}}\,
\operatorname{erf}\!\left(\dfrac{r_{ij}}{\sqrt{2(\sigma_i^2+\sigma_j^2)}}\right), & i\neq j
\end{cases}
\tag{15}
$$

由 $\{q\_i\}$ 代回式 (13) 得 $E\_{\mathrm{total}}$；$\mathbf F\_i=-\partial E\_{\mathrm{total}}/\partial \mathbf R\_i$ 时 $q\_i$ 对 $\mathbf R$ 的依赖须 **可微地** 穿过式 (14)–(15) 求解器。

### 2.4 为何 Shaidu 中「$\chi\_i,J\_i$ 可固定」？Chen 文又如何？

**Shaidu 等 [51]（Table 2 中的 VQeq 基准）**  
Taylor 系数 $E\_i^{(0,1,2)}$ 由 NN 给出；**$\chi\_i$ 与 $J\_i$ 不随环境由 NN 再学一套**，故 QEq 线性系统里一次/二次系数由 **固定元素参数 + $E\_i^{(1)},E\_i^{(2)}$** 承担。

**Chen & Luber（本文 MACE_LR）** 在式 (13) 中采用 **更一般的分解**（原文 §2.3）：

| 项 | 来源 | 角色 |
|----|------|------|
| $\chi\_i(\mathbf R,\boldsymbol\epsilon)$ | **网络预测** | 电负性（可场依赖） |
| $E\_i^{(1)}(\mathbf R,\boldsymbol\epsilon)$ | **网络预测** | 对 $\chi\_i$ 的 **可学习、环境/场依赖修正** |
| $J\_i$ | **实验固定** | 原子硬度（**不**由 NN 预测） |
| $E\_i^{(2)}(\mathbf R,\boldsymbol\epsilon)$ | **网络预测** | 对 $J\_i$ 的 **可学习修正** |
| $1/(\sigma\_i\sqrt{\pi})$ | 几何/原子半径 | Gaussian 自能修正 |

因此：

- **「$J\_i$ 可固定」** 在本文中 **字面成立**：$J\_i$ 用实验值，环境/场响应主要靠 $E\_i^{(2)}$ 修正。  
- **$\chi\_i$ 在 Chen 文中并非固定**：与 $E\_i^{(1)}$ 一并由网络输出；固定的是 **Shaidu 基准** 里的做法。  
- 线性项系数进入式 (14) 的是 **$\chi\_i + E\_i^{(1)}$**；二次项对角元是 **$J\_i + E\_i^{(2)} + 1/(\sigma\_i\sqrt{\pi})$**——Taylor 展开 **不是** 替代 $\chi\_i$，而是在式 (13) 中与 $\chi\_i$、$J\_i$ **并列**。

**外场如何进入**（原文，非额外 $-\sum q\_i\boldsymbol\epsilon\cdot\mathbf R\_i$ 项）：  
经典带外场的 QEq 常需显式场耦合项 [52]；本文将 $\boldsymbol\epsilon$ **嵌入原子表示**（式 5–10），使 $E\_i^{(0,1,2)}$、$\chi\_i$ 场依赖，从而在 **数据驱动** 下得到随 $\boldsymbol\epsilon$ 变化的电荷再分布；偶极/极化率对 **MACE_LR_EF_dE** 仍由式 (2)–(4) 对式 (13) 的 $E\_{\mathrm{total}}$ 求导（液态水 MD 中亦用 $\boldsymbol\mu=-\partial E/\partial\boldsymbol\epsilon$，$\boldsymbol\alpha=\partial\boldsymbol\mu/\partial\boldsymbol\epsilon$，原文 Fig. 8 说明）。

### 2.5 训练目标（原文 §2.3–2.4）

原文：**损失函数只包含总能量与力**；**不使用** 原子电荷作为训练标签（尽管 4G-HDNNP 等可用参考电荷）。$q\_i$ 由式 (14)–(15) 在每次前向中解出。NMA 等验证集上可额外报告 $\boldsymbol\mu$、$\boldsymbol\alpha$ 的 RMSE；**MACE_LR_EF_dE** 训练偶极/极化率时亦可以能量、力为主、响应用于评估（见 Table 3 与 §2.4 叙述）。

**模型对应**（Table 1）：**MACE_LR** 用式 (13)–(15)；**MACE_LR_EF_dE** 用式 (2) 求 $\boldsymbol\mu$；**MACE_LR_Q** 用式 (1) 与式 (14) 得到的 $q\_i$。

![表 1：MACE、MACE_LR、MACE_μ、MACE_EF、MACE_EF_dE、MACE_LR_EF_dE、MACE_LR_Q 等方法对比（特征、实现、可预测量）。](/img/posts/2026-06-04-chen-vqeq-field-aware-mace/table01-mace-variants-comparison.png)

---

## 三、模型对照实验：NMA 数据集（~1.3 万构型）

**N-甲基乙酰胺（NMA）** 数据集（Zhang 等 FIREANN 工作）：外场沿 $x$ **0–0.4 V/Å**，含能量、力、$\mu$、$\alpha$（DFT）；训练 **10 512** / 验证 **1 289**。

### 3.1 逐档改进（Table 3）

| 模型 | E (meV) | F (meV/Å) | μ (Debye) | α (a.u.) |
|------|---------|-----------|-----------|----------|
| MACE_μ（无场输入，直接预测 μ） | 108.520 | 56.770 | 0.241 | — |
| MACE_EF（场嵌入，直接预测 μ） | 1.667 | 7.076 | 0.016 | 0.130 |
| MACE_EF_dE（场+能量导数，无 LR） | 1.591 | 7.911 | 0.010 | 0.079 |
| **MACE_LR_EF_dE** | **1.208** | **5.983** | **0.007** | **0.063** |
| FIREANN（文献） | 5.286 | — | 0.028 | 0.506 |

**结论链**：

1. **必须显式输入 $\varepsilon$**——否则无法学场强依赖（MACE_μ 惨败）。  
2. **能量导数求 μ** 优于直接头预测（0.016 → 0.007 Debye）。  
3. **加入 QEq 长程（MACE_LR）** 在 NMA 上 **略优** 于 MACE_EF_dE，但训练/推理更慢——精度与算力权衡。

![表 3：NMA 验证集 RMSE（外场 0–0.4 V/Å）。](/img/posts/2026-06-04-chen-vqeq-field-aware-mace/table03-nma-validation-rmse.png)

### 3.2 电荷物理一致性（Fig. 2）

**MACE_LR_Q**（电荷求和 $\mu=\sum q\_i\mathbf R\_i$）在五种重原子上，**随场变化的电荷 z-score 热图** 与 **DFT Mulliken** 几乎一致；**MACE_LR_EF_dE** 的「有效电荷」热图则偏差大——说明 **能量导数偶极** 与 **QEq 电荷偶极** 是不同可观测量，但 **MACE_LR_Q** 在 **0.011 Debye** 仍与能量导数 **0.007 Debye** 同量级，**物理自洽**。

![图 2：NMA 五原子电荷 z-score——DFT vs MACE_LR_EF_dE vs MACE_LR_Q。](/img/posts/2026-06-04-chen-vqeq-field-aware-mace/fig02-nma-atomic-charge-zscore.png)

### 3.3 极化率（Fig. 3）

**MACE_LR_EF_dE** 对 $\alpha$ 张量展平误差的 violin 分布 **最窄**（优于 MACE_EF、MACE_EF_dE）。

![图 3：α 预测误差分布（NMA 验证集）。](/img/posts/2026-06-04-chen-vqeq-field-aware-mace/fig03-polarizability-error-violin.png)

---

## 四、振动光谱：NMA 与场致频移

### 4.1 零场 NMA（300 K，Fig. 4）

**ML-TRPMD**（热环聚合物 MD + ML 势）与实验 IR/Raman **高度一致**；普通 **ML-MD** 在高频区（~3000 cm⁻¹）有明显频移——说明 **核量子效应/更精确动力学** 对光谱峰位重要，但 **ML 势本身** 在 TRPMD 框架下已足够准。

![图 4：NMA 实验 vs ML-TRPMD vs ML-MD 的 IR 与 Raman。](/img/posts/2026-06-04-chen-vqeq-field-aware-mace/fig04-nma-ir-raman-ml-trpmd.png)

### 4.2 外加场 0 / 0.2 / 0.4 V/Å（Fig. 5）

**MACE_LR_EF_dE** 驱动 ML-MD：**酰胺 I 带 ~1700 cm⁻¹** 随场强 **红移**（箭头标注），IR、Raman-iso、Raman-aniso 三通道一致——证明 **单场统一模型** 可模拟 **场致振动光谱响应**。

![图 5：NMA 在 0、0.2、0.4 V/Å 下的 IR 与 Raman 谱。](/img/posts/2026-06-04-chen-vqeq-field-aware-mace/fig05-nma-field-dependent-ir-raman.png)

---

## 五、长程基准：四个带电/表面数据集（Table 2）

在 **Ag₃⁺/⁻、Na₈/₉Cl₈⁺、Au₂/MgO(001)、C₁₀H₂/C₁₀H₃⁺** 上对比 **4G-HDNNP、VQeq（Shaidu 等）、MACE、MACE_LR**（能量 meV/atom，力 meV/Å）：

- **MACE_LR** 在四类集上 **全面低于** 原始 MACE；  
- 相对 **VQeq** 与 **4G-HDNNP**，MACE_LR 在多数集上 **能量/力更优**（如 Ag₃⁺/⁻ 力 RMSE **20 vs 79** meV/Å）；  
- 说明 **场感知 + 变分 QEq 嵌入 MACE** 对 **带电与界面** 体系同样有效。

![表 2：四数据集验证 RMSE。](/img/posts/2026-06-04-chen-vqeq-field-aware-mace/table02-validation-rmse-four-datasets.png)

---

## 六、周期体系：分层液态水

### 6.1 对称约束水层与能量–场曲线（Fig. 6）

Zhang 等 **200 构型**：四层水，场沿 $x$，**偶极投影可相互抵消**——考验模型是否真懂 **场致能量**，而非死记硬背偶极。  
**MACE_LR_EF_dE** 预测 $E(\epsilon\_x)$ **抛物线** 与 DFT 重合；测试集力 RMSE **37.8 meV/Å**、$\alpha$ **2.2 a.u.**，略优于 MACE_EF_dE 与 FIREANN。

![图 6：$E$ 随 $\epsilon\_x$（-2–2 V/Å）与 DFT 对比。](/img/posts/2026-06-04-chen-vqeq-field-aware-mace/fig06-field-energy-parabola.png)

### 6.2 IR/Raman 与场致峰移（Fig. 7）

**~1870 构型**（64 分子立方盒，CP2K AIMD，场 0–0.3 V/Å）：训练 **能、力、α**（**未**用 Berry 相位偶极训练，避免 PBC 偶极多值跳变；IR 用时域 **$\mu=-\partial E/\partial\varepsilon$**）。

- **零场**：ML-MD IR 再现 O–H 伸缩（3400–3600 cm⁻¹）、弯曲（~1650）、 librational（600–800）、氢键伸缩（~200 cm⁻¹），与实验/AIMD 一致；Raman 亦吻合。  
- **0.1–0.3 V/Å**：O–H 带 **红移**，librational **蓝移**（场致取向、转动受限）。

![图 7：液态水零场与不同场强 IR；零场 Raman。](/img/posts/2026-06-04-chen-vqeq-field-aware-mace/fig07-water-ir-raman-vs-field.png)

### 6.3 轨迹上的 μ 与 α（Fig. 8）

沿 AIMD 轨迹，ML 预测的 **$\mu\_x$、$\alpha\_{zz}$** 与 DFT 波动一致（偶极常数项平移对齐）。多分量对比见 SI Fig. S7–S8。

![图 8：不同场强下 $\mu\_x$ 与 $\alpha\_{zz}$ 时间序列（AIMD vs MACE_LR_EF_dE）。](/img/posts/2026-06-04-chen-vqeq-field-aware-mace/fig08-water-dipole-polarizability-trajectory.png)

### 6.4 强场外推与微调（Fig. 9）

训练场 **0–0.3 V/Å**，测试 **0.4 V/Å**：**直接预测** IR 在高频区偏差大；仅用 **200 个 0.4 V/Å 构型 **微调** 后，IR、$\mu\_x$、$\alpha\_{zz}$ 轨迹 **明显贴近** AIMD——说明框架可 **外推+小样本修正**。

![图 9：0.4 V/Å 外推——直接预测 vs 微调后 IR/μ/α。](/img/posts/2026-06-04-chen-vqeq-field-aware-mace/fig09-water-strong-field-finetune.png)

### 6.5 Cu²⁺ 水溶液极化率（Fig. 10）

非场测试集：**MACE_LR_EF_dE** 对 9 分量 $\alpha$ 的 RMSE **3.02 a.u.**，略优于 MACE_EF_dE（3.21 a.u.）。

![图 10：Cu²⁺ 液态水 $\alpha$ 预测 vs DFT 相关图。](/img/posts/2026-06-04-chen-vqeq-field-aware-mace/fig10-cu-water-polarizability-correlation.png)

---

## 七、讨论、概念辨析与相关路线

### 7.1 贡献小结

- **统一架构**：在显式输入均匀外场 $\boldsymbol\varepsilon$ 的前提下，构造 **一个** $E\_{\mathrm{total}}(\mathbf R,\boldsymbol\varepsilon)$，由其对 $\mathbf R$、$\boldsymbol\varepsilon$ 的导数得到 **力、偶极、极化率**，并接 **IR/Raman**（偶极/极化率自相关 + Fourier）。  
- **物理约束**：变分 QEq 长程 + **不监督** 分区电荷；**MACE_LR_Q**（$\boldsymbol\mu=\sum q\_i\mathbf r\_i$）与 **MACE_LR_EF_dE**（能量导数 μ）可 **相互校验**。  
- **分子 + 周期**：NMA、液态水、带电团簇/表面四数据集。

### 7.2 概念辨析：场感知 MACE、变分 QEq、「一场 $\boldsymbol\varepsilon$、一个 $E\_{\mathrm{total}}$」

**场感知 MACE vs 标准 MACE**

标准 MACE 学 $E(\mathbf R)$；**场感知**（文中 **MACE_EF**）仅在初始节点特征上增加 **O(3) 等变** 的 $\psi\_{\mathrm{field}}(\boldsymbol\varepsilon)$，与 $\psi\_{\mathrm{atom}}(z\_i)$ 拼接，使能量面变为 $E(\mathbf R,\boldsymbol\varepsilon)$。消息传递骨架与 Batatia 等 MACE 同源，**不是**零场 MD 的通用加速版，而是 **外场为控制变量** 时的架构扩展。**MACE_LR**、**MACE_LR_EF_dE** 在此基础上再叠 **变分 QEq 长程** 或 **$\mu,\alpha$ 能量导数读出**。

**「变分」用在哪里**

变分指 **式 (13)** 对 $\{q\_i\}$ 求极小（等价于式 (14) 线性系统）。**MACE_LR** 用式 (13)–(15)；$J\_i$ **实验固定**，$\chi\_i$ 与 $E\_i^{(0,1,2)}$ 由网络预测（§2.4）。无长程时总能量为式 (11)–(12)。

**「一场 $\boldsymbol\varepsilon$、一个 $E\_{\mathrm{total}}$」何意**

二者 **都不是** 自然界给定的常数：

| 量 | 训练 | 推理 / MD |
|----|------|-----------|
| $\boldsymbol\varepsilon$ | 每条 DFT 样本附带的 **计算条件**（如 NMA 沿 $x$ 的 0–0.4 V/Å） | **用户指定的控制参数**（零场 MD 取 $\mathbf 0$；场依赖光谱取 0.2、0.4 V/Å 等） |
| $E\_{\mathrm{total}}$ | **标签**：拟合 DFT 总能量 | **MACE_LR**：式 (13)；**MACE_EF**：式 (11) |

Slogan 的含义是：对 **每个** $(\mathbf R,\boldsymbol\varepsilon)$，只建 **一个标量** 能量面，$\mathbf F$、$\boldsymbol\mu$、$\boldsymbol\alpha$ 应来自 **同一** $E\_{\mathrm{total}}$ 的导数，而非三个互不耦合的预测头（Table 3 中无场输入的 **MACE_μ** 即反例）。

### 7.3 变分 QEq（variant QEq）与经典 QEq 的区别

| | **经典 QEq + ML** | **variant QEq（VQeq，Shaidu 等）** | **Chen MACE_LR** |
|---|-------------------|-----------------------------------|------------------|
| 网络输出 | 常预测 $\chi\_i,J\_i$ 等 QEq 参数 | 预测 $q\_i\approx 0$ 附近的 **Taylor 系数** $E\_i^{(0,1,2)}$ | 同上，且系数 **依赖 $\boldsymbol\varepsilon$** |
| $\chi\_i,J\_i$ | NN 常预测 $\chi\_i,\,J\_i$ | Shaidu：**$\chi\_i,J\_i$ 固定** + $E\_i^{(k)}$ 由 NN | Chen：$\chi\_i$ 与 $E\_i^{(k)}$ **NN 预测**，$J\_i$ **实验固定**（式 13） |
| 求电荷 | 变分 / 线性求解 | **同样** 变分求 $\{q\_i\}$ | 同左 |
| 训练标签 | $E,\mathbf F$；电荷标签易歧义 | 强调 **只拟合 $E,\mathbf F$**，不监督 Mulliken/Hirshfeld | 同左；Fig. 2 与 Hirshfeld 仍高度一致 |
| 在 Chen 文中 | 概念对照 | Table 2 的 **VQeq 基准势** | **场感知 MACE + variant QEq** |

**VQeq 并未替换库仑定律**，而是替换 **NN→QEq 的接口**；Chen 再让该接口 **随均匀外场变化**。Table 2 的 **VQeq** 指 Shaidu 等的 **独立 ML 势**；**MACE_LR** 是 **同一思想嵌入场感知 MACE**。

### 7.4 偶极 μ 与极化率 α：本文 vs Kim 2026（可极化多极矩 / MACELES）

Kim 等（*Polarizable atomic multipoles for learning long-range electrostatics*, arXiv:2605.05746, 2026）将 **可学习原子多极矩**（$q,\mathbf u,Q$）+ **Ewald 长程** + **非自洽诱导**（$\Delta q\_i=-\kappa\_i\Phi\_i$，$\Delta\mathbf u\_i=\boldsymbol\alpha\_i\cdot\mathbf E\_i$）接到短程 MLIP 上，主训练标签为 **零场（或 bulk）能量、力**，电响应多 **后处理** 涌现。与 Chen 文 **μ、α 定义** 对比如下。

| | **Chen & Luber（MACE_LR_EF_dE）** | **Kim et al.（MACELES-uiu 等）** |
|---|-----------------------------------|----------------------------------|
| **外场** | **显式均匀 $\boldsymbol\varepsilon$** 进 $E\_{\mathrm{total}}$ | 训练势 **通常不含** 均匀外加场；$\Phi,\mathbf E$ 来自 **固定多极矩产生的局域场** |
| **μ** | $\boldsymbol\mu=-\partial E\_{\mathrm{total}}/\partial\boldsymbol\varepsilon$（热力学响应） | $\mathbf P=\sum\_i(q\_i+\Delta q\_i)\mathbf r\_i+\sum\_i(\mathbf u\_i+\Delta\mathbf u\_i)$（式 26，多极矩求和） |
| **α** | $\partial^2 E\_{\mathrm{total}}/\partial\varepsilon\_i\partial\varepsilon\_j$ 或 $\partial\boldsymbol\mu/\partial\boldsymbol\varepsilon$ | $\boldsymbol\alpha\_{\mathrm{sys}}=\sum\_i\boldsymbol\alpha\_i$（式 30，原子潜极化率张量 **相加**；bulk 水用 $\varepsilon\_\infty$ 标度） |
| **与 DFT 对齐** | NMA：**μ** RMSE **0.007 Debye**，**α** **0.063 a.u.**（field-sweep 数据） | 水：**BEC** RMSE **0.022 e**；各向同性/异性 **α** 与 DFPT 的 Pearson **r≈0.86–0.89** |
| **IR/Raman** | ML-MD 下用 **导数 μ**（及极化率相关函数）；可 **扫 $\boldsymbol\varepsilon$** 得场致频移 | 用 **BEC** 与预测的 **$\boldsymbol\alpha\_i$** 在轨迹上算光谱；主结果侧重 **零场 bulk** |
| **Debye** | 与 DFT 相同单位换算；**非** 方法论差异 | 同左 |

**哪种更「物理」取决于问题设定**（并非二选一）：

- **均匀外场下的 μ、α、$E(\epsilon)$ 曲线、场致 IR 红移**（NMA、FIREANN 类数据）：Chen 的 **能量导数** 与 **field-coupled DFT** 定义一致，且保证 μ、α 与 **同一** $E(\mathbf R,\boldsymbol\varepsilon)$ 自洽；直接预测 μ 头（**MACE_μ**，无 $\boldsymbol\varepsilon$ 输入）在 NMA 上 μ RMSE **0.241 Debye**，说明 **场必须进能量面**。  
- **零场 bulk、BEC、DFPT 本征极化率、强非局域电荷转移**（水、MAPbI₃、带电团簇）：Kim 的 **显式多极矩 + Ewald + 诱导响应** 更贴近 **极化力场 / LES** 传统，原子级 $q,\mathbf u,\boldsymbol\alpha\_i$ **可解释**；但 **$\sum\_i\boldsymbol\alpha\_i$** 是原子可加性近似，**非自洽诱导** 也弱于全自洽极化力场。  
- **互补**：Kim 解决「**无外场标签时** 把长程电学学进 MLIP」；Chen 解决「**外场即控制量时** 把 μ、α 绑回同一势能面」。液态水训练中 Chen 文常 **不用** Berry 相位偶极作标签，IR 仍用 **$-\partial E/\partial\boldsymbol\varepsilon$** 的 μ，与 Kim 用 **BEC + 原子 $\boldsymbol\alpha\_i$** 进光谱的链路不同。

### 7.5 局限（原文与 SI）

- QEq 长程 **增加算力**；小数据集上物理先验 **未必** 更省样本（200 构型水层上 MACE_EF_dE 偶优于 MACE_LR_EF_dE）。  
- **周期偶极** 多值性：训练常避开 Berry 相位偶极；PBC 下需专门分支处理。  
- 外场仅 **均匀矢量场**；与真实电极 **非均匀** 场、恒电势（ConstP）边界仍有距离。

### 7.6 与 Gao 2025 PQEq、Hu 2025 DP-QEq 的对照

| 工作 | 长程 | 外场/电化学边界 |
|------|------|-----------------|
| **Gao PQEq**（Nat. Commun. 2025） | **可极化 PQEq** + D3 + GNN $E\_0$ | 中性/带电 foundation，**非** ConstP 循环 MD |
| **Hu DP-QEq**（Nat. Commun. 2025） | 经典 **QEq** + DP 短程 | **ConstP** 双电极，枝晶成核 |
| **Chen 本文** | **变分 QEq** + 场感知 MACE | **均匀 $\boldsymbol\varepsilon$** + 能量导数 $\mu,\alpha$ |
| **Kim 2026** | **Ewald + 多极矩 + 诱导** | 零场/bulk 为主；BEC、$\alpha$、IR/Raman **涌现** |

**可结合方向（研究层面）**：

- Chen 的 **场嵌入 + 能量导数** 可与 Gao 的 **PQEq 可极化长程** 替换经典 QEq 项，改善 **强场下分子极化**；  
- Hu 的 **ConstP** 解决 **电极开放边界**；与 Chen 合并需把 **电势移动** 写进 **PQEq/变分 QEq 巨势** 并在多种偏压轨迹上重训；  
- Kim 的 **Ewald 多极矩** 与 Chen 的 **field-aware $E(\mathbf R,\boldsymbol\varepsilon)$** 可分别在「零场长程电学」与「扫场光谱」场景互补，而非简单替代。

---

## 八、方法摘要

| 项目 | 内容 |
|------|------|
| 基座 | MACE（高阶等变消息传递） |
| 场 | $\psi\_{\mathrm{field}}(\boldsymbol\varepsilon)$ 等变嵌入；训练 $\varepsilon\in[0,0.4]$ V/Å（NMA）等 |
| 长程 | 式 (13)–(15)：$\chi\_i+E\_i^{(1)}$，$J\_i+E\_i^{(2)}$，Gaussian $\operatorname{erf}$ 库仑 |
| 响应 | 式 (2)–(4)：$\boldsymbol\mu=-\partial E/\partial\boldsymbol\epsilon$，$\boldsymbol\alpha=-\partial^2 E/\partial\boldsymbol\epsilon^2$ 或 $\partial\boldsymbol\mu/\partial\boldsymbol\epsilon$ |
| 光谱 | ML-MD / ML-TRPMD + 偶极/极化率 ATF → IR/Raman |
| 软件 | MACE 扩展、CP2K（水 AIMD）、自动微分 |

---

## 九、一句话总结

Chen & Luber 2026 在 **MACE** 上实现 **外电场等变嵌入 + 变分 QEq 长程 + 能量导数统一求 $\mu$ 与 $\alpha$**，从 **NMA 验证** 到 **液态水 IR/Raman 与强场外推**，为 **场依赖响应性质与振动光谱** 提供可迁移的物理知情 MLIP 路线；与 **Kim 2026 多极矩–Ewald**（零场 BEC/极化率）、**PQEq foundation**、**DP-QEq ConstP** 等路线 **问题设定不同、可互补**，共同指向「**长程电学 + 场响应 + 电化学边界**」的下一代 MLIP。

---

## 延伸阅读（站内）

- [恒电势 MD 看见锂枝晶成核：Hu 等用 DP-QEq 把 QEq 长程静电接进 Deep Potential——Hu 等 2025 解读](/2026/06/03/恒电势-MD-看见锂枝晶成核-Hu-等用-DP-QEq-把-QEq-长程静电接进-Deep-Potential——Hu/)


## 参考文献（精选）

- Chen & Luber, *ChemRxiv* (2026). [10.26434/chemrxiv.15002402/v1](https://doi.org/10.26434/chemrxiv.15002402/v1)  
- Batatia et al., MACE, *J. Chem. Phys.* **159**, 054801 (2023).  
- Shaidu et al., variant QEq + NN, 相关文献 [51]（文中 VQEq 基准）.  
- Zhang & Jiang, FIREANN, 场诱导网络（NMA 数据）.  
- Gao et al., PQEq foundation MLIP, *Nat. Commun.* **16**, 10484 (2025).  
- Hu et al., DP-QEq ConstP dendrite, *Nat. Commun.* **16**, 7379 (2025).  
- Kim et al., *Polarizable atomic multipoles for learning long-range electrostatics*, arXiv:2605.05746 (2026).
