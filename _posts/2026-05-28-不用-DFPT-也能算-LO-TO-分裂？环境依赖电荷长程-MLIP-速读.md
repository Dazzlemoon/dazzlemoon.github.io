---
layout:     post
title:      不用 DFPT 也能算 LO-TO 分裂？环境依赖电荷长程 MLIP 速读
subtitle:   Korogod 等 2026 arXiv：MTP+EDQRd 用环境依赖点电荷 + 电荷守恒，仅用能量/力/应力训练即可构造 NAC，复现 NaCl 的 LO-TO 分裂与 ε₀/ε∞；本文从方法、公式推导到数值结果完整速读。
date:       2026-05-28
author:     qqz
header-img: img/post-bg-desk.jpg
catalog:    true
tags:
    - MLIP
    - MTP
    - LO-TO
    - NAC
    - 长程电静学
    - 声子
    - 介电常数
    - NaCl
    - Korogod
    - 高精度谱学
---
![一图总结](/img/posts/korogod-2026-loto-mlip/cover.png)

> 基于原文：Korogod, Shapeev & Novikov (2026)  
> *Long-range machine-learning potentials with environment-dependent charges enable predicting LO-TO splitting and dielectric constants*  
> arXiv:2603.06396 · [PDF 链接](https://arxiv.org/abs/2603.06396)

做极性晶体的声子，你大概都踩过这个坑：

- 短程 MLIP 在 **Γ 点** 把 LO / TO 光学支**合并成一条**；
- 即便加了固定电荷的库仑长程项，**LO-TO 分裂还是出不来**；
- 传统补救是再跑一遍 **DFPT**，单独算 Born 有效电荷 $Z^{*}$ 和高频介电常数 $\varepsilon_\infty$，手工喂给 Phonopy 的 **NAC**。

2026 年 3 月，Korogod 等人提出 **MTP+EDQ / MTP+EDQRd**：用**随局域环境变化的原子点电荷**描述长程静电，并给出一条关键结论——

> **对各向同性极性晶体，NAC 所需的信息可以完全从 MLIP 预测的原子电荷及其导数中提取，不必额外做 DFPT。**

NaCl 上 LO-TO 分裂与 DFT 高度一致；偶极矩涨落给出的 $\varepsilon_0/\varepsilon_\infty = 2.71 \pm 0.07$，实验值 2.53，偏差约 7%。

---

## 01. 这篇文章在解决什么？

在极性晶体与分子体系中，**长程静电相互作用**（库仑作用）对能量、力、声子色散和介电性质都至关重要。传统 MLIP 多为**短程局域模型**，即便叠加固定电荷的库仑项，也往往无法准确描述：

- 分子二聚体在**大分离距离**上的结合能曲线；
- 极性晶体在 **Γ 点** 的 **LO-TO 分裂**（纵光学声子与横光学声子频率之差）；
- 由偶极矩涨落得到的**静态/高频介电常数之比**。

**痛点**：现有长程 MLIP（**DPLR**、**CACE+LES**、**MTP+QRd** 等）已经证明「短程 + 库仑」可行，但 **QRd 固定类型电荷** 在环境变化剧烈时精度不够——典型例子是有机阴离子–芳香二聚体在大分离距离上的结合曲线。

**方案**：提出两类显式长程静电模型，与短程 **MTP** 耦合，系统验证从分子二聚体 → NaCl 晶体 → 四方相 PbTiO₃ 的预测能力：

| 模型 | 电荷怎么定 | 总电荷守恒 | 适用 |
|------|------------|------------|------|
| **QRd** | 仅依赖元素类型 $z_i$，参数 $b, s$ 再分配 | ✅ | 基准长程模型 |
| **EDQ** | $q_i = V(n_i)$，局域 MTP 预测 | ❌ | 真空分子 |
| **EDQRd** | EDQ + QRd 再分配 | ✅ | 周期极性晶体 |

**EDQRd** 可理解为：把 QRd 中类型常数 $b_{z_i}$ 替换为环境依赖项 $V(n_i, \mathbf p)$，兼顾 EDQ 的精度与 QRd 的廉价电荷守恒。短程部分一律采用 **MTP**；电荷预测也使用较低层级的 MTP（如 2 级、6 级、10 级等），与主 MTP 分开训练。

---

## 02. 背景：LO-TO 分裂与 MLIP 的长程短板

### 2.1 什么是 LO-TO 分裂？

在极性材料中，长光学（LO）与横光学（TO）声子在 **布里渊区中心 Γ 点** 频率不同，这一现象称为 **LO-TO 分裂**。其物理根源是**宏观电场**与**原子位移**之间的长程偶极-偶极耦合。

要在 Γ 点附近正确描述这一效应，通常需要在动力学矩阵中加入 **非解析修正（Non-Analytical Correction, NAC）**。经典做法依赖：

- 高频介电张量 $\hat\varepsilon_\infty$；
- Born 有效电荷 $Z^{*}$。

二者通常由 DFT 密度泛函微扰理论（DFPT）或实验获得，计算成本不低。Korogod 等的贡献是：**不必单独做 DFPT**，即可通过 MLIP 预测的原子电荷及其导数构造 NAC。

### 2.2 现有长程 MLIP 的脉络

文中回顾了多种显式纳入长程静电的 MLIP：

- **DPLR**（Deep Potential + 长程电荷位点）；
- **CACE+LES**（Cartesian ACE + Latent Ewald Summation）；
- **MTP+QRd**（固定类型依赖电荷 + 电荷守恒再分配，见 Korogod 等 2026 JCP）。

这些工作表明：**短程 MLIP + 长程库仑** 是可行路线，但固定电荷模型在环境变化剧烈时精度有限。本文进一步让**电荷随局域原子环境变化**，并给出**仅依赖 MLIP 即可构造 NAC** 的方案。

---

## 03. 方法概览：势函数、训练与 NAC 构造

### 3.1 总能量分解

总能量写为短程与静电两部分之和：

$$
E_{\text{long}}(\mathbf x, \theta, \mathbf a) = E_{\text{short}}(\mathbf x, \theta) + E_{\text{elec}}(\mathbf x, \mathbf a)
$$

**短程 MTP 能量**（原文式 1–2）：

$$
E_{\text{short}}(\mathbf x, \theta) = \sum_{i=1}^{N} V(n_i, \theta), \qquad
V(n_i, \theta) = \zeta_{z_i} + \sum_\alpha \xi_\alpha B_\alpha(n_i, \hat C)
$$

- $n_i = \{\mathbf r_{ij}, z_i, z_j\}$：以原子 $i$ 为中心、截断半径 $R_{\text{cut}}$ 内的局域环境；
- $B_\alpha$：由 moment tensor 描述符收缩得到的基函数，层级由 `lev_MTP` 控制；
- $\theta = (\zeta, \xi, \hat C)$：可训练参数。

**静电能量**为点电荷库仑和，周期体系用 **Ewald 求和** [8]：

$$
E_{\text{elec}} = \sum_{j<i} \frac{q_i q_j}{r_{ij}}
$$

### 3.2 三种电荷赋值方案

**QRd**（固定类型 + 电荷守恒，原文式 6）：

$$
q_i(z, \mathbf b, \mathbf s) = b_{z_i} + s_{z_i}\,\frac{Q_{\text{total}} - \sum_j b_{z_j}}{\sum_j s_{z_j}}
$$

$Q_{\text{total}}$ 为体系总电荷；$b,s$ 为可训练向量。**缺陷**：$q_i$ 不随局域几何变化，无法描述氢键/取向依赖极化。

**EDQ**（环境依赖，原文式 7）：

$$
q_i = V(n_i, \mathbf a)
$$

电荷由**另一个**（通常较低层级的）MTP 预测。**缺陷**：$\sum_i q_i$ 一般不等于 $Q_{\text{total}}$，仅适用于真空分子。

**EDQRd**（环境依赖 + 守恒，原文式 8）——**周期体系的主力**：

$$
q_i(\mathbf x) = V(n_i, \mathbf p) + s_{z_i}\,\frac{Q_{\text{total}} - \sum_j V(n_j, \mathbf p)}{\sum_j s_{z_j}}
$$

即用环境依赖的 $V(n_i,\mathbf p)$ 替换 QRd 中的 $b_{z_i}$，再以 $s_{z_i}$ 做全局再分配，保证总电荷守恒。

### 3.3 训练：损失函数与两阶段拟合

对 $K$ 个 DFT 参考构型，最小化能量、力、应力的加权平方误差（原文式 9）：

$$
\mathcal L(\Omega) = \sum_{k=1}^{K} \left[
w_e \big(E(\mathbf x^{(k)}, \Omega) - E_{\text{DFT}}^{(k)}\big)^2
+ w_f \sum_{i,l} \big(F_{i,l}(\mathbf x^{(k)}, \Omega) - F_{\text{DFT},i,l}^{(k)}\big)^2
+ w_s \sum_{a,b} \big(\sigma_{a,b}(\mathbf x^{(k)}, \Omega) - \sigma_{\text{DFT},a,b}^{(k)}\big)^2
\right]
$$

- $F_{i,l} = -\partial E / \partial r_{i,l}$：力，由自动微分得到；
- $\sigma_{a,b}$：应力张量，同样对应变求导；
- $w_e, w_f, w_s$：权重；NaCl 训练时 $w_s > 0$ 以保证晶格常数/密度正确。

优化器：**BFGS**。

**两阶段 EDQRd 训练**（NaCl / PbTiO₃）：

1. **阶段一**：拟合 **MTP+EDQ**（2000 步 BFGS），同时优化短程参数 $\theta$ 与电荷参数 $\mathbf p$；
2. **阶段二**：以阶段一参数为初值，拟合 **MTP+EDQRd**（同一损失函数），引入 $s_{z_i}$ 再分配项。

这样电荷预测网络先学到合理的局域电荷，再施加守恒约束。

### 3.4 主动学习选点（NaCl）

NaCl 训练集（304 构型）并非随机采样，而是 MD 中 **MTP 主动学习** [6]：

1. 用 maxvol 算法 [9] 从初始训练集选出 $m$ 个**几何上最线性无关**的构型，构成 $m\times m$ 矩阵 $A$；
2. MD 每一步计算**外推等级**：

$$
\gamma = \max_{1\le j\le m} |c_j|, \qquad
\mathbf c = \left(\frac{\partial E_{\text{MTP}}}{\partial \theta_1}, \ldots, \frac{\partial E_{\text{MTP}}}{\partial \theta_m}\right) A^{-1}
$$

3. 若 $\gamma < \gamma_{\text{th}}$：当前势可靠，继续 MD；
4. 若 $\gamma_{\text{th}} \le \gamma \le \Gamma_{\text{th}}$：构型进入**预选集**；
5. 若 $\gamma > \Gamma_{\text{th}}$：**终止** MD，强制 DFT 计算该构型；
6. 将新 DFT 构型加入训练集，重训 MTP，更新 $A$，重启 MD。

**目的**：用最少 DFT 调用覆盖 MD 轨迹上的「困难」构型，为后续 EDQRd 提供高质量训练数据。PbTiO₃ 使用文献 [7] 的 SCAN-DFT 数据集（3545 训练 + 600 验证）。

### 3.5 核心方法：仅用 MLIP 构造 NAC

对**各向同性**材料（如 NaCl），当 $\hat\varepsilon_\infty = \varepsilon_\infty \mathbf I$ 时，标准 NAC 项中的 $Z^{*}$ 与 $\varepsilon_\infty$ 可以合并。作者利用 Zhong 等 [11] 的极化定义，引入**与 $\varepsilon_\infty$ 无关的标度 Born 电荷** $Z^0$，使得 NAC 最终可写为（式 17，详见 §05）——右侧**完全由 MLIP 预测的原子电荷及其对坐标的导数**确定，无需额外 DFPT 步骤。

---

## 04. LO-TO 分裂的数学原理

本节从声子理论出发，说明 LO-TO 分裂**为什么出现**、**在公式里长什么样**，以及**为什么短程 MLIP 天然会失败**。

### 4.1 声子本征值问题

对含 $N$ 个原子的晶体，在波矢 $\mathbf q$ 处，第 $s$ 支声子频率 $\omega_{s,\mathbf q}$ 由广义本征值问题给出：

$$
\sum_{\kappa\beta} D_{\alpha\kappa,\,\beta\kappa'}(\mathbf q)\, e_{\beta\kappa',s}(\mathbf q) = \omega_{s,\mathbf q}^2\, e_{\alpha\kappa,s}(\mathbf q)
$$

动力学矩阵 $D$ 由力常数 $\Phi$ 傅里叶变换得到：

$$
D_{\alpha\kappa,\,\beta\kappa'}(\mathbf q) = \frac{1}{\sqrt{m_\kappa m_{\kappa'}}} \sum_{\mathbf R} \Phi_{\alpha\beta}(0\kappa;\, \mathbf R\kappa')\, e^{i\mathbf q\cdot\mathbf R}
$$

其中

$$
\Phi_{\alpha\beta}(0\kappa;\, \mathbf R\kappa') = \frac{\partial^2 E}{\partial u_{\alpha\kappa}(0)\,\partial u_{\beta\kappa'}(\mathbf R)}
$$

$m_\kappa$ 为原子质量，$u$ 为位移。

**关键**：非极性材料里 $\Phi$ 随距离快速衰减，$D(\mathbf q)$ 在 $\mathbf q \to 0$ 解析。极性材料里长程库仑使 $\Phi$ 按 $1/r^3$ 衰减 → $D(\mathbf q)$ 在 $\mathbf q \to 0$ 出现**非解析性** → 光学支在 Γ 点**频率不连续**，这就是 LO-TO 分裂的数学根源。

### 4.2 TO 与 LO：宏观电场差在哪？

在 $\mathbf q \to 0$ 的极限下，光学支对应晶格的整体极化振动：

| 模式 | 极化 vs 传播 | 宏观电场 |
|------|--------------|----------|
| **TO**（横光学） | 极化 $\perp$ 传播方向 $\mathbf q$ | 可被抵消，$\mathbf E \to 0$ |
| **LO**（纵光学） | 极化 $\parallel$ 传播方向 $\mathbf q$ | 保留退极化场，$\mathbf E \neq 0$ |

TO 模式「感受不到」长程宏观电场，LO 模式则额外受到**退极化场**（depolarization field）的恢复力 → **$\omega_{\text{LO}} > \omega_{\text{TO}}$**。在声子色散图上，表现为光学支在 **Γ 点** 出现**频率跳跃**（不连续），即 LO-TO 分裂。

对各向同性极性晶体（NaCl），**Lyddane–Sachs–Teller（LST）关系**把分裂与介电常数联系起来：

$$
\frac{\omega_{\text{LO}}^2}{\omega_{\text{TO}}^2} = \frac{\varepsilon_0}{\varepsilon_\infty}
$$

NaCl 实验 $\varepsilon_0/\varepsilon_\infty \approx 2.53$，意味着 LO 频率显著高于 TO——不是拟合误差小就能自动出现，必须在 Γ 点**显式处理长程偶极耦合**。

---

## 05. NAC 推导：从 DFPT 标准式到「只用 MLIP 电荷」

这是全文最核心的公式链条。Phonopy 等工具在极性材料声子计算中，会在 $\mathbf q \to 0$ 对力常数加入 **非解析修正（NAC）**。

### 5.1 标准 NAC 形式（Gonze 等）

偶极–偶极长程项在力常数中写为（原文式 13–14）：

$$
\Phi^{\text{dd}}_{\alpha\beta}(0\kappa;\, j\kappa') =
\sum_{\alpha'\beta'} \frac{Z^*_{\kappa,\alpha\alpha'}\, Z^*_{\kappa',\beta\beta'}}{\varepsilon_\infty}
\left(\frac{\delta_{\alpha'\beta'}}{|\mathbf d|^3} - \frac{3 d_{\alpha'} d_{\beta'}}{|\mathbf d|^5}\right)
$$

其中：

- $Z^{*}_{\kappa,\alpha\beta} = \partial P_\alpha / \partial u_{\beta\kappa}$：**Born 有效电荷（BEC）**，描述原子 $\kappa$ 在 $\beta$ 方向位移时 $\alpha$ 方向极化的响应；
- $\varepsilon_\infty$：高频介电常数（各向同性时 $\hat\varepsilon_\infty = \varepsilon_\infty \mathbf I$）；
- $\mathbf d$：原胞内 $\kappa$ 原子与第 $j$ 个晶胞中 $\kappa'$ 原子的相对位置。

分子形状因子 $\delta_{\alpha'\beta'}/\lVert \mathbf d \rVert^3 - 3d_{\alpha'}d_{\beta'}/\lVert \mathbf d \rVert^5$ 是经典点偶极相互作用张量的特征形式。

在 Phonopy 工作流中：

1. 由 MLIP/DFT 计算**解析部分**力常数 $\Phi^{\text{short}}$（有限超胞）；
2. 在 $\Gamma$ 点叠加 $\Phi^{\text{dd}}$（NAC）；
3. 对角化得到含 LO-TO 分裂的声子色散。

**传统瓶颈**：步骤 2 需要单独做 DFPT 或实验得到 $Z^{*}$ 和 $\varepsilon_\infty$。Korogod 等的贡献是证明——对 EDQRd 这类长程 MLIP，$Z^{*}$ 与 $\varepsilon_\infty$ 的组合可以从**同一套可微电荷模型**中一次性提取。

### 5.2 引入与 $\varepsilon_\infty$ 无关的标度 Born 电荷 $Z^0$

沿用 Zhong 等 [11] 的极化定义（原文式 15）：

$$
P_\alpha = \sqrt{\varepsilon_\infty}\, P^0_\alpha, \qquad
P^0_\alpha = \frac{1}{2\pi i}\sum_\beta R_{\alpha\beta} \sum_{j=1}^{N} q_j \exp(2\pi i \rho_{j,\beta})
$$

- $q_j$：MLIP（EDQRd）预测的原子电荷；
- $\mathbf R$：晶胞矩阵；
- $\rho_j$：原子 $j$ 的约化坐标。

**注意**：$P^0$ 只依赖电荷分布，**不含** $\varepsilon_\infty$——这是后续消去 $\varepsilon_\infty$ 的支点。

Born 电荷写为（原文式 16）：

$$
Z^*_{l,\alpha\beta} = \sqrt{\varepsilon_\infty}\, Z^0_{l,\alpha\beta}, \qquad
Z^0_{l,\alpha\beta} = \mathrm{Re}\!\left[\exp(-2\pi i \rho_{l,\alpha})\,\frac{\partial P^0_\alpha}{\partial r_{l,\beta}}\right]
$$

$Z^0$ 是**标度 Born 电荷（scaled BEC）**，完全由 MLIP 电荷 $q_j(\mathbf x)$ 及其对坐标的导数确定。

### 5.3 $\varepsilon_\infty$ 精确相消 → 最终 NAC（原文式 17）

将 $Z^{*} = \sqrt{\varepsilon_\infty}\, Z^0$ 代入标准 NAC，$\varepsilon_\infty$ 在分子、分母中**精确相消**：

$$
\boxed{
\Phi^{\text{dd}}_{\alpha\beta}(0\kappa;\, j\kappa') =
\sum_{\alpha'\beta'} Z^0_{\kappa,\alpha\alpha'}\, Z^0_{\kappa',\beta\beta'}
\left(\frac{\delta_{\alpha'\beta'}}{|\mathbf d|^3} - \frac{3 d_{\alpha'} d_{\beta'}}{|\mathbf d|^5}\right)
}
$$

**物理含义**：

- NAC 所需的「有效偶极耦合强度」由**电荷对位移的响应**（$Z^0$）完全决定；
- $\varepsilon_\infty$ 只是中间变量，**不必单独输入**；
- 对 EDQRd 势，$q_j = q_j(\mathbf x)$ 可微 → $Z^0$ 可通过自动微分从 $P^0(\mathbf x)$ 对坐标求导得到。

### 5.4 声子计算完整流程

对 NaCl，完整流程如下：

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 训练好的 MTP+EDQRd 势                                     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 有限超胞（如 3×3×3，216 原子）位移法 → 力常数 Φ_short    │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 由 EDQRd 电荷计算 P⁰ → 求导得 Z⁰ → 组装 Φ_dd（式 17）    │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Phonopy：D(q) = FT(Φ_short + Φ_dd)，对角化 → ω_s(q)      │
└─────────────────────────────────────────────────────────────┘
```

**BEC 超胞**：计算 $Z^0$ 时用更大超胞（NaCl：$10\times10\times10$，8000 原子）减小有限尺寸效应；力常数仍用较小超胞（$3\times3\times3$，216 原子）。

**ensemble**：训练 5 个独立 MTP+EDQRd，声子谱取平均，阴影为 1σ 置信区间。

### 5.5 为什么 MTP / MTP+EDQRd（无 NAC）在 Γ 点会失败？

| 模型 | 长程静电 | Γ 点行为 |
|------|----------|----------|
| **MTP** | 无 | $D(\mathbf q)$ 解析 → **无分裂**；还可能错误合并分支 |
| **MTP+EDQRd** | 有库仑，但 Phonopy 默认只用有限超胞 $\Phi_{\text{short}}$ | 长程 $1/r^3$ 在有限超胞截断 → TO 可近似正确，**LO 支仍缺失** |
| **MTP+EDQRd+NAC** | 库仑 + 显式 $\Phi^{\text{dd}}$ | $Z^0$ 由 EDQRd 导出 → **复现 LO-TO 分裂** |

分裂不是「能量/力 RMSE 小」就能自动出现——必须在 Γ 点**显式注入** $\Phi^{\text{dd}}$。

![Figure 2：NaCl 声子谱——MTP、MTP+EDQRd、MTP+EDQRd+NAC 与 DFT 对比](/img/posts/korogod-2026-loto-mlip/fig2-nacl-phonon-spectra.png)

*图 2 | NaCl 声子色散（ensemble 平均，阴影为 1σ）。**(a)** 纯 MTP：Γ 点**无 LO-TO 分裂**，U–W–K 路径部分支合并，TO 频率偏高。**(b)** MTP+EDQRd：Γ 点 TO 频率正确，但仍缺分裂；X、L 点改善。**(c)** MTP+EDQRd+NAC：全路径与 DFT 极好一致，LO-TO 分裂略低估——**只有加 NAC 才闭合 Γ 点**。**(d)** 三种 MLIP 叠加对比：仅 MTP+EDQRd+NAC 在 Γ 点给出最高光学支（LO）。DFT 参考的 NAC 用文献实验 BEC 与 $\varepsilon_\infty$；MLIP 的 NAC **完全由 EDQRd 电荷导出**。*

| 子图 | 对比 | 主要问题 / 结论 |
|------|------|-----------------|
| (a) | MTP vs DFT | Γ 点**无 LO-TO 分裂**；U-W-K 路径上部分支合并；TO 频率偏高 |
| (b) | MTP+EDQRd vs DFT | Γ 点 TO 频率正确，但仍**缺 LO-TO 分裂**；X、L 点改善 |
| (c) | MTP+EDQRd+NAC vs DFT | **全路径与 DFT 极好一致**；LO-TO 分裂略低估 |
| (d) | 三种 MLIP 叠加 | 仅 **MTP+EDQRd+NAC** 在 Γ 点达到最高光学支频率 |

---

## 06. 介电常数：偶极矩涨落与 LST 自洽

基于电子连续介质 MD 理论 [12]，由偶极矩涨落估计静态/高频介电常数之比（原文式 18–19）：

$$
\frac{\varepsilon_0}{\varepsilon_\infty} = 1 + \frac{4\pi}{3 V \kappa T}\left(\langle M^2 \rangle - \langle M \rangle^2\right)
$$

总偶极矩：

$$
M = \sum_{i=1}^{N} q_i\, \mathbf r_i^{u}
$$

$\mathbf r_i^{u}$ 为 **unwrapped** 坐标（跨越周期边界时不折叠）。NaCl 5832 原子、290 K、2 ns NVT-MD 得到 **$2.71 \pm 0.07$**，与 LST 关系及实验 **2.53** 一致——说明 EDQRd 电荷不仅用于 NAC，也能正确描述**宏观极化涨落**。

![Figure 3：NaCl 的 ε₀/ε∞ 随 MD 时间收敛](/img/posts/korogod-2026-loto-mlip/fig3-nacl-dielectric-ratio.png)

*图 3 | MTP+EDQRd 偶极矩涨落给出的 $\varepsilon_0/\varepsilon_\infty$ 随模拟时间变化（阴影 1σ；黑线实验值 2.53 [19]）。约 **0.5 ns** 后均值与标准差收敛；最终 **2.71 ± 0.07** 仅比实验高约 7%，与 §04 的 LST 关系及 §05 的 NAC 声子结果**自洽**。*

---

## 07. 主要数值结果（读图）

### 7.1 各算例参数对照

| 体系 | 短程 MTP | 电荷 MTP | 训练数据 | NAC |
|------|----------|----------|----------|-----|
| 有机二聚体 | lev 8 | EDQ lev 2 | 312/313 构型 [3] | 不适用（非周期） |
| NaCl | lev 12 | EDQRd lev 6 | 304 构型（主动学习） | 式 17，$Z^0$ 来自 EDQRd |
| PbTiO₃ | lev 20 | EDQRd lev 16 | 3545 训练 + 600 验证 [7] | 同式 17（近似，非严格各向同性） |

### 7.2 有机二聚体：MTP+EDQ 改善长程结合曲线

体系：真空中的 $\text{CH}_3\text{COO}^- +$ 4-甲基苯酚、$\text{CH}_3\text{COO}^- +$ 4-甲基咪唑。短程 MTP 8 级，EDQ 电荷模型 2 级 MTP，各训练 5 个 ensemble 成员。

![Table I：两个有机二聚体体系的能量与力拟合 RMSE](/img/posts/korogod-2026-loto-mlip/table-i-fitting-errors-dimers.png)

*表 I | **MTP+EDQ** 能量 RMSE 比 MTP / MTP+QRd **降低 3–9 倍**，力 RMSE 约降 30%。固定电荷 QRd 几乎不能改善能量误差。*

| 体系 | 模型 | 能量 RMSE (meV/atom) | 力 RMSE (meV/Å) |
|------|------|----------------------|-----------------|
| 乙酸根 + 4-甲基苯酚 | MTP | 0.45 ± 0.02 | 22.0 ± 1.9 |
| | MTP+QRd | 0.52 ± 0.16 | 23 ± 3 |
| | **MTP+EDQ** | **0.13 ± 0.05** | **14.8 ± 0.8** |
| 乙酸根 + 4-甲基咪唑 | MTP | 0.731 ± 0.009 | 16.4 ± 0.8 |
| | MTP+QRd | 0.719 ± 0.017 | 15.9 ± 0.4 |
| | **MTP+EDQ** | **0.078 ± 0.005** | **10.9 ± 0.2** |

![Figure 1：DFT 与 MTP / MTP+QRd / MTP+EDQ 的结合曲线对比](/img/posts/korogod-2026-loto-mlip/fig1-binding-curves-dimers.png)

*图 1 | 分子间距离–结合能曲线（ensemble 平均）。(a) 4-甲基苯酚、(b) 4-甲基咪唑。**MTP** 与 **MTP+QRd**（紫、蓝虚线）在 6–12 Å 出现非物理振荡，长程平台偏离 DFT（黑实线）。**MTP+EDQ**（红虚线）在大、小分离距均与 DFT 重合，尤其正确再现咪唑体系的**单调上升**长程行为——**电荷必须随局域环境变化**。*

这说明：**电荷必须随局域化学环境变化**，才能描述阴离子-芳香体系中的方向依赖极化与长程静电。

### 7.3 NaCl 晶体：EDQRd 降误差 + NAC 复现 LO-TO

304 构型（216 原子 NPT-MD + MTP 主动学习）；CP2K GTH-PBE 参考。代表模型：**MTP-12** vs **MTP-12+EDQRd(MTP-6)**。

![Table II：NaCl 的能量、力、应力拟合 RMSE](/img/posts/korogod-2026-loto-mlip/table-ii-fitting-errors-nacl.png)

*表 II | EDQRd 使能量/应力 RMSE 约降 **3 倍**，力 RMSE 约降 **5 倍**；单纯提高 MTP 层级（12→16）收益有限。两种 EDQRd 变体均显著优于纯 MTP。*

| 模型 | 能量 RMSE (×10⁻⁵ eV/atom) | 力 RMSE (meV/Å) | 应力 RMSE (eV) |
|------|---------------------------|-----------------|----------------|
| MTP-12 | 12.1 ± 0.5 | 12.9 ± 0.2 | 0.28 ± 0.04 |
| MTP-16 | 11.6 ± 0.4 | 12.17 ± 0.10 | 0.27 ± 0.04 |
| **MTP-12+EDQRd(MTP-6)** | **3.6 ± 0.5** | **2.69 ± 0.03** | **0.091 ± 0.011** |
| **MTP-16+EDQRd(MTP-10)** | **2.5 ± 0.5** | **2.2 ± 0.4** | **0.066 ± 0.014** |

**宏观性质**：晶格常数 **5.66 Å**（MTP 与 EDQRd 均与 DFT 一致）；300 K 密度 **2.05 g/cm³**（实验 2.156 g/cm³，偏差 ~5%）。声子与 LO-TO 见 **§05.5 图 2**；介电比值见 **§06 图 3**。

### 7.4 PbTiO₃：NAC 向单轴晶体的延伸

四方相 PbTiO₃ 的 ordinary / extraordinary $\varepsilon_\infty$ 相差 ~10%，式 (17) **严格推导仅对各向同性成立**。作者仍用同一 NAC 流程试探：MTP-20 + EDQRd(MTP-16)，SCAN 训练集 [7]；验证误差 0.6 meV/atom（能量）、91.8 meV/Å（力），接近 CACE+LES [11] 的 0.4 / 79.8 meV/Å。

![Figure 4：PbTiO₃ 声子谱——MTP+EDQRd 与 MTP+EDQRd+NAC 对比 DFT](/img/posts/korogod-2026-loto-mlip/fig4-pbtio3-phonon-spectrum.png)

*图 4 | 四方 PbTiO₃ 声子色散（DFT 参考为 PBEsol）。(a) **MTP+EDQRd**：远离 Γ 点总体合理，Γ 点**无法给出 LO-TO 分裂**（类比 NaCl 图 2b）。(b) **MTP+EDQRd+NAC**：Γ 点附近**急剧改善**，分裂趋势与 DFT 一致——即便材料**非严格各向同性**，EDQRd 电荷构造的 NAC 仍是实用近似。*

---

## 08. 结论与意义

1. **EDQ / EDQRd** 以简洁的库仑点电荷形式纳入**环境依赖**长程静电，可与任意短程 MLIP 组合；本文以 **MTP** 为例。
2. 对分子二聚体，**MTP+EDQ** 在能量/力误差和**长程结合曲线**上均大幅优于 MTP+QRd。
3. 对 NaCl，**MTP+EDQRd** 显著降低拟合误差，并在加入 **NAC** 后**首次仅用 MLIP 训练数据**（能量、力、应力）即可复现 LO-TO 分裂；偶极涨落给出的 $\varepsilon_0/\varepsilon_\infty$ 与实验一致。
4. 对 PbTiO₃，同一 NAC 流程在**非严格各向同性**情况下仍能大幅改善 Γ 点声子色散，验证方法的**更广适用性**。
5. **未来工作**：为长程模型开发专用主动学习；计算**各向异性**材料的完整介电张量。

---

## 09. 和「高精度谱学」的关系

若你关心 **LO-TO 分裂、极性声子、介电响应、IR 活性**，本文的价值在于：

1. **统一框架**：长程库仑 + 环境依赖电荷 + 声子 NAC 在同一 MLIP 里闭合，不必额外 DFPT 流水线；
2. **可微链条**：$q(\mathbf x) \to P^0 \to Z^0 \to \Phi^{\text{dd}}$ 全程可自动微分，为后续与 Kim 式电响应、PFT 式 Hessian 监督**拼接**留接口；
3. **谱学预测路径**：MD 偶极涨落 → $\varepsilon_0/\varepsilon_\infty$；Phonopy + NAC → LO/TO 频率 → 可与 LST 关系交叉验证；
4. **高通量潜力**：降低对额外 DFPT（BEC、$\varepsilon_\infty$）的依赖，使**大规模 MD + 声子 + 介电**流水线更连贯，为极性晶体、铁电体（如 PbTiO₃）的高通量谱学预测提供可复现路径。

和 **PFT（直接监督 Hessian）** 的对比：PFT 让短程 MLIP 的**解析力常数**更准；Korogod 等则把**长程 $1/r^3$ 部分**通过 NAC 显式补全——二者解决的是声子误差的不同来源，理论上可叠加在同一总能量 $E_{\text{long}}$ 上。

**算法逻辑一句话**：EDQRd 把「长程库仑」和「Born 电荷信息」统一在**同一套可微电荷模型**里；NAC 只是把这份信息在 Γ 点**显式注入**动力学矩阵，从而闭合 LO-TO 分裂的数学链条。

---

## 10. 局限与未来

- **EDQ 不守恒总电荷**，只适用于真空分子；周期体系必须用 **EDQRd**。
- NAC 严格推导假设**各向同性** $\hat\varepsilon_\infty = \varepsilon_\infty \mathbf I$；PbTiO₃ 等单轴材料目前是**实用近似**。
- 主动学习目前基于短程 MTP，作者计划为长程模型开发专用主动学习；各向异性介电张量计算是未来方向。

---

## 11. 一句话总结

> Korogod 等把「环境依赖点电荷 + 电荷守恒」与「Γ 点 NAC」连成一条可微公式链：  
> **$q(\mathbf x) \to Z^0 \to \Phi^{\text{dd}}$**，不必单独跑 DFPT，就能从 EFS 训练的 MLIP 复现 LO-TO 分裂与介电常数。

---

## 延伸阅读（站内）

- [一篇讲透：为什么“可极化多极矩”能让材料力场更懂电学？](/2026/05/16/Kim2026-可极化多极矩长程电静学/)
- [速览笔记：PFT 如何让 MLIP 真正「听懂」声子？](/2026/05/20/Koker2026-PFT声子微调/)
- [红外光谱、拉曼光谱与声子谱：原理、数学与电池研究中的应用](/2026/05/18/红外拉曼声子谱-原理数学与电池应用/)
- [DPA4 短程有多强？SOG 长程能否接棒 MACE-POLAR-1？](/2026/05/23/DPA4-SOG-长程能否接棒MACE-POLAR-1/)


## 参考

1. Korogod D, Shapeev A V, Novikov I S. **Long-range machine-learning potentials with environment-dependent charges enable predicting LO-TO splitting and dielectric constants**. arXiv:2603.06396, 2026. https://arxiv.org/abs/2603.06396  
2. Zhong P, Kim D, King D S, Cheng B. **Machine learning interatomic potential can infer electrical response**. arXiv:2504.05169, 2025.  
3. Gonze X, Charlier J-C, Allan D, Teter M. Interatomic force constants from first principles: the case of $\alpha$-quartz. *Phys. Rev. B* **50**, 13035 (1994).  
4. Korogod D, et al. Incorporating coulomb interactions with fixed charges in moment tensor potentials and equivariant tensor network potentials. *J. Chem. Phys.* **164**, 064120 (2026).

**文献信息**：Dmitry Korogod, Alexander V. Shapeev, Ivan S. Novikov · Skoltech, MIPT, Digital Materials LLC, HSE University · 2026-03-09
