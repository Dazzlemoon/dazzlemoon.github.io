---
layout:     post
title:      极性材料长程静电三路线对比：Monacelli 插件式 LO-TO、Korogod 可学电荷 NAC 与 Kim 可极化多极矩光谱
subtitle:   2026 三篇 MLIP 长程静电工作横向解读：共同物理核 (★) q→0 非解析 LO-TO；Monacelli 固定 Z/ε 偶极插件、Korogod EDQRd+NAC、Kim 多极+诱导响应+MD IR/Raman；§六详论 ε_∞/ε_e/Z 构型依赖；谱学任务选型与能否互相替代。
date:       2026-06-07
author:     天月将白
header-img: img/post-bg-desk.jpg
catalog:    true
tags:
    - LO-TO
    - MLIP
    - Monacelli
    - Korogod
    - Kim
    - 长程静电
    - 高精度谱学
    - 谱学选型
---

# 极性材料长程静电三路线对比：Monacelli、Korogod 与 Kim

> **Monacelli & Marzari (2026)** · *Electrostatic interactions in atomistic and machine-learned potentials for polar materials* · **Phys. Rev. B** 113, 094101 · [DOI 10.1103/7ygl-8db2](https://doi.org/10.1103/7ygl-8db2)  
> **Korogod, Shapeev & Novikov (2026)** · *Long-range MLIP with environment-dependent charges enable predicting LO-TO splitting and dielectric constants* · arXiv:2603.06396  
> **Kim, King, Park et al. (2026)** · *Polarizable atomic multipoles for learning long-range electrostatics* · arXiv:2605.05746  

**本地长文解读**：Monacelli → `Monacelli的LOTO文章/2026-monacelli-loto-electrostatic-mlip-blog.md`；Korogod → `LOTO文章/Korogod-2026-LO-TO-MLIP-文章介绍.md`；Kim → `Kim-文章介绍/2026kim-polarizable-multipoles-wechat.md`。

---

## 一、为什么要放在一起比？

2026 年三条「长程静电 + MLIP」路线几乎同时出现，解决**同一类痛点**——短程 MLIP 在极性体系中：

- $\Gamma$ 点附近 **LO/TO 光学支合并**；
- 长程库仑结合能、介电响应、IR/Raman 失真；
- 训练数据在有限胞 PBC 下**不含**完整 $1/r^3$ 偶极–偶极项。

但三者的**数学形式、训练策略、主交付物**截然不同。本文在**同一 LO-TO 非解析核**下对照公式，并单独讨论 **$\varepsilon\_\infty$、$\varepsilon\_e$、$Z^{\ast}$ 是否随构型/相变变化**——这是选型时最容易混淆的一点。

**结论先行**：

> 三者在 **$\mathbf q\to 0$ 偶极–偶极非解析核 (★)** 上**同宗**，在**实现与交付**上**异路**——**不能**互相简单替代，也**不宜**在同一体系无脑叠加两套长程项。

---

## 二、一句话对照表

| | **Monacelli 2026** | **Korogod 2026** | **Kim 2026** |
|---|---|---|---|
| **首要目标** | **插件式**补长程 $E,\mathbf f,\boldsymbol\sigma$ | **训练**长程 MLIP，从电荷**导出 NAC** | **训练**可极化多极矩，**预测 IR/Raman** |
| **长程变量** | 固定 **$Z^{\ast}$、$\varepsilon\_\infty$**（DFPT） | **可学** **$q\_i(\mathbf x)$**（EDQRd） | **可学** $q,u,Q$ + 诱导 $\Delta q,\Delta u$ |
| **长程能量** | **偶极双线性** $Z\Delta R$ + $k$-Ewald | **单极库仑** $q\_iq\_j/r$ + Ewald | **多极 Ewald** $\|S(\mathbf k)\|^2/k^2$ |
| **LO-TO** | Hessian 式 (19) **内建** | 有限超胞不够 → **Phonopy + NAC** | **非主交付**；MD 自相关谱 |
| **重训短程势** | **否** | **是**（EFS） | **是**（EFS） |
| **Benchmark** | BaTiO$\_3$ 声子 | NaCl LO-TO + $\varepsilon\_0/\varepsilon\_\infty$ | bulk water / MAPbI$\_3$ IR/Raman |

---

## 三、共同物理核心：$q\to 0$ 非解析项 (★)

极性绝缘体 LO/TO 分裂来自**宏观退极化场**。Gonze–Lee DFPT 标准长程力常数（Korogod 式 13，Monacelli 式 19 同源）：

$$
\Phi^{\rm dd}_{\alpha\beta}(0\kappa;\,j\kappa') =
\sum_{\alpha'\beta'} \frac{Z^*_{\kappa,\alpha\alpha'}\, Z^*_{\kappa',\beta\beta'}}{\varepsilon_\infty}
\left(\frac{\delta_{\alpha'\beta'}}{|\mathbf d|^3} - \frac{3 d_{\alpha'} d_{\beta'}}{|\mathbf d|^5}\right)
$$

傅里叶变换 **$\mathbf q\to 0$**：

$$
\lim_{\mathbf q\to 0} D_{ij}^{\alpha\beta}(\mathbf q)
= \frac{1}{\Omega}\,
\frac{Z_{j\nu\beta}\, q_\nu q_\mu Z_{i\mu\alpha}}
{\mathbf q\cdot\boldsymbol\varepsilon_\infty\cdot\mathbf q}
\tag{★}
$$

**含义**：动力学矩阵 $D(\mathbf q)$ 在 $\Gamma$ 点对 **$\mathbf q/\lvert \mathbf q \rvert$ 不连续** → 光学支频率跳跃 = LO-TO 分裂。  
**短程 MLIP 失败机制**：有限超胞位移法只得**解析** $\Phi^{\rm short}$，$\Gamma$ 点 **LO 支被抹平**。

---

## 四、Monacelli：固定 $Z^{\ast},\varepsilon\_\infty$ 的偶极双线性插件

### 4.1 核心公式

原子偶极（Born 有效电荷，现代极化理论）：

$$
\mu_{i\alpha} = \sum_\beta Z_{i\alpha\beta}\,(R_{i\beta}-\bar R_{i\beta})
$$

长程能量（**非** $q\_iq\_j/r$，PRB 式 (9)）：

$$
E_{\rm LR}(\mathbf R) = \frac{1}{2\Omega}\sum_{ij\alpha\beta\mu\nu}
\Delta R_{i\alpha}\,\Delta R_{j\mu}\, Z_{i\beta\alpha} Z_{j\nu\mu}
\sum_{\mathbf k\neq 0}
\frac{k_\beta k_\nu\, e^{-\eta^2 k^2/2}}
{\mathbf k\cdot\boldsymbol\varepsilon_\infty\cdot\mathbf k}
\, e^{-i\mathbf k\cdot(\mathbf R_j-\mathbf R_i)}
$$

- **$\eta$**：高斯 smearing；下限 = 训练集最大原子对距离（含 PBC），以保证**不重训**短程 GAP；
- **总能量**：$E\_{\rm tot}=E\_{\rm SR}+E\_{\rm LR}$。

### 4.2 LO-TO 如何出现

对 $E\_{\rm LR}$ 求 Hessian → Monacelli 式 (17)(18)，$q\to 0$ 即 (★)（式 (19)）。**不必** Phonopy 手工 NAC；力、应力（式 (11)(12)–(15)）一并自洽。

### 4.3 显式假设（Discussion）

> dielectric tensor and effective charges are **independent of atomic coordinates**

即 **$Z$、$\varepsilon\_\infty$ 在参考结构 $\bar{\mathbf R}$ 上 DFPT 算一次后全程冻结**；模拟中只有 $\Delta R$ 变。跨相（立方 $Z,\varepsilon$ 用于四方 BaTiO$\_3$，Fig. 2）是**可迁移性测试**，非模型内禀自由度。

---

## 五、Korogod：环境依赖 $q\_i(\mathbf x)$ + 声子后处理 NAC

### 5.1 训练态能量

$$
E_{\rm tot} = E_{\rm short}^{\rm MTP}(\mathbf x,\theta) + \mathrm{Ewald}\!\left[\sum_{j<i}\frac{q_i q_j}{r_{ij}}\right]
$$

**EDQRd 电荷**（周期体系主力）：

$$
q_i(\mathbf x) = V(n_i,\mathbf p) + s_{z_i}\,
\frac{Q_{\rm total}-\sum_j V(n_j,\mathbf p)}{\sum_j s_{z_j}}
$$

NaCl 训练 304 构型来自 **MTP 主动学习 MD** → $q(\mathbf x)$ 已在**多种热畸变形态**上拟合。

### 5.2 LO-TO：两步，第二步不可省

| 步骤 | 内容 | $\Gamma$ 点 |
|------|------|-------------|
| 1 | MTP+EDQRd MD 能量含长程库仑 | 有限超胞 $\Phi\_{\rm short}$ 仍**截断** $1/r^3$ |
| 2 | 从 $q(\mathbf x)$ 得 $Z^0$，Phonopy **加** $\Phi^{\rm dd}$ | **LO-TO 出现** |

**NAC 链**（Korogod 式 15–17）：

$$
P^0_\alpha = \frac{1}{2\pi i}\sum_\beta R_{\alpha\beta}\sum_j q_j(\mathbf x)\, e^{2\pi i \rho_{j,\beta}}
$$

$$
Z^{*}_\ell = \sqrt{\varepsilon_\infty}\, Z^0_\ell, \qquad
Z^0_{l,\alpha\beta} = \mathrm{Re}\!\left[
e^{-2\pi i \rho_{l,\alpha}}\,\frac{\partial P^0_\alpha}{\partial r_{l,\beta}}
\right]
$$

代入标准 NAC 后 **$\varepsilon\_\infty$ 精确相消**：

$$
\Phi^{\rm dd}_{\alpha\beta} =
\sum_{\alpha'\beta'} Z^0_{\kappa,\alpha\alpha'}\, Z^0_{\kappa',\beta\beta'}
\left(\frac{\delta_{\alpha'\beta'}}{|\mathbf d|^3} - \frac{3 d_{\alpha'} d_{\beta'}}{|\mathbf d|^5}\right)
$$

**与 Monacelli 差异**：训练能量是**单极库仑**；LO-TO **不在** $E\_{\rm elec}$ 梯度里自动闭合，而靠 **phonon 后处理**；但 (★) 核相同。

### 5.3 介电验证

MD 偶极涨落（Korogod 式 18–19）得 $\varepsilon\_0/\varepsilon\_\infty = 2.71\pm 0.07$（NaCl，实验 2.53），与 LST 及 NAC 声子**自洽**。

---

## 六、Kim：可极化多极矩 + MD 振动光谱

### 6.1 长程势

screened 核 $\varphi(r)=\mathrm{erf}(r/\sqrt{2\sigma})/r$；周期 Ewald（Kim 式 14–15）：

$$
U^{\rm elec} = \frac{1}{2\varepsilon_0 V}\sum_{0<|\mathbf k|<k_c}
\frac{e^{-\sigma^2 k^2/2}}{k^2}\,|S(\mathbf k)|^2
$$

**多极结构因子**：

$$
S(\mathbf k) = \sum_i \left(
q_i^{\rm les} + i\mathbf k\cdot\mathbf u_i^{\rm les}
- \frac{1}{2}\mathbf k\cdot\mathbf Q_i^{\rm les}\cdot\mathbf k
\right) e^{i\mathbf k\cdot\mathbf r_i}
$$

**诱导响应**（非自洽，一次线性响应）：

$$
\Delta q_i = -\kappa_i \Phi(\mathbf r_i), \qquad
\Delta u_i = \boldsymbol\alpha_i\cdot\mathbf E(\mathbf r_i)
$$

$$
U = U_{\rm sr} + U^{\rm elec} + \sum_i U_i^q + \sum_i U_i^u
\quad\text{（Kim 式 22）}
$$

### 6.2 电响应反演与光谱

**BEC**（Kim 式 28）：

$$
Z^*_{i\alpha\beta} = \frac{\partial P^u_\alpha}{\partial r_{i\beta}}
+ \lim_{k\to 0}\Re\!\left[
e^{-ik r_{i\alpha}}\,\frac{\partial P^q_\alpha(k)}{\partial r_{i\beta}}
\right]
$$

**IR**（Kim 式 52）：

$$
I_{\rm IR}(\omega) \propto \int_0^T dt\,
\left\langle \mathbf J(0)\cdot\mathbf J(t)\right\rangle e^{-i\omega t},
\qquad
\mathbf J(t) = \sum_i \mathbf Z^*_i(t)\cdot\mathbf v_i(t)
$$

**Raman**（Kim 式 53–54）：

$$
R_{\rm iso}(\omega) \propto \omega^2 \int \langle \alpha(0)\alpha(t)\rangle e^{-i\omega t} dt,
\qquad
R_{\rm aniso}(\omega) \propto \omega^2 \int \langle \mathrm{Tr}[\boldsymbol\beta(0)\boldsymbol\beta(t)]\rangle e^{-i\omega t} dt
$$

Kim **原文未做** phonon dispersion / LO-TO / NAC；MAPbI$\_3$ 验证 **Raman/IR**，bulk water 为**液态**展宽带。

---

## 七、$\varepsilon\_\infty$、$\varepsilon\_e$ 与 $Z^{\ast}$：谁随「形态」变？

这是三篇最容易混为一谈之处。先分**三个层次**：

| 层次 | 符号 | 含义 |
|------|------|------|
| 宏观高频介电 | $\varepsilon\_\infty$ | 电子云屏蔽（DFPT 冻结核 / 实验） |
| Born 有效电荷 | $Z^{\ast}=\partial P/\partial u$ | 位移→极化（LO-TO 核 (★) 里的 $Z$） |
| MLIP 参数化 | $q\_i(\mathbf x)$、$\varepsilon\_e$、latent 多极 | 势函数里算长程静电的变量 |

**物理上**：$\varepsilon\_\infty$ 与 $Z^{\ast}$ **都可以**随结构、相、温度变。三篇差别是**模型允许哪一层随 $\mathbf x$ 变**。

### 7.1 Monacelli：**$Z$ 与 $\varepsilon\_\infty$ 全程冻结**

- 参考相 $\bar{\mathbf R}$ 上 DFPT **一次** → 常数 $Z$、常数 $\boldsymbol\varepsilon\_\infty$；
- 式 (9) 中只有 $\Delta R$ 进入，**系数不更新**；
- **最「硬」的谐波/单相近似**；跨相 Fig. 2 是**迁移性实验**，不是内禀 $Z(\mathbf R)$。

### 7.2 Korogod：**MD 中 $q(\mathbf x)$ 变；NAC 常取单相 $Z^0$；$\varepsilon\_\infty$ 在 LO-TO 公式里消失**

| 环节 | 是否随构型变 |
|------|-------------|
| MD / 训练 $E\_{\rm elec}$ | **是** — $q\_i(n\_i)$ 随局域环境变 |
| NAC 声子 $Z^0=\partial P^0/\partial r$ | **原则上随 $\mathbf x$**；实践在**该相平衡构型**算**一套**（NaCl 10×10×10 超胞） |
| LO-TO 公式里的 $\varepsilon\_\infty$ | **不需要输入**（相消） |
| LST 对照 $\varepsilon\_0/\varepsilon\_\infty$ | 用**实验常数** $\varepsilon\_\infty$；涨落来自**时变** $q(t)$ |

**不是**「和 Monacelli 一样全冻结」：长程库仑在 MD 里**已随热畸变适配**；声子 NAC 仍偏**单参考几何**，但 $Z^0$ 来自**同一 MLIP** 而非外部 DFPT。

### 7.3 Kim：**$\varepsilon\_e(\mathbf x)$ 显式随构型变**（有诱导偶极时）

**无诱导项**（-les / -u）：$\varepsilon\_e=\varepsilon\_\infty$ **常数**（LES 原版）。

**有诱导偶极**（-uiu 等，Kim 式 23–25）：

$$
\chi = \frac{1}{\varepsilon_0 V}\sum_i \alpha_i, \qquad
\varepsilon_e + \chi = \varepsilon_\infty, \qquad
\varepsilon_e(\mathbf x) = \frac{\varepsilon_\infty}{1+\chi_{\rm les}(\mathbf x)}
$$

- **锚点**：宏观 $\varepsilon\_\infty$ 固定（如水 **1.78** 实验值）；
- **拆分**：每个构型把 $\varepsilon\_\infty$ 分给 **$\varepsilon\_e(\mathbf x)+\chi(\mathbf x)$**，二者都变、之和恒定；
- 训练 $E/F$ 时 $\varepsilon\_e$ 吸收进 latent，Ewald 只用 $\varepsilon\_0$；**恢复物理 $Z^{\ast}$、$\alpha$ 时逐构型算 $\varepsilon\_e$**。

此外 $q,u,Q,\kappa,\alpha$ **均**为局域描述符函数 → **构型依赖最广**。

### 7.4 三篇对照（构型依赖）

| 量 | Monacelli | Korogod | Kim（-uiu） |
|----|-----------|---------|-------------|
| $\varepsilon\_\infty$ 锚点 | DFPT **固定** | NAC 中**不出现**；LST 用实验常数 | **实验/DFPT 常数**约束 |
| 势函数有效介电 | $\boldsymbol\varepsilon\_\infty$ **固定**在 $k$ 求和 | 无 $\varepsilon\_e$；屏蔽在 $q$ 里 | **$\varepsilon\_e(\mathbf x)$ 显式变** |
| 电荷/多极 | 无 $q$；$Z\Delta R$ | **$q\_i(\mathbf x)$ 变** | **$q,u,Q,\alpha$ 变** |
| $Z^{\ast}$ / LO-TO | **固定** DFPT $Z$ | **$Z^0(\mathbf x)$ 可导**；声子常用一相一套 | **$Z^{\ast}(\mathbf x)$ 反演** |
| 跨一级相变 | 分相各一套 $(Z,\varepsilon)$ | 换相重算/重训 $Z^0$ | 靠训练覆盖；无 LO-TO benchmark |

**精确表述**（纠正「Kim 独变、另两篇不变」）：

- **Monacelli**：$Z$、$\varepsilon\_\infty$ **冻在最死**；
- **Korogod**：**$q(\mathbf x)$ 在 MD 已松**；NAC 仍常**单相 $Z^0$**；
- **Kim**：**$\varepsilon\_e(\mathbf x)$ 与 $\chi(\mathbf x)$ 显式拆分**，表达力最强、假设层也最多。

---

## 八、公式地图

```
共同物理 (★):  q→0 时 D(q) ∝ Z q q Z / (q·ε·q)

Monacelli:  μ=Z·ΔR (固定 Z,ε) → E_LR 双线性 k-Ewald (9) → Hessian → (19)
            插件 GAP，不重训

Korogod:    q_i(x) 可学 → Ewald q_iq_j/r → P⁰ → Z⁰=∂P⁰/∂r → Phonopy 外加 Φ^dd
            EFS 训练；MD 与 phonon NAC 步骤分离

Kim:        q,u,Q + Δq,Δu 可学 → Ewald |S(k)|² → 反演 Z*, α
            MD 自相关 → IR/Raman；ε_e(x) 随构型（有 -iu 时）
```

---

## 九、「表达能力」分任务排序

| 维度 | 更强者 | 说明 |
|------|--------|------|
| 长程参数化自由度 | **Kim** | 单极+偶极+四极+诱导；Korogod 仅 $q$；Monacelli 固定 $Z\Delta R$ |
| 环境依赖电响应 | **Kim** | $q,u,Q,\kappa,\alpha,\varepsilon\_e$ 均可随环境变 |
| 液体 / 界面 / 无序 | **Kim** | water、MAPbI$\_3$ 多相 MD |
| IR + 各向异性 Raman | **Kim** | 唯一系统交付 $I\_{\rm IR}$、$R\_{\rm iso/aniso}$ |
| 外场 Stark 谱 | **Kim** | $E\_0=0.05$–$0.15\ \mathrm{V/Å}$ MD（Fig. 6） |
| 晶体 $\Gamma$ LO-TO 色散 | **Monacelli ≈ Korogod** | BaTiO$\_3$ / NaCl benchmark；Kim **原文未做** |
| 不重训既有短程势 | **Monacelli** | 唯一后验 ASE 插件 |
| 免 DFPT 读 $Z,\varepsilon$ | **Kim ≈ Korogod** | 从可微电荷/多极反演 |
| 变胞应力 / SSCHA | **Monacelli** | 式 (12)–(15) + ForwardDiff 应力 |

**归纳**：Kim 在**观测量种类与电响应宽度**上最广；Monacelli 在**晶体 LO-TO Hessian 内建 + 插件部署**上最专；Korogod 在**可学电荷 + NAC 闭合 LO-TO（NaCl 已验证）**上最完整。**不是单一总分排序**。

---

## 十、Kim 能否观测 LO-TO？

| 问题 | 答案 |
|------|------|
| 原文是否报告 LO-TO 色散？ | **否**（无 phonon / NAC / LO-TO 关键词） |
| 液体 water 的 IR/Raman | **无**晶体 $\Gamma$ LO-TO；展宽带 $\neq$ 色散支分裂 |
| MAPbI$\_3$ | 有 LO-TO **物理背景**，但只算 **Stokes Raman / IR**，未给 $\omega\_{\rm LO}-\omega\_{\rm TO}$ |
| 原则上势函数能否延伸？ | **可能** — 对 $U$ 求 Hessian 或 Phonopy+NAC 得 (★)；**未实现、未 benchmark** |
| MD 谱能否间接看 LO-TO？ | **间接、展宽、混叠**；不能替代 phonon dispersion 诊断 |

Kim 对 water IR 的成功说明 **$Z^{\ast}(t)$ 在光谱意义下可靠**；**能否据此得到 LO-TO 色散曲线**见 **§十二**。

---

## 十一、谱学 / 模拟任务选型

| 任务 | 推荐 |
|------|------|
| 极性晶体 LO/TO、铁电声子 | **Monacelli** 或 **Korogod** |
| LST $\omega\_{\rm LO}^2/\omega\_{\rm TO}^2=\varepsilon\_0/\varepsilon\_\infty$ | **Korogod**（NaCl 已验）；Monacelli 可交叉检验 |
| 液体 / 无序 IR | **Kim** |
| Raman（各向异性、组合带） | **Kim** |
| 保留 GAP/NNP，只补 LO-TO | **Monacelli** |
| 不想单独 DFPT | **Korogod** 或 **Kim** |
| 跨一级相变路径 | **三者都弱**；分相参数或分相训练 |

---

## 十二、Kim 的 $Z^{\ast}(t)$ 可靠，能否直接得到 LO-TO 色散？

### 12.1 Kim 原文实际验证的是什么

Kim 对 bulk water 的 benchmark 是：沿 MD 轨迹用式 (28) 得到 **$Z^{\ast}\_i(t)$**，代入

$$
\mathbf J(t)=\sum_i \mathbf Z^*_i(t)\cdot\mathbf v_i(t), \qquad
I_{\rm IR}(\omega)\propto \mathrm{FT}\big[\langle \mathbf J(0)\cdot\mathbf J(t)\rangle\big]
$$

并与实验 IR 对比（Fig. 2b）。$Z^{\ast}$ 与 DFT 的 parity 可达 RMSE $\sim 0.022\ e$（MACELES-uiu）。

这证明的是：**时变 Born 电荷足以驱动合理的红外吸收强度与峰位**（在液体、有限温、展宽条件下）。  
**没有**验证：$\Gamma$ 点 **$\omega\_{\rm LO}\neq\omega\_{\rm TO}$** 的声子色散支，或沿 BZ 高对称线的 **$\omega\_s(\mathbf q)$** 曲线。

### 12.2 为什么「$Z^{\ast}(t)$ 可靠」$\neq$ 「LO-TO 色散准确」

LO-TO 色散图需要的是**动力学矩阵** $D(\mathbf q)$，尤其是 **$\mathbf q\to 0$ 的非解析部分 (★)**：

$$
\lim_{\mathbf q\to 0} D_{ij}^{\alpha\beta}(\mathbf q)
= \frac{1}{\Omega}\,
\frac{Z_{j\nu\beta}\, q_\nu q_\mu Z_{i\mu\alpha}}
{\mathbf q\cdot\boldsymbol\varepsilon_\infty\cdot\mathbf q}
$$

| 对象 | 数学角色 | Kim 的 $Z^{\ast}(t)$ 是否覆盖 |
|------|----------|-------------------------|
| **LO-TO 色散** | $D(\mathbf q)$ 本征值；$\Gamma$ 点 LO/TO 为**不同 $\mathbf q/\lvert \mathbf q \rvert$ 极限** | **否** — 需 $\Phi\_{ij}=\partial^2 U/\partial r\_i\partial r\_j$（含长程）或显式 NAC |
| **IR 光谱** | $\langle \mathbf J(0)\cdot\mathbf J(t)\rangle$，$\mathbf J=\sum Z^{\ast}\cdot\mathbf v$ | **是** — 原文主交付 |
| **Raman** | $\langle\alpha(0)\alpha(t)\rangle$，需 **$\alpha(t)$** 而非仅 $Z^{\ast}$ | **部分** — $Z^{\ast}$ 不进入 Raman 主导项 |

**核心区别**：

1. **$Z^{\ast}$ 是一阶电响应**（$\partial P/\partial u$）；LO-TO 来自 **力常数矩阵的二阶导** $\partial^2 E/\partial u\_i\partial u\_j$ 在 **$q\to 0$ 的方向依赖非解析项**。$Z^{\ast}$ 准确是 (★) 的**必要 ingredient**，但**不自动给出** $D(\mathbf q)$ 对角化结果。

2. **MD 自相关谱**给出的是**热涨落下所有模式的叠加**（展宽、各向同性平均、频移），**不是**冷晶格上 harmonic phonon 色散 $\omega\_s(\mathbf q)$。液体 water **本身没有**晶体 $\Gamma$ LO-TO 分裂；MAPbI$\_3$ 即便有，Kim 也只算了 Raman/IR **强度曲线**，未做 Phonopy 色散。

3. Kim 的 **$Z^{\ast}\_i(t)$ 随构型变**（含 $\mathbf u$、$\Delta\mathbf u$、$\varepsilon\_e(\mathbf x)$），而 LO-TO 标准推导在**平衡/reference 结构**上对 **$q\to 0$ 极限**取**冻结**的 $Z^{\ast}$、$\varepsilon\_\infty$（Monacelli）或平衡构型上的 **$Z^0$**（Korogod NAC）。轨迹上的 $Z^{\ast}(t)$ **时间平均或瞬时值**都不能直接替代这一极限。

**结论**：**仅凭 Kim 已验证的 $Z^{\ast}(t)$ MD 管线，不能准确得到 LO-TO 色散曲线**；最多在极性晶体 IR 中间接看到 LO/TO 相关带的**间距模糊**，不能替代 phonon dispersion 诊断。

### 12.3 若要在 Kim 势上补 LO-TO：可借 Monacelli 或 Korogod 的哪一段

Kim 的总势 $U=U\_{\rm sr}+U^{\rm elec}+\cdots$ **已含长程 Ewald**；补 LO-TO 不是「再加一个 $Z^{\ast}$」，而是**在 $\Gamma$ 点补上 (★) 所缺的那一块非解析 Hessian**。两条可行嫁接思路：

#### 路线 A：借 **Korogod** — 用 Kim 的电荷/多极 **导出 NAC**（免 DFPT，与 Kim 训练一致）

与 Korogod 同族：在**该相平衡结构** $\mathbf x\_0$ 上，用 Kim 模型算

$$
Z^*_{i\alpha\beta}(\mathbf x_0) \quad\text{（式 28，含 }q,u,\Delta u\text{）}
$$

或从 Kim 的 $q\_i^{\rm les}(\mathbf x\_0)$ 构造 $P^0$、$Z^0=\partial P^0/\partial r$（Korogod 式 15–16），再组装

$$
\Phi^{\rm dd}=Z^0 Z^0 \otimes T_{\rm dipole}(|\mathbf d|)
$$

工作流：

1. **有限超胞位移**（在 **完整 Kim 势 $U$** 上）→ $\Phi\_{\rm short}$；
2. 平衡构型上 **Kim 导出** $Z^0$（或 $Z^{\ast}$）→ 加 NAC；
3. Phonopy 对角化 → $\omega\_s(\mathbf q)$，$\Gamma$ 点 LO-TO。

**优点**：与 Kim **同一套** EFS 训练势；$Z^{\ast}$ 已在 IR 上校验，$Z^0$ 用于 NAC **逻辑自洽**。  
**注意**：Kim 的 $Z^{\ast}$ 比 Korogod 多 **$\mathbf u,\Delta\mathbf u$** 贡献；NAC 标准式 (17) 源自**单极电荷** $P^0(q)$，对 MAPbI$\_3$ 等极性晶体需 **benchmark 与 DFPT** 是否复现 (★)。PbTiO$\_3$ 上 Korogod 已表明「非严格各向同性仍可用 NAC 近似」，Kim 可沿同一路试。

#### 路线 B：借 **Monacelli** — DFPT 的 $Z,\varepsilon\_\infty$ **插件式补 (★)**（不重训 Kim）

Monacelli 式 (9) 在 **$E\_{\rm tot}=E\_{\rm SR}+E\_{\rm LR}$** 中把 (★) **内建进 Hessian**。嫁到 Kim 上需避免**长程双重计数**：

- **做法 1（推荐概念）**：把 Kim 势**拆成**「已含长程的 $U$」与「短程等价部分」；仅在 **Monacelli 与 Kim 长程差异**处修正，或 **仅用 Monacelli 替换 Kim 用于 LO-TO 的 phonon 计算子程序**（Kim 仍负责 MD 光谱）。
- **做法 2（实用）**：MD / IR / Raman 仍用 **完整 Kim**；声子单独用 **Kim 的 $U\_{\rm sr}$（关掉 LES 多极项）+ Monacelli $E\_{\rm LR}$**，其中 $Z,\varepsilon\_\infty$ 来自 **该相一次 DFPT**，$\eta$ 大于 Kim 训练最大原子对距。

**优点**：LO-TO 与 BaTiO$\_3$ 一样**闭合在 (9)(19)**，不依赖 Phonopy 外挂 NAC 的 monopole 近似；DFPT $Z,\varepsilon$ **物理明确**。  
**代价**：需 **额外 DFPT**；Kim 与 Monacelli **长程形式不同**（多极 Ewald vs 偶极双线性），须谨慎处理 **double counting**。

#### 路线 C：对 **完整 Kim $U$** 直接求 Hessian（不借公式，计算最贵）

对大超胞有限差分 $\Phi\_{ij}=\partial^2 U/\partial r\_i\partial r\_j$，若超胞足够大且 $U$ 含完整 Ewald，**原则上** $\Gamma$ 附近可出现 LO-TO。  
Kim 多极 + 诱导项使 Hessian **贵且复杂**；原文未做。Monacelli/Korogod 的价值正是**避免**在有限胞上硬算 (★)。

### 12.4 简要建议

| 目标 | 建议 |
|------|------|
| 只要 IR/Raman（Kim 已验证） | 继续 **$Z^{\ast}(t)$ + $\alpha(t)$** MD 自相关 |
| 极性晶体 **LO-TO 色散**，已有 Kim 势、想免 DFPT | **路线 A**：Kim 平衡构型 **$Z^0$ / $Z^{\ast}$ + Korogod 式 NAC + Phonopy** |
| 已有 DFPT $Z,\varepsilon$，要最稳 LO-TO | **路线 B**：**Monacelli 插件**嫁到 phonon 工作流（注意去重 Kim 长程） |
| 统一 MD 光谱 + 声子 | **Kim 做 MD 谱** + **Korogod-NAC 或 Monacelli 做 phonon**（分计算、同一参考相） |

**一句话**：Kim 的 **$Z^{\ast}(t)$ 可靠解决的是「动量空间下的电流–光谱」**，不是「$\mathbf q$ 空间下的 LO-TO 色散」；要 LO-TO 曲线，**不能停在一阶 $Z^{\ast}(t)$**，必须引入 **(★) 所在的二阶非解析项**——**优先试 Korogod 式 NAC（用 Kim 自己的 $Z^{\ast}$）**，或有 DFPT 时用 **Monacelli 插件**更干净。

---

## 推荐方案：Kim 提供 $Z^{\ast}$，Monacelli / Korogod 提供 LO-TO 非解析拼接

§12.3 给了三条嫁接路线；本节把其中最自洽的一条——**「Kim 出电荷响应、Monacelli/Korogod 出非解析项」**——写成可落地的数学方案。核心思想：

> **分工**：Kim 已用 IR 校验过的 **$Z^{\ast}\_{\rm Kim}$** 作为 (★) 的输入 ingredient；声子色散的解析部分仍由 Kim 短程势给出；$\Gamma$ 点缺失的非解析 Hessian 由 Monacelli/Korogod 的闭式补上。这样**不重训** Kim，也**不需要**额外 DFPT 跑 $Z^{\ast}$（只在用 Monacelli 介电核时需要一次 $\varepsilon\_\infty$）。

### 12·5.1 为什么这条路最自洽

LO-TO 的非解析 Hessian (★) 需要两样东西：**Born 电荷 $Z^{\ast}$** 与 **介电核 $\mathbf q\cdot\boldsymbol\varepsilon\_\infty\cdot\mathbf q$**。

- Kim 的 $Z^{\ast}\_{\rm Kim}$ 由式 (28) 反演，已在 bulk water 上对齐 DFT（RMSE $\sim$0.02 $e$）→ **可直接复用**；
- 介电核可用 Kim 自己的 $\varepsilon\_e/\varepsilon\_\infty$，或外部一次 DFPT $\varepsilon\_\infty$；
- 解析声子由 **完整 Kim 势的有限位移 $\Phi\_{\rm short}$** 提供。

于是**三个 ingredient 各取所长**，避免了「Kim 完全退化成 Monacelli」（关掉多极、丢失 IR/Raman 表达力）的代价。

### 12·5.2 数学拼接：解析 + 非解析

声子动力学矩阵分解为解析与非解析两部分（标准 Gonze–Lee 框架）：

$$
D_{ij}^{\alpha\beta}(\mathbf q)
= \underbrace{D_{ij}^{\alpha\beta,\,\rm an}(\mathbf q)}_{\text{Kim 有限位移}}
+ \underbrace{D_{ij}^{\alpha\beta,\,\rm NA}(\mathbf q)}_{\text{Monacelli/Korogod}}
$$

**解析部分**：在完整 Kim 势 $U$ 上做有限位移（大超胞），得实空间力常数后傅里叶变换：

$$
\Phi_{ij}^{\alpha\beta,\,\rm an} = \frac{\partial^2 U_{\rm Kim}}{\partial u_{i\alpha}\,\partial u_{j\beta}}\bigg|_{\bar{\mathbf R}},
\qquad
D_{ij}^{\alpha\beta,\,\rm an}(\mathbf q) = \frac{1}{\sqrt{m_i m_j}}\sum_{\mathbf R}\Phi_{ij}^{\alpha\beta,\,\rm an}\,e^{i\mathbf q\cdot\mathbf R}
$$

**非解析部分**（即 (★)），用 Kim 反演的 $Z^{\ast}\_{\rm Kim}$ 代入：

$$
\boxed{\;
\lim_{\mathbf q\to0} D_{ij}^{\alpha\beta,\,\rm NA}(\mathbf q)
= \frac{1}{\sqrt{m_i m_j}}\,\frac{1}{\Omega}\,
\frac{\big(\mathbf q\cdot\mathbf Z^*_{i}\big)_\alpha\,\big(\mathbf q\cdot\mathbf Z^*_{j}\big)_\beta}
{\mathbf q\cdot\boldsymbol\varepsilon_\infty\cdot\mathbf q}
\;}
$$

展开成分量：

$$
D_{ij}^{\alpha\beta,\,\rm NA}(\mathbf q\to0)
= \frac{1}{\sqrt{m_i m_j}\,\Omega}\,
\frac{\big(\sum_\mu q_\mu Z^*_{i,\mu\alpha}\big)\big(\sum_\nu q_\nu Z^*_{j,\nu\beta}\big)}
{\sum_{\gamma\delta} q_\gamma\,\varepsilon_{\infty,\gamma\delta}\,q_\delta}
$$

其中 $Z^{\ast}\_{i}$ 即 Kim 式 (28) 的 $Z^{\ast}\_{i\alpha\beta}=\partial P\_\alpha/\partial R\_{i\beta}$。**这一步就是「Kim 提供 $Z^{\ast}$、Monacelli/Korogod 提供拼接」**：分子的 $Z^{\ast}$ 来自 Kim，分母的 $1/(\mathbf q\cdot\boldsymbol\varepsilon\_\infty\cdot\mathbf q)$ 与拼接结构来自 Monacelli (式 19) / Korogod (式 17) 的非解析核。

### 12·5.3 介电核 $\varepsilon\_\infty$ 的三种取法

(★-Kim) 还需要一个 $\boldsymbol\varepsilon\_\infty$。按是否愿意跑 DFPT，有三档：

| 取法 | $\boldsymbol\varepsilon\_\infty$ 来源 | 特点 |
|------|--------------------------------------|------|
| **K1（全 Kim）** | Kim 的 $\varepsilon\_e+\chi=\varepsilon\_\infty$（式 25），取轨迹平均或参考构型值 | 完全免 DFPT；但 Kim 的 $\varepsilon\_\infty$ 锚点本身常来自实验/DFPT |
| **K2（Korogod 式相消）** | 用 scaled BEC $Z^{\ast}=\sqrt{\varepsilon\_\infty}\,Z^0$，则 $\varepsilon\_\infty$ 在 $\Phi^{\rm dd}$ 中**精确相消** | 只需 Kim 的 $q$ 构造 $Z^0$，**无需显式 $\varepsilon\_\infty$**（各向同性最干净） |
| **K3（Monacelli 式显式）** | 一次 DFPT 的 $\boldsymbol\varepsilon\_\infty$ | 各向异性张量最准；代价是一次 DFPT |

其中 **K2** 最值得强调：套用 Korogod 的 $P^0$、$Z^0=\mathrm{Re}[\,e^{-2\pi i\rho\_l}\,\partial P^0/\partial r\_l\,]$，非解析项写成

$$
\Phi_{ij}^{\alpha\beta,\,\rm NA}
= \sum_{\alpha'\beta'} Z^0_{i,\alpha\alpha'}\,Z^0_{j,\beta\beta'}
\left(\frac{\delta_{\alpha'\beta'}}{|\mathbf d|^3}-\frac{3 d_{\alpha'}d_{\beta'}}{|\mathbf d|^5}\right)
$$

$\varepsilon\_\infty$ 已被 $\sqrt{\varepsilon\_\infty}$ 因子两两抵消——**完全用 Kim 的电荷分布即可闭合 LO-TO，不碰 DFPT**。区别仅在于 $P^0$ 里要不要带上 Kim 的偶极/四极贡献（见下条）。

### 12·5.4 关键风险：$Z^{\ast}\_{\rm Kim}$ 含 $\mathbf u,\Delta\mathbf u$，与单极 NAC 不完全同源

标准 NAC 式 (17) 与 Korogod 的 $Z^0$ 都源自**单极极化** $P^0(q)$。而 Kim 的 $Z^{\ast}$（式 28）包含：

$$
Z^*_{i\alpha\beta}
= \underbrace{\frac{\partial P^u_\alpha}{\partial r_{i\beta}}}_{\text{偶极/诱导项}}
+ \underbrace{\lim_{k\to0}\Re\!\Big[e^{-ikr_{i\alpha}}\frac{\partial P^q_\alpha(k)}{\partial r_{i\beta}}\Big]}_{\text{单极项}}
$$

含义与处理：

- **若直接用整支 $Z^{\ast}\_{\rm Kim}$ 代入 (★-Kim)**：这是物理上**更完整**的 Born 电荷（单极+偶极+诱导都进了 $\partial P/\partial u$），原则上比纯单极 $Z^0$ 更准；但要确认 Kim 的总极化 $P=P^q+P^u$ 在该晶体上与 DFPT $Z^{\ast}$ 一致（water 已验证，极性晶体需再 benchmark）。
- **若走 K2 的 $Z^0$ 相消路线**：$P^0$ 必须和 Korogod 一样**只用单极** $q\_i^{\rm les}$，否则 $\sqrt{\varepsilon\_\infty}$ 的相消关系不再成立。即：要么「整支 $Z^{\ast}$ + 显式 $\varepsilon\_\infty$（K1/K3）」，要么「纯单极 $Z^0$ + 相消（K2）」，**两者不可混用**。

### 12·5.5 落地工作流

```
参考相 x0（高对称、可含虚频）
        │
        ├─[1] 完整 Kim 势 U 上有限位移（大超胞）→ Φ_an → D_an(q)
        │
        ├─[2] Kim 反演 Z*_Kim（式 28）  或  纯单极 Z0（Korogod 式 15-16）
        │
        ├─[3] 介电核：K1 Kim ε∞ / K2 相消 / K3 DFPT ε∞
        │
        └─[4] 组装 D_NA(q)（★-Kim）→ D = D_an + D_NA → 对角化 → ω_s(q)
                                                         │
                                          Γ 点 LO-TO splitting
MD / IR / Raman 仍由完整 Kim（Z*(t), α(t)）单独跑，与声子分计算、共用同一参考相。
```

**无 double counting 的保证**：解析部分 $D\_{\rm an}$ 来自有限超胞，其长程 $1/r^3$ 已被超胞截断（正是 §三所述短程 MLIP 在 $\Gamma$ 抹平 LO-TO 的原因）；非解析部分 $D\_{\rm NA}$ 只补 $\mathbf q\to0$ 的**奇异极限**。两者天然互补，不重复——这与传统 DFPT/Phonopy 的 NAC 完全同构，差别只在 $Z^{\ast}$ 由 Kim 提供而非 DFPT。

### 12·5.6 与三条原始路线的关系

| 本节方案 | 等价于 §12.3 | 介电核 | $Z$ 来源 |
|----------|--------------|--------|----------|
| K2（相消） | **路线 A（Korogod 式 NAC）** | 相消，免 $\varepsilon\_\infty$ | Kim 单极 $q\Rightarrow Z^0$ |
| K3（显式 $\varepsilon\_\infty$） | 介于 A / B | DFPT $\boldsymbol\varepsilon\_\infty$ | Kim 整支 $Z^{\ast}$ |
| 整支 $Z^{\ast}$ + Kim $\varepsilon\_\infty$（K1） | **路线 A 的「全 Kim」变体** | Kim $\varepsilon\_e+\chi$ | Kim 整支 $Z^{\ast}$ |

**推荐默认**：各向同性或弱各向异性体系优先 **K2**（最省、与 Kim 训练自洽）；强各向异性介电张量（如四方 PbTiO$\_3$）用 **K3** 拿 DFPT $\boldsymbol\varepsilon\_\infty$ 配 Kim 整支 $Z^{\ast}$。两种都**保留** Kim 的 IR/Raman/外场表达力。

### 12·5.7 小结

> **「Kim 提供 $Z^{\ast}$，Monacelli/Korogod 提供非解析拼接」= 把 (★) 的分子 ($Z^{\ast}$) 交给 Kim、分母与拼接结构交给 NAC**。它不要求改写 Kim 的多极 Ewald，也不要求把 Kim 退化成偶极双线性；代价只是 (a) 确认 Kim 的 $Z^{\ast}$/$P^0$ 在目标晶体上对齐 DFPT，(b) 在「整支 $Z^{\ast}$+显式 $\varepsilon\_\infty$」与「单极 $Z^0$+相消」之间**二选一**，不可混用。

---

## 十三、三路线总结公式

**Monacelli**：

$$
E_{\rm LR}=\frac{1}{2\Omega}\sum_{ij}\Delta\mathbf R_i^{\mathsf T}\mathbf Z_i\,
\Big[\sum_{\mathbf k\neq 0}\frac{\mathbf k\mathbf k^{\mathsf T}}{\mathbf k^{\mathsf T}\boldsymbol\varepsilon\mathbf k}\,e^{-\eta^2k^2/2}e^{-i\mathbf k\cdot(\mathbf R_j-\mathbf R_i)}\Big]
\mathbf Z_j\Delta\mathbf R_j
$$

**Korogod**：

$$
E_{\rm train}=E_{\rm MTP}+\mathrm{Ewald}\Big[\sum_{i<j}\frac{q_i(\mathbf x)q_j(\mathbf x)}{r_{ij}}\Big], \qquad
\Phi^{\rm dd}=Z^0 Z^0 \otimes T_{\rm dipole}(|\mathbf d|)
$$

**Kim**：

$$
U=U_{\rm sr}+\frac{1}{2\varepsilon_0 V}\sum_{\mathbf k}\frac{e^{-\sigma^2k^2/2}}{k^2}|S(\mathbf k)|^2
-\sum_i\frac{1}{2}\kappa_i\Phi_i^2-\sum_i\frac{1}{2}\mathbf E_i\cdot\boldsymbol\alpha_i\cdot\mathbf E_i
$$

$$
\varepsilon_e(\mathbf x)=\frac{\varepsilon_\infty}{1+\chi_{\rm les}(\mathbf x)}, \qquad
I_{\rm IR}\propto\mathrm{FT}[\langle\mathbf J(0)\cdot\mathbf J(t)\rangle], \quad
R\propto\omega^2\,\mathrm{FT}[\langle\alpha(0)\alpha(t)\rangle]
$$

---

## 十四、总结

三篇 2026 工作共享 **(★)** 这一 LO-TO 物理核，但走三条路：

1. **Monacelli** — DFPT 的 $Z,\varepsilon\_\infty$ **冻结** + 偶极双线性 **插件** → LO-TO **内建于 Hessian**，最适合**已有短程势 + 极性晶体声子/变胞 MD**；
2. **Korogod** — **可学** $q\_i(\mathbf x)$ + MD 长程库仑 + **Phonopy NAC**（$\varepsilon\_\infty$ 相消）→ 最适合**端到端 MLIP + 不想单独跑 DFPT 的 LO-TO**；
3. **Kim** — **可极化多极矩** + **$\varepsilon\_e(\mathbf x)$ 拆分** + MD **IR/Raman** → 最适合**液体/多相/界面谱学**；**$Z^{\ast}(t)$ 可靠不等于 LO-TO 色散**（§十二），补 LO-TO 宜 **Korogod-NAC（用 Kim 的 $Z^{\ast}$）** 或 **Monacelli 插件（DFPT $Z,\varepsilon$）**。

选型口诀：**声子 LO-TO 深 → Monacelli/Korogod；振动光谱宽 → Kim；Kim 势上要 LO-TO → 借 NAC 或 Monacelli，不能单靠 $Z^{\ast}(t)$ MD 谱。**

落地推荐（§十二·五）：**$D=D\_{\rm an}^{\rm Kim}+D\_{\rm NA}^{(★)}$**——解析部分用完整 Kim 有限位移，非解析部分用 (★-Kim) 把 Kim 的 $Z^{\ast}$ 填进 Monacelli/Korogod 的核；各向同性优先 **K2 相消（免 DFPT）**，强各向异性用 **K3（DFPT $\varepsilon\_\infty$ + Kim 整支 $Z^{\ast}$）**；「整支 $Z^{\ast}$+显式 $\varepsilon\_\infty$」与「单极 $Z^0$+相消」**二选一不可混用**。

---

## 延伸阅读（站内）

- [极性材料长程静电与 LO-TO 分裂：Monacelli & Marzari 2026 插件式 MLIP 长程修正解读](/2026/06/07/极性材料长程静电与-LO-TO-分裂-Monacelli-&-Marzari-2026-插件式-MLIP-长程修正解读/)
- [不用 DFPT 也能算 LO-TO 分裂？环境依赖电荷长程 MLIP 速读](/2026/05/28/不用-DFPT-也能算-LO-TO-分裂-环境依赖电荷长程-MLIP-速读/)
- [一篇讲透：为什么“可极化多极矩”能让材料力场更懂电学？](/2026/05/16/Kim2026-可极化多极矩长程电静学/)


## 参考文献

1. Monacelli L, Marzari N. *Electrostatic interactions in atomistic and machine-learned potentials for polar materials*. Phys. Rev. B **113**, 094101 (2026). https://doi.org/10.1103/7ygl-8db2  
2. Korogod D, Shapeev A V, Novikov I S. *Long-range machine-learning potentials with environment-dependent charges enable predicting LO-TO splitting and dielectric constants*. arXiv:2603.06396 (2026).  
3. Kim D, King D S, Park Y, et al. *Polarizable atomic multipoles for learning long-range electrostatics*. arXiv:2605.05746 (2026).  
4. Gonze X, Lee C. Dynamical matrices, Born effective charges, dielectric permittivity tensors, and interatomic force constants from DFPT. Phys. Rev. B **55**, 10355 (1997).  
5. Cochran W, Cowley R A. Dielectric constants and lattice vibrations. J. Phys. Chem. Solids **23**, 447 (1962).
