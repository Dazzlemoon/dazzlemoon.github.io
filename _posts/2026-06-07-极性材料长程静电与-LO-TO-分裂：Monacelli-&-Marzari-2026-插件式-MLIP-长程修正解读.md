---
layout:     post
title:      极性材料长程静电与 LO-TO 分裂：Monacelli & Marzari 2026 插件式 MLIP 长程修正解读
subtitle:   Phys. Rev. B 2026：Born 有效电荷 + 高频介电张量 + η 高斯 smearing + Ewald 傅里叶求和 → 式 (9) 长程能量、(11) 力、(12)–(15) 应力；式 (16)–(19) 复现 LO-TO；可叠加既有 GAP 无需重训；BaTiO₃ benchmark。与 Korogod/Kim 横向对比见 `Monacelli、Korogod（LOTO 文章）与 Kim 多极矩 MLIP/` 专文。
date:       2026-06-07
author:     天月将白
header-img: img/post-bg-desk.jpg
catalog:    true
tags:
    - LO-TO
    - MLIP
    - Born charge
    - 长程静电
    - Phys.Rev.B
    - 高精度谱学
    - BaTiO3
---

![一图总结](/img/posts/2026-06-07-monacelli-loto-electrostatic-mlip/cover.png)

# 极性材料长程静电与 LO-TO 分裂：插件式 MLIP 长程修正如何复现 BaTiO₃ 声子色散？

> **论文**：Lorenzo Monacelli & Nicola Marzari, *Electrostatic interactions in atomistic and machine-learned potentials for polar materials*, **Phys. Rev. B** 113, 094101 (**2026**). Editors' Suggestion.  
> **DOI**：[10.1103/7ygl-8db2](https://doi.org/10.1103/7ygl-8db2)  
> **机构**：EPFL THEOS / NCCR MARVEL  
> **代码**：[GitHub 开源](https://github.com/lmonacell/longrangecoulomb)（Python + Julia，GPLv3，ASE 力场插件）  
> **数据**：[Materials Cloud](https://archive.materialscloud.org/)（BaTiO₃ 动力学矩阵、介电张量、Born 有效电荷）

---

## 一、背景：极性绝缘体里，短程 MLIP 为何「看不见」LO-TO？

在**极性材料**中，离子偏离平衡位置会产生**电偶极矩**；其产生的电场按 **$1/r^3$** 衰减，并与远处离子的局域偶极耦合。金属中电子屏蔽使长程场仅在离子动力学中重要；**绝缘体**无传导电子，**零频**下长程电场全程存在，直接决定：

- **$\Gamma$ 点 LO-TO 分裂**（纵向/横向光学声子）[3–6]；
- 声子色散 → 热膨胀、比热、晶格热导、**Raman / IR** 等一切与声子相关的性质。

第一性原理（DFPT 或有限差分）在**有限 $q$** 下已能正确处理长程静电 [4,7]；但在 **$q\to 0$（$\Gamma$ 点）**，周期位移产生**宏观极化**，动力学矩阵对 **$q/\lvert q \rvert$ 方向非解析**——这是短程截断势无法捕捉的**立方衰减**长程效应。

近年 **GAP / 神经网络势** 等 MLIP 可在百万原子尺度接近 DFT 精度 [15,16]，但常见做法：

- 只拟合**局域环境**；
- 训练数据在**周期边界**下不含显式长程项；
- 于是 $\Gamma$ 附近力常数矩阵 **$D(q)$ 随 $q$ 平滑**，**LO-TO 分裂消失**。

已有补救路线包括：

| 路线 | 代表 | 局限 |
|------|------|------|
| 二/三代 MLIP：点电荷 + 环境依赖 $q\_i$ | [18–23] | 部分电荷非物理可观测量，定义不唯一 |
| Wannier / 极化 / 电荷密度 + 四代 charge equilibration | [24–35] | 需额外 FP 数据，难复用既有 FPMD |
| 长程坐标描述符 / 消息传递 | [36,37] | 训练胞须足够大以采样长程 |

**本文策略**：从第一性原理推导**长程偶极–偶极**对 **能量、力、应力** 的贡献，**仅依赖**平衡态 **Born 有效电荷 $Z$** 与 **高频介电张量 $\varepsilon$**（DFPT 可观测量），以**插件**形式叠加到**已有短程势**（如 BaTiO₃ GAP [52]），**无需重训**。

---

## 二、偶极形式体系：从式 (1)–(4) 到 Born 有效电荷

### 2.1 偶极–场耦合能量

中性体系中，长程静电的**主导项**是偶极–偶极耦合。每个原子偶极 $\boldsymbol{\mu}\_i$ 产生按 **$1/r^3$** 衰减的电场 $\mathbf{E}(\mathbf{R}\_i)$，总耦合能为：

$$
E = -\frac{1}{2}\sum_{i=1}^{N} \mathbf{E}(\mathbf{R}_i)\cdot\boldsymbol{\mu}_i
\tag{1}
$$

式中 **$\frac{1}{2}$** 避免对全部原子对双重计数；粗体表示矢量/矩阵。

### 2.2 原子偶极与 Born 有效电荷

晶体总极化 $\boldsymbol{\mu}$ 可由**现代极化理论** [30–32] 严格定义。将原子 $i$ 的位移对总极化的贡献记为**原子偶极** $\boldsymbol{\mu}\_i$：

$$
Z_{i\alpha\beta} = \frac{\partial \mu_\alpha}{\partial R_{i\beta}}, \qquad
\mu_\alpha(\mathbf{R}) = \mu_\alpha^{(0)} + \sum_{i=1}^{N} \mu_{i\alpha}(\mathbf{R})
\tag{2}
$$

$$
\mu_{i\alpha}(\mathbf{R}) = \sum_\beta Z_{i\alpha\beta}\,(R_{i\beta} - \bar{R}_{i\beta})
\tag{3}
$$

- $Z\_{i\alpha\beta}$：**Born 有效电荷张量**（$\alpha,\beta$ 为笛卡尔分量）；
- $\bar{\mathbf{R}}\_i$：原子 $i$ 的**参考（平衡）位置**；
- $\mu\_\alpha^{(0)}$：原子处于 $\bar{\mathbf{R}}$ 时的晶胞总偶极，定义至**极化量子**。

### 2.3 辅助点电荷偶极

为构造可 Ewald 求和的电荷分布，将每个 $\boldsymbol{\mu}\_i$ 表为**等量异号**两点电荷：

$$
\boldsymbol{\mu} = q\,\mathbf{d}
\tag{4}
$$

$\lvert q\mathbf{d} \rvert$ 固定时，$q$ 与 $d$ 的选取只影响**多极展开高阶项**；在**偶极极限**下，最终能量、力、应力**与 $q,d$ 无关**（附录 C 详证）。

---

## 三、高斯 smearing 电荷密度与 Ewald 电场：式 (5)–(8)

### 3.1 高斯电荷密度 $\rho(\mathbf{r})$

用**缓慢变化**的高斯电荷密度代替 $\delta$ 点电荷，在保持长程偶极–偶极行为的同时**抹平**短程奇点；参数 **$\eta$** 为 smearing 宽度：

$$
\rho(\mathbf{r}) = \sum_{j=1}^{2N} \frac{q_j}{\sqrt{8\pi^3\eta^2}}
\exp\!\left(-\frac{(\mathbf{r}-\tilde{\mathbf{R}}_j)^2}{2\eta^2}\right)
\tag{5}
$$

$\tilde{\mathbf{R}}\_j$ 为辅助电荷位置（附录 A 给出 $\tilde{\mathbf{R}}\_{i\pm}$ 与 $Z$ 的关系）。**实用意义**：

- 若超胞线尺度 **$L \lesssim \eta$**，高斯电荷相互作用**不显著改变**短程 GAP 在训练胞上的能量/力；
- 因而可在**不重训**短程势的前提下叠加长程项；
- **$\eta$ 下限**：训练数据中任意原子对距离（含 PBC）的最大值——更小则须从训练集**扣除**长程贡献并重训短程部分。

### 3.2 傅里叶空间 Ewald 电场

式 (5) 的极性电荷分布产生**条件收敛**的宏观表面电荷问题；对**三维 bulk**，在傅里叶空间用 Ewald 求和（附录 B）：

$$
\mathbf{E}(\mathbf{r}) = \frac{i}{\Omega}\sum_{\mathbf{k}_j\neq 0}
\mathbf{k}_j\, e^{-\eta^2 k_j^2/2}\, e^{i\mathbf{k}_j\cdot\mathbf{r}}
\left[\frac{\mathbf{k}\,\boldsymbol{\varepsilon}\,\mathbf{k}}{\mathbf{k}\cdot\boldsymbol{\varepsilon}\cdot\mathbf{k}}\right]_{\alpha\beta} S(\mathbf{k}_j)
\tag{6}
$$

- $\Omega$：超胞体积；$i$ 来自 Maxwell 方程傅里叶形式；
- **结构因子**（辅助电荷体系）：

$$
S(\mathbf{k}) = \sum_{j=1}^{2N} q_j\, e^{-i\mathbf{k}\cdot\tilde{\mathbf{R}}_j}
\tag{7}
$$

- **$\mathbf{k}$ 网格**（倒格矢 $\mathbf{a}^{\ast},\mathbf{b}^{\ast},\mathbf{c}^{\ast}$ 的整数线性组合）：

$$
\mathbf{k}_{(l,m,n)} = l\mathbf{a}^* + m\mathbf{b}^* + n\mathbf{c}^*, \quad l,m,n\in\mathbb{Z},\;\mathbf{k}\neq 0
\tag{8}
$$

**注意**：2D/1D 体系式 (6) 不再成立，须用非均匀 $\varepsilon(\mathbf{r})$ 处理 [7,38,39]。

---

## 四、核心结果：长程能量式 (9)

将式 (5) 的静电能与式 (6) 的 $\mathbf{E}$ 联立（附录 C 由式 (1) 代入 Ewald 场并对 $\sin$ 一阶展开），得**本文中心公式**——仅含 **$Z$、$\varepsilon$、位移 $(R-\bar{R})$** 的长程能量：

$$
E(\mathbf{R}) = \frac{1}{2\Omega}\sum_{ij\alpha\beta\mu\nu}
(R_{i\alpha}-\bar{R}_{i\alpha})(R_{j\mu}-\bar{R}_{j\mu})\, Z_{i\beta\alpha} Z_{j\nu\mu}
\sum_{\mathbf{k}\neq 0}
\frac{k_\beta k_\nu\, e^{-\eta^2 k^2/2}}{\mathbf{k}\cdot\boldsymbol{\varepsilon}\cdot\mathbf{k}}
\, e^{-i\mathbf{k}\cdot(\mathbf{R}_j-\mathbf{R}_i)}
\tag{9}
$$

**要点**：

1. 只依赖 **DFPT 可观测量** $Z$、$\boldsymbol{\varepsilon}$（平衡结构 $\bar{\mathbf{R}}$）；
2. 非物理的 $q\_j$、$\tilde{\mathbf{R}}\_j$ 在偶极极限**相消**；
3. **唯一可调参数**为 **$\eta$**；
4. 总能量 = **短程 MLIP** + 式 (9)。

---

## 五、力与应力：式 (10)–(15)

### 5.1 长程力

$$
f_{i\alpha} = -\frac{\partial E}{\partial R_{i\alpha}}
\tag{10}
$$

完整表达式（数值实现须与 $E$ 梯度一致，保留 $k^2$ 小量项）：

$$
\begin{aligned}
f_{i\alpha} =\;& \sum_{j\beta\mu\nu} (R_{j\mu}-\bar{R}_{j\mu})\, Z_{i\beta\alpha} Z_{j\nu\mu}
\sum_{\mathbf{k}\neq 0} \frac{k_\beta k_\nu\, e^{-\eta^2 k^2/2}}{\mathbf{k}\cdot\boldsymbol{\varepsilon}\cdot\mathbf{k}}
\cos\!\big[\mathbf{k}\cdot(\mathbf{R}_j-\mathbf{R}_i)\big] \\
&+ \sum_{j\beta\gamma\mu\nu} (R_{i\gamma}-\bar{R}_{i\gamma})(R_{j\mu}-\bar{R}_{j\mu})\, Z_{i\beta\gamma} Z_{j\nu\mu} \\
&\quad \times \sum_{\mathbf{k}\neq 0} \frac{k_\alpha k_\beta k_\nu\, e^{-\eta^2 k^2/2}}{\mathbf{k}\cdot\boldsymbol{\varepsilon}\cdot\mathbf{k}}
\sin\!\big[\mathbf{k}\cdot(\mathbf{R}_j-\mathbf{R}_i)\big]
\end{aligned}
\tag{11}
$$

**长波极限**：仅**第一项**存活（第二项 $\propto k^2$）；实现中仍保留全式 (11) 以保证 **$\mathbf{f}=-\nabla E$** 数值一致。

### 5.2 平移不变性与声学求和规则（附录 D）

式 (3) 的原子偶极划分**破坏**平移不变性：有效电荷声学求和规则

$$
\sum_i Z_{i\alpha\beta} = 0 \quad \forall\,\alpha,\beta
\tag{D1}
$$

只能保证刚性平移不产生净偶极，**不能**阻止 $\boldsymbol{\mu}\_i$ 随全局平移改变 → 质心可能受非零净力。

**修复**：重定义质心位移，消除刚性平移：

$$
R_{i\alpha} = \bar{R}_{i\alpha}^{(0)} + \frac{1}{N}\sum_j \big(R_{j\alpha}-\bar{R}_{j\alpha}^{(0)}\big)
\tag{D2}
$$

在此定义下式 (9) 对全局平移不变，导数满足全部声学求和规则。质心力的后验修正：

$$
f_{i\alpha}^{\mathrm{ASR}} = -\sum_{j\beta}\frac{\partial E}{\partial R_{j\beta}}\,\frac{\delta_{\beta\alpha}}{N}
= \frac{1}{N}\sum_j \frac{\partial E}{\partial R_{j\alpha}}
= -\frac{1}{N}\sum_j f_{j\alpha}
\tag{D3}
$$

数值上可在式 (11) 之后施加。**应力**：因 (D2) 写在能量子程序内部，ForwardDiff.jl 自动微分会链式包含 $\partial \bar{\mathbf{R}}/\partial \boldsymbol{\varepsilon}$，同步修正应力的 ASR。

### 5.3 应力张量

应变 $\boldsymbol{\varepsilon}$ 下（**clamped-ion**，原子随应变移动）：

$$
\sigma_{\alpha\beta} = -\frac{1}{\Omega}\frac{\partial E}{\partial \varepsilon_{\alpha\beta}}
\tag{12}
$$

$$
R'_\alpha(\boldsymbol{\varepsilon}) = R_\alpha + \sum_\beta \varepsilon_{\alpha\beta} R_\beta
\tag{13}
$$

倒格矢随应变：

$$
k'_\alpha(\boldsymbol{\varepsilon}) = k_\alpha - \sum_\beta \varepsilon_{\alpha\beta} k_\beta
\tag{14}
$$

（故 $\mathbf{k}\cdot\mathbf{R}$ 在应变下不变。）体积：

$$
\Omega'(\boldsymbol{\varepsilon}) = \Omega\left(1 + \sum_{\alpha=1}^{3}\varepsilon_{\alpha\alpha}\right)
\tag{15}
$$

将 $\Omega'(\boldsymbol{\varepsilon})$、$\mathbf{R}'(\boldsymbol{\varepsilon})$、$\mathbf{k}'(\boldsymbol{\varepsilon})$ 代入式 (9)，用 **ForwardDiff.jl** 对独立应变分量算法微分得 $\boldsymbol{\sigma}$——对 **NPT MD、SSCHA** [40–44] 与 EOS/热膨胀至关重要。

---

## 六、LO-TO 分裂：力常数矩阵式 (16)–(19)

长程势必须复现 **$q\to 0$** 非解析动力学矩阵 → **LO-TO 分裂**。

### 6.1 原子间力常数

$$
\Phi_{ij}^{\alpha\beta} = \frac{d^2 E}{d R_{i\alpha}\, d R_{j\beta}} = -\frac{d f_{i\alpha}}{d R_{j\beta}}
\tag{16}
$$

平衡位形 $\mathbf{R}=\bar{\mathbf{R}}$ 下：

$$
\Phi_{ij}^{\alpha\beta} = \frac{1}{\Omega}\sum_{\mathbf{k}\neq 0} k_\nu k_\mu\, Z_{j\nu\beta} Z_{i\mu\alpha}\,
\frac{e^{-\eta^2 k^2/2}}{\mathbf{k}\cdot\boldsymbol{\varepsilon}\cdot\mathbf{k}}
\cos\!\big[\mathbf{k}\cdot(\mathbf{R}_i-\mathbf{R}_j)\big]
\tag{17}
$$

### 6.2 傅里叶变换与 $q\to 0$ 极限

$$
\begin{aligned}
D_{ij}^{\alpha\beta}(\mathbf{q}) =\;& \frac{1}{2\Omega}\sum_{\mathbf{k}=\mathbf{q}+\mathbf{G}} k_\nu k_\mu\, Z_{j\nu\beta} Z_{i\mu\alpha}\,
\frac{e^{-\eta^2 k^2/2}}{\mathbf{k}\cdot\boldsymbol{\varepsilon}\cdot\mathbf{k}}\, e^{i\mathbf{k}\cdot(\mathbf{R}_i-\mathbf{R}_j)} \\
&+ \frac{1}{2\Omega}\sum_{\mathbf{k}=-\mathbf{q}+\mathbf{G}} k_\nu k_\mu\, Z_{j\nu\beta} Z_{i\mu\alpha}\,
\frac{e^{-\eta^2 k^2/2}}{\mathbf{k}\cdot\boldsymbol{\varepsilon}\cdot\mathbf{k}}\, e^{-i\mathbf{k}\cdot(\mathbf{R}_i-\mathbf{R}_j)}
\end{aligned}
\tag{18}
$$

与 QE [4,46,47]、ABINIT [5,6,48] 的傅里叶插值 ansatz 同型。当 **$\lvert G \rvert\gg 1/\eta$**：

$$
\lim_{\mathbf{q}\to 0} D_{ij}^{\alpha\beta}(\mathbf{q}) = \frac{1}{\Omega}\,
\frac{Z_{j\nu\beta} q_\nu q_\mu Z_{i\mu\alpha}}{\mathbf{q}\cdot\boldsymbol{\varepsilon}\cdot\mathbf{q}}
\tag{19}
$$

（CGS 单位下 $\varepsilon\_0=(4\pi)^{-1}$ 后与 Ref. [4] 式 (18) 一致。）式 (19) 即标准 **LO-TO 非解析修正**；证明式 (9)(11) 可描述任意极性材料的 LO-TO。

---

## 七、模型使用：参数、叠加与 $\eta$ 选取

### 7.1 输入量

| 量 | 来源 | 说明 |
|----|------|------|
| $\bar{\mathbf{R}}\_i$ | 高对称参考结构 | 可为 saddle point（虚频）；利于子群对称 |
| $Z\_{i\alpha\beta}$ | DFPT | 极性晶体中随位移变化小 |
| $\boldsymbol{\varepsilon}$ | DFPT（高频） | 与 $Z$ 同平衡结构 |
| $\eta$ | 用户选取 | **唯一**自由参数 |

总能量/力/应力：

$$
E_{\mathrm{tot}} = E_{\mathrm{SR}} + E_{\mathrm{LR}},\quad
\mathbf{f}_{\mathrm{tot}} = \mathbf{f}_{\mathrm{SR}} + \mathbf{f}_{\mathrm{LR}},\quad
\boldsymbol{\sigma}_{\mathrm{tot}} = \boldsymbol{\sigma}_{\mathrm{SR}} + \boldsymbol{\sigma}_{\mathrm{LR}}
$$

### 7.2 $\eta$ 的权衡

- **$\eta$ 越小**：最近偶极–偶极作用距离越短，精度越高；但 $k$ 求和更贵，且被忽略的多极项变重要；
- **$\eta$ 下限**：训练集最大原子对距离（含 PBC）——否则长程项改变训练标签，须**预处理扣除** $E\_{\mathrm{LR}}$ 并重训 $E\_{\mathrm{SR}}$；
- **结构偏离 $\bar{\mathbf{R}}$**：$Z$、$\varepsilon$ 固定近似误差增大；**液体、扩散、强一级相变**不适用；可分相设不同 $(Z,\varepsilon)$。

---

## 八、BaTiO₃ 立方相 benchmark（Fig. 1）

**体系**：$\alpha$-BaTiO$\_3$ 立方钙钛矿（5 原子原胞，**Pm$\overline{3}$m**，空间群 221）。该结构在全 BZ 有**两条虚频**（高温涨落稳定化 saddle point [52]）。

**短程势**：Ref. [52] 的 **solid BaTiO$\_3$ GAP**（2×2×2 超胞训练），**不重训**。

**长程修正**：式 (11)，**$\eta = 2.5\,\text{\AA}$**（正文；图注另写 2.8 Å，以正文计算为准）。

**DFT 参考**：QE 7.0 + PBEsol，SSSP 1.2.1 efficiency PAW/USPP，截断 60/600 Ry，$8\times8\times8$ $k$ 网格，$a=4.035\,\text{\AA}$（附录 E）。

![图 1：立方 BaTiO₃ 声子色散与高频 DOS——DFT、纯 GAP、GAP+长程修正对比（原文 Fig. 1）。虚频表示 saddle point；纯 GAP 在 $\Gamma$ 附近 LO/TO 合并；长程修正恢复 500–600 cm$^{-1}$ 带隙。](/img/posts/2026-06-07-monacelli-loto-electrostatic-mlip/fig01-batio3-phonon-lr-gap-dos.png)

**观察**：

1. **BZ 边缘**：GAP 已与 DFPT 较好一致（与 2×2×2 训练 $q$ 点 commensurate）；
2. **$q\to\Gamma$**：纯 GAP **LO/TO 合并**，500–600 cm$^{-1}$ **带隙消失**（DOS 右 panel）；
3. **+LR**：定性恢复带隙，虚频模式数目与 DFT 一致（纯 GAP 在 $\Gamma$ 附近多一条虚频）；
4. 更大训练胞时长程更敏感，可先**从训练集减去** $E\_{\mathrm{LR}}$ 再改进短程拟合。

---

## 九、讨论：固定 $Z,\varepsilon$ 的适用范围与 Fig. 2 可迁移性

**核心假设**：$\boldsymbol{\varepsilon}$、$Z$ 与原子坐标**无关**（固定为参考相对位形）。

**适用**：晶格保持、键不断裂、无自由扩散的**固相热力学**；一级相变可分相模拟 + 自由能积分 / SSCHA / SCP / TDEP [41–44,55–59]。

**不适用**：液体、强键重排、$Z$ 强依赖位移（多声子 IR 散射罕见即因 $Z$ 通常较稳 [49,50]）。

**改进方向**：用 equivariant ML 参数化 $\boldsymbol{\varepsilon}(\mathbf{R})$、$Z(\mathbf{R})$、质心 [28,29]，反解式 (3) 定义 $\bar{\mathbf{R}}(\mathbf{R})$，代入式 (9)——类似三/四代 MLIP [17]。

### 附录 F：四方相可迁移性（Fig. 2）

用**立方相** DFPT 的 $Z$、$\boldsymbol{\varepsilon}$ 计算**四方相**色散（$\eta=2.5\,\text{\AA}$）：

![图 2：沿 $\Gamma\to(1/2,0,0)$ 的色散——DFT、纯 GAP、立方相参数的长程修正（原文 Fig. 2）。](/img/posts/2026-06-07-monacelli-loto-electrostatic-mlip/fig02-tetragonal-phonon-transferability.png)

**介电性质可迁移**：LO-TO 分裂尤其**高能支**被正确捕获；结构远离参考位形时精度下降。

**延伸阅读（三路线对比）**：与 Korogod (2026) MTP+EDQRd+NAC、Kim (2026) 可极化多极矩 MLIP 的公式对照、$\varepsilon\_e$ 构型依赖与谱学选型，见站内延伸阅读专文（Monacelli / Korogod / Kim 三路线对比，待同步博客）。

---

## 十、附录公式推导脉络（原理向完整索引）

### 附录 A：辅助电荷体系

$$
q_i = \frac{1}{3}\sum_\alpha Z_{i\alpha\alpha}
\tag{A1}
$$

正负电荷位置（偶极中心在位移中点）：

$$
\tilde{R}_{i\alpha}^{+} = R_{i\alpha} + \frac{1}{2q_i}\sum_\beta Z_{i\alpha\beta}(R_{i\beta}-\bar{R}_{i\beta})
\tag{A2a}
$$

$$
\tilde{R}_{i\alpha}^{-} = R_{i\alpha} - \frac{1}{2q_i}\sum_\beta Z_{i\alpha\beta}(R_{i\beta}-\bar{R}_{i\beta})
\tag{A2b}
$$

### 附录 B：Maxwell 方程 → 式 (6)

$$
\nabla\cdot\mathbf{D} = \rho
\tag{B1}
$$

$$
\sum_{\alpha\beta}\frac{\varepsilon_{\alpha\beta}}{\varepsilon_0}\frac{\partial E_\beta}{\partial r_\alpha} = \rho
\tag{B2}
$$

$$
-\sum_{\alpha\beta}\frac{\varepsilon_{\alpha\beta}}{\varepsilon_0}\frac{\partial^2 V}{\partial r_\alpha\partial r_\beta} = \rho
\tag{B3}
$$

$$
-V(\mathbf{k})\sum_{\alpha\beta}\varepsilon_{\alpha\beta} k_\alpha k_\beta = \frac{\rho(\mathbf{k})}{\varepsilon_0}
\tag{B4}
$$

对式 (5) 傅里叶变换：

$$
\rho(\mathbf{k}) = \frac{1}{\Omega}\int d\mathbf{r}\, e^{-i\mathbf{k}\cdot\mathbf{r}}
\sum_i \frac{q_i}{\sqrt{8\pi^3\eta^2}} e^{-( \mathbf{r}-\mathbf{R}_i)^2/(2\eta^2)}
\tag{B5}
$$

$$
\rho(\mathbf{k}) = \frac{1}{N_k \Omega}\sum_i e^{-i\mathbf{k}\cdot\mathbf{R}_i}
\int d\mathbf{r}\, e^{-i\mathbf{k}\cdot(\mathbf{r}-\mathbf{R}_i)}
\frac{q_i}{\sqrt{8\pi^3\eta^2}} e^{-(\mathbf{r}-\mathbf{R}_i)^2/(2\eta^2)}
\tag{B6}
$$

$$
\rho(\mathbf{k}) = \frac{1}{N_k \Omega}\sum_i e^{-i\mathbf{k}\cdot\mathbf{R}_i}\, q_i\, e^{-\eta^2 k^2/2}
\tag{B7}
$$

$$
V(\mathbf{k}) = -\frac{1}{N_k \Omega\,\varepsilon_0}\sum_i q_i e^{-i\mathbf{k}\cdot\mathbf{R}_i} e^{-\eta^2 k^2/2}
\left(\mathbf{k}\cdot\boldsymbol{\varepsilon}\cdot\mathbf{k}\right)^{-1}
\tag{B8}
$$

$$
\mathbf{E}(\mathbf{r}) = -\nabla V(\mathbf{r}),\quad
\mathbf{E}(\mathbf{k}) = -i\mathbf{k} V(\mathbf{k})
= \frac{i}{N_k \Omega}\sum_i q_i \mathbf{k}\, e^{-i\mathbf{k}\cdot\mathbf{R}_i} e^{-\eta^2 k^2/2}
\left(\mathbf{k}\cdot\boldsymbol{\varepsilon}\cdot\mathbf{k}\right)^{-1}
\tag{B9}
$$

实空间（排除 $\mathbf{k}=0$，中性体系 $\sum\_i q\_i=0$）：

$$
\mathbf{E}(\mathbf{r}) = \frac{i}{N_k \Omega}\sum_{ij,\,\mathbf{k}_j\neq 0}
q_i \mathbf{k}_j e^{-i\mathbf{k}_j\cdot\mathbf{R}_i} e^{-\eta^2 k_j^2/2}
\left(\mathbf{k}\cdot\boldsymbol{\varepsilon}\cdot\mathbf{k}\right)^{-1} e^{i\mathbf{k}_j\cdot\mathbf{r}}
\tag{B10}
$$

$$
\mathbf{E}(\mathbf{r}) = \frac{i}{N_k \Omega}\sum_{ij,\,\mathbf{k}_j\neq 0}
q_i \mathbf{k}_j e^{-\eta^2 k_j^2/2}
\left(\mathbf{k}\cdot\boldsymbol{\varepsilon}\cdot\mathbf{k}\right)^{-1} e^{i\mathbf{k}_j\cdot(\mathbf{r}-\mathbf{R}_i)}
\tag{B11}
$$

结构因子定义：

$$
S(\mathbf{k}) = \frac{1}{N_k}\sum_i q_i e^{-i\mathbf{k}\cdot\mathbf{R}_i}
\tag{B12}
$$

合并得正文式 (6) / (B13)：

$$
\mathbf{E}(\mathbf{r}) = \frac{i}{\Omega}\sum_{\mathbf{k}_j\neq 0}
\mathbf{k}_j e^{-\eta^2 k_j^2/2} e^{i\mathbf{k}_j\cdot\mathbf{r}}
\left(\mathbf{k}\cdot\boldsymbol{\varepsilon}\cdot\mathbf{k}\right)^{-1} S(\mathbf{k}_j)
\tag{B13}
$$

### 附录 C：由式 (1) 推出式 (9)

$$
E = -\frac{1}{2}\sum_{i\alpha\beta} (R_{i\alpha}-\bar{R}_{i\alpha}) Z_{i\beta\alpha} E_\beta(\mathbf{R}_i)
\tag{C1}
$$

代入 Ewald 场 (6)：

$$
\begin{aligned}
E =\;& -\frac{1}{2}\sum_{i\alpha\beta} (R_{i\alpha}-\bar{R}_{i\alpha})\,\frac{i}{\Omega}\sum_{\mathbf{k}\neq 0,j}
Z_{i\beta\alpha} k_\beta e^{-\eta^2 k^2/2}\left(\mathbf{k}\cdot\boldsymbol{\varepsilon}\cdot\mathbf{k}\right)^{-1} \\
&\times q_j e^{-i\mathbf{k}\cdot(\mathbf{R}_j-\mathbf{R}_i)}
\Big[ e^{-i\sum_{\mu\nu} k_\nu Z_{j\nu\mu}(R_{j\mu}-\bar{R}_{j\mu})/(2q_j)}
- e^{+i\sum_{\mu\nu} k_\nu Z_{j\nu\mu}(R_{j\mu}-\bar{R}_{j\mu})/(2q_j)} \Big]
\end{aligned}
\tag{C2}
$$

$$
E = \frac{1}{2}\sum_{i\alpha\beta} (R_{i\alpha}-\bar{R}_{i\alpha})\,\frac{1}{\Omega}\sum_{\mathbf{k}\neq 0,j}
Z_{i\beta\alpha} k_\beta e^{-\eta^2 k^2/2}\left(\mathbf{k}\cdot\boldsymbol{\varepsilon}\cdot\mathbf{k}\right)^{-1}
\times 2 q_j e^{-i\mathbf{k}\cdot(\mathbf{R}_j-\mathbf{R}_i)}
\sin\!\left(\frac{\sum_{\mu\nu} k_\nu Z_{j\nu\mu}(R_{j\mu}-\bar{R}_{j\mu})}{2q_j}\right)
\tag{C3}
$$

对 $\sin$ **一阶展开**（$e^{-\eta^2 k^2/2}$ 使有效 $k$ 小）得：

$$
E = \frac{1}{2\Omega}\sum_{ij\alpha\beta\mu\nu}
(R_{i\alpha}-\bar{R}_{i\alpha})(R_{j\mu}-\bar{R}_{j\mu})\, Z_{i\beta\alpha} Z_{j\nu\mu}
\sum_{\mathbf{k}\neq 0}
\frac{k_\beta k_\nu\, e^{-\eta^2 k^2/2}}{\mathbf{k}\cdot\boldsymbol{\varepsilon}\cdot\mathbf{k}}
\, e^{-i\mathbf{k}\cdot(\mathbf{R}_j-\mathbf{R}_i)}
\tag{C4}
$$

即正文式 (9)；$q$ 依赖消失，**不依赖任意电荷划分**。

### 附录 E：DFT 计算细节

| 项 | 设置 |
|----|------|
| 软件 | QUANTUM ESPRESSO 7.0 |
| XC | PBEsol [54] |
| 赝势 | SSSP 1.2.1 efficiency [64] |
| 截断 | 60 Ry（波函数）/ 600 Ry（密度） |
| $k$ 网格 | $8\times8\times8$，无 offset |
| 晶格常数 | $a = 4.035\,\text{\AA}$ |

---

## 十一、实现与验证

- **Python + Julia** 实现式 (9)(11)；应力由 **ForwardDiff.jl** 对含 (13)–(15) 的能量子程序微分；
- **ASE** 力场计算器，可**自动叠加**任意 ASE 短程势；
- **测试**：BaTiO$\_3$ 的 $q\to 0$ LO-TO 与式 (19) 及 **QE 7.3** 实现对比 [46]；
- **数据**：Materials Cloud [63] 公开动力学矩阵、$\boldsymbol{\varepsilon}$、$Z$。

---

## 十二、方法摘要

| 模块 | 内容 |
|------|------|
| 物理输入 | Born $Z$、高频 $\boldsymbol{\varepsilon}$（DFPT，平衡结构） |
| 长程核 | 偶极–偶极，Ewald $k$ 求和 + 高斯 smearing $\eta$ |
| 输出 | $E$ (9)、$\mathbf{f}$ (11)、$\boldsymbol{\sigma}$ (12)–(15) |
| 声子 | $\Phi$ (16)(17)、$D(\mathbf{q})$ (18) → LO-TO (19) |
| 集成 | $E\_{\mathrm{tot}}=E\_{\mathrm{SR}}+E\_{\mathrm{LR}}$，**无需重训** GAP |
| 局限 | 固定 $Z,\varepsilon$；非 liquid/强相变；2D/1D 需改 Ewald |

---

## 十三、总结

Monacelli & Marzari 给出**第一性原理严格**的长程静电修正：**只用量子力学可观测量** $Z$ 与 $\boldsymbol{\varepsilon}$，通过式 **(9)(11)** 及算法微分应力，**插件式**叠加既有短程 MLIP，在 **BaTiO$\_3$** 上恢复 **LO-TO 分裂**与高频声子带隙；**四方相**用立方相参数仍捕获主要介电色散特征。对**谱学、SSCHA、极性材料大规模 MD**——凡依赖正确 **$\Gamma$ 点非解析声子**的场景——该框架提供了一条**不重训**即可启用长程静电的实用路径；下一步是将 $Z(\mathbf{R})$、$\boldsymbol{\varepsilon}(\mathbf{R})$ **环境依赖化**，向三/四代 MLIP 收敛。

与 Korogod / Kim 等 2026 长程 MLIP 路线的横向对比见文末「延伸阅读（站内）」及三路线对比专文（待同步博客）。

---

## 延伸阅读（站内）

- [不用 DFPT 也能算 LO-TO 分裂？环境依赖电荷长程 MLIP 速读](/2026/05/28/不用-DFPT-也能算-LO-TO-分裂环境依赖电荷长程-MLIP-速读/)
- [一篇讲透：为什么“可极化多极矩”能让材料力场更懂电学？](/2026/05/16/Kim2026-可极化多极矩长程电静学/)


## 参考文献（精选）

- [3] Cochran & Cowley, J. Phys. Chem. Solids **23**, 447 (1962) — LO-TO 经典理论  
- [4] Giannozzi et al., Phys. Rev. B **43**, 7231 (1991) — DFPT 与 LO-TO 实现  
- [52] Stengel & Spaldin, Nat. Mater. **5**, 477 (2006) 等 — BaTiO$\_3$ GAP 训练  
- [7] Royo & Stengel, Phys. Rev. X **11**, 041027 (2021) — 2D 长程介电屏蔽  

完整 64 条文献见 Monacelli 原文。
