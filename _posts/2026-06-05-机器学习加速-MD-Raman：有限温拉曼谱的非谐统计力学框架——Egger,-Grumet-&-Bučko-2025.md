---
layout:     post
title:      机器学习加速 MD-Raman：有限温拉曼谱的非谐统计力学框架——Egger, Grumet & Bučko 2025 视角解读
subtitle:   TUM Egger 等（JCP 2025 Perspective）：谱密度/极化率速度自相关求 I(ω)；DFPT 算 α(t) 占 SiO₂ MD-Raman 成本 ~85%（MLFF 后 ~98%）；λ-SOAP/Δ-ML 学 α_μν，50 次 DFPT 训练即可近 DFPT 全谱。
date:       2026-06-05
author:     qqz
header-img: img/post-bg-desk.jpg
catalog:    true
tags:
    - Raman spectroscopy
    - MD-Raman
    - machine learning
    - DFPT
    - materials science
---

![一图总结](/img/posts/2026-06-05-egger-ml-md-raman-materials/cover.png)

# 机器学习加速 MD-Raman：有限温拉曼谱的非谐统计力学框架

> **论文**：David A. Egger, Manuel Grumet & Tomáš Bučko, *Machine learning accelerates Raman computations from molecular dynamics for materials science*, **J. Chem. Phys.** 163, 120901 (**2025**).  
> **DOI**：[10.1063/5.0287358](https://doi.org/10.1063/5.0287358)  
> **类型**：Perspective（视角文章）  
> **机构**：Technical University of Munich；Comenius University Bratislava / Slovak Academy of Sciences.  
> **工具/数据**：[TheoFEM-TUM/MD-Raman](https://github.com/TheoFEM-TUM/MD-Raman)（GitHub，含 SiO$\_2$ 示例）

---

## 一、背景：谐声子拉曼不够，MD-Raman 又太贵

拉曼效应源于光与 **极化率 $\boldsymbol\alpha$ 调制** 的振动耦合；只有改变 $\boldsymbol\alpha$ 的模式拉曼激活。晶体谐近似下，由 **空间群对称性** 即可判定一阶拉曼活性声子。

但 **强非谐振动** 日益重要：热膨胀、热导、相变，乃至 **半导体带隙** [3–14] 等均受非谐影响。实验上，立方卤化物钙钛矿本应对称性 **Raman 沉默**，却出现显著强度，尤以低频 **Raman central peak（中央峰）** 为著 [15–21]——与非谐八面体倾斜等运动相关。

**谐声子路线**（DFT 声子 + DFPT 求极化率导数）：

$$
\frac{\partial \alpha_{\mu\nu}}{\partial Q_p}
$$

$Q\_p$ 为第 $p$ 个声子正则坐标，$\alpha\_{\mu\nu}$ 为极化率张量分量；由此得 **谐 Raman 谱**（Fig. 1 上栏）。

**MD-Raman**（统计力学 + MD + 极化率时间序列 $\boldsymbol\alpha(t)$）可 **原则上精确包含非谐** 与有限温效应（Fig. 1 下栏），但沿轨迹反复 **DFPT** 求 $\boldsymbol\alpha(t)$ 极贵，长期阻碍普及 [23–26]。

本文视角：**ML 力场（MLFF）** 已大幅加速 MD；**ML 学 $\boldsymbol\alpha(t)$** 近年进展显著 [30–31, 74–102]——二者结合使 **ML 加速 MD-Raman** 成为分子与材料表征的可行工具。

![图 1：谐声子法 vs MD-Raman 概念对比（原文 Fig. 1）——上：小位移 → $\partial\alpha\_{\mu\nu}/\partial Q\_p$ → 谐谱；下：全轨迹 → $\boldsymbol\alpha(t)$ 时间序列 → 非谐谱。](/img/posts/2026-06-05-egger-ml-md-raman-materials/fig01-phonon-vs-md-raman.png)

---

## 二、MD-Raman 统计力学框架（原文式 (1)–(11)）

以下公式 **直接对应** Egger et al. JCP 2025 §II 编号；符号与原文一致。

### 2.1 谱密度、Wiener–Khinchin 定理与位置自相关（式 (1)–(4)）

考虑 **平稳** 随机过程 $\chi(t)$。以粒子 $j$ 的笛卡尔位置分量 $r\_{\mu,j}(t)$（$\mu = x,y,z$）为例，**谱密度**：

$$
S_{r_{\mu,j}}(\omega) = \lim_{T\to\infty}\frac{1}{2T}\left|A_{r_{\mu,j}}(\omega)\right|^2
\tag{1}
$$

其中 $A\_{r\_{\mu,j}}(\omega)$ 为 $r\_{\mu,j}(t)$ 的 Fourier 变换：

$$
A_{r_{\mu,j}}(\omega) = \int_{-T}^{T} \mathrm{d}t\, r_{\mu,j}(t)\, e^{-i\omega t}
\tag{2}
$$

**Wiener–Khinchin（WK）定理**：

$$
S_{\chi}(\omega) = \int_{-\infty}^{\infty} \mathrm{d}t\, C_{\chi}(t)\, e^{i\omega t}
\tag{3}
$$

位置 **自相关函数**：

$$
C_{r_{\mu,j}}(t) = \lim_{T\to\infty}\frac{1}{2T}\int_{-T}^{T}\mathrm{d}\tau\, r_{\mu,j}(\tau)\, r_{\mu,j}(\tau+t) \equiv \left\langle r_{\mu,j}(\tau)\cdot r_{\mu,j}(\tau+t)\right\rangle_{\tau}
\tag{4}
$$

Fig. 2 中展示 **归一化自相关**（以 Si 晶体 300 K MD 为例）：

$$
\tilde{C}_{r_{x,j}}(t) = \frac{C_{r_{x,j}}(t)}{C_{r_{x,j}}(0)}
$$

![图 2：bulk Si @ 300 K 的时序、归一化自相关与谱密度（原文 Fig. 2）。(a)–(c) 位置 $r\_{x,j}$；(d)–(f) 速度 $\dot{r}\_{x,j}$；(g)–(i) 介电张量速度 $\dot{\alpha}\_{xx}$（涨落与极化率张量 $\boldsymbol\alpha$ 成正比）。](/img/posts/2026-06-05-egger-ml-md-raman-materials/fig02-si-md-correlation-spectral.png)

### 2.2 速度自相关、VDOS 与极化率速度自相关（式 (5)–(10)）

采用 **速度** $\dot{r}\_{\mu,j}(t)$ 更利于概念与数值 [33]。**速度自相关（VACF）**：

$$
C_{\dot{r}_{\mu,j}}(t) = \left\langle \dot{r}_{\mu,j}(\tau)\cdot \dot{r}_{\mu,j}(\tau+t)\right\rangle_{\tau}
\tag{5}
$$

由 WK 定理 [式 (3)]，**速度谱密度**：

$$
S_{\dot{r}_{\mu,j}}(\omega) = \int_{-\infty}^{\infty} \mathrm{d}t\, C_{\dot{r}_{\mu,j}}(t)\, e^{i\omega t}
\tag{6}
$$

$N$ 粒子体系总速度谱密度：

$$
S_{\dot{\mathbf r}}(\omega) = \sum_{j=1}^{N}\sum_{\mu=x,y,z} S_{\dot{r}_{\mu,j}}(\omega)
\tag{7}
$$

**振动态密度（VDOS）** 与质量加权速度谱密度相关 [33, 34]（本文取无额外前置因子的定义，纯谐情形下 $g(\omega)$ 与声子 DOS 仅差 **频率无关、温度有关** 的因子）：

$$
g(\omega) \propto \sum_{j=1}^{N} m_j\, S_{\dot{\mathbf r}_j}(\omega)
\tag{8}
$$

$g(\omega)$ 反映体系 **动能的频率分解**。

拉曼核心量是 **极化率时间变化**。定义 **极化率速度自相关（PACF）**：

$$
C_{\dot{\alpha}_{\mu\nu}}(t) = \left\langle \dot{\alpha}_{\mu\nu}(\tau)\cdot \dot{\alpha}_{\mu\nu}(\tau+t)\right\rangle_{\tau}
\tag{9}
$$

$\boldsymbol\alpha$ 及其时间导数为 **非局域** 量，无法唯一分解为原子贡献；但可分析张量分量 $\alpha\_{\mu\nu}$ 的时域关联。由 WK 定理：

$$
S_{\dot{\alpha}_{\mu\nu}}(\omega) = \int_{-\infty}^{\infty} \mathrm{d}t\, C_{\dot{\alpha}_{\mu\nu}}(t)\, e^{i\omega t}
\tag{10}
$$

此即 **极化率时间变化** 的频率分解，直接决定拉曼强度量级。

### 2.3 拉曼强度 $I(\omega)$（式 (11)）

在 **Born–Oppenheimer**、非共振、纯振动（Placzek）过渡假设下，经 **球平均** 可得广泛适用的强度公式 [1, 35]：

$$
I(\omega) \propto \frac{(\omega_{\mathrm{in}}-\omega)^4}{\omega}\,\frac{1}{1-\exp\!\left(-\dfrac{\hbar\omega}{k_{\mathrm B}T}\right)}\,\frac{45\,S_{a^2} + 7\,S_{\gamma^2}}{45}
\tag{11}
$$

- $\omega\_{\mathrm{in}}$：入射激光频率；
- $S\_{a^2}$、$S\_{\gamma^2}$：张量不变量 $a^2$（平均极化率）与 $\gamma^2$（各向异性）**时间导数**的谱密度；
- 各向同性/各向异性拉曼谱由 $\dot{\alpha}\_{\mu\nu}$ 的时域关联给出：各向同性部分反映 **迹的平均**，各向异性部分反映 **张量不对称涨落** [33]。

谐声子框架中，$\partial\alpha\_{\mu\nu}/\partial Q\_p$ 在角色上对应 $S\_{\dot{\alpha}\_{\mu\nu}}$。

**MD-Raman 小结**：从 **极化率速度谱密度** $S\_{\dot{\alpha}\_{\mu\nu}}$ 经式 (11) 得 $I(\omega)$；前提是沿 MD 轨迹计算 $\boldsymbol\alpha(t)$。

---

## 三、瓶颈分析：DFPT 主导，MLFF 使问题更尖锐（§III）

**测试体系**：SiO$\_2$，$3\times3\times3$ 超胞，**300 K**，**20 ps** DFT-MD [32]。

![图 3：SiO₂ MD-Raman 成本与 MLFF 影响（原文 Fig. 3）。(a) 无 ML：DFPT ~85% vs DFT-MD；(b) DFT-MD 与 MLFF-MD 的 VDOS 一致；(c) MLFF-MD 后 DFPT 占 ~98%。](/img/posts/2026-06-05-egger-ml-md-raman-materials/fig03-sio2-cost-vdos-mlff.png)

| 情形 | MD 成本占比 | DFPT ($\boldsymbol\alpha(t)$) 占比 |
|------|-------------|-------------------------------------|
| DFT-MD + DFPT | ~15% | **~85%** |
| MLFF-MD + DFPT | **~2%**（VDOS 计算总成本降 **98%**） | **~98%** |

**采样与精度**（Nyquist + Verlet）：

- 可分辨最高频率 = 时间序列采样率之半；
- 频率分辨率随 **极化率快照数**（时间序列长度）增加；
- 典型 MD 步长 **1 fs**：~3000 cm$^{-1}$ 模式 Verlet 误差可达 **~40 cm$^{-1}$**；~1000 cm$^{-1}$ 仅 **~1 cm$^{-1}$** [25, 33]；
- 准确分辨 Raman 各峰通常需 **数百至约 1000** 个 $\boldsymbol\alpha$ 快照；
- 典型 DFPT 标度 **$\mathcal{O}(N^3)$**，大体系瓶颈更严重。

**结论**：MLFF 解决 MD 后，**预测 $\boldsymbol\alpha(t)$** 成为 MD-Raman 的 **决定性瓶颈**。

---

## 四、ML 加速 $\boldsymbol\alpha(t)$：工作流与模型族（§IV）

### 4.1 ML 加速 MD-Raman 概念流程（Fig. 4）

![图 4：ML 加速 MD-Raman 概念图（原文 Fig. 4）——轨迹拆训练集 $\{\mathcal{X}\_T\}$ 与生产集 $\{\mathcal{X}\_P\}$；$\alpha\_{\mu\nu}(\{\mathcal{X}\_T\})$ 由 DFPT 标注并训练 ML；ML 预测生产集得完整 $\alpha\_{\mu\nu}(t)$。](/img/posts/2026-06-05-egger-ml-md-raman-materials/fig04-ml-accelerated-md-raman-workflow.png)

设目标体系已有目标温度 **足够长 MD 轨迹**：

1. 抽取原子坐标训练集 $\{\mathcal{X}\_T\}$（可 **主动学习** 选点 [69]）；
2. DFPT 得标签 $\alpha\_{\mu\nu}(\{\mathcal{X}\_T\})$；
3. 训练 ML 模型：输入坐标 → 输出 $\boldsymbol\alpha$；
4. 对生产集 $\{\mathcal{X}\_P\}$ 预测 → $\alpha\_{\mu\nu}(t)$ → 按 §II 求 $I(\omega)$。

亦可用同一轨迹的 train/val/test 划分，在 **独立生产轨迹** 上预测 Raman。

### 4.2 噪声容忍（Fig. 5 + Appendix）

![图 5：合成信号噪声对谱密度的影响（原文 Fig. 5）——100 THz 纯余弦 vs 每步 i.i.d. 高斯噪声；噪声降低信噪比但不移峰位。](/img/posts/2026-06-05-egger-ml-md-raman-materials/fig05-noise-spectral-density.png)

Appendix 合成数据：频率 **100 THz**，步长 **1 fs**，长度 **1000** 点；噪声为振幅相对 $\sigma=1.0$ 的高斯分布。**ML 引入的 $\boldsymbol\alpha(t)$ 噪声** 在时间统计独立且 SNR 足够时，**不改变峰位**，对 $I(\omega)$ 有一定容忍度。

### 4.3 极化率模型与 ML 方法脉络

**经典局域模型**（Raman 长期沿用 [1, 43]）：

| 模型 | 思路 |
|------|------|
| 原子极化率加和 | 标量原子 $\alpha$ 相加，缺方向性 |
| **BPM**（Long & Bell）[40] | 键极化率加和，部分方向性 |
| **Thole** [41, 42] | 原子极化率 + 诱导偶极自洽 |

沿 MD 轨迹用 BPM 等近似 $\boldsymbol\alpha(t)$ 近年重新受关注 [61, 64–66]。

**核方法（kernel）**

- **λ-SOAP** [74]：在 GPR 中纳入张量旋转对称；SOAP [75] 的张量推广；
- Raimbault et al. [78]：分子晶体 Raman；SOAP vs λ-SOAP 显著差异；
- **Δ-ML**：学「快速基线 − DFPT」；Egger 组 SiO$\_2$ [32]：Δ-ML 所需 DFPT 样本 **< 直接 ML 一半**；基线为原子位移线性响应近似；
- 其他：CM / BoB / MBTR [79, 80]；AlphaML [82, 83]；肽段氨基酸迁移 [84]；从电子密度学响应 [85–87]。

**神经网络**

- Sommers et al. [93]：局域 $\boldsymbol\alpha$ + 双网络嵌入/拟合；配 NN-MLFF 算液态水 Raman；
- **PAINN** 等变消息传递 [96, 97]；
- 高阶张量传递 [98]、高维势 [99]、外场响应网络 [100, 101]、**NEP 张量性质** [102] 等持续进展。

**应用亮点**（不完全列举）：超 Raman [103]、溶剂效应 [104]、LO/TO 分裂 [105]、界面/体相光谱 [106, 107]；开源包 [108] 降低集成门槛。

### 4.4 SiO$\_2$ 验证：50 次 DFPT 训练 ≈ 1000 次全 DFPT 谱（Fig. 6）

![图 6：SiO₂ @ 300 K Raman——同一条轨迹上 DFPT 与 ML 极化率（原文 Fig. 6；改编自 Grumet et al. J. Phys. Chem. C 128, 6464 (2024)）。](/img/posts/2026-06-05-egger-ml-md-raman-materials/fig06-sio2-raman-dfpt-vs-ml.png)

| 方案 | DFPT 次数 | CPU 时间（量级） |
|------|-----------|------------------|
| 全 DFPT MD-Raman | **1000** | **~800 core-h** |
| ML 学 $\boldsymbol\alpha$（λ-SOAP / Δ-ML 管线 [32]） | **50**（训练） | **~40 core-h** |

计算负担降 **~95%**，Raman 谱 **目视无精度损失**。

---

## 五、结论与展望（§V）

**核心信息**

1. **MD-Raman** 无需谐近似与单一平衡结构，适合 **强非谐、无明确平衡结构** 的分子/晶体/非晶/液体；
2. 瓶颈始终是 **$\boldsymbol\alpha(t)$ 的量子力学计算**（DFPT），常占 **80%–90%+**；MLFF 加速 MD 后占比可升至 **~98%**；
3. **物理先验 + ML**（张量等变、Δ-ML、BPM 基线等）与 **核方法 / 神经网络** 两条线并行，已有多款 **低成本、第一性原理** $\boldsymbol\alpha(t)$ 预测器；
4. **MLFF + ML-$\boldsymbol\alpha$** 使 MD-Raman 从「数十年前的昂贵方案」变为 **材料科学常规工具**。

**未来方向**（原文讨论）

| 方向 | 要点 |
|------|------|
| 核量子效应 | 彩色 Langevin 恒温 [109]；全量子 MD ML |
| 超大体系 | 百万原子 MLFF-MD 下，DFPT 训练仍贵；深度 NN 绕过 DFPT 标签 [110] |
| DFT 精度 | 半局域泛函 **绝对极化率** 常不准 [111]；需系统 benchmark |
| 化学空间泛化 | 非局域 $\boldsymbol\alpha$ 表示、共振 Raman、跨体系协议 |

---

## 六、方法摘要（Quick Reference）

| 模块 | 内容 |
|------|------|
| 谐 Raman | $\partial\alpha\_{\mu\nu}/\partial Q\_p$ → 谐 $I(\omega)$ |
| MD-Raman 核心 | 式 (1)–(10)：$S\_{\dot{\alpha}\_{\mu\nu}}(\omega)$；式 (11)：$I(\omega)$ |
| 关键输入 | MD 轨迹 + $\boldsymbol\alpha(t)$（DFPT 或 ML） |
| 瓶颈 | DFPT $\boldsymbol\alpha(t)$；$\mathcal{O}(N^3)$；需 ~$10^2$–$10^3$ 快照 |
| ML 加速 | $\{\mathcal{X}\_T\}\xrightarrow{\mathrm{DFPT}}\alpha\_{\mu\nu}(\{\mathcal{X}\_T\})\xrightarrow{\mathrm{train}}\mathrm{ML}\xrightarrow{\mathrm{predict}}\alpha\_{\mu\nu}(t)$ |
| 示例 | SiO$\_2$：50 DFPT / ~40 core-h → 谱 ≈ 1000 DFPT / ~800 core-h |
| 软件 | [MD-Raman](https://github.com/TheoFEM-TUM/MD-Raman)；featomic / metatensor [38] |

---

## 七、总结

Egger, Grumet & Bučko 在 JCP 2025 视角文中系统回顾 **MD-Raman 的统计力学基础**（式 **(1)–(11)** 全链：从 $C\_{\dot{\alpha}\_{\mu\nu}}$ 到 Bose 因子修正的 $I(\omega)$），指出 **DFPT 求 $\boldsymbol\alpha(t)$** 为首要瓶颈，并综述 **λ-SOAP、Δ-ML、PAINN、NEP** 等 ML 路线如何与 **MLFF** 协同，将 SiO$\_2$ 等案例的计算量降低 **一个数量级以上** 而保持谱形。对 **钙钛矿中央峰、固态离子导体、蛋白质与非晶** 等需 **有限温非谐拉曼** 的场景，ML 加速 MD-Raman 正成为与实验互补的 **第一性原理表征** 工具。

---

*解读基于 Egger et al., J. Chem. Phys. **163**, 120901 (2025) 正文；公式编号与原文 §II 式 (1)–(11) 对齐；Appendix 合成噪声参数见原文。*

## 延伸阅读（站内）

- [外电场下的偶极与振动光谱：Chen 等用场感知 MACE + 变分 QEq 统一预测响应性质——Chen & Luber 2026 解读](/2026/06/04/外电场下的偶极与振动光谱-Chen-等用场感知-MACE-+-变分-QEq-统一预测响应性质——Chen-&-Luber/)

