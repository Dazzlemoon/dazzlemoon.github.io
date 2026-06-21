---
layout:     post
title:      倒空间机器学习方法谱系：ReciNet、RSNN、EquiEwald、PRDNet 与 IBZ-PRD 的区别
subtitle:   从任务目标、倒空间变量、结构因子、对称性、融合位置和谱学迁移六个维度，对比 ReciNet、RSNN、EquiEwald、PRDNet 及 PRDNet 的 IBZ 路径改写：哪些方法适合晶体性质预测，哪些适合长程 MLIP，哪些更接近高精度 IR/Raman 与声子色散。
date:       2026-06-21
author:     qqz
header-img: img/post-bg-desk.jpg
catalog:    true
tags:
    - 倒空间
    - ReciNet
    - RSNN
    - EquiEwald
    - PRDNet
    - IBZ-PRD
    - MLIP
    - 方法对比
---

这几篇方法表面上都在说“倒空间”“Fourier”“长程”，但它们并不是同一种东西。真正关键的差别在于：**倒空间被当作什么物理对象**。

- **ReciNet**：把倒空间当作晶体性质预测中的**长程特征通道**；
- **RSNN**：把倒空间当作机器学习势能中的**可学习长程能量核 / 描述符**；
- **EquiEwald**：把倒空间当作 SO(3) 等变 MLIP 中的**张量级长程消息传递空间**；
- **PRDNet**：把倒空间当作晶体性质预测中的**伪粒子衍射全局指纹**；
- **IBZ-PRD 改写**：把 PRDNet 的 Miller 衍射采样换成**声子/能带色散 native 的 $\mathbf{q}$ 路径采样**。

如果用一句话概括：**ReciNet 和 PRDNet 更像“全局结构表示增强器”，RSNN 和 EquiEwald 更像“长程物理相互作用建模器”，IBZ-PRD 则是把 PRDNet 往谱学坐标系迁移的工程化桥梁。**

---

## 一、先给结论：五种方法分别解决什么问题

| 方法 | 主任务 | 倒空间变量 | 核心输出 | 最强优势 | 主要边界 |
| ------ | ------ | ------ | ------ | ------ | ------ |
| **ReciNet** | 晶体性质预测 | 分数坐标 $\mathbf{f}$ + 倒格矢 $\mathbf{k}\_m$ | 原子/图级长程表示 | JARVIS / MP / MatBench 多性质强，端到端、可插拔 | 主要预测标量性质，不直接给能量-力自洽 MLIP |
| **RSNN** | 长程 MLIP + 强度性质 | 倒格矢 $\mathbf{k}$，结构因子 $S(\mathbf{k})$ | $E\_{\mathrm{long}}$、$h\_{\mathrm{long}}$ | 可学习任意径向长程核，适合库仑/色散/带电缺陷 | 标量不变路线，角向/多极各向异性表达有限 |
| **EquiEwald** | 等变 MLIP，能量/力 | 倒易波矢 $k\_\alpha$ + irrep 特征 | 等变长程更新 $x\_i^{\mathrm{Ewald}}$ | 保留 SO(3) 张量结构，适合各向异性多极长程 | 架构更复杂，成本高于标量倒空间方法 |
| **PRDNet** | 晶体性质预测 | Miller 集 $H=(h,k,l)$ | 全局衍射指纹 $\mathbf{F}\_{\mathrm{concat}}$ | 伪粒子形式因子解决局域表示碰撞，消融证据强 | 静态衍射坐标，不等于声子/IR/Raman 物理 |
| **IBZ-PRD** | 谱学导向表示迁移 | IBZ 高对称路径 $\{\mathbf{q}\_\ell\}$ | 路径响应指纹 $\mathcal{R}(\mathbf{q}\_\ell)$ | 采样坐标贴近声子色散、LO-TO、低 $q$ 电响应 | 路径是轻量探针，不是完整动力学矩阵或 DFPT 替代 |

---

## 二、共同母题：局域 GNN 为什么不够

这些方法都从同一个矛盾出发：晶体不是有限分子，而是无限周期体系。局域 GNN 只在截断半径 $r\_{\mathrm{cut}}$ 内聚合邻居，天然擅长短程化学键，却容易忽略：

1. 离子晶体中的长程 Coulomb 相互作用；
2. 缺陷/带电超胞中的背景场；
3. 带隙、弹性、稳定性等对全局周期敏感的性质；
4. LO-TO 分裂、声子色散、IR/Raman 强度中的低 $q$ 长程电响应；
5. 局域环境相似但晶格整体不同导致的 representation collision。

倒空间的优势在于：周期性结构天然可以写成相位因子求和。最基本的结构因子形式是：

$$
S(\mathbf{k})=\sum_j a_j\,e^{-i\mathbf{k}\cdot\mathbf{r}_j}
$$

不同方法的差别就在于：$a\_j$ 是什么、$\mathbf{k}$ 怎么选、$S(\mathbf{k})$ 后面如何处理。

---

## 三、ReciNet：性质预测里的“倒空间长程消息通道”

ReciNet 的目标是 **crystal property prediction（CPP）**，即从晶体结构预测形成能、带隙、模量、$E\_{\mathrm{hull}}$ 等标量性质。它不是 MLIP，因此不以能量-力自洽、MD 稳定性为核心指标。

### 3.1 它怎么用倒空间

ReciNet 用两套表示并行更新：

- **local branch**：半径图 GNN，处理短程几何与化学；
- **global branch**：ReciprocalBlock，处理分数坐标和倒格矢。

它的倒空间聚合类似：

$$
\mathbf{r}_m=\sum_{j\in\mathcal{I}_m}\mathbf{h}_{j,\mathrm{global}}^\ell
\exp(-i\mathbf{k}_m^\top\mathbf{f}_j)
$$

再逆变换回原子级全局特征：

$$
\tilde{\mathbf{h}}^\ell_{\mathrm{global}}
=
\sum_{j\in\mathcal{I}_m}
\exp(i\mathbf{k}_m^\top\mathbf{f}_j)\,
\mathbf{r}_m\,\mathbf{W}_{\mathrm{filter}}
$$

这里的关键不是显式写一个 Coulomb 核，而是让模型学到一个 **long-range feature update**。

### 3.2 优势

ReciNet 的优势在“工程有效性”：

- 用分数坐标 $\mathbf{f}$，天然对齐周期边界；
- 不需要 EwaldMP 那种人为超胞网格；
- ReciprocalBlock 可以插到 CGCNN、Matformer 等骨干中；
- 在 JARVIS、Materials Project、MatBench 上大面积刷新或接近 SOTA；
- ReciNet-MT 用 MoE 做多性质预测，OPT/MBJ 带隙之间出现正迁移。

### 3.3 边界

ReciNet 的倒空间模块输出的是**表征**，不是显式物理能量项。它适合回答：

> “这个晶体的形成能、带隙、模量大概是多少？”

但不直接回答：

> “这个体系在 MD 中每一步的能量、力、应力是否严格自洽？”

因此它更接近 **property predictor**，而非 **force field**。

---

## 四、RSNN：从 Ewald 到“可学习长程核”

RSNN 的核心更接近物理 MLIP：它显式把总能量拆成短程局域项和倒空间长程项：

$$
E_{\mathrm{total}}
=
\sum_j \varepsilon_j^{\mathrm{short}}
+E_{\mathrm{long}}
+E_{\mathrm{back}}
$$

### 4.1 RSP：可学习倒空间势

传统 Ewald 倒空间长程能量可写为：

$$
E_{\mathrm{long}}^{\mathrm{Ewald}}
=
\frac{1}{2V}
\sum_{\mathbf{k}\neq\mathbf{0}}
\Phi_n(\lvert\mathbf{k}\rvert)\,
\lvert S(\mathbf{k})\rvert^2
$$

RSNN 的关键替换是：

$$
E_{\mathrm{long}}
=
\frac{1}{V}
\sum_{\mathbf{k}\neq\mathbf{0}}
\mathrm{FCN}(\lvert\mathbf{k}\rvert)\,
\lvert S(\mathbf{k})\rvert^2
$$

也就是说，RSNN 不再预设长程相互作用一定是 $1/r$ 或 $1/r^6$，而是让神经网络学习倒空间核函数。

### 4.2 RSD：倒空间描述符

RSNN 还有 RSD，用于带隙等强度性质：

$$
h_{\mathrm{long}}
=
\sum_{\mathbf{k}\neq\mathbf{0}}
\mathrm{FCN}\left(\lvert\mathbf{k}\rvert,\frac{\lvert S(\mathbf{k})\rvert}{V}\right)
\left\lvert\frac{S(\mathbf{k})}{V}\right\rvert^2
$$

这里的 $\lvert S(\mathbf{k})\rvert/V$ 是为了保证强度性质对单胞选择不敏感。

### 4.3 优势

RSNN 的强项是**径向长程物理**：

- 能统一拟合 Coulomb、van der Waals 以及混合长程核；
- 对带电缺陷、离子晶体、极性体系有清晰物理动机；
- $E\_{\mathrm{long}}$ 可对结构求导，因而能进入力和应力；
- RSP/RSD 分工清楚：一个做广延能量，一个做强度描述符。

### 4.4 边界

RSNN 的结构因子本质上是标量：

$$
S(\mathbf{k})=\sum_j q_j e^{-i\mathbf{k}\cdot\mathbf{r}_j}
$$

能量依赖 $\lvert S(\mathbf{k})\rvert^2$ 和 $\mathrm{FCN}(\lvert\mathbf{k}\rvert)$，所以它主要表达**各向同性径向核**。这对 Coulomb / 色散很自然，但对偶极-偶极、四极、多极矩方向耦合这类角向结构更吃力。

---

## 五、EquiEwald：把倒空间长程提升到 SO(3) 等变张量层

EquiEwald 与 RSNN 共享“倒空间学习长程”的思想，但技术路线不同。RSNN 是 **E(3) 不变标量路线**，EquiEwald 是 **SO(3) 等变张量路线**。

### 5.1 结构因子从标量变成 irrep

RSNN 的结构因子是标量 $S(\mathbf{k})$；EquiEwald 对每个角动量度 $\ell$ 和磁分量 $m$ 构造结构因子：

$$
s_{\alpha,m}^{(\ell)}
=
\sum_j
\exp(i k_\alpha\cdot r_j)\,
x_j^{(\ell,m)}
$$

这里 $x\_j^{(\ell,m)}$ 不是一个标量电荷，而是等变 GNN 输出的 irrep 特征。这样，倒空间里保留了角向信息。

### 5.2 长程更新回到表示空间

EquiEwald 不是先算一个标量长程能量再加到总能量里，而是计算一个等变长程更新：

$$
x_i^{\mathrm{Ewald}}
=
\mathrm{Gate}\left(
W_g h_{i,\ell=0},
\bigoplus_{\ell=0}^{\ell_{\max}}
\mathrm{MLP}^{(\ell)}
\left([M_m^{(\ell)}(r_i)]_{m=-\ell}^{\ell}\right)
\right)
$$

再与局域更新融合：

$$
x_i^{t+1}
=
\frac{1}{\sqrt{3}}
\left(x_i^t+x_i^{\mathrm{Local},t}+x_i^{\mathrm{Ewald},t}\right)
$$

### 5.3 优势

EquiEwald 的优势是**各向异性长程相互作用**：

- 保留 SO(3) 等变性，适合能量-力预测；
- 能表达偶极、四极、多极耦合等方向相关长程效应；
- 对分子二聚体、蛋白质、超分子、周期催化体系都有改进；
- 长程信息在表示空间与局域等变特征融合，而不只是能量标量相加。

### 5.4 边界

EquiEwald 更重，也更依赖等变骨干、irrep 分解和通道设计。若任务只是大规模晶体标量性质预测，ReciNet/PRDNet 这类轻量全局表征可能更划算；若长程相互作用主要是标量径向核，RSNN 可能更直接。

---

## 六、PRDNet：伪粒子衍射，全局结构指纹而非力场

PRDNet 的任务与 ReciNet 更接近：都是晶体性质预测。但 PRDNet 的倒空间不是 Ewald 物理核，而是**伪粒子衍射指纹**。

### 6.1 从 XRD 结构因子出发

传统衍射结构因子：

$$
F(\mathbf{Q})
=
\sum_j f_j(\mathbf{Q})e^{-i\mathbf{Q}\cdot\mathbf{r}_j}
$$

PRDNet 把固定 X 射线形式因子改成可学习形式因子：

$$
f_i^*(H)=\mathrm{MLP}_{\mathrm{form}}(h_i^{(L)})
$$

然后在 Miller 集 $H=(h,k,l)$ 上计算 Re/Im 结构因子，拼成全局倒空间指纹。

### 6.2 它解决的核心问题

PRDNet 针对的是 **representation collision**：两个晶体局部邻域几乎一样，但晶格形状、全局周期不同，局域 GNN 可能给出相似表示。衍射分支通过全晶格相干求和补足全局周期信息。

### 6.3 优势

- 与晶体衍射物理直觉一致；
- 可学习伪形式因子能区分同元素不同局域环境；
- 保留 Re/Im，比只用强度 $\lvert F\rvert^2$ 信息更丰富；
- 与图分支模态级融合，消融中 NoDiff 退化明显；
- 适合标量晶体性质预测中的全局周期补强。

### 6.4 边界

PRDNet 的 Miller 集是静态衍射坐标，适合 XRD / 电子衍射式结构指纹。但它不等于：

- 声子动力学矩阵 $D(\mathbf{q})$；
- Born 有效电荷 $Z^{\ast}$；
- 极化率导数 $\partial\alpha/\partial Q$；
- IR/Raman 强度。

因此，“借 PRDNet 的伪粒子思想做谱学”是合理的；“把 XRD 结构因子直接当 Raman 谱”是不合理的。

### 6.5 PRDNet 和 ReciNet 到底哪里不同

PRDNet 与 ReciNet 都是晶体性质预测方法，也都试图补足局域 GNN 的长程盲区。但二者的“倒空间”角色完全不同。

**第一，倒空间对象不同。**

ReciNet 在倒空间里变换的是**神经网络原子嵌入**：

$$
\mathbf{r}_m
=
\sum_j \mathbf{h}_{j,\mathrm{global}}
\exp(-i\mathbf{k}_m^\top\mathbf{f}_j)
$$

也就是说，倒空间是一个 **message passing / feature update 的计算域**。原子特征进入倒空间，经过可学习滤波，再回到原子级表示。

PRDNet 在倒空间里计算的是**伪衍射结构因子**：

$$
F_{\mathbf{h}}^*
=
\sum_j f_j^*(\mathbf{h})\,
\exp(-2\pi i\,\mathbf{h}\cdot\mathbf{r}_{j,\mathrm{frac}})
$$

这里的 $f\_j^{\ast}(\mathbf{h})$ 是环境敏感的“伪粒子形式因子”。倒空间分支更像给整个晶体拍一张 learned diffraction fingerprint，而不是更新每个原子的隐藏状态。

**第二，信息回流位置不同。**

ReciNet 的倒空间信息在每个 ReciNet Block 内回流到节点表示，属于**层内融合**：

$$
\mathbf{h}^{\ell+1}
\leftarrow
\mathrm{LocalGNN}(\mathbf{h}^{\ell})
+
\mathrm{ReciprocalBlock}(\mathbf{h}^{\ell},\mathbf{f},\mathbf{k})
$$

PRDNet 的衍射分支通常在结构级别生成 $\mathbf{F}\_{\mathrm{concat}}$，再与图分支的全局向量拼接，属于**模态级融合**：

$$
\mathbf{z}
=
\mathrm{MLP}_{\mathrm{fusion}}
\left(
\mathbf{g}_{\mathrm{graph}}
\oplus
\mathbf{d}_{\mathrm{PRD}}
\right)
$$

所以 ReciNet 更像“每一层都让局域 GNN 看见倒空间长程”；PRDNet 更像“图分支看局域，衍射分支单独看全局，最后合并判断”。

**第三，物理归纳偏置不同。**

ReciNet 的偏置来自**周期 Fourier 表示 + 分数坐标一致性**。它关心的是：长程周期信息怎样变成更好的 learned representation。

PRDNet 的偏置来自**晶体衍射**。它关心的是：不同晶体在一组 Miller 探针下的相干响应是否能区分局域 GNN 分不开的全局结构。

**第四，适合的问题不同。**

| 问题 | ReciNet 更自然 | PRDNet 更自然 |
| ------ | ------ | ------ |
| 想让倒空间信息参与深层节点更新 | 是 | 否 |
| 想得到一个强全局周期指纹 | 可以，但不是主设计 | 是 |
| 担心 representation collision | 有帮助 | 这是核心动机 |
| 想解释为 learned diffraction | 不太自然 | 很自然 |
| 想往声子/谱学路径迁移 | 需要改造 $\mathbf{k}$ 采样和监督 | 可沿 PRD → IBZ-PRD 迁移 |

因此二者不是“谁替代谁”。**ReciNet 更像倒空间消息传递模块，PRDNet 更像倒空间全局观测模块**。如果做高精度谱学，两者甚至可以组合：ReciNet 提供层内长程更新，PRD/IBZ 分支提供路径级全局响应指纹。

---

## 七、IBZ-PRD：把 PRDNet 的 Miller 集改成谱学 $\mathbf{q}$ 路径

IBZ-PRD 不是一篇独立的完整论文方法，更像一个**PRDNet 谱学迁移方案**：保持“可学习响应 × Bloch 相位相干求和 × 全局融合”的骨架，但把采样坐标从 Miller 整数点换成 IBZ 高对称路径。

### 7.1 原版 PRDNet 的采样

Miller 版：

$$
\mathbf{G}_{\mathbf{h}}
=
h\mathbf{b}_1+k\mathbf{b}_2+l\mathbf{b}_3,
\quad
(h,k,l)\in\mathbb{Z}^3
$$

适合静态衍射和全局晶体指纹。

### 7.2 谱学为什么要换成 IBZ 路径

声子色散、能带、低 $q$ 介电响应都天然是 $\mathbf{q}$ 的函数：

$$
D(\mathbf{q})\mathbf{e}_\nu(\mathbf{q})
=
\omega_\nu^2(\mathbf{q})\mathbf{e}_\nu(\mathbf{q})
$$

高对称路径是 IBZ 的一维骨架：

$$
\mathcal{Q}_{\mathrm{path}}
=
\{\mathbf{q}(t):t\in[0,1]\}
\subset
\mathrm{IBZ}
$$

离散后得到 $\{\mathbf{q}\_\ell\}\_{\ell=1}^{N\_q}$。这比高 $\lvert\mathbf{h}\rvert$ Miller 点更贴近 LO-TO、光学支、低 $q$ 电响应。

### 7.3 IBZ-PRD 的最小接口

可学习路径响应：

$$
\mathcal{R}(\mathbf{q}_\ell)
=
\sum_i r_{i,\ell}\,
\exp(2\pi i\,\mathbf{q}_\ell\cdot\mathbf{r}_{i,\mathrm{frac}})
$$

这里 $r\_{i,\ell}$ 可以是标量、向量或张量响应，取决于要预测的对象：

| 目标 | 可学习响应 | 物理类比 |
| ------ | ------ | ------ |
| 声子频率/稳定性 | 标量 $r\_{i,\ell}$ | 路径几何探针 |
| IR 强度 | 向量 $\boldsymbol{\mu}\_i^{\ast}(\mathbf{q}\_\ell)$ | 偶极/Born 电荷 proxy |
| Raman 强度 | 张量 $\boldsymbol{\alpha}\_i^{\ast}(\mathbf{q}\_\ell)$ | 极化率响应 proxy |
| LO-TO / 极性效应 | 低 $q$ 响应通道 | 长程静电 proxy |

### 7.4 在 $\mathbf{q}\_\ell$ 上求和，与 Miller 集上求和有什么区别

两者数学外形相似，都是“可学习权重 × 相位因子 × 全原子相干求和”。但物理坐标不同，导致它们看见的东西不同。

**Miller 集求和**：

$$
F_{\mathbf{h}}^*
=
\sum_j f_j^*(\mathbf{h})
\exp(-2\pi i\,\mathbf{h}\cdot\mathbf{r}_{j,\mathrm{frac}})
$$

其中 $\mathbf{h}=(h,k,l)\in\mathbb{Z}^3$，对应倒格点 $\mathbf{G}\_{\mathbf{h}}$。它的物理原型是 XRD / ED 的 Bragg 衍射：晶体对一组离散倒格点的静态散射响应。

**IBZ 路径求和**：

$$
\mathcal{R}(\mathbf{q}_\ell)
=
\sum_j r_{j,\ell}
\exp(2\pi i\,\mathbf{q}_\ell\cdot\mathbf{r}_{j,\mathrm{frac}})
$$

其中 $\mathbf{q}\_\ell$ 是 BZ / IBZ 高对称路径上的连续波矢离散点，通常不是整数 Miller 指数。它的物理原型不是 Bragg 衍射峰，而是声子、能带、介电响应沿 $\mathbf{q}$ 的色散。

差别可以这样看：

| 维度 | Miller 集 $H$ | IBZ 路径 $\{\mathbf{q}\_\ell\}$ |
| ------ | ------ | ------ |
| 坐标类型 | 整数倒格点 | BZ 内连续路径点 |
| 典型相位 | $e^{-2\pi i\mathbf{h}\cdot\mathbf{r}\_{\mathrm{frac}}}$ | $e^{2\pi i\mathbf{q}\_\ell\cdot\mathbf{r}\_{\mathrm{frac}}}$ |
| 物理原型 | Bragg 衍射 / 静态结构因子 | 声子色散 / 能带 / 低 $q$ 响应 |
| 采样重点 | 多个倒格点方向和频率 | $\Gamma$、高对称点、路径段 |
| 适合输出 | 全局结构指纹、晶体性质 | 谱学路径指纹、色散相关性质 |
| 风险 | 不能直接解释为 IR/Raman | 路径稀疏，不能替代全 BZ 积分 |

最重要的是低 $q$ 区域。Miller 集的非零点一般是倒格点，最小非零 $\mathbf{G}$ 仍是 Bragg 周期尺度；而谱学里的 LO-TO、介电响应、极性声子强度通常绑定在 $\Gamma$ 附近的 $\mathbf{q}\to 0$ 行为。因此如果目标是 IR/Raman 或声子色散，$\mathbf{q}\_\ell$ 路径比 Miller 集更贴近物理自变量。

但也不能把 $\mathcal{R}(\mathbf{q}\_\ell)$ 误认为真正的动力学矩阵。它只是一个沿谱学坐标采样的 learned response fingerprint。要获得真正频率，仍需学习：

$$
D(\mathbf{q}_\ell)\mathbf{e}_\nu(\mathbf{q}_\ell)
=
\omega_\nu^2(\mathbf{q}_\ell)\mathbf{e}_\nu(\mathbf{q}_\ell)
$$

或至少用 $\omega\_\nu(\mathbf{q}\_\ell)$ 作为监督信号。

### 7.4.1 为什么还需要模式分辨形式

当前的 IBZ 路径指纹：

$$
\mathcal{R}(\mathbf{q}_\ell)
=
\sum_j r_{j,\ell}
\exp(2\pi i\,\mathbf{q}_\ell\cdot\mathbf{r}_{j,\mathrm{frac}})
$$

只有 $\mathbf{q}\_\ell$ 一个标签。它能告诉模型“这个路径点附近的全局响应长什么样”，但不能区分同一 $\mathbf{q}\_\ell$ 上的不同声子分支。真实声子模式的标签是：

$$
(\mathbf{q}_\ell,\nu)
$$

其中 $\nu$ 是声子分支 / 模式编号，$\mathbf{e}\_{\kappa\alpha,\nu}(\mathbf{q}\_\ell)$ 是胞内第 $\kappa$ 个原子沿方向 $\alpha$ 的模式本征矢。换句话说，$\mathbf{q}\_\ell$ 只描述**不同晶胞之间的相位关系**，而 $\nu$ 和 $\mathbf{e}\_{\kappa\alpha,\nu}$ 才描述**胞内哪些原子、沿哪些方向、以什么相对相位在振动**。

因此，如果目标只是预测一条整体谱：

$$
\hat{I}(\omega)
=
\mathrm{Decoder}\left(\{\mathcal{R}(\mathbf{q}_\ell)\}_{\ell}\right)
$$

那么当前形式可以作为 spectrum-level learned descriptor。它可能学到“结构对应什么谱形”，但缺少模式解释。

如果目标是高精度 IR/Raman，最好升级为模式分辨：

$$
\mathcal{R}_{\nu}(\mathbf{q}_\ell)
=
\sum_{\kappa,\alpha}
r_{\kappa\alpha}^{*}(\mathbf{q}_\ell)\,
e_{\kappa\alpha,\nu}(\mathbf{q}_\ell)
$$

这里 $r\_{\kappa\alpha}^{\ast}$ 是 learned response proxy，可以是伪力常数响应、伪偶极响应或伪极化率响应；$e\_{\kappa\alpha,\nu}$ 则把响应投影到第 $\nu$ 支真实声子模式上。

这样最终光谱可由模式贡献合成：

$$
\hat{I}(\omega)
=
\sum_{\nu}
\hat{A}_{\nu}\,
g(\omega-\hat{\omega}_{\nu})
$$

其中 $\hat{\omega}\_{\nu}$ 是模式峰位，$\hat{A}\_{\nu}$ 是 IR/Raman 强度，$g$ 是 Gaussian / Lorentzian 展宽函数。这个形式更接近实验光谱的物理来源：实验看到的是整体谱，但整体谱由多个模式峰叠加而来。

### 7.5 优势与边界

IBZ-PRD 的优势是**采样几何对齐谱学**：它让 PRDNet 的全局相干求和不再只看静态衍射点，而是沿声子/能带常用路径建立指纹。

但它仍然只是 embedding 探针。若要成为真正的高精度谱学模型，还需要：

- 对 $\omega\_\nu(\mathbf{q})$、力常数或 Hessian 的监督；
- 对 IR 强度、Born 电荷或偶极导数的监督；
- 对 Raman 强度、极化率张量导数的监督；
- 明确处理小群、简并、选择定则和 LO-TO 非解析项。

### 7.6 RSA 与 IBZ-PRD 的衔接

**Reciprocal-Space Attention（RSA）**（Ramasubramanian et al., arXiv:2510.13055, 2025）将线性 attention 映射到 Fourier 倒空间：以 **Fourier Positional Encoding（FPE）** $e^{i\mathbf{k}\cdot r}$ 与 **Ewald 权重** $w\_k$ 耦合 MACE 短程骨干，学习无预定义电荷的长程 electrostatics/dispersion（bulk water 的 $\chi\_{zz}(k\to 0)$ 等）。IBZ-PRD 最小接口见 **§7.3**；下文讨论二者如何组合。

#### 7.6.1 数学同构与任务分工

RSA 的 FPE 与 IBZ-PRD 的 Bloch 相位同属 $e^{i\mathbf{q}\cdot\mathbf{r}}$ 型全胞相干耦合；差别在于：

| 维度 | RSA | IBZ-PRD |
|------|-----|---------|
| 主任务 | 长程 **MLIP**（$E,F$ 自洽） | **谱学/色散**全局指纹 |
| 倒空间变量 | 倒格矢 $\mathbf{k}$ + Ewald $w\_k$ | IBZ 路径 $\mathbf{q}\_\ell$ |
| 核心运算 | FPE + 线性 attention | §7.3 一次相干求和 |
| 长程 electrostatics | 显式（$\chi\_{zz}$ 等） | 低 $q$/LO-TO 需额外处理 |

**结论**：二者**可衔接、不互替**——IBZ-PRD 解决「沿哪条 $\mathbf{q}$ 采样」，RSA 解决「长程如何在 per-atom 上耦合」。

#### 7.6.2 三种融合路线（由易到难）

**方案 A（推荐优先）**：以 IBZ 路径点（及 $\Gamma$ 附近低 $q$ 加密点）作为 RSA 的 $\mathbf{k}$ 网格，局域 GNN 与双支路并行：

```
局域 Graph → h_i^(local)
RSA@{q_ℓ}  → h_i^(LR)     ─┐
IBZ-PRD    → z_path(q_ℓ)  ─┴→ 融合 → 伪 μ*/α* → 模式/谱
```

与 **ReciNet + IBZ-PRD** 同构，但以 **FPE + Ewald 权重 attention** 替代 ReciNet 的 Fourier 块。

**方案 B**：在固定 $\mathbf{q}\_\ell$ 上用 RSA 型 attention 替代独立 $r\_{i,\ell}$，再对原子 pooling 得 $\mathcal{R}(\mathbf{q}\_\ell)$——同一 $\mathbf{q}\_\ell$ 上原子长程关联由 attention 学习，代价是失去 PRD 式轻量相干求和。

**方案 C（双任务）**：RSA 作 MLIP 保 $E,F$ 与 MD 自洽；IBZ-PRD 作谱学 head；联合色散/DFPT/实验谱监督——适合 SR-PhononNet 类路线，数据与工程成本最高。

#### 7.6.3 模式分辨与低 $q$

§7.4.1 的路径指纹只有 $\mathbf{q}\_\ell$ 标签；高精度 IR/Raman 宜升级为模式分辨 $\mathcal{R}\_{\nu}(\mathbf{q}\_\ell)$（见该节公式）。**RSA 的价值点**：由 per-atom 长程嵌入 $h\_m^{\mathrm{LR}}$ 生成 $r\_{\kappa\alpha}^{\ast}$，而非标量 $r\_{i,\ell}$；**低 $q$ / LO-TO** 正是 RSA 相对纯 IBZ 指纹的补强方向（RSA 原文 bulk water $\chi\_{zz}(k\to 0)$ benchmark）。

#### 7.6.4 主要困难与取舍

1. **任务不一致**：RSA 训 $E,F$；IBZ-PRD 训性质/谱——需联合监督或明确分工。  
2. **路径稀疏**：$N\_q\sim 10^1\text{–}10^2$ 够色散图，不够全 BZ 积分；$\Gamma$ 附近宜加密。  
3. **$w\_k$ 是否用于谱学支路**：MLIP/RSA 可保留 Ewald 权重；IBZ 探针支路可仅用 FPE attention。  
4. **等变与张量**：RSA 正文为标量；IR/Raman 需等变 FPE 扩展或伪 $\boldsymbol{\mu}^{\ast},\boldsymbol{\alpha}^{\ast}$ head。  
5. **正交单胞**：RSA 现实现限制 vs IBZ 对任意 Bravais 格通用——需统一分数坐标与 FPE 实现。

| 问题 | 判断 |
|------|------|
| 能否接入 IBZ-PRD？ | **能**，且与「Graph + IBZ 路径 + 长程响应」主线一致 |
| 最稳接法 | **方案 A**：$\{\mathbf{q}\_\ell\}$ 作 RSA 的 $\mathbf{k}$ 集，与 IBZ 指纹双支路融合 |
| 最大价值 | $\Gamma$/低 $q$/LO-TO、极性体系、模式分辨谱学 |
| 最大风险 | 双任务训练、等变 head、路径≠全 BZ |

---

## 八、五者的核心差异：不是“谁更先进”，而是“物理对象不同”

### 8.1 倒空间对象不同

| 方法 | 倒空间中被学习的对象 |
| ------ | ------ |
| ReciNet | 原子嵌入的 Fourier 全局更新 |
| RSNN | 长程能量核 $\mathrm{FCN}(\lvert\mathbf{k}\rvert)$ 与结构因子描述符 |
| EquiEwald | irrep 特征的倒空间等变消息 |
| PRDNet | 可学习伪粒子衍射指纹 |
| IBZ-PRD | 沿高对称 $\mathbf{q}$ 路径的谱学响应指纹 |

### 8.2 融合位置不同

| 方法 | 融合位置 | 含义 |
| ------ | ------ | ------ |
| ReciNet | 每层 block 内局域/全局更新融合 | 长程信息参与深层表征学习 |
| RSNN | 能量或图级描述符层融合 | 物理能量分解清楚 |
| EquiEwald | 等变表示空间融合 | 长程信息保留张量方向性 |
| PRDNet | 结构级模态融合 | 衍射分支作为全局指纹 |
| IBZ-PRD | 路径响应指纹 + 实空间特征融合 | 谱学坐标下的全局补充 |

### 8.3 对称性假设不同

| 方法 | 对称性路线 |
| ------ | ------ |
| ReciNet | 周期性 + 分数坐标一致性，主要面向不变标量性质 |
| RSNN | E(3) 不变 + 单胞选择一致性 |
| EquiEwald | SO(3) 等变，能量不变、力等变 |
| PRDNet | E(3) 不变，Miller 集闭包确保全局指纹稳定 |
| IBZ-PRD | IBZ / star / 小群约化，需额外关照谱学选择定则 |

---

## 九、如果面向高精度谱学，应该怎么选

### 9.1 只想提高晶体标量性质预测

优先考虑 **ReciNet 或 PRDNet**。

- 如果希望长程信息进入每层表征更新，选 ReciNet 思路；
- 如果特别担心局部表示碰撞、想引入强全局周期指纹，选 PRDNet 思路。

### 9.2 想做能量-力自洽 MLIP，并处理长程静电/缺陷

优先看 **RSNN**。

它的 RSP 直接对应 $E\_{\mathrm{long}}$，可导出力/应力；对库仑、色散、带电超胞等问题更自然。

### 9.3 想做极性、分子、界面、蛋白质等各向异性长程力场

优先看 **EquiEwald**。

它比 RSNN 更复杂，但能保留 SO(3) 等变张量信息，适合偶极/多极/极化等方向性强的体系。

### 9.4 想把 PRDNet 用到 IR/Raman 或声子色散

不要直接用 Miller 衍射强度。更合理的是 **IBZ-PRD**：

1. 用 seekpath / spglib 生成高对称路径；
2. 在 $\mathbf{q}\_\ell$ 上做可学习响应相干求和；
3. 分头监督频率、IR 强度、Raman 强度；
4. 对低 $q$、$\Gamma$ 点、LO-TO 分裂单独建模或加物理修正；
5. 做 L1/L2/L3 消融：频率、IR、Raman 不要混成一个 loss。

### 9.5 PRDNet → SR-PhononNet：把伪粒子形式因子改成伪偶极/伪极化率响应算子

如果要把 PRDNet 的思想迁移到 IR/Raman，高层逻辑不是“用衍射强度预测谱”，而是把 **可学习形式因子** 改写成 **可学习光谱响应算子**。

PRDNet 原版是：

$$
f_i^*(\mathbf{h})
=
\mathrm{MLP}_{\mathrm{form}}(\mathbf{h}_i^{(L)})
$$

然后求：

$$
F_{\mathbf{h}}^*
=
\sum_i f_i^*(\mathbf{h})
\exp(-2\pi i\,\mathbf{h}\cdot\mathbf{r}_{i,\mathrm{frac}})
$$

SR-PhononNet 可以把这一步改成两类响应算子。

**第一类：伪偶极响应算子，用于 IR。**

IR 强度来自振动模式引起的偶极矩变化。可令每个原子在路径点/频率通道上输出向量响应：

$$
\boldsymbol{\mu}_i^*(\mathbf{q}_\ell)
=
\mathrm{MLP}_{\mathrm{IR}}
\left(
\mathbf{h}_i^{(L)},\mathbf{q}_\ell
\right)
\in\mathbb{R}^{3}
$$

相干求和得到路径级偶极响应：

$$
\mathbf{M}(\mathbf{q}_\ell)
=
\sum_i
\boldsymbol{\mu}_i^*(\mathbf{q}_\ell)
\exp(2\pi i\,\mathbf{q}_\ell\cdot\mathbf{r}_{i,\mathrm{frac}})
$$

若有声子本征矢 $\mathbf{e}\_{i\nu}(\mathbf{q}\_\ell)$，可进一步让响应沿模式投影：

$$
\mathbf{M}_{\nu}(\mathbf{q}_\ell)
=
\sum_i
\boldsymbol{\mu}_i^*(\mathbf{q}_\ell)
\cdot
\mathbf{e}_{i\nu}(\mathbf{q}_\ell)
$$

IR 强度可用近似头学习，或采用物理形式：

$$
I_{\mathrm{IR},\nu}(\mathbf{q}_\ell)
\propto
\left\lVert
\mathbf{M}_{\nu}(\mathbf{q}_\ell)
\right\rVert^2
$$

**第二类：伪极化率响应算子，用于 Raman。**

Raman 强度来自振动导致的极化率张量变化。可令每个原子输出二阶张量：

$$
\boldsymbol{\alpha}_i^*(\mathbf{q}_\ell)
=
\mathrm{MLP}_{\mathrm{Raman}}
\left(
\mathbf{h}_i^{(L)},\mathbf{q}_\ell
\right)
\in\mathbb{R}^{3\times 3}
$$

相干求和：

$$
\mathbf{A}(\mathbf{q}_\ell)
=
\sum_i
\boldsymbol{\alpha}_i^*(\mathbf{q}_\ell)
\exp(2\pi i\,\mathbf{q}_\ell\cdot\mathbf{r}_{i,\mathrm{frac}})
$$

若有模式本征矢：

$$
\mathbf{A}_{\nu}(\mathbf{q}_\ell)
=
\sum_i
\boldsymbol{\alpha}_i^*(\mathbf{q}_\ell)
\cdot
\mathbf{e}_{i\nu}(\mathbf{q}_\ell)
$$

Raman 强度可以从 $\mathbf{A}\_\nu$ 的不变量构造，例如各向同性部分与各向异性部分，也可以先用监督学习近似：

$$
I_{\mathrm{Raman},\nu}(\mathbf{q}_\ell)
=
\mathrm{Head}_{\mathrm{Raman}}
\left(
\mathbf{A}_{\nu}(\mathbf{q}_\ell)
\right)
$$

### 9.6 一个可落地的 SR-PhononNet 路线

最小可行版本可以分三层，不必一开始就追求全物理闭环。

**L1：频率路线。**

先预测 $\omega\_\nu(\mathbf{q}\_\ell)$ 或若干目标谱线峰位。输入为图分支特征 + IBZ 路径响应指纹：

$$
\hat{\omega}_\nu(\mathbf{q}_\ell)
=
\mathrm{Head}_{\omega}
\left[
\mathbf{g}_{\mathrm{graph}}
\oplus
\mathcal{R}(\mathbf{q}_\ell)
\right]
$$

这一层验证的是：IBZ 路径相干指纹是否比纯局域 GNN 更能补足色散信息。

**L2：IR 强度路线。**

引入 $\boldsymbol{\mu}\_i^{\ast}(\mathbf{q}\_\ell)$，监督对象可以是 DFPT Born 有效电荷、模式有效电荷、IR intensity 或实验 IR 谱。目标是让模型学会“哪些模式带偶极活性”。

**L3：Raman 强度路线。**

引入 $\boldsymbol{\alpha}\_i^{\ast}(\mathbf{q}\_\ell)$，监督对象可以是 DFPT 极化率导数、Raman tensor 或 Raman spectrum。目标是让模型学会“哪些模式调制极化率”。

这三层最好分头训练/消融：

| 层级 | 学什么 | 需要的监督 | 风险 |
| ------ | ------ | ------ | ------ |
| L1 | 频率 / 色散 | 声子频率、Hessian、力常数 | 峰位准但强度不准 |
| L2 | IR 活性 | Born 电荷、偶极导数、IR 强度 | 偶极方向与选择定则处理不足 |
| L3 | Raman 活性 | 极化率导数、Raman tensor | 张量等变性与偏振几何较难 |

更稳妥的 SR-PhononNet 不应只直接输出整条谱，而应采用**模式级 + 谱级**两级结构。

**第一级：模式级预测。**

对每个 $(\mathbf{q}\_\ell,\nu)$ 预测：

$$
\left(
\hat{\omega}_{\nu}(\mathbf{q}_\ell),
\hat{A}^{\mathrm{IR}}_{\nu}(\mathbf{q}_\ell),
\hat{A}^{\mathrm{Raman}}_{\nu}(\mathbf{q}_\ell)
\right)
$$

其中 $\hat{\omega}\_{\nu}$ 是峰位 / 频率，$\hat{A}^{\mathrm{IR}}\_{\nu}$ 和 $\hat{A}^{\mathrm{Raman}}\_{\nu}$ 是模式强度。若只有 $\Gamma$ 点一阶 IR/Raman，则 $\mathbf{q}\_\ell$ 可先固定为 $\Gamma$ 或 $\Gamma$ 邻域。

**第二级：谱级合成。**

用展宽函数把模式级输出合成为整体谱：

$$
\hat{I}_{\mathrm{IR}}(\omega)
=
\sum_{\nu}
\hat{A}^{\mathrm{IR}}_{\nu}\,
g(\omega-\hat{\omega}_{\nu})
$$

$$
\hat{I}_{\mathrm{Raman}}(\omega)
=
\sum_{\nu}
\hat{A}^{\mathrm{Raman}}_{\nu}\,
g(\omega-\hat{\omega}_{\nu})
$$

如果训练数据只有实验整体谱，没有逐模式标签，可以再加一个小的 residual head：

$$
\hat{I}(\omega)
=
\hat{I}_{\mathrm{mode}}(\omega)
+
\Delta I_{\mathrm{NN}}(\omega)
$$

这样做的好处是：模式级 head 保留峰位、强度、活性来源的可解释性；谱级 head 保留对实验展宽、峰重叠、非谐修正和仪器响应的拟合能力。相比单纯用 $\mathcal{R}(\mathbf{q}\_\ell)$ 直接输出整条谱，这种两级模型更不容易把“谱形拟合”误当成“声子物理理解”。

### 9.7 这条路最需要注意的边界

1. **伪偶极/伪极化率不是物理可观测量本身**。它们是 learned proxy，必须通过 DFPT 或实验谱监督校准。
2. **$\Gamma$ 点与低 $q$ 要特殊处理**。IR/Raman 的一阶光学过程常在 $\Gamma$ 附近，LO-TO 分裂还需要非解析长程项。
3. **张量对称性不能忽略**。IR 是向量响应，Raman 是二阶张量响应；若只用标量 head，容易丢掉偏振与选择定则。
4. **路径采样不是完整 BZ 积分**。IBZ path 适合色散图与轻量指纹；热力学、态密度、宽谱积分仍需更密的 $\mathbf{q}$ 网格。
5. **PRDNet 的优点是“可学习探针”，不是“XRD 物理直接迁移”**。迁移时应保留相干求和和环境敏感响应，替换掉不适合谱学的 Miller/形式因子物理。

用一句话说，**SR-PhononNet 可走的路是：Graph 学局域力常数环境，IBZ 相干求和学全局相位与长程响应，伪偶极/伪极化率算子分别服务 IR/Raman 强度，最后用 DFPT/实验谱把这些 proxy 拉回物理可观测量。**

---

## 十、最终判断：它们之间的互补关系

这几条路线不是互相替代，而是可以组合：

- **ReciNet + IBZ-PRD**：晶体性质预测中同时加入倒空间长程更新和谱学路径指纹；
- **RSA + IBZ-PRD**：以 IBZ 路径点为 RSA 的 $\mathbf{k}$ 网格，双支路提供 per-atom 长程耦合与路径级谱学指纹（§7.6，方案 A）；
- **RSNN + PRD/IBZ descriptor**：MLIP 的能量长程项之外，再加入图级谱学/强度描述符；
- **EquiEwald + IBZ path supervision**：等变长程力场用于生成或约束声子色散；
- **PRDNet → SR-PhononNet**：把伪粒子形式因子改写成伪偶极/伪极化率响应算子。

最关键的取舍是：

| 你关心的问题 | 更合适的方法 |
| ------ | ------ |
| “晶体全局结构表示怎么补强？” | PRDNet / ReciNet |
| “长程能量项怎么可学习且可导？” | RSNN / **RSA**（Ewald-attention MLIP） |
| “各向异性长程力怎么保持等变？” | EquiEwald |
| “谱学色散该沿哪里采样？” | IBZ-PRD |
| “IR/Raman 强度怎么建模？” | IBZ-PRD + 伪响应算子 + DFPT/张量监督 |

一句话总结：

> **ReciNet 是倒空间表征增强，RSNN 是倒空间能量核学习，EquiEwald 是倒空间等变消息传递，PRDNet 是倒空间伪衍射指纹，IBZ-PRD 是把伪衍射迁移到声子/谱学 $\mathbf{q}$ 路径的坐标改写。**

如果后续目标是“高精度谱学思路”，最值得推进的不是单独押注某一个模型，而是把它们拆成可复用模块：**ReciNet 的层内长程更新、PRDNet 的可学习探针、RSNN/RSA 的长程物理核、EquiEwald 的等变张量约束、IBZ-PRD 的谱学路径采样**（RSA 与 IBZ-PRD 的衔接见 §7.6）。真正强的谱学模型，大概率会从这些模块的组合里长出来。

## 延伸阅读（站内）

- [ReciNet：倒空间感知的晶体长程建模——Nie 等 2026 解读](/2026/06/18/ReciNet-倒空间感知的晶体长程建模-Nie-等-2026-解读/)

