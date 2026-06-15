---
layout:     post
title:      EquiEwald：基于倒易空间的SO(3)等变神经势能——长程相互作用的全新范式
subtitle:   EquiEwald将Ewald求和嵌入不可约表示框架，通过在倒易空间进行等变消息传递来捕捉各向异性的多极长程相互作用，在带电分子二聚体、蛋白质构象动力学和周期性催化材料中均显著优于现有方法
date:       2026-06-15
author:     qqz
header-img: img/post-bg-desk.jpg
catalog:    true
tags:
    - EquiEwald
    - MLIP
    - SO(3)
    - 长程相互作用
    - Ewald
    - 倒易空间
    - irrep
    - 高精度谱学
---

![一图总结](/img/posts/2026-06-15-equiewald-so3-long-range/cover.png)

## 一、背景：长程相互作用的根本挑战

分子动力学（MD）模拟已成为探究化学反应性、凝聚态结构、材料发现和生物分子功能的核心工具。然而，在真实材料和分子体系所需的空间和时间尺度上实现量子级精度，仍是核心挑战。

机器学习原子间势能（MLIP）旨在通过从 $ab\ initio$ 数据中学习高维势能面来克服这一权衡。在此领域，SO(3)等变图神经网络——包括NequIP、Allegro和MACE——已成为短程MLIP的当前最优方法。但这些模型的精度根本上依赖于**严格的局域性假设**：体系总能被分解为原子贡献，每个原子的贡献仅由有限截断半径内的环境决定。

这一假设与长程相互作用（如静电、偶极-偶极耦合和集体极化）支配的体系**本质不相容**——这些相互作用随距离缓慢衰减，并呈现显著的各向异性。

### 1.1 等变表示的基本原理

从数学上说，SO(3)等变表示天然具有角动量分解的结构。度-$\ell$ 原子特征 $x\_i^{(\ell)} \in \mathbb{C}^{2\ell+1}$ 在全局旋转 $R \in SO(3)$ 下必须满足：

$$x_i^{(\ell)} \mapsto x_i^{(\ell)\prime} = D^{(\ell)}(R) x_i^{(\ell)}$$

其中 $D^{(\ell)}(R)$ 是度 $\ell$ 的 Wigner-D 矩阵。这保证了能量预测具有不变性（$E^\prime = E$），而力作为矢量正确变换（$F\_i^\prime = RF\_i$）。

然而，当所有相互作用被局域化在有限截断半径内时，**跨越截断的分子间长程静电作用、偶极-偶极耦合等现象根本不可能被模型表征**。此前的长程扩展要么破坏了 SO(3) 等变性，要么无法维持能量-力的自洽性。

---

## 二、方法：EquiEwald的创新框架

### 2.1 总体架构

EquiEwald 是一个统一的 SO(3) 等变神经原子间势能，它通过将 Ewald 求和的物理洞察嵌入**不可约表示（irrep）框架**，在倒易空间中进行等变消息传递，建立了处理长程相互作用的全新范式。

![EquiEwald模型总体架构](/img/posts/2026-06-15-equiewald-so3-long-range/fig01-model-architecture.png)

**图1**: EquiEwald 的整体结构与长程模块细节。**(a)** 完整的模型结构，包含短程编码器和长程 Ewald 模块，两者通过残差融合形成最终原子表示用于能量和力预测。**(b)** EquiEwald 长程模块的详细设计：节点特征 $\{h\_i\}$ 和波矢 $\{k\}$ 输入 SO(3) 线性映射和 gating，然后分解为每度 $\ell$ 的实部/虚部两个分支，通过 $\langle k, r\_j \rangle$ 驱动的 k 空间滤波器处理，经逆傅里叶变换和 MLP 后返回每度 $\ell$ 的实空间更新；各度输出经拼接和 gating 后与归一化的局域特征融合。

模型由**两个协同的表征通路**构成：

1. **短程编码器**：基于局域图消息传递，捕捉局域化学环境
2. **长程谱编码器**：在倒易空间通过消息传递引入非局域信息

两个通路共享同一原子级输入，其输出通过残差更新融合。

### 2.2 核心数学：在倒易空间进行等变消息传递

**步骤1：构建 irrep 解析的结构因子**

对于波矢集合 $\{k\_{\alpha}\}\_{\alpha=1}^{N\_k}$，首先对每度 $\ell$ 计算结构因子：

$$s_{\alpha,m}^{(\ell)} = \sum_{j} \exp(i k_{\alpha} \cdot r_j) \, x_j^{(\ell,m)} \qquad (*)$$

其中 $x\_j^{(\ell,m)}$ 是原子 $j$ 上度 $\ell$、磁分量 $m$ 的特征，$k\_{\alpha}$ 是倒易波矢。与 RSNN 的标量结构因子不同，这里的结构因子对**每个度 $\ell$** 包含 $2\ell+1$ 个磁分量 $m$，保留了完整的角分辨信息。

**步骤2：应用可学习的k空间滤波器**

在逆变换之前对每度 $\ell$ 内部的所有磁分量 $m$ 应用**共享的**可学习滤波器 $F(k\_{\alpha})$：

$$M_m^{(\ell)}(r_i) = \sum_{\alpha=1}^{N_k} \exp(i k_{\alpha} \cdot r_i) \, F(k_{\alpha}) \, s_{\alpha,m}^{(\ell)}$$

关键设计在于：$F(k\_{\alpha})$ **仅混合通道维度**——它对同一度 $\ell$ 内所有磁分量 $m$ 的施加方式完全相同，这是保持 SO(3) 等变性的核心约束。

**步骤3：逆累积与非线性精炼**

长程更新通过**度级逆累积**、**度级特定非线性精炼**和**标量条件门控**一步完成：

$$x_i^{\text{Ewald}} = \text{Gate}\left(W_g h_{i,\ell=0},\ \bigoplus_{\ell=0}^{\ell_{\max}} \text{MLP}^{(\ell)}\left( [M_m^{(\ell)}(r_i)]_{m=-\ell}^{\ell} \right) \right)$$

其中 $h\_{i,\ell=0}$ 是当前原子表示的标量（$\ell=0$）块，$g\_i = W\_g h\_{i,\ell=0}$ 充当门控系数——对每个度 $\ell \ge 1$ 提供一个 $C$ 维门控，广播到该度内的所有磁分量 $m$。

**步骤4：信息融合**

在相互作用层 $t$，运行表示、局域更新和长程更新通过残差融合：

$$x_i^{t+1} = \frac{1}{\sqrt{3}} \left(x_i^t + x_i^{\text{Local},t} + x_i^{\text{Ewald},t}\right)$$

![EquiEwald 模型细节图](/img/posts/2026-06-15-equiewald-so3-long-range/fig03-model-details.png)

**图3**: EquiEwald 模型的补充架构细节。左侧展示了从输入原子到输出能量/力的完整数据流，包含球谐嵌入、短程模块、长程Ewald模块、信息融合和输出头；右侧放大展示了"分解 irrep Ewald 模块"的内部结构——节点特征 $\{h\_i\}$ 经 SO(3) 线性层和 GATE 激活后，分解为每度 $\ell$ 的特征，通过实/虚部分支进行 k 空间滤波和逆变换，最终经 MLP 和拼接后输出每度的等变长程更新。

### 2.3 周期体系 vs 非周期体系的区别

**周期体系**：倒易向量在由模拟晶胞诱导的倒格子上采样。对于直接晶格基 $(a\_1, a\_2, a\_3)$，倒易基 $(b\_1, b\_2, b\_3)$ 满足 $b\_i \cdot a\_j = 2\pi\delta\_{ij}$：

$$b_1 = \frac{2\pi}{V}(a_2 \times a_3),\ b_2 = \frac{2\pi}{V}(a_3 \times a_1),\ b_3 = \frac{2\pi}{V}(a_1 \times a_2)$$

通过索引盒子 $\{-N\_x,\dots,N\_x\} \times \{-N\_y,\dots,N\_y\} \times \{-N\_z,\dots,N\_z\}$ 枚举整数三元组形成倒易向量 $k\_{n\_x,n\_y,n\_z} = n\_x b\_1 + n\_y b\_2 + n\_z b\_3$。在此情况下，滤波器退化为共享的、$k$ 无关的通道混合器 $F \in \mathbb{R}^{C \times C}$，以低秩瓶颈形式参数化：

$$F = W_{\text{up}} W_{\text{down}},\quad W_{\text{down}} \in \mathbb{R}^{C_{\downarrow} \times C},\ W_{\text{up}} \in \mathbb{R}^{C \times C_{\downarrow}}$$

**非周期体系**：在固定笛卡尔体素网格上采样。给定频率分辨率 $\Delta k$ 和截断 $k\_{\text{max}}$，枚举整数三元组并保留 $\|k\| \le k\_{\text{max}}$ 的点。使用可分的体素窗口函数减小离散化伪影：

$$D(k,r) = \prod_{d \in \{x,y,z\}} \text{sinc}\left(\frac{\Delta k\, r_d}{2}\right)$$

每个倒易点使用径向嵌入 $\psi(\|k\_{\alpha}\|)$ 和瓶颈投影实现频率感知的谱门控：

$$f_{\alpha} = W_{\text{up}}(W_{\text{down}}\, \psi(\|k_{\alpha}\|)) \in \mathbb{R}^C$$

---

## 三、实验结果

### 3.1 分子二聚体：长程静电捕捉能力检验

分子二聚体由阳离子 $C\_4N\_2H\_6$ 和阴离子 $C\_3NOH\_7$ 组成，分子间偶极-偶极相互作用主导。训练集覆盖质心间距 5-12 Å，测试集延伸至 12-15 Å——远超短程模型 5 Å 截断半径。

![分子二聚体基准对比](/img/posts/2026-06-15-equiewald-so3-long-range/fig02-dimer-benchmark.png)

**图2**: 短程模型与长程模型在带电分子二聚体上的对比。**a** eSCN（纯短程）无法捕捉正确的渐近衰减，MAE 高达 21.08 meV。**b** eSCN+EwaldMP（标量长程）MAE 1.18 meV。**c** eSCN+LES（标量长程）MAE 2.28 meV。**d** eSCN+EquiEwald 跨越全量程精准预测，MAE 仅 **0.78 meV**，且在外推区域（12-15 Å）仍保持与参考值的高度一致。

### 3.2 AIMD-Chig 数据集

AIMD-Chig 数据集来自 Chignolin 蛋白质的 $ab\ initio$ 分子动力学模拟，其柔性骨架和链内长程相互作用形成了极具挑战的测试场景。

| 模型 | 测试能量MAE (meV) | 测试力MAE (meV/{\AA}) |
|------|:---:|:---:|
| eSCN | 193.9 | 23.1 |
| eSCN+EwaldMP（标量）| 132.8 | 20.3 |
| **eSCN+EquiEwald** | **109.0** | **18.1** |

EquiEwald 相比 eSCN，能量 MAE 降低 **44%**（193.9 → 109.0 meV），力 MAE 降低 **21%**（23.1 → 18.1 meV）。更引人注目的是热力学性质预测。利用 Boltzmann 重加权估算折叠自由能差 $\Delta G$：

$$P(s) \propto \sum_{i \in s} \exp(-\beta E_i)$$

$$\Delta G = -k_B T \ln\frac{P_{\text{folded}}}{P_{\text{unfolded}}}$$

EquiEwald 将能量预测误差从 eSCN 的 1.15 kcal/mol 降至 **0.67 kcal/mol**（相对降低约 42%）。

### 3.3 Buckyball Catcher 超分子体系与 OC20 周期体系

在 Buckyball Catcher 超分子体系中，eSCN 测试能量 MAE 为 36.0 meV，EquiEwald 将其降至 **18.1 meV**（改善近 50%）。在周期性催化材料数据集 OC20 上：

| 模型 | 测试能量MAE (meV) | 测试力MAE (meV/{\AA}) |
|------|:---:|:---:|
| eSCN | 347.0 | 24.7 |
| **eSCN+EquiEwald** | **321.2** | **24.1** |
| EquiformerV2 | 541.0 | 46.4 |
| **EquiformerV2+EquiEwald** | **453.0** | **38.4** |

EquiEwald 在两种骨干网络上均展现一致的改进。

### 3.4 训练设置

模型使用复合损失函数优化：

$$\mathcal{L} = \lambda_E \|E_{\text{pred}} - E_{\text{ref}}\|_1 + \lambda_F \frac{1}{3N}\sum_{i=1}^N \|F_i^{\text{pred}} - F_i^{\text{ref}}\|_1$$

其中 $\lambda\_E = 1$，$\lambda\_F = 100$。对于非周期体系，球谐度数 $\ell\_{\max}=3$，倒易空间截断 $0.6\ \text{\AA}^{-1}$，网格间距 $0.2\ \text{\AA}^{-1}$，频域滤波器为 128 维高斯径向基函数。

![训练与补充设置](/img/posts/2026-06-15-equiewald-so3-long-range/fig04-training-settings.png)

**图4**: EquiEwald 的补充训练设置与性能数据。周期体系（OC20）使用 6.0 Å 局域截断半径，倒易空间沿三方向的频率数考虑平均各向异性（1, 1, 3）；非周期体系中 $\ell\_{\max}=3$、$k\_{\text{max}}=0.6\ \text{\AA}^{-1}$、$\Delta k=0.2\ \text{\AA}^{-1}$。文中还包含了 Buckyball Catcher 等补充数据，以及详细的超参数配置。

---

## 四、与 RSNN 方法的对比：从标量不变到等变张量

Guo 等（2026）提出的 **RSNN（Reciprocal-Space Neural Network）** 是在本工作之前——早在 2022 年即已预印本发表——另一基于倒易空间的长程 MLIP 方案。两者共享"在倒空间学习长程核"的核心理念，但在**对称性处理**、**结构因子构造**和**滤波器机制**上存在本质差异。下面从数学上系统对比。

### 4.1 设计哲学的根本差异：E(3) 不变 vs SO(3) 等变

| 维度 | RSNN (Guo et al. 2026) | EquiEwald (Zhang et al. 2026) |
|------|------------------------|-------------------------------|
| 对称性 | E(3) **不变**（各向同性） | SO(3) **等变**（各向异性） |
| 结构因子 | 标量 $\lvert S(\mathbf{k}) \rvert^2$（标量求和） | irrep 张量 $s\_{\alpha,m}^{(\ell)}$（分度求和） |
| k 空间滤波器 | FCN($\lvert \mathbf{k} \rvert$)，仅依赖 $\lvert \mathbf{k} \rvert$ | 通道混合器 $F \in \mathbb{R}^{C\times C}$ 或谱门控 $F(k\_{\alpha}) = \text{diag}(f\_{\alpha})$ |
| 原子特征 | 标量电荷 $q\_j$（ChargeMLP 预测） | irrep 特征 $x\_j^{(\ell,m)}$（等变 GNN 输出） |
| 训练策略 | 三步训练（three-step） | 端到端端联合训练 |
| 适用体系 | 周期晶体为主 | 周期 + 非周期统一框架 |
| 长程类型 | 库仑 + 色散（$1/r^n$ 通用） | 静电、多极、极化（张量级各向异性） |

### 4.2 结构因子：标量求和 vs irrep 分解

**RSNN** 采用经典的 Ewald 结构因子形式，将原子信息压缩为标量电荷特征 $q\_j$：

$$S(\mathbf{k}) = \sum_{j=1}^{N} q_j \, e^{-i\mathbf{k}\cdot\mathbf{r}_j}$$

长程能量仅依赖结构因子的模平方 $\lvert S(\mathbf{k}) \rvert^2$，该量在旋转下不变——**完全丢失了角分布信息**。这意味着 RSNN 无法区分同一 $\lvert \mathbf{k} \rvert$ 方向不同角度上的各向异性相互作用。

**EquiEwald** 将这一过程推广到不可约表示层面：

$$s_{\alpha,m}^{(\ell)} = \sum_{j} \exp(i k_{\alpha} \cdot r_j) \, x_j^{(\ell,m)} \qquad (对比式 *)$$

关键区别在于：

1. **每度 $\ell$ 独立求和**：不同角动量通道的特征独立传播到倒空间，而非坍缩为单一标量
2. **每度内 $2\ell+1$ 个磁分量 $m$ 完整保留**：方向信息不会因取模方而丢失
3. **原子特征 $x\_j^{(\ell,m)}$ 而非标量电荷**：特征本身即是 SO(3) 等变张量，在倒空间的操作维持其变换性质

### 4.3 k 空间滤波器：径向标量 vs 等变通道混合

**RSNN** 的滤波器是 $\lvert \mathbf{k} \rvert$ 的标量函数：

$$E_{\text{long}}^{\text{RSNN}} = \frac{1}{V}\sum_{\mathbf{k}\neq \mathbf{0}} \text{FCN}(|\mathbf{k}|) \, |S(\mathbf{k})|^2$$

输入仅 $\lvert \mathbf{k} \rvert$ 意味着**所有 $\mathbf{k}$ 方向的贡献模式相同**——这是各向同性假设。

**EquiEwald** 的滤波器在周期体系中为通道混合器 $F \in \mathbb{R}^{C \times C}$：

$$\tilde{s}_{\alpha,m}^{(\ell)} = F \, s_{\alpha,m}^{(\ell)}$$

$F$ 的作用是**在同一度 $\ell$、同一磁分量 $m$ 内混合 $C$ 个通道**，且对所有 $m$ 共享权重——这保证了等变性。在非周期体系中进一步扩展为 $k$ 依赖的谱门控 $F(k\_{\alpha}) = \text{diag}(f\_{\alpha})$，其中：

$$f_{\alpha} = W_{\text{up}}(W_{\text{down}}\, \psi(\|k_{\alpha}\|)) \in \mathbb{R}^C$$

相比 RSNN 的标量 FCN($\lvert \mathbf{k} \rvert$)，EquiEwald 的滤波器作用于**通道空间**而非 $\lvert \mathbf{k} \rvert$ 空间，保留了丰富的特征表达能力。

### 4.4 逆变换与特征融合：简单求和 vs 度级精炼

**RSNN** 的逆变换极为简洁：将滤波后的结构因子模平方乘以前置因子后直接求和得到标量长程能量，再从能量对原子位置求导得到力。原子电荷特征 $q\_j$ 经由独立的 ChargeMLP 支路学习，整个 RSP 流程与局域网络之间保持相对独立。

**EquiEwald** 的逆变换是度级结构化的：

$$x_i^{\text{Ewald}} = \text{Gate}\left(W_g h_{i,\ell=0},\ \bigoplus_{\ell=0}^{\ell_{\max}} \text{MLP}^{(\ell)}\left( [M_m^{(\ell)}(r_i)]_{m=-\ell}^{\ell} \right) \right)$$

长程更新以等变向量（tensor）的形式返回，与局域更新的 irrep 特征在同等表示空间中融合——这种"在表示空间融合"而非"在能量空间相加"的方式，是 EquiEwald 区别于 RSNN 的核心架构差异。

### 4.5 关于背景修正

RSNN 引入了带电超胞的背景电荷修正项：

$$E_{\text{back}} = -\frac{\alpha_n}{V}\left(\sum_{j=1}^{N} q_j\right)^2$$

EquiEwald 的框架不依赖显式的背景修正——因为其原子特征 $x\_j^{(\ell,m)}$ 不是标量电荷，无需约束总电荷。在等变框架中，体系的总体带电状态由 $\ell=0$（标量）通道自然编码。

### 4.6 能力范围对比

RSNN 的优势在于**核函数形式灵活**：FCN($\lvert \mathbf{k} \rvert$) 可逼近任意 $1/r^n$ 衰减的实空间相互作用，对混合库仑+色散体系（如 NaCl）效果显著。但其各向同性的本质使其无法描述偶极-偶极、四极-四极等**角度依赖**的相互作用。

EquiEwald 的优势在于**张量级各向异性长程关联**：通过 irrep 分解保留了完整的角分布信息，在带电分子二聚体（展示清晰的角度依赖静电）和蛋白质构象（展示复杂的非各向同性链内作用）上表现尤为突出。

两者共同确立了"在倒易空间学习长程相互作用"这一范式，但分别选择了**标量不变**和**等变张量**两条技术路线，分别适用于不同物理特征的长程体系。对以各向同性长程为主的材料（如离子晶体中的纯库仑+色散），RSNN 的简单标量框架已足够高效；对需要精确描述各向异性多极耦合的体系（如极性分子、生物分子、界面体系），EquiEwald 的等变框架提供了正确的归纳偏置。

---

## 五、总结

EquiEwald 通过在**倒易空间**进行 **SO(3) 等变消息传递**，将 Ewald 求和的思想与等变图神经网络在表示空间层面有机统一。其核心创新——为每度 $\ell$ 独立计算 irrep 结构因子、应用等变保持的 k 空间滤波器、以及度级非线性精炼——使得各向异性的张量级长程关联得以在不破坏旋转对称性的前提下被学习。与 RSNN 的标量不变路线形成互补，两者共同推进了倒易空间 MLIP 的理论前沿。实验结果表明，EquiEwald 在非周期体系（带电分子二聚体、蛋白质构象、超分子组装）和周期体系（催化材料）中均显著超越现有方案。

## 延伸阅读（站内）

- [用倒空间神经网络捕获长程相互作用——Guo 等 2026 解读](/2026/06/14/用倒空间神经网络捕获长程相互作用-Guo-等-2026-解读/)
- [原子机器学习中的长程静电：物理视角梳理——Grasselli 等 2026 解读](/2026/06/09/原子机器学习中的长程静电-物理视角梳理-Grasselli-等-2026-解读/)

