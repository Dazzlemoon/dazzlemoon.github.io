---
layout:     post
title:      布里渊区、不可约布里渊区与高对称路径——从固体物理基础到 PRDNet 的 IBZ 迁移
subtitle:   从正格/倒格与 Bloch 定理出发，用公式系统介绍 BZ、IBZ、高对称点与能带极值，给出立方/六方实例，并简述将 PRDNet 的 Miller 集替换为 IBZ 路径的思路。
date:       2026-05-29
author:     qqz
header-img: img/post-bg-desk.jpg
catalog:    true
tags:
    - 固体物理
    - 布里渊区
    - IBZ
    - 高对称点
    - 声子色散
    - PRDNet
    - 倒易空间
---

![一图总结](/img/posts/ibz-prdnet-bz-path/ibz-path-prdnet-cover.png)

## 0. 导读：为什么要学 IBZ 路径

在固体物理与材料计算里，**能带图**和**声子色散图**几乎总是画在「高对称路径」上——例如 $\Gamma\!\to\!X\!\to\!M\!\to\!\Gamma$。这条路径不是随意折线，而是 **第一布里渊区（BZ）** 中 **不可约布里渊区（IBZ）** 的骨架。

对 **PRDNet** 一类「倒易空间全局指纹」模型而言：

| 采样方式 | 变量 | 典型物理场景 |
|----------|------|--------------|
| **Miller 集** $H$ | 整数 $(h,k,l)$，对应离散倒格点 $\mathbf{G}$ | X 射线 / 电子衍射（静态结构因子） |
| **IBZ 路径** $\{\mathbf{q}_\ell\}$ | 连续波矢 $\mathbf{q}$ 沿高对称折线 | 声子色散、介电响应、红外/拉曼 |

二者都是 Bloch 相位 $e^{i\mathbf{q}\cdot\mathbf{r}}$ 下的**全结构相干求和**，但 **$\mathbf{q}$ 才是谱学色散的 native 坐标**。下文从基本概念出发建立公式链，最后一节简述 Miller → IBZ 的 PRD 改写思路。

---

## 1. 出发点：周期晶格与 Bloch 定理

### 1.1 正格（Direct Lattice）

Bravais 格矢 $\mathbf{a}_1,\mathbf{a}_2,\mathbf{a}_3$ 张成三维周期结构。任意格点为

$$
\mathbf{R} = n_1\mathbf{a}_1 + n_2\mathbf{a}_2 + n_3\mathbf{a}_3, \quad n_i \in \mathbb{Z}.
$$

单胞内原子位置 $\boldsymbol{\tau}_\kappa$（$\kappa=1,\ldots,n_{\mathrm{basis}}$），完整晶体坐标

$$
\mathbf{r}_{\kappa} = \boldsymbol{\tau}_\kappa + \mathbf{R}.
$$

用晶胞矩阵 $\mathbf{L}=[\mathbf{a}_1,\mathbf{a}_2,\mathbf{a}_3]$（列向量，单位 Å），分数坐标 $\mathbf{r}_{\mathrm{frac}}$ 与笛卡尔坐标关系

$$
\mathbf{r}_{\mathrm{cart}} = \mathbf{r}_{\mathrm{frac}}\,\mathbf{L}, \qquad
\mathbf{r}_{\mathrm{frac}} = \mathbf{r}_{\mathrm{cart}}\,\mathbf{L}^{-1}.
$$

### 1.2 倒格（Reciprocal Lattice）

倒格基矢 $\mathbf{b}_i$ 满足

$$
\mathbf{a}_i \cdot \mathbf{b}_j = 2\pi\,\delta_{ij}.
$$

显式公式（$\mathbf{V}=\mathbf{a}_1\cdot(\mathbf{a}_2\times\mathbf{a}_3)$ 为原胞体积）：

$$
\mathbf{b}_1 = 2\pi\,\frac{\mathbf{a}_2\times\mathbf{a}_3}{\mathbf{V}}, \quad
\mathbf{b}_2 = 2\pi\,\frac{\mathbf{a}_3\times\mathbf{a}_1}{\mathbf{V}}, \quad
\mathbf{b}_3 = 2\pi\,\frac{\mathbf{a}_1\times\mathbf{a}_2}{\mathbf{V}}.
$$

任意倒格矢

$$
\mathbf{G} = h\mathbf{b}_1 + k\mathbf{b}_2 + l\mathbf{b}_3, \quad (h,k,l)\in\mathbb{Z}^3.
$$

$(h,k,l)$ 即 **Miller 指数**；$\mathbf{G}$ 的端点构成倒易点阵。

倒格矩阵（列向量）与正格的关系

$$
\mathbf{B} = [\mathbf{b}_1,\mathbf{b}_2,\mathbf{b}_3] = 2\pi\,\mathbf{L}^{-\mathsf{T}}.
$$

### 1.3 Bloch 定理

在周期势 $V(\mathbf{r}+\mathbf{R})=V(\mathbf{r})$ 中，单电子波函数（或晶格动力学 Bloch 模）可写为

$$
\psi_{n\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}}\,u_{n\mathbf{k}}(\mathbf{r}), \qquad
u_{n\mathbf{k}}(\mathbf{r}+\mathbf{R}) = u_{n\mathbf{k}}(\mathbf{r}).
$$

**波矢 $\mathbf{k}$** 是倒易空间中的标签；$\mathbf{k}$ 与 $\mathbf{k}+\mathbf{G}$ 给出物理上等价的态。因此独立波矢只需在第一布里渊区内取代表。

---

## 2. 第一布里渊区（Brillouin Zone, BZ）

### 2.1 Wigner–Seitz 构造

**第一布里渊区** $\mathrm{BZ}$ 是倒易点阵的 Wigner–Seitz 原胞：包含原点、且到任意非零倒格点 $\mathbf{G}\neq\mathbf{0}$ 的距离不小于到原点距离的 $\mathbf{k}$ 集合：

$$
\mathrm{BZ} = \left\{\mathbf{k}\in\mathbb{R}^3 \;\middle|\; \|\mathbf{k}\| \le \|\mathbf{k}-\mathbf{G}\|,\ \forall\,\mathbf{G}\in\mathbb{Z}^3\setminus\{\mathbf{0}\}\right\}.
$$

等价地：$\mathrm{BZ}$ 是围绕 $\Gamma$ 点（$\mathbf{k}=\mathbf{0}$）的 Voronoi 胞。

**几何直觉**：从 $\Gamma$ 出发，作到各近邻倒格点的垂直平分面；这些面围成的有界多面体即为 $\mathrm{BZ}$。

### 2.2 为何 BZ 是「独立 $\mathbf{k}$ 的容器」

由于 $e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}} = e^{i\mathbf{k}\cdot\mathbf{r}}$ 在周期函数意义下等价，所有不等价波矢在 $\mathrm{BZ}$ 内**恰取一次**。因此：

- **能带** $E\_n(\mathbf{k})$：电子本征能量；
- **声子色散** $\omega\_\nu(\mathbf{k})$：第 $\nu$ 支格波频率；
- **介电函数** $\varepsilon\_{\alpha\beta}(\mathbf{k},\omega)$、**Born 有效电荷** $Z^{\ast}\_{\alpha\beta}(\mathbf{k})$ 等

均为 $\mathbf{k}$（或 $\mathbf{q}$，晶格动力学中常记作 $\mathbf{q}$）的函数，定义域为 $\mathrm{BZ}$。

### 2.3 坐标约定

计算软件（VASP、Quantum ESPRESSO、phonopy、seekpath）常用两套坐标：

| 名称 | 符号 | 与笛卡尔关系 |
|------|------|--------------|
| 分数倒易坐标 | $\mathbf{k}_{\mathrm{frac}}=(k_1,k_2,k_3)$ | $\mathbf{k}_{\mathrm{cart}} = k_1\mathbf{b}_1+k_2\mathbf{b}_2+k_3\mathbf{b}_3 = \mathbf{k}_{\mathrm{frac}}\,\mathbf{B}$ |
| 笛卡尔倒易坐标 | $\mathbf{k}_{\mathrm{cart}}$ | 单位 Å$^{-1}$，满足 $e^{i\mathbf{k}\cdot\mathbf{r}}$ |

**注意**：Miller 指数 $(h,k,l)$ 与 $\mathbf{k}_{\mathrm{frac}}$ 在数值上常一致（即 $\mathbf{k}_{\mathrm{frac}}=(h,k,l)$ 时 $\mathbf{k}_{\mathrm{cart}}=\mathbf{G}$），但 **BZ 内的 $\mathbf{k}$ 一般是分数坐标，不必为整数**。

---

## 3. 不可约布里渊区（Irreducible Brillouin Zone, IBZ）

### 3.1 点群与 $\mathbf{k}$ 的 star

晶体 **点群** $\mathcal{G}$（不含平移）作用于波矢：$g\in\mathcal{G}$ 将 $\mathbf{k}$ 映射为 $g\mathbf{k}$（模 $\mathbf{G}$）。

**星（star）** 定义为

$$
\mathrm{star}(\mathbf{k}) = \{g\mathbf{k} \;\mod\mathbf{G} : g\in\mathcal{G}\}.
$$

同一 star 上的波矢给出对称相关的本征值（可能简并，或属于等价表示）。

### 3.2 IBZ 的定义

**不可约布里渊区** $\mathrm{IBZ}\subset\mathrm{BZ}$ 是 $\mathrm{BZ}$ 的一个子楔区，使得 $\mathrm{BZ}$ 中每个 $\mathbf{k}$ 至少属于某个 $\mathrm{star}(\mathbf{k}_0)$，且 $\mathbf{k}_0\in\mathrm{IBZ}$ **唯一**代表该等价类：

$$
\mathrm{BZ} = \bigcup_{\mathbf{k}_0\in\mathrm{IBZ}} \mathrm{star}(\mathbf{k}_0) \quad (\text{模倒格矢}).
$$

**计算意义**：

- Monkhorst–Pack 网格、DFPT 声子计算只需在 IBZ 内取点；
- 通过对称性展开（star 求和）可恢复全 BZ 信息，**省计算量**且**不破坏晶体对称**。

### 3.3 IBZ 路径：IBZ 的一维骨架

完整 IBZ 是三维楔区；**高对称路径**是 IBZ 内连接高对称点的折线，是一维子集：

$$
\mathcal{Q}_{\mathrm{path}} = \{\mathbf{q}(t) : t\in[0,1]\} \subset \mathrm{IBZ}.
$$

离散为 $N_q$ 个点 $\{\mathbf{q}_\ell\}_{\ell=1}^{N_q}$ 后，可用于：

1. 画色散图（能带 / 声子）；
2. 作为神经网络的全局倒易探针（PRD 的 IBZ 路径模式）。

**重要区分**：IBZ **路径模式** ≠ 完整 IBZ **体积分点**；前者是谱学作图与轻量指纹，后者用于态密度、热力学积分等。

---

## 4. 高对称点（High-Symmetry Points）

### 4.1 定义

在 $\mathrm{BZ}$ 的**顶点、棱中点、面心**等处，保留 $\mathbf{k}$ 的对称操作集合 **增大**（小群 $L_{\mathbf{k}}$ 变大），本征值常出现**简并**或**特殊模式**（如 $\Gamma$ 点声学支 $\omega\to 0$）。

这些点称为 **高对称点**，用标准符号标记（由空间群与 seekpath 库给出）。

### 4.2 小群（Little Group）

对给定 $\mathbf{q}$，**小群**为

$$
L_{\mathbf{q}} = \{g\in\mathcal{G} : g\mathbf{q} = \mathbf{q} + \mathbf{G}\}.
$$

- 在 $\Gamma$：$L\_{\Gamma}=\mathcal{G}$（完整点群）；
- 在一般 $\mathbf{q}$：$L\_{\mathbf{q}}$ 是 $\mathcal{G}$ 的真子群。

声子分支、红外/拉曼活性按 $L_{\mathbf{q}}$ 的不可约表示分类；实验强度需将偶极、极化率张量投影到相应表示）。

### 4.3 常见高对称点（立方晶系，原胞为立方）

| 符号 | 分数坐标 $\mathbf{k}_{\mathrm{frac}}$ | 几何位置 |
|------|--------------------------------------|----------|
| $\Gamma$ | $(0,0,0)$ | BZ 中心 |
| $X$ | $(1/2,0,0)$ | 面心（$k_x$ 方向） |
| $M$ | $(1/2,1/2,0)$ | 棱中点 |
| $R$ | $(1/2,1/2,1/2)$ | 体对角顶点 |

六方晶系常见：$\Gamma$, $M$, $K$, $A$ 等。

---

## 5. 高对称路径与能带 / 声子极值

### 5.1 路径参数化

连接高对称点 $A\to B$ 的直线段：

$$
\mathbf{q}(t) = (1-t)\,\mathbf{q}_A + t\,\mathbf{q}_B, \quad t\in[0,1].
$$

多段路径（如 $\Gamma\!\to\!X\!\to\!M\!\to\!\Gamma$）在每段上分别参数化，再按弧长或等分点数 $N_q$ 离散。

**seekpath** 根据空间群自动给出标准路径标签（如 `GXMG`）及分数坐标。

### 5.2 能带与色散

**电子能带**：

$$
\hat{H}_{\mathbf{k}}\,\psi_{n\mathbf{k}} = E_n(\mathbf{k})\,\psi_{n\mathbf{k}}.
$$

沿路径绘制 $E_n\big(\mathbf{q}(t)\big)$ 即 **能带图**。

**声子色散**（谐波近似，动力学矩阵 $D(\mathbf{q})$）：

$$
D(\mathbf{q})\,\mathbf{e}_\nu(\mathbf{q}) = \omega_\nu^2(\mathbf{q})\,\mathbf{e}_\nu(\mathbf{q}).
$$

$\omega_\nu(\mathbf{q})$ 为第 $\nu$ 支声子频率；沿路径绘图即 **声子色散关系**。

### 5.3 能带极值与有效质量

**能带极值点** $\mathbf{k}^{\ast}$ 满足

$$
\nabla_{\mathbf{k}} E_n(\mathbf{k})\big|_{\mathbf{k}=\mathbf{k}^*} = \mathbf{0}.
$$

在极值附近 Taylor 展开：

$$
E_n(\mathbf{k}) \approx E_n(\mathbf{k}^*) + \frac{1}{2}\sum_{\alpha\beta} \frac{\partial^2 E_n}{\partial k_\alpha\partial k_\beta}\Big|_{\mathbf{k}^*} \Delta k_\alpha \Delta k_\beta.
$$

**有效质量张量**（电子学中常用 $ \hbar^2 $ 约定）：

$$
\left(\frac{1}{m^*}\right)_{\alpha\beta} = \frac{1}{\hbar^2}\,\frac{\partial^2 E_n}{\partial k_\alpha\partial k_\beta}\Big|_{\mathbf{k}^*}.
$$

- **价带顶 / 导带底** 常在 $\Gamma$ 或高对称点（直接/间接带隙）；
- 也可能在 **高对称路径内部**（非高对称点的极值）——此时需沿路径扫描 + 数值求导定位。

**声子极值**：同理，$\omega_\nu(\mathbf{q})$ 在 $\mathbf{q}^{\ast}$ 处 $\nabla_{\mathbf{q}}\omega_\nu=\mathbf{0}$；$\Gamma$ 点声学支 $\omega\to 0$ 是 **全局** 极小（Goldstone 模），光学支极值常出现在 $X,M,R$ 等边界点。

### 5.4 群速度

$$
\mathbf{v}_n(\mathbf{k}) = \frac{1}{\hbar}\,\nabla_{\mathbf{k}} E_n(\mathbf{k}).
$$

色散关系斜率 $|dE/dk|$ 大处群速度高；极值点处群速度为零。

### 5.5 $\Gamma$ 点的 LO–TO 分裂（与 BZ 的关系）

光学声子可按原子位移方向相对波矢 $\mathbf{q}$ 分类：

| 模式 | 位移方向 | 符号 |
|------|----------|------|
| 横光学 | $\perp\,\mathbf{q}$ | **TO**（Transverse Optical） |
| 纵光学 | $\parallel\,\mathbf{q}$ | **LO**（Longitudinal Optical） |

在 **非极性** 晶体中，$\Gamma$ 点（$\mathbf{q}=\mathbf{0}$）上 LO 与 TO 光学支通常 **简并**：$\omega_{\mathrm{LO}}(\Gamma)=\omega_{\mathrm{TO}}(\Gamma)$。  
在 **极性/离子晶体**（如 NaCl、PbTiO$_3$）中，即便在 $\Gamma$ 点二者也 **不再简并**：

$$
\omega_{\mathrm{LO}}(\Gamma) \neq \omega_{\mathrm{TO}}(\Gamma),
$$

这一现象称为 **LO–TO 分裂**（LO–TO splitting）。

**为何与布里渊区有关？** 分裂是 **BZ 中心 $\mathbf{q}=\mathbf{0}$ 处、$q\to 0$ 长波长极限** 的效应，而非 $X,M,R$ 等高 $|q|$ 边界点的专属现象：

1. **物理机制**：极性晶体中，LO 模伴随 **宏观极化** $\mathbf{P}$；在 $q\to 0$ 时激发 **退极化宏观电场** $\mathbf{E}$，抬高 LO 频率。TO 模位移 $\perp\mathbf{q}$，不产生该宏观场，频率较低。
2. **$\Gamma$ 是色散图的锚点**：标准 IBZ 路径（如 $\Gamma\!\to\!X\!\to\!M\!\to\!\Gamma$）**必须回到 $\Gamma$**，才能在图上读出 LO/TO 是否在中心分开。
3. **动力学矩阵在 $\Gamma$ 非解析**：仅含短程力常数的 $D(\mathbf{q})$ 在 $\mathbf{q}\to\mathbf{0}$ 时对纵/横行为不同；DFPT/Phonopy 需在 $\Gamma$ 邻域加入 **非解析修正（NAC）**，依赖高频介电常数 $\varepsilon\_\infty$ 与 **Born 有效电荷** $Z^{\ast}$：

$$
D_{\alpha\beta}(\mathbf{q}\to\mathbf{0}) \;\;\text{对 LO/TO 给出不同极限}.
$$

4. **与 MLIP / PRD 的关联**：纯短程 MLIP 或固定电荷库仑项常在 $\Gamma$ **合并 LO/TO 为一条**；CACE+LES、MTP+EDQRd（Korogod et al., 2026）等 **环境依赖电荷 + 长程静电** 路线，目标正是从势函数导数构造 NAC，恢复 $\Gamma$ 点分裂。  
   这也是 PRD 从 Miller 高 $|\mathbf{h}|$ 采样转向 **含 $\Gamma$ 的 IBZ 低 $q$ 路径** 的动机之一——$\Gamma$ 邻域电响应与 LO–TO、介电常数直接相关。

> **小结**：LO–TO 分裂 **绑定在 BZ 中心 $\Gamma$**；IBZ 路径负责在色散图上 **展示** 它，分裂本身由 **极性 + 长程静电** 决定，而非路径形状（GXMG 等）本身。

---

## 6. 实例

### 6.1 简单立方（SC）→ 立方 BZ

正格：$\mathbf{a}_1=a\hat{x}$, $\mathbf{a}_2=a\hat{y}$, $\mathbf{a}_3=a\hat{z}$。

倒格：$\mathbf{b}_i = \frac{2\pi}{a}\hat{e}_i$。最近邻倒格点 $(\pm1,0,0)$ 等，$\mathrm{BZ}$ 为边长 $\frac{2\pi}{a}$ 的 **立方体**，$\Gamma$ 在中心，$X$ 在面心。

**标准路径** `GXR` 或 `GXMGR`（取决于原胞/惯用胞选择）；对 SC 单原子胞，常见：

$$
\Gamma(0,0,0) \to X(1/2,0,0) \to M(1/2,1/2,0) \to \Gamma.
$$

**物理示例**：若某能带在 $\Gamma$ 为带顶、在 $X$ 为带底，则为 **间接带隙** $\Delta E = E_X - E_\Gamma$。

### 6.2 面心立方（FCC）→ 倒易为 BCC

FCC 正格倒格为 BCC 点阵，Wigner–Seitz 胞为 **截角十二面体**（rhombic dodecahedron），高对称点仍记 $\Gamma,X,W,K,L,U$ 等（具体坐标依赖惯用胞；计算时用 seekpath 读取）。

**声子示例（NaCl，rocksalt）**：DFPT 沿 $\Gamma\!\to\!X\!\to\!W\!\to\!L\!\to\!\Gamma$ 可看到 TO/LO 光学支；$\Gamma$ 点的 **LO–TO 分裂** 见 §5.5。

### 6.3 六方晶系（以 hP 原胞为例）

常见高对称点：

| 符号 | 典型 $\mathbf{k}_{\mathrm{frac}}$ |
|------|-----------------------------------|
| $\Gamma$ | $(0,0,0)$ |
| $M$ | $(1/2,0,0)$ |
| $K$ | $(1/3,1/3,0)$ |
| $A$ | $(0,0,1/2)$ |

标准路径例：$\Gamma\!\to\!M\!\to\!K\!\to\!\Gamma\!\to\!A\!\to\!L\!\to\!H\!\to\!A$（标签因空间群而异）。

**二维材料（石墨烯）**：BZ 为六边形，$K$ 与 $K'$ 为 Dirac 点，$E(\mathbf{k})$ 线性色散 $E\sim |k-K|$，有效质量 $m^{\ast}\to 0$。

### 6.4 数值离散示例

设路径 `GXMG`，$N_q=20$，在 $\Gamma\!\to\!X$ 段：

$$
\mathbf{q}_\ell = \frac{\ell}{19}\,\mathbf{q}_X, \quad \ell=0,\ldots,19, \quad \mathbf{q}_X=(\tfrac{2\pi}{a},0,0)_{\mathrm{cart}}.
$$

对每点计算 $\omega_\nu(\mathbf{q}_\ell)$，横轴用累积弧长 $s_\ell$ 绘图——即 phonopy / VASP 输出的色散图格式。

---

## 7. IBZ 路径的 Bloch 相干和（与 PRD 的数学接口）

无论 Miller 集还是 IBZ 路径，PRD 类方法的核心结构是 **可学习形式因子 × Bloch 相位求和**。

### 7.1 Miller 集（PRDNet 原版）

Miller 集 $H$（$|h|,|k|,|l|\le C_{\max}$，gcd=1，对称闭包）：

$$
F_{\mathbf{h}} = \sum_{i=1}^{N} f_i^{*}(\mathbf{h})\,\exp\!\big(-2\pi i\,\mathbf{h}\cdot\mathbf{r}_{\mathrm{frac},i}\big).
$$

$\mathbf{h}\cdot\mathbf{r}_{\mathrm{frac}}$ 为无量纲相位；$\mathbf{h}$ 对应倒格点 $\mathbf{G}_{\mathbf{h}}$，适合 **静态衍射** 物理。

### 7.2 IBZ 路径（谱学导向改写）

路径点 $\mathbf{q}_\ell\in\mathcal{Q}_{\mathrm{path}}$，可学习标量/向量响应 $r_{i,\ell}$：

$$
\mathcal{R}(\mathbf{q}_\ell) = \sum_{i\in\mathrm{cell}} r_{i,\ell}\,\exp\!\big(i\,\mathbf{q}_\ell\cdot\mathbf{r}_{\mathrm{cart},i}\big).
$$

取实部/虚部拼接 → MLP → 全局条件向量 $\mathbf{z}$，再与图网络池化特征融合，调制局域 Head（如 CACEles 的 $q_i,\mathbf{u}_i,\alpha_i$）。

**关键**：$\mathcal{R}(\mathbf{q})$ 是 **embedding 探针**，不是 $D(\mathbf{q})$ 的本征值，也不是物理 $Z^{\ast}(\mathbf{q})$ 本身——但 **采样几何** 与声子/介电色散一致。

---

## 8. PRDNet：Miller 集 → IBZ 路径的迁移思路（简述）


### 8.1 为何要换

| 维度 | Miller 集 $H$ | IBZ 路径 $\{\mathbf{q}_\ell\}$ |
|------|---------------|----------------------------------|
| 物理场景 | XRD / 电子衍射，$F_{\mathbf{h}}^2$ | 声子 $\omega_\nu(\mathbf{q})$，$\varepsilon(\mathbf{q})$, IR/Raman |
| 变量 | 整数 $\mathbf{h}$，离散 $\mathbf{G}$ | 连续 $\mathbf{q}$，含 $\Gamma$ 邻域低 $q$ |
| 相位 | $2\pi\mathbf{h}\cdot\mathbf{r}_{\mathrm{frac}}$ | $\mathbf{q}\cdot\mathbf{r}_{\mathrm{cart}}$ |
| 对称 | Friedel 对、Miller 置换闭包 | IBZ + star($\mathbf{q}$) + 小群 $L_{\mathbf{q}}$ |

PRDNet 用 Miller 集解决 **晶体性质预测（CPP）** 中的表示碰撞；迁移到 **高精度谱学 / CACEles 多极** 时，**$\mathbf{q}$ 路径更贴近 LO–TO、光学支、低 $q$ 电响应**，而高 $|\mathbf{h}|$ Miller 点偏「短波长衍射」。

### 8.2 最小改动（L1，工程上已实现思路）

```
晶体 M
  ├─ 图分支 → GlobalPool → g          （不变）
  └─ PRD 分支：
        原：Miller 集 H → F_h → d
        新：seekpath 路径 → {q_ℓ} → R(q_ℓ) → d_rec
              ↓
        z = MLP_fusion([g ⊕ d_rec]) → 调制 Head
```

**替换清单**：

1. **采样器**：`MillerIndexSampler` → `IBZQPathSampler`（环境变量如 `GXMG`，默认 $N\_q=20$）；
2. **相位公式**：$\exp(-2\pi i\,\mathbf{h}\cdot\mathbf{r}\_{\mathrm{frac}})$ → $\exp(i\,\mathbf{q}\_\ell\cdot\mathbf{r}\_{\mathrm{cart}})$；
3. **形式因子**：$f\_i^{\ast}(\mathbf{h})$ → $r\_{i,\ell}(h\_i^{(L)})$（每个路径点可独立 MLP 通道）；
4. **融合接口**：$\mathbf{d}\_{\mathrm{rec}}$ 维数 $2N\_q$（Re/Im）替代 $2|H|$，**结构级 fusion 不变**。

### 8.3 不变与边界

**保持不变**：

- 图分支 + 全局 fusion 的双塔架构；
- 可学习「伪粒子」形式因子思想（环境敏感探针）；
- 晶体学不变性仍由结构输入 + 对称路径约定保证（非完整 $P\_\mu$ 投影）。

**液态 / 弱周期体系**：IBZ 路径反映的是 **模拟盒周期边界**，非真实长程晶体序。

### 8.4 公式链一览

$$
\boxed{
\begin{aligned}
&\text{BZ / IBZ / 高对称路径} \Rightarrow \{\mathbf{q}_\ell\} \\[4pt]
&\mathcal{R}(\mathbf{q}_\ell)=\sum_i r_{i,\ell}(h_i)\,e^{i\mathbf{q}_\ell\cdot\mathbf{r}_i}
\Rightarrow \mathbf{d}_{\mathrm{rec}}
\Rightarrow \mathbf{z}=\mathrm{MLP}([\mathbf{g}\oplus\mathbf{d}_{\mathrm{rec}}]) \\[4pt]
&\text{Head 输出 } q_i,\mathbf{u}_i,\alpha_i \Rightarrow \text{LES Ewald / MD 谱学}
\end{aligned}
}
$$

与 Miller 版 PRDNet 同构，**仅倒易采样从 $\mathbf{h}$ 网格换为 $\mathbf{q}$ 路径**——这是「把 PRD 用于声子几何」的精确含义。

---

## 9. 小结

| 概念 | 一句话 |
|------|--------|
| **BZ** | 倒易点阵 Wigner–Seitz 原胞，独立波矢 $\mathbf{k}$ 的 Fundamental domain |
| **IBZ** | BZ 在对称操作下的不可约楔区，数值积分只需扫 IBZ |
| **高对称点** | BZ 边界/顶点，小群变大，简并与特殊模式（$\Gamma$ 声学支等） |
| **高对称路径** | IBZ 内连接高对称点的折线，色散图与 PRD 探针的一维采样 |
| **能带极值** | $\nabla_{\mathbf{k}}E_n=\mathbf{0}$；有效质量由 Hessian 给出 |
| **LO–TO 分裂** | 极性晶体在 BZ 中心 $\Gamma$（$q\to 0$）处 LO/TO 光学支频率不等；需 NAC 与 $Z^{\ast},\varepsilon_\infty$ |
| **Miller → IBZ** | PRD 全局分支从衍射整数 $\mathbf{h}$ 改为谱学 $\mathbf{q}$ 路径，fusion 架构不变 |

---

## 10. 延伸阅读

- Ashcroft & Mermin，《固体物理学》— 第 12 章倒格与 BZ
- seekpath / spglib 文档 — 标准高对称路径与 IBZ

## 延伸阅读（站内）

- [当晶体「照镜子」：PRDNet 如何用伪粒子衍射补全长程盲区？](/2026/05/26/当晶体-照镜子-PRDNet-如何用伪粒子衍射补全长程盲区/)
- [不用 DFPT 也能算 LO-TO 分裂？环境依赖电荷长程 MLIP 速读](/2026/05/28/不用-DFPT-也能算-LO-TO-分裂环境依赖电荷长程-MLIP-速读/)

