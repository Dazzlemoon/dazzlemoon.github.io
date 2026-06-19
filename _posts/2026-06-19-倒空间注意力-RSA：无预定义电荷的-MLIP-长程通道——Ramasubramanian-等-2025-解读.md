---
layout:     post
title:      倒空间注意力 RSA：无预定义电荷的 MLIP 长程通道——Ramasubramanian 等 2025 解读
subtitle:   CMU / Argonne / Buffalo / UCSD Ramasubramanian 等（arXiv 2025）：FPE + Ewald 权重耦合 MACE 的 LR-RSA；收录正文全部 display 公式，并延伸讨论与 IBZ-PRD、EFA 的衔接与对比（高精度谱学 / 长程 MLIP 谱系）。
date:       2026-06-19
author:     qqz
header-img: img/post-bg-desk.jpg
catalog:    true
tags:
    - MLIP
    - 倒空间
    - RSA
    - 长程相互作用
---

![一图总结](/img/posts/2026-06-19-rsa-reciprocal-space-attention/cover.png)

# 倒空间注意力 RSA：无预定义电荷的 MLIP 长程通道

> **论文**：Hariharan Ramasubramanian, Alvaro Vazquez-Mayagoitia, Ganesh Sivaraman & Atul C. Thakur, *Reciprocal Space Attention for Learning Long-Range Interactions*, **arXiv** (**2025**, 2510.13055).  
> **预印本**：[arXiv:2510.13055](https://arxiv.org/abs/2510.13055)  
> **机构**：Carnegie Mellon University；Argonne National Laboratory；University at Buffalo；UC San Diego.  
> **代码与数据**：[rfhari/reciprocal_space_attention](https://github.com/rfhari/reciprocal_space_attention)

---

## 一、背景：局域 MLIP 与长程物理的缺口

机器学习原子间势（**MLIP**）以 ab initio 能量与力为监督，在单步成本上比 DFT 低数个量级。主流 **MACE、NequIP** 等 GNN 在截断 $r\_{\mathrm{cut}}$ 内做消息传递；$L$ 层感受野 $\approx L\cdot r\_{\mathrm{cut}}$，属半局域。均匀 bulk 中长程常可均值场处理；但在**表面、界面、带电/极性介质**等场景，Coulomb、极化与色散等非局域效应不可忽略。

已有策略分两类：**电荷增强**（PhysNet 等）与**全局长程模块**（LODE、Ewald MP、SpookyNet、SCFNN 等）。本文 **Reciprocal-Space Attention（RSA）** 将线性注意力映射到 **Fourier 倒空间**，用 **Fourier Positional Encoding（FPE）** 与 Ewald 权重学习长程相互作用，**无需预定义电荷**；与 **MACE** 短程（SR）并联得 **LR-MACE**。

纳入长程模块的五项设计目标（原文 Introduction）：**(i)** 端到端可微、能量–力一致；**(ii)** 全局感受野且 scaling 可控；**(iii)** 兼容 PBC；**(iv)** 与短程骨干无缝集成；**(v)** 避免非可观测量（如预定义电荷）。

---

## 二、理论 §II：实空间注意力 → 倒空间 RSA

### 2.1 标准 dot-product 自注意力（式 (1)）

序列长度 $N$；$m$ 为 query 位置，$n$ 为 key/value 位置。$Q\in\mathbb R^{N\times d\_k}$，$K\in\mathbb R^{N\times d\_k}$，$V\in\mathbb R^{N\times d\_v}$；$\langle Q\_m,K\_n\rangle=Q\_m^{\top}K\_n$。

$$
A_m(Q,K,V)=\frac{\sum_{n=1}^{N}\exp\bigl(\langle Q_m,K_n\rangle\bigr)\,V_n}{\sum_{n=1}^{N}\exp\bigl(\langle Q_m,K_n\rangle\bigr)}
\tag{1}
$$

**含义**：标准 softmax 自注意力；计算与内存 **$O(N^2)$**。

### 2.2 RoPE 集成的线性注意力（式 (2)–(3)）

**RoPE** 对 query/key 施加位置相关旋转 $R\_m$；$\phi$ 为非负特征映射：

$$
A_m(Q,K,V)=\frac{\sum_{n=1}^{N}\bigl(R_m\phi(Q_m)\bigr)^{\top} R_n\phi(K_n)\,V_n^{\top}}{\sum_{n=1}^{N}\phi(Q_m)^{\top}\phi(K_n)}
\tag{2}
$$

对固定 $Q\_m$，$\sum\_{n=1}^{N}\phi(K\_n)V\_n^{\top}$ 与 $m$ 无关，可预计算为 K–V cache，式 (2) 化为：

$$
A_m(Q,K,V)=\frac{\bigl(R_m\phi(Q_m)\bigr)^{\top}\sum_{n=1}^{N} R_n\phi(K_n)\,V_n^{\top}}{\phi(Q_m)^{\top}\sum_{n=1}^{N}\phi(K_n)}
\tag{3}
$$

**含义**：整体计算 **$O(N)$**。RSA 将式 (3) 中 RoPE 替换为倒空间 **FPE**，并嵌入 Ewald 长程结构。

### 2.3 Ewald 势分割（式 (4)）

$$
V(r)=v_{\mathrm{SR}}(r)+v_{\mathrm{LR}}(r)=\frac{\mathrm{erfc}\!\left(\dfrac{r}{\sqrt{2}\sigma}\right)}{r}+\frac{\mathrm{erf}\!\left(\dfrac{r}{\sqrt{2}\sigma}\right)}{r}
\tag{4}
$$

$\mathrm{erf}$、$\mathrm{erfc}$ 为误差函数与互补误差函数；$\sigma$ 为 screening 参数。$v\_{\mathrm{SR}}$ 由 MACE 等 MLIP 表示；RSA 学习 $v\_{\mathrm{LR}}$ 的数据驱动等价物。

### 2.4 Ewald 长程能（式 (5)）

电中性体系（排除 $k=0$）：

$$
E_{\mathrm{LR}}=\frac{2\pi}{V}\sum_{k\neq 0}\frac{e^{-k^2\sigma^2/2}}{k^2}\sum_{m=1}^{N}\sum_{n=1}^{N}\tilde q_m\tilde q_n\,e^{ik\cdot(r_m-r_n)}=\frac{2\pi}{V}\sum_{k\neq 0}\frac{e^{-k^2\sigma^2/2}}{k^2}\left\lvert S(k)\right\rvert^2
\tag{5}
$$

$V$ 为胞体积；$k$ 为倒格矢；$\tilde q\_m$、$r\_m$ 为第 $m$ 个原子的电荷与坐标。正文说明 $S(k)S(-k)$ 通过 $e^{ik\cdot(r\_m-r\_n)}$ 全局耦合所有原子；对固定倒空间网格，$\lvert S(k)\rvert^2$ 形式将复杂度从 $O(N^2)$ 降至 **$O(N)$**。

### 2.5 Fourier 位置编码 FPE（式 (6)–(8)）

$$
\mathrm{FPE}_k(x,\vec r_m)=x\cdot e^{i\vec k\cdot\vec r_m}
\tag{6}
$$

PBC 下相位对晶格平移 $\mathbf T$ 不变：

$$
e^{ik\cdot[(r_m-r_n)+T]}=e^{ik\cdot(r_m-r_n)}
\tag{7}
$$

复 query/key 的内积 $\langle Q,K\rangle=Q^{\top}\bar K$（$\bar K$ 为复共轭）。FPE 下：

$$
\left\langle Q_m e^{ik\cdot r_m},\,K_n e^{ik\cdot r_n}\right\rangle=\langle Q_m,K_n\rangle\,e^{ik\cdot(r_m-r_n)}
\tag{8}
$$

**含义**：式 (8) 使 attention 内积携带相对位移相位，与 Ewald 求和同构，并保证平移不变与 PBC 兼容。

### 2.6 倒空间二次 attention（式 (9)–(10)）

不设 row-wise softmax，定义：

$$
\mathrm{RSA}_m(Q,K,V)=V_m=\sum_{k\neq 0}\sum_{n=1,\ldots,N}(Q_m,K_n)\,e^{ik\cdot(r_m-r_n)}\,V_n
\tag{9}
$$

与 per-atom Ewald 长程势对照：

$$
V_m^{\mathrm{LR}}=\frac{2\pi}{V}\sum_{k\neq 0}\sum_{n=1}^{N}\frac{e^{-k^2\sigma^2/2}}{k^2}\,\tilde q_n\,e^{ik\cdot(r_m-r_n)}
\tag{10}
$$

**含义**：式 (9) 对 $n$ 为 $O(N^2)$；式 (10) 为经典 Ewald per-atom 长程势，二者结构对应。

### 2.7 线性 RSA（式 (11)）

$$
\mathrm{RSA}_m(Q,K,V)\simeq\sum_{k\neq 0} w_k\,\mathrm{FPE}(\phi(Q_m),r_m)^{\top}\left(\sum_{n=1}^{N}\mathrm{FPE}(\phi(K_n),r_n)\,V_n^{\top}\right)
\tag{11}
$$

$w\_k=\exp(-k^2\sigma^2/2)/k^2$ 为 Ewald 权重；$\mathrm{FPE}(\phi(Q\_m),r\_m)$、$\mathrm{FPE}(\phi(K\_n),r\_n)$ 为经特征映射与 FPE 旋转的 query/key。内层对 $n$ 求和与 $m$ 无关，整体 **$O(N)$**（固定 $k$ 集合）。

![Fig. 1：SR/LR 并联 GNN 架构与 RSA 模块示意（原文 Fig. 1）](/img/posts/2026-06-19-rsa-reciprocal-space-attention/fig01-rsa-architecture.png)

**图 1 解读**

- **(a)**：$Z\_i$、晶格 $C$、坐标 $R\_i$、位移 $R\_{ij}=R\_j-R\_i$；每层 **SR-MP** 与 **LR-MP** 并联求和，Readout 输出能量。
- **(b)**：$h\_m^L,h\_n^L$ 经 FFN 与 $\sigma\_+$ 得 $\phi(Q\_m),\phi(K\_n)$，再 FPE 旋转；$\tilde K\_n\otimes V\_n$ 为 key–value cache，与 $\tilde Q\_m$ 收缩后对 $k$ 求和。

---

## 三、方法 §VI：MPNN 骨干与 RSA 实现

### 3.1 标准 MPNN（Methods A）

Gilmer 等 **MPNN** 框架：节点特征 $h\_m\in\mathbb R^H$、坐标 $r\_m\in\mathbb R^3$；$n\in\mathcal N(m)$ 且 $\lVert x\_n-x\_m\rVert<r\_{\mathrm{cut}}$ 时连边。第 $t$ 层：

$$
M_m^{(t+1)}=\sum_{n\in\mathcal N(m)} f_{\mathrm{int}}\!\left(h_m^{(t)},h_n^{(t)},e_{mn}\right),
$$

$$
h_m^{(t+1)}=f_{\mathrm{upd}}\!\left(h_m^{(t)},M_m^{(t+1)}\right).
$$

$f\_{\mathrm{int}}$ 聚合邻域消息，$f\_{\mathrm{upd}}$ 更新节点嵌入；readout 映射为原子能量贡献。**SR MACE** 每层即局域 MP 块。

### 3.2 LR-MACE 并联更新（式 (14)）

$$
M_m^{(t+1),\mathrm{nl}}=\mathrm{RSA}^{(t)}\!\left(H^{(t)},\delta\right)_m,\qquad h_m^{(t+1),\mathrm{nl}}=f_{\mathrm{upd}}\!\left(h_m^{(t)},M_m^{(t+1),\mathrm{nl}}\right).
\tag{14}
$$

$H^{(t)}=(h\_1^{(t)},\ldots,h\_N^{(t)})$；$\delta=(r\_1,\ldots,r\_N)$。Fig. 1(a) 中 SR 与 LR 分支逐层相加。

### 3.3 RSA 实现（Methods B，式 (15)）

由原子特征 $h\_m\in\mathbb R^H$ 投影得 query、key、value：

$$
Q_m=h_m W_Q,\quad K_m=h_m W_K,\quad V_m=h_m W_V,
$$

$W\_Q,W\_K,W\_V\in\mathbb R^{H\times D}$，文中 $D=H$。按式 (6)，经特征映射 $\phi$ 施加 FPE：

$$
\tilde Q_m=\mathrm{FPE}(\vec r_m,\phi(Q_m)),\qquad \tilde K_m=\mathrm{FPE}(\vec r_m,\phi(K_m)).
$$

$\tilde Q,\tilde K\in\mathbb R^{k\times H}$，$k$ 为倒空间向量数；实现限于**正交单胞**；非线性用 **SiLU**。先 $O(N)$ 计算 $\tilde K$ 与 $V$ 的外积 cache，再左乘 $\tilde Q\_m$：

$$
\mathrm{RSA}_m(Q,K,V)\simeq\sum_{k\neq 0} w_k\,\mathrm{FPE}(\phi(Q_m),r_m)^{\top}\left(\sum_{n=1}^{N}\mathrm{FPE}(\phi(K_n),r_n)\,V_n^{\top}\right)
\tag{15}
$$

每步对系统大小线性，整体 **$O(N)$**（正文亦对照式 (9)）。

### 3.4 损失函数（式 (16)）

$$
\mathcal L=\frac{1}{|\mathcal D|}\sum_{S\in\mathcal D}\left\lvert\hat E(S)-E(S)\right\rvert+\frac{\lambda}{|\mathcal D|}\sum_{S\in\mathcal D}\frac{1}{|S|}\sum_{i\in S}\left\lVert\hat F_i(S)-F_i(S)\right\rVert_1
\tag{16}
$$

$\lambda\ge 0$ 控制力项权重；$\lVert\cdot\rVert\_1$ 即 MAE。

### 3.5 Bulk water MD（Methods D）

300 水分子、$1\,\mathrm{g/mL}$、300 K、NVT、1 fs、ASE Langevin；SR/LR 各 10 条轨迹、每条 $\ge 300\,\mathrm{ps}$。

---

## 四、讨论 §IV：等变扩展（式 (12)–(13)）

### 4.1 球张量 FPE 增广（式 (12)）

$$
\tilde T_m^{(\ell)}(r,k)=e^{ik\cdot r}\,T_m^{(\ell)}(r)
\tag{12}
$$

$m\in\{-\ell,\ldots,\ell\}$。指数因子 $e^{ik\cdot r}$ 在 $\mathbf r$、$\mathbf k$ 同步旋转下不变，FPE 为 $\ell$ 无关标量相位；RSA 亦依赖 $e^{ik\cdot(r\_m-r\_n)}$（式 (9)），保持平移不变。

### 4.2 Plane-wave 展开（式 (13)）

$$
e^{ik\cdot r}=4\pi\sum_{\ell'=0}^{\infty}\sum_{m'=-\ell'}^{\ell'} i^{\ell'} j_{\ell'}(kr)\,Y_{\ell'm'}(\hat r)\,Y_{\ell'm'}^*(\hat k)
\tag{13}
$$

$j\_{\ell'}$ 为球 Bessel 函数；$Y\_{\ell'm'}$ 为球谐；$\hat r,\hat k$ 为单位向量。将式 (13) 与式 (12) 结合，对 $T\_m^{(\ell)}(r)=f\_\ell(r)Y\_m^{(\ell)}(\hat r)$ 得：

$$
\tilde T_m^{(\ell)}(r,k)=4\pi\sum_{\ell'=0}^{\infty}\sum_{m'=-\ell'}^{\ell'} i^{\ell'} Y_{\ell'm'}^*(\hat k)\times f_\ell(r)\,j_{\ell'}(kr)\,Y_{\ell'm'}(\hat r)\,Y_m^{(\ell)}(\hat r).
$$

该式与 **LODE** 势场描述符结构相近。本文实现 MACE **标量通道**（$\ell\_{\max}=0$）；可扩展为 $m\_{\mathrm{inv}}^{\mathrm{LR}}\otimes m\_{\ell m}^{\mathrm{SR}}$ 混合方案。

### 4.3 Scaling 与局限

固定密度下全文倒格矢求和为 **$O(N^{3/2})$**；截断前 $K$ 个低频模近 **$O(N)$**。可接 PME/SPME/PPPM；$\mathbf k$ 网格与盒尺寸可能有弱依赖；当前限于**正交单胞**。

---

## 五、结果 §III：全部 Fig/Table

### 5.1 SN2 反应（Fig. 2）

F/I 子集；MACE $r\_{\mathrm{cut}}=5\,\mathrm{\AA}$、2 层 MP（感受野 $\approx 10\,\mathrm{\AA}$）；RSA $\sigma=5\,\mathrm{\AA}$。

![Fig. 2：SN2 反应势能曲线（原文 Fig. 2）](/img/posts/2026-06-19-rsa-reciprocal-space-attention/fig02-sn2-reaction-energy.png)

**图 2 解读**：LR-MACE 全程贴合 DFT；SR-MACE 超感受野后能量饱和。

### 5.2 二聚体结合曲线（Fig. 3）

BFDB，$30\,\mathrm{\AA}$ PBC；CP 类 PBE0+MBD（步长 $0.1\,\mathrm{\AA}$）；水二聚体参考 SPC/E。

![Fig. 3：CP 与水二聚体结合曲线（原文 Fig. 3）](/img/posts/2026-06-19-rsa-reciprocal-space-attention/fig03-dimer-binding-curves.png)

**图 3 解读**：LR-MACE 恢复长程 tail；SR-MACE 截断外饱和。

### 5.3 随机电荷与熔融 NaCl（Table I）

随机电荷：128 原子、$\pm 1e$ 各 64；$5\,\mathrm{\AA}$、2 层、$\sigma=5\,\mathrm{\AA}$。NaCl：1014 构型、128 原子、80/20；单层、$6\,\mathrm{\AA}$。

![Table I：随机电荷与熔融 NaCl MAE（原文 Table I）](/img/posts/2026-06-19-rsa-reciprocal-space-attention/table01-benchmark-mae.png)

| 数据集 | 模型 | 能量 MAE (meV/atom) | 力 MAE (meV/Å) |
|--------|------|---------------------|----------------|
| Random Charges | LR MACE (10 Å) | **2.5** | **71.6** |
| Random Charges | SR MACE (10 Å) | 3.0 | 97.1 |
| Liquid NaCl | LR MACE (6 Å) | **6.8** | **141.9** |
| Liquid NaCl | SR MACE (6 Å) | 8.7 | 175.1 |

### 5.4 磷烯剥离（Fig. 4）

Deringer 等 DFT+MBD；SR $6\,\mathrm{\AA}$、2 层；LR $\sigma=5\,\mathrm{\AA}$。

![Fig. 4：磷烯层间剥离示意与能量曲线（原文 Fig. 4）](/img/posts/2026-06-19-rsa-reciprocal-space-attention/fig04-phosphorene-exfoliation.png)

**图 4 解读**：LR-MACE 全距离贴合 DFT+MBD，无需经验 $R\_6$。

### 5.5 Bulk water（Fig. 5–6）

1593 构型 × 64 分子，revPBE0-D3；$6\,\mathrm{\AA}$、2 层；RSA $\sigma=5\,\mathrm{\AA}$。

![Fig. 5：bulk water 快照与 O–O RDF（原文 Fig. 5）](/img/posts/2026-06-19-rsa-reciprocal-space-attention/fig05-bulk-water-rdf.png)

**图 5 解读**：$g\_{\mathrm{OO}}(r)$ 两模型与实验一致，局域结构由 SR 主导。

![Fig. 6：纵向 dipole–density 关联 $\chi\_{zz}(k)$（原文 Fig. 6）](/img/posts/2026-06-19-rsa-reciprocal-space-attention/fig06-dipole-density-correlation.png)

**图 6 解读**：$\chi\_{zz}(k)$ 在 $k\to 0$ 关联介电常数；SPC/E 固定电荷评估 dipole。SR-MACE 小 $k$ 发散；LR-MACE 正确给出长波长行为。

---

## 六、延伸讨论：RSA 与 EFA 的对比

### 6.1 一句话定位

| 方法 | 核心 |
|------|------|
| **EFA** | **欧氏实空间** ERoPE + 球面积分 → $\mathrm{sinc}$ 核，$O(N)$ 全局线性 attention；强调 $SO(3)$ 与各向异性长程 |
| **RSA** | **倒空间** FPE + Ewald $w\_k$，$O(N)$ 线性 attention；强调 PBC、electrostatics/dispersion、无预定义电荷 |

共同根源：线性 attention（K–V cache）、复数位置编码、局域 MP $\parallel$ 全局长程块。分岔在**编码空间**与**物理先验**。

### 6.2 ERoPE（EFA）与 FPE（RSA）

Frank 等 **ERoPE**：

$$
\mathrm{ERoPE}_u(x,r):=x\cdot e^{i\omega u\cdot r}
\tag{EFA-1}
$$

$u\in S^2$ 为单位方向，$\omega$ 为频率。对方向积分得旋转不变、仅依赖距离的核：

$$
\frac{1}{4\pi}\int_{S^2} e^{i\omega u\cdot r_{mn}}\,du=\frac{\sin(\omega r_{mn})}{\omega r_{mn}}=\mathrm{sinc}(\omega r_{mn})
\tag{EFA-2}
$$

**RSA 的 FPE** 见式 (6)(8)：离散倒格矢 $\mathbf{k}$ 求和，并乘 $w\_k=\exp(-k^2\sigma^2/2)/k^2$（式 (11)），与 Ewald 长程势（式 (10)）同构。

| 维度 | EFA | RSA |
|------|-----|-----|
| 编码 | 实空间 $r$、方向 $u$、$\omega$ | 倒格矢 $\mathbf{k}$、$\sigma$ |
| 长程形状 | $\mathrm{sinc}(r)$ 几何先验 + 学习 | Ewald $w\_k$ 先验 + 学习 |
| PBC | 非核心（部分 benchmark 有） | **核心设计** |
| 线性 attention | $\sum\_n \psi(q\_m)^\top\psi(k\_n)v\_n^\top$（K–V cache） | 式 (11) 型 FPE cache |

### 6.3 对称性与表达力

**EFA** 等变版在 attention 中引入球谐 $Y(u)$：$\ell=0$ 看距离，$\ell\ge 1$ 编码**距离+取向**。已在**非局域电荷转移**、**累积烯**二面角（~0.7 eV 势垒 vs MP 零势垒）、**DES370K 二聚体**长程系数等场景验证。

**RSA** 正文为 **MACE 标量**（$\ell\_{\max}=0$）；等变在 Discussion 式 (12)(13)，benchmark 未落地。

- **charge–dipole、二面角离域、取向依赖长程** → 当前 **EFA 更完整**  
- **周期晶体、bulk 介电、层间色散** → **RSA 的 Ewald–Fourier 更贴物理**

### 6.4 实验场景对照

| 场景 | EFA（Nat MI 2026） | RSA（arXiv 2025） |
|------|-------------------|-------------------|
| SN₂ | MP+EFA 全路径 + MD 正确 | LR-MACE 贴合 DFT（Fig. 2） |
| NaCl / 静电 | 团簇 scaling：单层 MP+EFA $\gg$ 多层 MP | 熔融 NaCl、随机电荷（Table I） |
| 二聚体 tail | DES370K | BFDB / 水二聚体（Fig. 3） |
| 色散 | 二聚体长程系数 | 磷烯剥离（Fig. 4） |
| 介电 / 低 $q$ | 未强调 | **$\chi\_{zz}(k)$**（Fig. 6） |
| 各向异性非局域 | 累积烯、Au$\_2$-MgO 等 | 未覆盖 |

重叠 benchmark 说明二者均能补长程；**差异化**：EFA 强 **分子/团簇 MD 与各向异性**；RSA 强 **PBC、介电、层状色散**。

### 6.5 各自优势与选型

**RSA 更擅长**：强 PBC 晶体/熔盐/层状材料；无预定义电荷/经验色散；低 $q$ dipolar screening；与 **IBZ-PRD / ReciNet / RSNN** 等倒空间模块同坐标系；与 MACE SR/LR 并联路径清晰。

**EFA 更擅长**：大分子/团簇/气相反应（不依赖倒格矢网格）；$\ell\ge1$ 各向异性长程已验证；NaCl 团簇「一层 MP+EFA vs 多层 MP」叙事；*Nature Machine Intelligence* 正式发表与更全 MD 对比。

### 6.6 与 IBZ-PRD 的侧向对照

RSA 与 IBZ-PRD 的完整衔接（融合路线、模式分辨、困难取舍）见倒空间方法谱系对比文 **§7.6**。此处仅作 EFA 对照下的侧向摘要：

| 长程支路 + IBZ-PRD | 直觉 |
|--------------------|------|
| **RSA + IBZ-PRD** | $\mathbf{q}\_\ell$ 可作 RSA 的 $\mathbf{k}$ 集；低 $q$/LO-TO 与 RSA electrostatics 同向 |
| **EFA + IBZ-PRD** | 双支路融合（实空间 per-atom 全局 + IBZ 路径指纹），类似 ReciNet + IBZ-PRD |
| **周期谱学 / 极性 solid** | 倾向 **RSA** 长程 |
| **分子谱学 / 各向异性非局域** | 倾向 **EFA**（尤其 $\ell\ge1$） |

### 6.7 谱系位置（综合）

```
局域 MP（MACE / 通用 MPNN）
        │
        ├── EFA：ERoPE → ∫_{S²} → sinc(r)   [实空间、SO(3)、分子/团簇]
        │
        └── RSA：FPE → Σ_k w_k             [倒空间、PBC、固体/介电]
```

二者**非替代关系**。若目标为 **高精度 IR/Raman + IBZ 色散 + 极性晶体**，**RSA 长程 + IBZ-PRD 指纹**在倒空间上一致性更好；若目标为 **气相反应、累积烯式非局域、电荷转移**，**EFA**（尤其等变版）证据更充分。谱系文建议的可复用模块组合中，RSA 可填入「**倒空间 Ewald-attention 长程核**」一格，与 ReciNet 层内更新、IBZ-PRD 路径采样、EFA 实空间全局 attention 并列。

---

## 七、方法摘要与结论

| 项目 | 内容 |
|------|------|
| **核心** | 式 (3) 线性 attention + 式 (6)–(8) FPE + 式 (11) Ewald 权重 → LR 通道 |
| **骨干** | MACE（$\ell\_{\max}=0$）$\parallel$ RSA → LR-MACE |
| **超参** | $r\_{\mathrm{cut}}$、MP 层数、$\sigma$（常 5–6 Å） |
| **代码** | [github.com/rfhari/reciprocal_space_attention](https://github.com/rfhari/reciprocal_space_attention) |

**RSA** 以 **FPE** 编码 Bloch 相位，配合 **Ewald 型 $w\_k$** 与线性 attention，在不引入部分电荷或经验色散项的前提下学习长程 electrostatics/dispersion。LR-MACE 在 SN2、二聚体、随机电荷、NaCl、磷烯剥离与 bulk water $\chi\_{zz}(k)$ 上系统恢复 SR 缺失的长程 asymptotics。延伸讨论（§六）表明 **与 EFA** 分属倒空间 Ewald-attention 与实空间 sinc-attention，面向周期固体/介电 vs 分子各向异性非局域。

---

## 延伸阅读（站内）

- [ReciNet：倒空间感知的晶体长程建模——Nie 等 2026 解读](/2026/06/18/ReciNet-倒空间感知的晶体长程建模-Nie-等-2026-解读/)
- [当原子"看见"彼此：EFA 如何让机器学习力场拥有全局视野](/2026/05/25/EFA全局注意力与势函数长程建模/)
- [布里渊区、不可约布里渊区与高对称路径——从固体物理基础到 PRDNet 的 IBZ 迁移](/2026/05/29/布里渊区-不可约布里渊区与高对称路径-从固体物理基础到-PRDNet-的-IBZ-迁移/)

## 参考文献（精选）

- Ramasubramanian et al., *Reciprocal Space Attention for Learning Long-Range Interactions*, arXiv:2510.13055 (2025).
- Frank, J. T., Chmiela, S., Müller, K.-R. & Unke, O., Euclidean Fast Attention, *Nat. Mach. Intell.* **8**, 388–402 (2026). https://doi.org/10.1038/s42256-026-01195-y
- Batatia et al., MACE, NeurIPS 2022.
- Grisafi & Ceriotti, LODE, JCP 2019.
- Kosmala et al., Ewald message passing, arXiv:2303.04791.
- Nie et al., ReciNet, TMLR 2026.
- Gao & Remsing, SCFNN, Nat. Commun. 2022.
- Anstine & Isayev, MLIP long-range review, JPCA 2023.
