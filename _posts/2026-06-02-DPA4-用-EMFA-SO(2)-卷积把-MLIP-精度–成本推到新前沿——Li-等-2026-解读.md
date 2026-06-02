---
layout:     post
title:      DPA4 用 EMFA SO(2) 卷积把 MLIP 精度–成本推到新前沿——Li 等 2026 解读
subtitle:   北大/DP Technology 等提出 DPA4：博客含式 (1)–(44) 全文推导 + S-1 算子 (45)–(47)、(57)–(67)、(69) 与四元数规范 (48–50)；GIE/FiLM、EMFA、Lebedev FFN、Native ZBL、保守损失；Matbench CPS 0.833，Air 42.9× 更少训练算力。
date:       2026-06-02
author:     qqz
header-img: img/post-bg-desk.jpg
catalog:    true
tags:
    - 机器学习势
    - 大原子模型
    - DPA4
    - 材料模拟
    - SE3等变
---

![一图总结](/img/posts/2026-06-02-dpa4-emfa-so2-convolution/cover.png)

# DPA4 用 EMFA SO(2) 卷积把 MLIP 精度–成本推到新前沿

> **论文**：Tiancheng Li, Wentao Li, Anyang Peng, Jianming Xue, Linfeng Zhang, Duo Zhang & Han Wang, *DPA4: Pushing the Accuracy–Cost Frontier of Interatomic Potentials with EMFA SO(2) Convolution*, **arXiv:2606.02419** (2026).  
> **预印本**：[arXiv:2606.02419](https://arxiv.org/abs/2606.02419)  
> **机构**：北京大学、北京科学智能研究院（AI for Science Institute）、清华大学、DP Technology、应用物理与计算数学研究所等。

---

## 一、背景：LAM 很准，但训练太贵

**机器学习原子间势（MLIP）** 正从「单体系专用」走向 **大原子模型（LAM）** 预训练：M3GNet、CHGNet、MACE、MatterSim、Orb、UMA、DPA 系列等，目标是用一个模型覆盖大量元素与化学空间，充当 DFT 替身跑 MD、高通量筛选与分子设计。

但两条矛盾同时存在：

| 维度 | 现状痛点 |
|------|----------|
| **精度** | NequIP、MACE、Equiformer、eSEN 等 **SE(3) 等变** 架构把方向当作一等特征，benchmark 上明显更省数据、更准 |
| **成本** | 高阶 Clebsch–Gordan 张量积随角动量 $L$ 暴涨；UMA-M 等 LAM 训练可达 **12.9 万 H200 GPU·小时** |
| **训练协议** | 保守 **能量–梯度** 训练（力由能量求导）需 **二阶反向**，难吃 LLM 式训练栈；不少 SOTA 靠 **DeNS 去噪** 或 **直接预测力** 预训练再微调 |

Li 等提出 **DPA4**：在 **边局域 SO(2) 等变卷积 EMFA** 上同时抬精度、压参数与训练算力，并用 **torch.compile 友好的保守能量梯度路径** 把墙钟训练加速约 **3×**；短程用 **Native ZBL Zone Bridging** 把解析 ZBL 排斥与可学习分支在同一标量能量里耦合，避免外接 ZBL 修正的力不连续。

![图 1：代表性 MLIP 的 Matbench Discovery 综合分 CPS 与 A100 训练成本（对数坐标）；气泡面积 ∝ 参数量。DPA4 Neo/Air/Plus/Pro 连成新帕累托前沿，DPA4-Air 相对 eSEN-30M-MP 标注 42.9× 更少训练算力。](/img/posts/2026-06-02-dpa4-emfa-so2-convolution/fig01-cps-vs-training-cost.png)

---

## 二、DPA4 架构总览

DPA4 是 **保守 SE(3) 等变** 消息传递图网络：在实空间不可约表示 $V\_{\le L}\otimes\mathbb{R}^C$ 上做 $N\_{\mathrm{layer}}$ 轮交互，只从 **$l=0$ 标量切片** 读出原子能量，力与应力由总能量对坐标/晶胞的梯度给出。

![图 2：DPA4 架构总览。(a) 共享边缓存 → GIE → $N\_{\mathrm{layer}}$ 个交互块（EMFA SO(2) 卷积 + 等变 FFN + RMSNorm）→ 原子能量头；(b) 边缓存与几何嵌入 GIE（FiLM + 球谐投影）；(c) EMFA SO(2) 卷积：低秩边–节点乘积 A1、多 focus A2、包络门控注意力 A3。](/img/posts/2026-06-02-dpa4-emfa-so2-convolution/fig02-dpa4-architecture-overview.png)


下文 **第三节至第十三节** 按论文 Section 4.2–4.4 与补充 S-1 系统推导；**式号与原文一致**。阅读顺序建议：总能量 → ZBL 桥接 → 群表示与边缓存 → GIE → EMFA → FFN → 辅助层与训练 → 对称性。

---

## 三、总能量、NN 分支分解与保守性

### 3.1 总势能与力的定义

$$
E(Z,R) = E_{\mathrm{NN}}^{\Theta}(Z,R) + E_{\mathrm{ZBL}}(Z,R)
\tag{1}
$$

- $\Theta$：可学习分支全部参数  
- **力**：$\mathbf F\_k = -\partial E / \partial R\_k$  
- **维里**：对晶胞应变 $\boldsymbol\varepsilon$ 求 $\partial E / \partial \boldsymbol\varepsilon$（周期性体系应力张量）  
- 两分支在同一标量 $E$ 上求导 → **严格保守**，适合 NVE/NPT 与结构弛豫

### 3.2 学习分支 $E\_{\mathrm{NN}}^{\Theta}$ 的数据流

$$
E_{\mathrm{NN}}^{\Theta}(Z,R)
= \sum_{i=1}^{N} \varepsilon_i\!\left(h_i^{(N_{\mathrm{layer}})}\right),
\qquad
h_i^{(0)} = \mathrm{GIE}(Z,R),
\qquad
h_i^{(\ell)} = \mathrm{Block}^{(\ell)}\!\left(h_i^{(\ell-1)}\right)
$$

每个 **Block** 含两次残差：

$$
h_i \leftarrow h_i + \mathcal N\!\left(\mathcal C_\Theta h\right)_i,
\qquad
h_i \leftarrow h_i + \mathcal N\!\left(\mathcal F_\Theta^{\mathrm{FFN}}(h)\right)_i
$$

$\mathcal N$ 为 **等变 RMSNorm**（式 66）;$\mathcal C\_\Theta$ 为 EMFA SO(2) 卷积（式 30）;$\mathcal F\_\Theta^{\mathrm{FFN}}$ 为 Lebedev FFN（式 31）。

**原子能量头** $\varepsilon\_i$ 只读取 $h\_i$ 的 **$l=0$ 不变切片** $h\_{i,l=0}\in\mathbb R^C$：

$$
\varepsilon_i = W_{\mathrm{out}}\,\sigma\!\bigl(W_{\mathrm{in}}\, h_{i,\,\iota(0,0),:}^{(N_{\mathrm{layer}})} + b_{\mathrm{in}}\bigr) + b_{\mathrm{out}}
$$

（实现上为作用于 $l=0$ 通道的 MLP 或线性层；$W\_{\mathrm{in}},W\_{\mathrm{out}}$ 不耦合 $l\ge 1$ 系数。）因 $\varepsilon\_i$ 仅依赖 $D\_0\equiv 1$ 的标量块，$E\_{\mathrm{NN}}=\sum\_i\varepsilon\_i$ 为 **旋转标量**。

### 3.3 特征空间与角动量截断

节点特征 $h\_i \in V\_{\le L}\otimes\mathbb R^C$，其中

$$
V_{\le L} = \bigoplus_{l=0}^{L} V_l,\quad \dim V_l = 2l+1
$$

旋转 $Q\in\mathrm{SO}(3)$ 作用下，系数按 **实 Wigner D-矩阵** $D\_l(Q)$ 块对角变换（式 45–46）：

$$
Y_l^m(Q^{-1}\hat{\mathbf r}) = \sum_{m'=-l}^{l} D_l(Q)_{m,m'}\, Y_l^{m'}(\hat{\mathbf r})
\tag{45}
$$

$$
D(Q) = \mathrm{diag}\bigl(D_0(Q), D_1(Q), \ldots, D_L(Q)\bigr),\quad D_0\equiv 1
\tag{46}
$$

系数按线性索引打包（式 47）：

$$
\iota(l,m) = l^2 + l + m,\qquad m=-l,\ldots,l,\quad l=0,\ldots,L
\tag{47}
$$

在此打包下，$D(Q)$ 为块对角，$(2l+1)\times(2l+1)$ 块 $D\_l(Q)$ 占据行/列 $\iota(l,-l),\ldots,\iota(l,+l)$。

EMFA 卷积内为省算力，只保留 $\lvert m \rvert\le M\le L$ 的 **$m$ 子空间**（式 57–59，补充 S-1.4）：

$$
\mathcal I_M = \{(l,m): 0\le l\le L,\ |m|\le \min(l,M)\},\qquad D_M = |\mathcal I_M|
\tag{57}
$$

$P\_M:\mathbb R^{(L+1)^2}\to\mathbb R^{D\_M}$ 为到该 reduced layout 的 **正交投影**；边局域旋转截断为 $D\_{\le M}(R\_{ij})=P\_M D(R\_{ij})$（式 58）。因 $P\_M^{\mathsf T}P\_M\neq I$（当 $M<L$），往返全局系会损失各 $l$ 块范数；抬回时用对角补偿（式 59）：

$$
(\Xi_M)_{\iota(l,m),\iota(l,m)} = \kappa_l = \sqrt{\frac{2l+1}{2\min(l,M)+1}},\qquad
\Xi_M = \mathrm{diag}(\kappa_0 I_1,\kappa_1 I_3,\ldots,\kappa_L I_{2L+1})
\tag{59}
$$

**原理**：全 SO(3) 张量积需 Clebsch–Gordan 耦合，代价 $\mathcal O(L^6)$ 量级;把每条边 $(i,j)$ 的键向对齐 $z$ 轴后，剩余对称性降为 **SO(2)**，$m$ 量子数解耦，卷积在边局域系里做 **低秩乘积 + 注意力**，计算与参数量大幅下降（eSCN / eSEN 路线，DPA4 进一步用 **全 $l$ 边特征** 与 **Multi-Focus** 加厚表达）。

---

## 四、Native ZBL Zone Bridging（§4.2.1 + §4.2.5）

### 4.1 设计动机

极近距离（$r\lesssim 1\,\text{\AA}$）DFT 训练样本稀疏，学习分支外推不可靠;核排斥应由 **ZBL 解析势** 主导。若用 **能量级开关** 拼接 ZBL 与 NN（DP-ZBL），对坐标求导会产生 **与能量失配成正比的切换力**（式 42），Fig. 4 中 DPA3 力曲线尖峰即源于此。

DPA4 策略：**不**在总能量上乘 $\lambda(R)$ 混合两分支，而在 **进入 NN 的几何量** 上做 $C^3$ 钳制与源冻结，使内区 NN 梯度为零、ZBL 用真实 $r\_{ij}$ 单独求导。

### 4.2 七次 Hermite 桥接多项式（式 2）

在 $[r\_{\mathrm{in}}, r\_{\mathrm{out}}]\subset(0,r\_c)$ 上定义 $t(r)=(r-r\_{\mathrm{in}})/(r\_{\mathrm{out}}-r\_{\mathrm{in}})$，

$$
h_c(t) := 20t^4 - 45t^5 + 36t^6 - 10t^7,\qquad
h_w(t) := 35t^4 - 84t^5 + 70t^6 - 20t^7
\tag{2}
$$

$h\_c$：左端常数、右端恒等，两端 **三阶导连续** → 钳制距离 $\tilde r(r)$ 为 $C^3$。  
$h\_w$：两端常数 0/1，**三阶导为零** → 门函数 $w(r)$ 在端点无虚假振荡。

### 4.3 钳制距离、门与源冻结（式 3–5）

$$
\tilde r(r) =
\begin{cases}
r_{\mathrm{in}}, & r\le r_{\mathrm{in}} \\
r_{\mathrm{in}} + (r_{\mathrm{out}}-r_{\mathrm{in}})\, h_c(t(r)), & r_{\mathrm{in}}<r<r_{\mathrm{out}} \\
r, & r\ge r_{\mathrm{out}}
\end{cases}
\tag{3}
$$

$$
w(r) =
\begin{cases}
0, & r\le r_{\mathrm{in}} \\
h_w(t(r)), & r_{\mathrm{in}}<r<r_{\mathrm{out}} \\
1, & r\ge r_{\mathrm{out}}
\end{cases}
\tag{4}
$$

学习分支边矢量（方向不变）：

$$
\tilde{\mathbf r}_{ij} = \tilde r(r_{ij})\,\hat{\mathbf r}_{ij},\qquad
\|\tilde{\mathbf r}_{ij}\| = \tilde r(r_{ij})
\tag{4b}
$$

对源原子 $j$ 的 **源冻结门**：

$$
\eta_j := \prod_{i\in\mathcal N_{\mathrm{out}}(j)} w(r_{ji}) \in [0,1]
\tag{5}
$$

**推论 1**：若存在邻居 $r\_{ji}\le r\_{\mathrm{in}}$，则 $\eta\_j=0$，来自 $j$ 的一切 NN 消息被乘零;且内区 $\tilde r$ 为常数 → NN 对 $r\_{ji}$ **梯度为零**，内区排斥 **仅由 $E\_{\mathrm{ZBL}}$** 提供。  
**推论 2**：$r\_{ij}\ge r\_{\mathrm{out}}$ 时 $\tilde r=r\_{ij}$、$\eta\_j=1$，NN 见真实几何。

凡进入 GIE / EMFA / FFN 的径向量、球谐、Wigner 旋转，凡依赖 $r\_{ij}$ 者均用 $\tilde r(r\_{ij})$;$E\_{\mathrm{ZBL}}$ **始终**用 $r\_{ij}$。

### 4.4 解析 ZBL 对势（式 39–40）

$$
E_{\mathrm{ZBL}}(Z,R) = \frac{1}{2}\sum_{i\ne j} E_{\mathrm{ZBL}}^{ij}(r_{ij})
\tag{39}
$$

$$
E_{\mathrm{ZBL}}^{ij}(r) = \frac{k_e Z_i Z_j}{r}\,
\Phi\!\left(\frac{r}{a_{ij}}\right),\quad
a_{ij} = \frac{0.88534\,a_0}{Z_i^{0.23}+Z_j^{0.23}}
\tag{39b}
$$

屏蔽函数（四指数标准形）：

$$
\Phi(x) = 0.18175\,e^{-3.1998x} + 0.50986\,e^{-0.94229x}
+ 0.28022\,e^{-0.4029x} + 0.028171\,e^{-0.20162x}
\tag{40}
$$

### 4.5 拼接势的切换力（式 41–42，对比用）

$$
E_{\mathrm{splice}} = \sum_i \Bigl[\lambda_i(R)\,E_{\mathrm{ZBL}}^{(i)} + \bigl(1-\lambda_i(R)\bigr)\,E_{\mathrm{NN}}^{(i)}\Bigr]
\tag{41}
$$

$$
\mathbf F_{\mathrm{splice}} = \mathbf F_{\mathrm{weighted}}
- \sum_i (\nabla_R \lambda_i)\,\bigl(E_{\mathrm{ZBL}}^{(i)} - E_{\mathrm{NN}}^{(i)}\bigr)
\tag{42}
$$

第二项即 **切换力**：$\lambda\_i$ 随构型变化且两分支能量在拼接窗不匹配时出现。Native ZBL **无** 此项。

---

## 五、共享边缓存与几何输入（§4.2.1）

每条有向边 $(i,j)$（$r\_{ij}<r\_c$）预计算并 **各交互层复用**：

| 量 | 定义 | 用途 |
|----|------|------|
| $r\_{ij}$, $\hat{\mathbf r}\_{ij}$ | 真实距离与单位键向 | ZBL;定义边局域旋转 $R\_{ij}$ |
| $\tilde r\_{ij}=\tilde r(r\_{ij})$ | 钳制标量距离 | 进入 NN 的径向基、包络 |
| $\boldsymbol\varphi(r\_{ij})$ | 正弦径向基式 (15) | $\rho\_{ij}$、注意力偏置 |
| $s\_5(r\_{ij}), s\_7(r\_{ij})$ | $C^3/C^6$ cutoff 包络 (7) | 权重、度归一化 |
| $D\_{ij}, D\_{ij}^{-1}$ | 边局域 Wigner 矩阵 | 式 (16)(17)(24) 规范变换 |
| $g(r\_{ij};Z\_i,Z\_j)$ | 边种类嵌入 | 式 (8) 聚合 |
| $\eta\_j$ | 源冻结门 (5) | 式 (8)(13) 消息权重 |

**边局域规范**（式 16）：选 $R\_{ij}\in\mathrm{SO}(3)$ 使 $R\_{ij}\hat{\mathbf r}\_{ij}=\mathbf e\_z$。实现上用 **双四元数图**（式 48–50，补充材料 S-1.2）在球面几乎处处光滑地构造 $R\_{ij}$，避免南北极奇点。

---

## 六、Geometry-Informed Embedding（GIE，§4.2.2）

GIE 在 **第一层** 同时注入 **化学**（元素与边种类）与 **几何**（局域配位），避免纯靠多轮 message passing「慢慢发现」环境。

### 6.1 $l=0$：Deep Potential 式描述符 + FiLM

**Step 1 — 边四向量**（式 6）：把标量距离与方向合成 **SO(3) 矢量** 的输入

$$
u_{ij,0} = \frac{s_5(r_{ij})}{r_{ij}},\qquad
u_{ij,k} = u_{ij,0}\,\hat r_{ij,k},\quad k=1,2,3
\tag{6}
$$

**Step 2 — $C^3$ 截断包络**（式 7），$x=r/r\_c$：

$$
s_p(r) =
\begin{cases}
1 + x^p\bigl(a_p + b_p x + c_p x^2 + d_p x^3\bigr), & x\in[0,1) \\
0, & x\ge 1
\end{cases}
\tag{7}
$$

系数由 $s\_p(r\_c)$ 及 $s\_p^{(1)},s\_p^{(2)},s\_p^{(3)}$ 在 $r\_c$ 处均为 0 唯一确定, 正文用 $s\_5$ 做边权与度, $s\_7$ 做径向基.

**Step 3 — 聚合矩阵**（式 8–9）：

$$
A_i = n_i \sum_{j:\,r_{ij}<r_c} \eta_j\, \mathbf u_{ij} \otimes g(r_{ij};Z_i,Z_j),
\quad
n_i = (d_i+\epsilon)^{-1/2},\quad
d_i = \sum_{j:\,r_{ij}<r_c} s_5(r_{ij})^2
\tag{8–9}
$$

$\mathbf u\_{ij}\in\mathbb R^4$ 的空间三分量按矢量表示变换，标量分量不变 → $A\_i$ 的 Gram 型收缩为 **旋转不变**。

**Step 4 — 局域环境描述符**（式 10）：

$$
D_i = A_i^{\mathsf T} A_i^{(:,1:K_{\mathrm{env}})} \in \mathbb R^{C_{\mathrm{env}}\times K_{\mathrm{env}}}
\tag{10}
$$

截断列数 $K\_{\mathrm{env}}$ 控制代价。

**Step 5 — FiLM 调制元素嵌入**（式 11–12）：

初值 $h\_{i,l=0,c}^{(0)}=T\_{Z\_i,c}$（可学习元素表 $T$）。几何通过 **Feature-wise Linear Modulation** 注入：

$$
h_{i,l=0}^{(0)} \leftarrow \boldsymbol\gamma_i \odot h_{i,l=0}^{(0)} + \boldsymbol\beta_i
\tag{11}
$$

$$
\boldsymbol\gamma_i = \mathbf 1 + e^{\beta_\gamma}\,\tanh\!\bigl(\mathcal N_0(W_\gamma\,\mathrm{vec}\,D_i)\bigr),\quad
\boldsymbol\beta_i = e^{\beta_\beta}\,\tanh\!\bigl(\mathcal N_0(W_\beta\,\mathrm{vec}\,D_i)\bigr)
\tag{12}
$$

$W\_\gamma,W\_\beta\in\mathbb R^{C\times C\_{\mathrm{env}}K\_{\mathrm{env}}}$;$\mathcal N\_0$ 为标量 RMSNorm;$\beta\_\gamma,\beta\_\beta$ 初始为 $\log(0.01)$ → 训练初 **$\boldsymbol\gamma\approx\mathbf 1,\ \boldsymbol\beta\approx\mathbf 0$**（近恒等），化学嵌入主导、几何微调。

**原理**：传统 GNN 先把 $Z\_i$ one-hot 嵌入再传消息;GIE 用 **与 Deep Potential 同构的不变描述符** 直接调制标量通道，一轮即携带 **配位数、键长分布、物种关联**。

### 6.2 $l\ge 1$：球谐投影 × 径向–物种 profile（式 13–15）

高阶无纯化学基线，必须从邻域 **方向** 出发：

$$
h_{i,\,\alpha(l,m),\,c}^{(0)} \mathrel{+}= n_i \sum_{j:\,r_{ij}<r_c}
\eta_j\, Y_l^m(\hat{\mathbf r}_{ij})\, \rho_{ij,l,c},\quad l\ge 1
\tag{13}
$$

$$
\rho_{ij,l,c} = \bigl[\phi_{\mathrm{rad}}(\boldsymbol\varphi(r_{ij}))\bigr]_{l,c}
+ \bigl[T_{\mathrm{edge}}(Z_i,Z_j)\bigr]_c
\tag{14}
$$

$$
\varphi_n(r) = \frac{\sin(\omega_n r)}{r}\, s_7(r),\quad
\omega_n = \frac{n\pi}{r_c},\quad n=1,\ldots,n_r
\tag{15}
$$

$\phi\_{\mathrm{rad}}$：无偏置 SiLU MLP，$\mathbb R^{n\_r}\to\mathbb R^{(L+1)\times C}$。  
**等变性**：$\rho\_{ij}$、$n\_i$、$\eta\_j$ 不变;$Y\_l^m(\hat{\mathbf r}\_{ij})$ 在全局旋转下按 $D\_l$ 变换 → $h\_{i}^{(0)}$ 在 $V\_{\le L}$ 上 **SO(3) 等变**。

---

## 七、EMFA SO(2) 卷积（§4.2.3，A1–A3）

算子 $\mathcal C\_\Theta: (h\_j)\_{j=1}^N \mapsto (\mathcal C\_\Theta h)\_i$ 对目标原子 $i$ 聚合邻居 $j$。六步闭式见式 (30)。

### 7.1 边局域化（式 16–17）

$$
R_{ij}\,\hat{\mathbf r}_{ij} = \mathbf e_z
\tag{16}
$$

$$
h_j' = L_{\mathrm{pre}}^{\mathrm{deg}}\, h_j,\qquad
x_{ij} = P_M\, D(R_{ij})\, h_j' \in \mathbb R^{D_M\times H}
\tag{17}
$$

$H=F\cdot C\_f$;$L\_{\mathrm{pre}}^{\mathrm{deg}}$ 为 **按角动量阶 $l$ 分块的通道升维**（式 62 族）。边特征 $\tilde\rho\_{ij}\in\mathbb R^{(L+1)\times H}$ 由式 (14) 经 $L\_{\mathrm{rad}}^{\mathrm{lift}}$ 提升。

### 7.2 A1：低秩边–节点 SO(2) 乘积（式 18–19）

边系中 $Y\_l(\hat{\mathbf r}\_{ij})$ 仅剩 **$m=0$** 分量，故 $\tilde\rho\_{ij}$ 充当各 $l$ 阶边 SO(2) 不可约表示的 **径向调制 $m=0$ 切片**。在固定 $\lvert m \rvert$ 子空间内，对 **不同 $l$** 做可学习混合（**不** 混合不同 $\lvert m \rvert$）：

$$
x_{ij,l,m,c} \mathrel{+}= \sum_{l':\,|m|\le l'\le L}
K_{l,l',|m|,c}(\tilde\rho_{ij})\, x_{ij,l',m,c}
\tag{18}
$$

$$
K_{l,l',|m|,c}(\tilde\rho_{ij}) = \sum_{r=1}^{R}
K^{(r)}_{l,l',|m|}(\tilde\rho_{ij})\, B_{r,c},\quad R\ll H
\tag{19}
$$

**与 eSEN/EqV3 差异**：后者边侧常只用 $l=0$ 不变标量;DPA4 边特征用 **全 $(L+1)$ 阶** $\tilde\rho\_{ij}$ 线性泛函生成 $K$，表达力更强。  
**与 SO(3) CG 张量积差异**：在 SO(2) 框架下 $\lvert m \rvert$ 解耦，$(-m,+m)$ 成对构成 2D 实表示，$K$ 不混 $\lvert m \rvert$ → 等变保持;代价远低于全 CG。

### 7.3 A2：Multi-Focus 双非线性（式 20–23）

**（i）按 focus 并行的 SO(2) 残差栈**（式 20–21）

隐宽分解 $x\_{ij}\in\mathbb R^{D\_M\times F\times C\_f}$，每 focus $f$ 上：

$$
x_{ij} \leftarrow \mathcal S_\Theta(x_{ij}),\quad
\mathcal S_\Theta = \text{复合 } S \text{ 层}
\tag{20}
$$

单层（式 21）：

$$
x_{ij} \leftarrow x_{ij} + \alpha_s\,\Phi_s\!\Bigl(
L_s^{\mathrm{SO(2)}}\,\mathcal N_s(x_{ij})
\Bigr),\quad s=1,\ldots,S
\tag{21}
$$

- $L\_s^{\mathrm{SO(2)}}$：边无关、**固定 $\lvert m \rvert$ 内跨 $l$ 混合** 的 SO(2) 等变线性（式 63–64）  
- $\mathcal N\_s$：等变 RMSNorm（式 66）  
- $\Phi\_s$：作用在 **$l=0$ 切片** 上的门控激活（式 67）  
- $\alpha\_s\in\mathbb R^{F\times C\_f}$：小初值残差缩放（$\sim 10^{-3}$）

**（ii）Cross-focus softmax 竞争**（式 22–23）

取栈入口的 $l=0$ 分量 $x\_{ij}^{(0)}\in\mathbb R^{F\times C\_f}$，经 focus-wise $\mathcal N\_0$ 后：

$$
\omega_{ij,f} = (1-\varepsilon)\,
\frac{\exp\!\bigl(\tau^{-1}\sum_c W_{c,f}^{\mathrm{cf}}\,\mathcal N_0(x_{ij}^{(0)})_{f,c}\bigr)}
{\sum_{f'}\exp(\cdots)+\varepsilon F}
+ \frac{\varepsilon}{F}
\tag{22}
$$

$\tau>0$ 温度;$\varepsilon\in[0,1)$ **label smoothing** 防止某 focus 权重塌缩到 0。重加权：

$$
x_{ij} \leftarrow \boldsymbol\omega_{ij}\,\odot\, x_{ij}
\quad (\boldsymbol\omega_{ij}\in\mathbb R^F \text{ 广播到 } l,m,c)
\tag{23}
$$

$\omega\_{ij,f}$ 只依赖 SO(2) 不变标量 → 不破坏等变。  
**原理**：单路 SO(2) 栈 + 宽通道 ≈ 大矩阵;拆成 $F$ 路窄栈 + softmax 竞争 → **参数量降、非线性 richer**（消融 A2 验证）。

### 7.4 抬回全局系（式 24）

$$
m_{ij} = \Xi_M\, D(R_{ij})^{-1}\, P_M^{\dagger}\, x_{ij} \in V_{\le L}\otimes\mathbb R^H
\tag{24}
$$

$P\_M^{\dagger}$：截断 $m$ 子空间嵌入回 $(L+1)^2$ 布局;$\Xi\_M$：度依赖标量重标定（式 59）。

### 7.5 A3：包络门控注意力（式 25–30）

将 $m\_{ij}$ 按 $H=F\cdot H\_a\cdot d\_a$ 拆成 **focus × head × 通道**。

**Query / Key**（式 25，对每 $(f,a)$）：

$$
q_i^{(f,a)} = Q^{(f)}\,\mathcal N_0\!\bigl(h_i'|_{l=0}\bigr)_{f,a,:},\quad
k_j^{(f,a)} = K^{(f)}\,\mathcal N_0\!\bigl(h_j'|_{l=0}\bigr)_{f,a,:}
\tag{25}
$$

**Logit**（式 26）：

$$
\ell_{ij}^{(f,a)} = \frac{\langle q_i^{(f,a)}, k_j^{(f,a)}\rangle}{\sqrt{d_a}}
+ \sum_{c=1}^{C_f} W_{c,f,a}^{\mathrm{rb}}\, \tilde\rho_{ij,0,c}
\tag{26}
$$

**注意力权重**（式 27）：

$$
w_{ij}^{(f,a)} =
\frac{s_5(r_{ij})^2\,\eta_j\,\exp(\ell_{ij}^{(f,a)})}
{\mathrm{softplus}(\alpha_{f,a}) + \sum_{k:\,r_{ik}<r_c} s_5(r_{ik})^2\,\eta_k\,\exp(\ell_{ik}^{(f,a)})}
\tag{27}
$$

- 分子 $s\_5^2\eta\_j$：cutoff 处 $C^3$ 趋于 0;内区 $\eta\_j=0$ 静默  
- 分母 **softplus$(\alpha\_{f,a})$** 保证无邻居时仍正定 → 避免 0/0

**聚合与输出门**（式 28–29）：

$$
A_i^{(f,a)} = \sum_{j:\,r_{ij}<r_c} w_{ij}^{(f,a)}\, m_{ij}^{(f,a)},\quad
\tilde A_i^{(f,a)} = G_i^{(f,a)}\, A_i^{(f,a)}
\tag{28–29}
$$

$$
G_i^{(f,a)} = \sigma\!\left(\sum_c W_{c,f,a}^{\mathrm{og}}\,
\mathcal N_0(h_i'|_{l=0})_{f,c}\right)\in(0,1)
$$

**卷积闭式**（式 30）：

$$
(\mathcal C_\Theta h)_i = L_{\mathrm{post}}^{\mathrm{deg}}\!
\left[\mathrm{concat}_{f,a}\,\tilde A_i^{(f,a)}\right]
\tag{30}
$$

$L\_{\mathrm{post}}^{\mathrm{deg}}$ **零初始化** → 训练起始 $\mathcal C\_\Theta\approx 0$，稳定深网。

---

## 八、等变 FFN 与 Lebedev 球面 SwiGLU（§4.2.4，A4）

### 8.1 残差 FFN（式 31）

$$
u_i = L_{\mathrm{ch}}^{\mathrm{in}}\, h_i,\quad
\mathcal F_\Theta^{\mathrm{FFN}}(h_i) = L_{\mathrm{ch}}^{\mathrm{out}}\!
\bigl[\mathcal S_{\mathrm{grid}}(u_i) + \mathcal S_{\mathrm{scalar}}(h_i|_{l=0})\bigr]
\tag{31}
$$

$$
h_i \leftarrow h_i + \mathcal F_\Theta^{\mathrm{FFN}}(h_i)
$$

$L\_{\mathrm{ch}}^{\mathrm{out}}$ 零初始化 → 初始为恒等残差。

### 8.2 Lebedev 求积与离散正交（式 32）

取代数精度 $p\ge 2L$ 的 Lebedev 节点 $\{q\_a,w\_a\}\_{a=1}^A$，$\sum\_a w\_a=1$，满足：对次数 $\le p$ 的球面多项式，离散平均等于连续球平均。

对实球谐（norm 约定）：

$$
\sum_{a=1}^{A} w_a\, Y_l^m(q_a)\, Y_{l'}^{m'}(q_a)
= \delta_{ll'}\delta_{mm'}\,\frac{1}{2l+1},\quad 0\le l,l'\le L
\tag{32}
$$

取 $p=2L$ 时节点数 $A$ 最小。相对 **经纬度乘积网格**（Equiformer 族），同精度下 $A$ 更小（$L=7$：170 vs 576，Table 3），且 fp64 下 **数值等变误差** 可压至 $\sim 10^{-14}$。

### 8.3 系数 ↔ 网格（式 33–35）

**正投影**（式 33 / 35）：

$$
U_{a,c} = \sum_{l=0}^{L}\sum_{m=-l}^{l} Y_l^m(q_a)\, u_{(l,m),c}
\tag{33}
$$

**逐点 SwiGLU**（式 34–36）：

$$
\mathrm{SwiGLU}(z) = \sigma(z_{\mathrm{gate}})\odot z_{\mathrm{gate}} \odot z_{\mathrm{val}},\quad
z=(z_{\mathrm{gate}},z_{\mathrm{val}})\in\mathbb R^{2H_{\mathrm{FFN}}}
\tag{34}
$$

$$
V_{a,:} = W_2\,\mathrm{SwiGLU}\!\bigl(W_1\, U_{a,:}\bigr),\quad
W_1\in\mathbb R^{2H_{\mathrm{FFN}}\times H_{\mathrm{FFN}}},\;
W_2\in\mathbb R^{H_{\mathrm{FFN}}\times H_{\mathrm{FFN}}}
\tag{36}
$$

**逆投影**（式 37）：

$$
\bigl[\mathcal S_{\mathrm{grid}}(u)\bigr]_{(l,m),c}
= (2l+1)\sum_{a=1}^{A} w_a\, Y_l^m(q_a)\, V_{a,c}
\tag{37}
$$

在 $V\_{\le L}$ 上，式 (33) 与 (37) 由 (32) **互逆**（$p\ge 2L$ 时精确）。

**等变证明梗概**：$u$ 定义球面函数 $U(\hat{\mathbf n})=\sum\_{l,m,c} u\_{(l,m),c} Y\_l^m(\hat{\mathbf n})$;旋转下 $U(\hat{\mathbf n})\mapsto U(R^{-1}\hat{\mathbf n})$;网格值 $U\_a=U(q\_a)$ 随之协变;逐点 SwiGLU 与 $\hat{\mathbf n}\mapsto R^{-1}\hat{\mathbf n}$ 可交换;逆变换线性且精确 → 回到系数仍等变。

### 8.4 辅助标量分支（式 38）

$$
\mathcal S_{\mathrm{scalar}}(h_i|_{l=0}) = \mathrm{SwiGLU}(W_3\, h_i|_{l=0})\in\mathbb R^{H_{\mathrm{FFN}}},\quad
W_3\in\mathbb R^{2H_{\mathrm{FFN}}\times C}
\tag{38}
$$

只作用 $l=0$，输出 **加回** $\mathcal S\_{\mathrm{grid}}$ 的 $l=0$ 槽，给模型一条 **纯不变** 非线性捷径。

![表 3：随机 SO(3) 旋转下全系数 $S^2$ 激活的等变误差;Lebedev 相对乘积网格在 fp64 上低 6–8 个数量级。](/img/posts/2026-06-02-dpa4-emfa-so2-convolution/table03-s2-activation-lebedev-equivariance.png)

---

## 九、等变线性层、RMSNorm 与 SO(2) 栈内层（式 60–67，补充 S-1.5）

Schur 引理给出 DPA4 所用线性算子的 **唯一形式**（式 60–61）：全局 SO(3) 下 **禁止混 $l$**，边局域 SO(2) 下可在 **固定 $\lvert m \rvert$** 内混不同 $l$，且 $m=0$ 为平凡表示、$\lvert m \rvert>0$ 为复二维对。

### 9.1 度分辨通道映射（式 62）

正文所有 $L^{\mathrm{deg}}\_{\Theta}$（含 $L^{\mathrm{pre}}\_{\mathrm{deg}},L^{\mathrm{post}}\_{\mathrm{deg}},L^{\mathrm{rad}}\_{\mathrm{lift}},L^{\mathrm{ch}}\_{\mathrm{in/out}}$）均为：

$$
\bigl(L^{\mathrm{deg}}_{\Theta} h\bigr)_{\iota(l,m),c'} = \sum_{c=1}^{C} W^{(l)}_{c,c'}\, h_{\iota(l,m),c}
\tag{62}
$$

每个角动量阶 $l$ 一张通道矩阵 $W^{(l)}\in\mathbb R^{C\times C'}$，对 $m\in\{-l,\ldots,l\}$ **相同** → 不混 $l$，保持 SO(3) 等变。

### 9.2 边局域 SO(2) 线性（式 63–64）

$\lvert m \rvert=0$（标量线，可含 $l=0$ 偏置 $b\_{0,c'}$）：

$$
\bigl(L^{\mathrm{SO(2)}}_{\Theta} x\bigr)_{(l,0),c'} = \sum_{l'=0}^{L}\sum_{c=1}^{C} A^{(l,l',0)}_{c,c'}\, x_{(l',0),c} + b_{0,c'}\,\delta_{l,0}
\tag{63}
$$

$\lvert m \rvert>0$（用 $(x\_{(l,-m),c},x\_{(l,+m),c})^{\mathsf T}$ 的 2D 实表示实现复乘法）：

$$
\begin{pmatrix}
(L^{\mathrm{SO(2)}}_{\Theta} x)_{(l,-m),c'} \\
(L^{\mathrm{SO(2)}}_{\Theta} x)_{(l,+m),c'}
\end{pmatrix}
= \sum_{l'\ge m}\sum_{c=1}^{C}
\begin{pmatrix}
U^{(l,l',m)}_{c,c'} & -V^{(l,l',m)}_{c,c'} \\
V^{(l,l',m)}_{c,c'} & U^{(l,l',m)}_{c,c'}
\end{pmatrix}
\begin{pmatrix}
x_{(l',-m),c} \\
x_{(l',+m),c}
\end{pmatrix}
\tag{64}
$$

式 (21) 中每层 $L\_s^{\mathrm{SO(2)}}$ 即此形式；**不混不同 $\lvert m \rvert$** 是 SO(2) 等变的核心约束。

### 9.3 等变 RMSNorm（式 65–66）

先算 **旋转不变** 的 per-atom 方差（$l=0$ 均值 $\bar h$ 只用于中心化 $l=0$）：

$$
\sigma^2(h) = \frac{1}{(L+1)C}\sum_{l=0}^{L}\sum_{m=-l}^{l}\sum_{c=1}^{C}
\bigl(h_{\iota(l,m),c} - \delta_{l,0}\bar h\bigr)^2,\quad
\bar h = C^{-1}\sum_c h_{\iota(0,0),c}
\tag{65}
$$

归一化（$\gamma\_l,\beta\_c$ 为可学习标量/向量）：

$$
(\mathcal N_{\gamma,\beta} h)_{\iota(l,m),c} = \gamma_l\,
\frac{h_{\iota(l,m),c}-\delta_{l,0}\bar h}{\sqrt{\sigma^2(h)+\varepsilon}} + \delta_{l,0}\beta_c
\tag{66}
$$

分子等变、分母不变 → 与 $D(Q)$ 对易。

### 9.4 标量门控非线性（式 67）

$$
(\Gamma_{\psi,G} h)_{\iota(l,m),c} =
\begin{cases}
\psi(h_{\iota(0,0),c}), & l=0, \\[4pt]
h_{\iota(l,m),c}\,\sigma\!\displaystyle\left(\sum_{c'} G^{(l)}_{c',c}\, h_{\iota(0,0),c'}\right), & l\ge 1.
\end{cases}
\tag{67}
$$

门只读 **不变** 的 $l=0$ 切片 → $\Phi\_s$ 在式 (21) 中保持等变。EMFA 第一层 $l=0$ 偏置还乘 $\tilde\rho\_{ij,0,c}\,s\_5(r\_{ij})$（式 69）以与 cutoff 同阶消失，避免「零距离边仍留常数偏移」。

以上组件保证 SO(2) 栈 **既深又严格等变**。

---

## 十、训练目标与编译保守梯度（§4.3–4.4，式 43–44）

### 10.1 损失函数（式 43）

DPA4 预测 $E\_\Theta$，力 $\mathbf F\_\Theta = -\nabla\_R E\_\Theta$，维里 $\Pi\_\Theta$ 由应变导数得到。mini-batch 损失：

$$
\mathcal L = \lambda_E \frac{1}{B}\sum_{b=1}^{B}\frac{|E_{\Theta,b}-E_b|}{N_b}
+ \lambda_F \frac{1}{\sum_b N_b}\sum_{b,i}\|\mathbf F_{\Theta,bi}-\mathbf F_{b,i}\|_2
+ \lambda_\Pi \frac{1}{B}\sum_{b=1}^{B}\frac{\|\Pi_{\Theta,b}-\Pi_b\|_1}{9 N_b}
\tag{43}
$$

能量、维里为 **按原子数归一的 MAE**;力为各原子力向量 $\ell\_2$ 残差平均。Matbench / SPICE 实验用 **EFSG**：力、应力均由能量梯度给出，非独立头直接回归。

### 10.2 为何力训练需要「二阶图」（式 44）

力项 $\|\mathbf F\_\Theta - \mathbf F\|^2$ 对参数 $\Theta$ 的梯度含

$$
\frac{\partial \mathcal L}{\partial \Theta} \supset
\frac{\partial^2 E_\Theta}{\partial R\,\partial \Theta}
\tag{44}
$$

即 **坐标–参数混合 Hessian 项**。标准 LLM 训练栈针对单次反传优化，对 MLIP 往往需 `make_fx` 先建 **力=能量梯度的内层图**，再对外层力残差反传;DPA4 用 **PyTorch Inductor + torch.compile** 编译该路径，bf16 混合精度下墙钟约 **3.1×** 加速、显存约 **40%**（Table S-2），且 **不改变** 保守力定义。

**优化器**：HybridMuon（矩阵块用 Muon、标量/Norm 用 Adam）;对度分辨线性层 **slice-mode Muon** 保持 $(2l+1)$ 块结构，避免展平破坏表示。

---

## 十一、正文主要公式覆盖清单

| 式号 | 内容 | 上文节 |
|------|------|--------|
| (1) | 总能量 NN+ZBL | §3 |
| (2)–(5) | Hermite、$\tilde r$、$w$、$\eta\_j$ | §4 |
| (6)–(15) | GIE 四向量、包络、$A\_i$、FiLM、球谐嵌入 | §6 |
| (10)–(12) | $D\_i$、FiLM $\gamma,\beta$ | §6 |
| (16)–(30) | EMFA 全流程 | §7 |
| (18)–(19) | A1 低秩乘积 | §7.2 |
| (20)–(23) | A2 Multi-Focus | §7.3 |
| (25)–(27) | A3 注意力 | §7.5 |
| (31)–(38) | FFN + Lebedev + SwiGLU | §8 |
| (39)–(42) | ZBL、拼接力对比 | §4 |
| (43)–(44) | 训练损失、混合导数 | §10 |
| (45)–(47) | Wigner、打包索引 | §3 |
| (57)–(59) | $m$ 截断投影、$\Xi\_M$ 范数补偿 | §3 |
| (60)–(61) | Schur 引理算子分类 | §9 |
| (62)–(67) | 度分辨线性、SO(2) 层、RMSNorm、门控 | §9 |
| (69) | cutoff 一致的 $l=0$ 偏置 | §9 |

补充材料另有更大超参表、HybridMuon (74–81) 等;正文 **Section 4.2 核心架构式 (1)–(44) 与 S-1 算子式 (45)–(47)、(57)–(67)、(69)** 均已写入。

---

## 十二、对称性小结（§4.2.6）

| 性质 | 机制 |
|------|------|
| 平移不变 | 仅 $\mathbf r\_{ij}=\mathbf R\_j-\mathbf R\_i$ |
| 置换不变 | 共享 $\Theta$;对同类原子求和/归一化注意力 |
| 旋转不变（$E$） | 中间层 SO(3) 等变;$\varepsilon\_i$ 只读 $l=0$;ZBL 只依赖 $r\_{ij}$ |
| $C^3$ 光滑 | $s\_5$、$\tilde r$、$\eta\_j$、softplus 分母;Hermite 桥接 |
| 保守力/应力 | 单一 $E$ 自动微分 |
| 周期边界 | 晶胞应变进入坐标导数链 |

---

## 十三、边局域旋转的构造（补充 S-1.2，式 48–50）

式 (16) 要求 $R\_{ij}\hat{\mathbf r}\_{ij}=\mathbf e\_z$。在球面 $S^2$ 上需 **光滑** 地选 $R\_{ij}$，避免单一欧拉角在 $\hat{\mathbf r}\parallel\pm\mathbf e\_z$ 处奇异。DPA4 用 **双四元数图** 混合：

$$
\mathbf q_+(\hat{\mathbf r}) = \frac{(1+z,\ y,\ -x,\ 0)^{\mathsf T}}{\sqrt{2(1+z)}},\qquad
\mathbf q_-(\hat{\mathbf r}) = \frac{(-x,\ 0,\ 1-z,\ y)^{\mathsf T}}{\sqrt{2(1-z)}}
\tag{48}
$$

$\mathbf q\_+$ 在南极奇异、$\mathbf q\_-$ 在北极奇异；重叠区取 $\langle\mathbf q\_+,\mathbf q\_-\rangle\ge 0$ 后混合：

$$
\mathbf q_{ij} = \frac{\lambda\,\mathbf q_+(\hat{\mathbf r}_{ij}) + (1-\lambda)\,\mathbf q_-(\hat{\mathbf r}_{ij})}
{\bigl\|\lambda\,\mathbf q_+ + (1-\lambda)\,\mathbf q_-\bigr\|},\qquad
\lambda = \frac{1+z}{2}
\tag{50}
$$

由 $\mathbf q\_{ij}$ 得 $R\_{ij}\in\mathrm{SO(3)}$，再用于式 (17)(24) 的 $D(R\_{ij})$。局部算子与绕 $\mathbf e\_z$ 的 SO(2) 对易，**规范选择不影响** 等变消息。

---

## 十四、Matbench Discovery 无机晶体 benchmark

**Matbench Discovery**（compliant 设定）：在 **MPtrj** 上训练，对 WBM 候选结构做弛豫，用形成能/凸包距离等组合指标；并报告 **κSRME**（与势光滑性、保守性相关的热导相关指标）。

![表 1：Matbench Discovery 排行榜节选（DPA4 与主要 baseline）。DPA4 全系为 EFSG（保守能量梯度力/应力）。](/img/posts/2026-06-02-dpa4-emfa-so2-convolution/table01-matbench-discovery-leaderboard.png)

### 14.1 核心数字（Table 1 + Fig. 1）

| 模型 | CPS↑ | F1↑ | κSRME↓ | 参数量 | 训练成本（A100·day） |
|------|------|-----|--------|--------|----------------------|
| **DPA4-Pro** | **0.833** | 0.859 | **0.255** | 20.91M | ~100 |
| EquiformerV3+DeNS-MP | 0.830 | **0.863** | 0.275 | 30.3M | 更高 |
| **DPA4-Air** | 0.804 | 0.828 | 0.302 | **2.76M** | **7.8** |
| eSEN-30M-MP | 0.797 | 0.831 | 0.340 | 30.1M | ~335 |

要点：

1. **DPA4-Pro** CPS **0.833**，为 compliant 榜 **最高**；F1 略低于 EqV3+DeNS，但 **参数量少约 31%**，且 **不用 DeNS / 直接力预训练**，全程保守能量梯度训练。  
2. **DPA4-Air**（2.76M）CPS **0.804**，高于 **eSEN-30M-MP**（0.797，30.1M），参数量 **约 1/11**；训练算力 **42.9×** 更少（7.8 vs 335 A100·day）。  
3. **DPA4-Plus**（5.40M）CPS 0.822；**DPA4-Neo**（1.60M）CPS 0.781，仍可比肩 10.4M 的 MatRIS-10M-MP。  
4. 作者指出：头部模型 CPS 差距已很小，**固定 MPtrj 榜的边际优化价值下降**，未来更宜比 **非 compliant 设定、更大数据集、多任务 LAM 预训练**。

---

## 十五、SPICE-MACE-OFF 有机分子 benchmark

**SPICE-MACE-OFF**：PubChem、DES370K 单体/二聚体、二肽、溶剂化氨基酸、水团簇、QMugs 等；参考 **ωB97M-D3(BJ)/def2-TZVPPD**；与 MACE-OFF、DPA3 相同划分。

![表 2：SPICE-MACE-OFF 各子集能量/力 MAE（meV/atom、meV/Å）及 LWAMAE 几何平均。](/img/posts/2026-06-02-dpa4-emfa-so2-convolution/table02-spice-mace-off-lwamae.png)

| 模型 | 参数量 | LWAMAE E | LWAMAE F | A100 训练（day） |
|------|--------|----------|----------|------------------|
| eSEN 6.5M | 6.5M | 0.14 | 2.58 | / |
| **DPA4-Plus** | 5.4M | **0.10** | **1.82** | 8 |
| **DPA4-Air** | 2.7M | 0.13 | 2.45 | 4 |
| DPA3-L24 | 4.9M | 0.22 | 5.78 | 288 |
| MACE(L) | 6.9M | 0.65 | 11.66 | 14 |

- **DPA4-Plus** 相对 6.5M eSEN：聚合能量/力误差各降 **29% / 30%**；相对 DPA3-L24 降 **55% / 69%**。  
- **DPA4-Air** 仍优于 6.5M eSEN，且训练仅 **4 A100·day**（DPA3-L24 为 288）。  

说明 DPA4 并非只擅长无机晶体，**有机分子力场** 同样刷新精度–参数前沿。

---

## 十六、训练与推理效率

### 16.1 训练：保守路径 + torch.compile

- 对照实验：**torch.compile + bf16 AMP** → 墙钟 **约 3.1×** 加速，峰值显存约降至 FP32 的 **40%**（补充表 S-2）。  
- 无需用直接力损失替代能量匹配，即可在工程上「训得动」高表达等变模型。

### 16.2 推理：LAMBench ASE 吞吐

在 **NVIDIA H20** 上经 ASE calculator 测 **端到端** 吞吐（含邻域表、几何预处理）：

![图 3：LAMBench inorganic_500 体系规模–吞吐；DPA4-Air/Neo 在小体系上可超过 cuEquivariance 优化后的 MACE-OPT；大体系因 O(N²) 邻域表瓶颈曲线下探。](/img/posts/2026-06-02-dpa4-emfa-so2-convolution/fig03-ase-inference-throughput-lambench.png)

- **DPA4-Air / Neo** 显著快于 DPA3-L16/L24；小原子数时甚至高于 **MACE-Omat OPT**。  
- **DPA4-Pro** 快于 EquiformerV3，且 Matbench CPS 更高。  
- 超大体系吞吐下降部分来自 DeePMD-kit **朴素全对邻域表** $O(N^2)$，属实现瓶颈而非架构上限。

---

## 十七、Native ZBL 数值验证（C–Si 二聚体）

上文第四至十节已给出 $\tilde r,\eta\_j$ 与 $E\_{\mathrm{ZBL}}$ 的公式；此处为 **3C-SiC / ABACUS** 训练后的 **C–Si 二聚体扫描**（亚 Å 区）实验对照：

![图 4：二聚体能量/力随间距；DPA3+外接 DP-ZBL 在 ~1.1 Å 力曲线尖峰；DPA4 Native ZBL 与解析 ZBL 力光滑一致。](/img/posts/2026-06-02-dpa4-emfa-so2-convolution/fig04-c-si-dimer-native-zbl-bridging.png)

- **DPA3 + DP-ZBL 外挂修正**：切换区出现 **非物理吸引力尖峰**（来自拼接窗内 ZBL 与学习能量不匹配）。  
- **DPA4**：内区跟随解析 ZBL 排斥，外区平滑接到学习力。  

注意：二聚体扫描是 **局域探针**；长时 MD 能量漂移、碰撞稳定性需另做多样本测试（原文 Discussion 承认）。

---

## 十八、机制消融（Section 2.6 摘要）

在 WBM 测试子集上 **单因素** 变动，确认增益来自目标设计而非偶然调参：

| 机制 | 结论（定性） |
|------|----------------|
| **A3 注意力聚合** | 相对 envelope 求和，两 focus 配置下力 MAE 约降 **6.5%**，训练成本略增 |
| **A1 低秩边–节点乘积** | 相对仅用 $l=0$ 边特征，精度提升、成本适中 |
| **A2 Multi-Focus** | 固定宽度下 **减参** 且精度优于单 focus |
| **A4 Lebedev** | 主要消除非线性分支 **数值对称破缺**；WBM MAE 变化 modest |

完整 sweep 见补充材料 Section S-3。

---

## 十九、讨论与局限

### 19.1 作者归纳的贡献

- **架构 + 训练协同设计**：EMFA SO(2) 避开全 SO(3) CG 成本又比旧 SO(2) 简化更表达；Lebedev 非线性 **机器精度等变**；**compile 友好保守训练** 使大模型可实用训练/消融。  
- **单任务、单数据集** 已刷新 Matbench 与 SPICE 两条前沿；**下一步是自然做 LAM 多任务预训练**（Section 3）。  
- **Native ZBL** 改善极近距力行为，利于高能碰撞/辐照等稀疏采样区。

### 19.2 局限

- 本文 **DPA4 各变体均为 per-dataset 单任务训练**，尚未展示跨任务 LAM 预训练与下游微调。  
- Matbench compliant 榜顶部差距很小，**继续刷榜收益递减**。  
- 推理在大体系上仍受 **邻域表实现** 限制；SO(2) 专用 kernel（类比 MACE 的 cuEquivariance）尚未发布。  
- ZBL 二聚体实验 **不替代** 长程 MD 稳定性验证。

---

## 二十、方法摘要

| 项目 | 无机（Matbench） | 有机（SPICE） |
|------|------------------|---------------|
| 训练数据 | MPtrj（GGA/GGA+U，89 元素） | SPICE-MACE-OFF 划分 |
| 评估 | WBM 弛豫 + CPS/κSRME 等 | 子集 E/F MAE + LWAMAE |
| 变体 | Neo 1.6M / Air 2.76M / Plus 5.4M / Pro 20.91M | Air、Plus 等 |
| 力/应力 | **EFSG** 保守梯度 | 同左 |
| 系统 | torch.compile + bf16；DeePMD-kit 生态 | 同左 |

---

## 二十一、与 SOG / 长程 DPA 路线的关联

若你关心 **DPA 家族能否接长程静电（SOG、多极、k 空间）**，本站已有：

- [DPA4 短程有多强？SOG 长程能否接棒 MACE-POLAR-1？](/2026/05/23/DPA4-SOG-长程能否接棒MACE-POLAR-1/)
- [Kim 2026：可极化多极矩长程电静学](/2026/05/16/Kim2026-可极化多极矩长程电静学/)

DPA4 本文聚焦 **短程等变消息传递 + 解析 ZBL + 训练/推理系统工程**；**不显式包含** SOG 式长程 Coulomb/色散。将 DPA4 骨干与长程模块耦合做 LAM，是合理的下一步（与 Jiang LAM-DPA、Gao PQEq 等路线并列但分工不同）。

---

## 二十二、一句话总结

**Li 等 2026 的 DPA4** 用 **EMFA SO(2) 卷积（低秩边–节点乘积 + 多 focus + 门控注意力）** 和 **Lebedev 等变非线性**，在 **Matbench Discovery** 与 **SPICE-MACE-OFF** 上同时推到新的 **精度–参数–训练成本** 帕累托前沿；**DPA4-Air** 以不到 **3M 参数、约 8 A100·day** 超过 30M 级 eSEN；**保守能量梯度 + torch.compile** 约 **3×** 训练加速；**Native ZBL** 消除近距力伪影——为下一代 **LAM 预训练骨干** 提供了强候选。

---

## 参考文献（精选）

1. Li T. et al., *DPA4: Pushing the Accuracy–Cost Frontier of Interatomic Potentials with EMFA SO(2) Convolution*, arXiv:2606.02419 (2026).  
2. Riebesell J. et al., Matbench Discovery, *npj Comput. Mater.* (2024).  
3. Batatia I. et al., MACE / MACE-OFF, *Nat. Commun.* / SPICE 相关文献.  
4. Fu X. et al., eSEN, UMA 架构与训练成本公开信息。  
5. Deng B. et al., CHGNet & MPtrj, *Nat. Mach. Intell.* (2023).
