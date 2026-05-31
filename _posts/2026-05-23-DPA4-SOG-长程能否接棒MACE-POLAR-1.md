---
layout:     post
title:      DPA4 短程有多强？SOG 长程能否接棒 MACE-POLAR-1？
subtitle:   DPA4（SeZM）短程强项、Kim 框架下 SOG 相对 LES 的长程优势、与 MACE-POLAR-1 的分场景边界，以及 DPA4+SOG 组合展望
date:       2026-05-23
author:     qqz
header-img: img/post-bg-desk.jpg
catalog:    true
tags:
    - MLIP
    - DPA4
    - SOG
    - 长程电静学
    - 势函数
---

![一图总结：DPA4 短程 × SOG 长程](/img/posts/2026-05-23-dpa4-sog-longrange/dpa4-sog-longrange-onepage.png)

> 背景串联：Kim et al. 2026 半局域可极化多极框架（[arXiv:2605.05746](https://arxiv.org/abs/2605.05746)）· DeepMD **DPA4/SeZM**（[dpa4 文档](https://github.com/deepmodeling/deepmd-kit)）· **MACE-POLAR-1**（arXiv:2602.19411）· 本组 **CACE-SOG** 长程核 · **SOG 开源实现**（[GitHub: Dazzlemoon/sog](https://github.com/Dazzlemoon/sog)）

如果你已经在用 **CACE-LES / CACE-SOG**，大概已经认同一件事：

> 短程 MLIP 负责「成键与局域量子效应」，长程头负责「库仑 + 极化 + 电响应」。

最近两条新闻线又把这个拼图补全了两块：

- **DPA4** 在 Matbench Discovery、SPICE-MACE-OFF 等基准上刷新 SOTA，但**本身不带显式长程静电**；
- **MACE-POLAR-1** 则是「MACE 短程 + 完整 GTO 长程 electrostatics foundation model」。

而 **SOG 核的设计初衷**，正是 Kim 框架里「可插拔长程」——**不限于 CACE**，理论上任何短程 backbone 都能接。

这篇稿子回答四个问题：

1. DPA4 在**短程**上到底改进了什么？  
2. SOG 相对 LES 的**长程**优势在哪？  
3. 组合起来会不会比 MACE-POLAR-1 更强？  
4. SOG 还能怎么改？**SOG + DPA4** 值得做吗？

---

## 01. 先对齐概念：三种「长程」不是一回事

| | 显式 $1/r$ 静电 | 典型代表 |
|---|:---:|---|
| **DPA4** | ✗ | 只在 `rcut`（~6 Å）内消息传递；可选 **ZBL** 管极短程核排斥 |
| **MACE-POLAR-1** | ✓ | GTO 多极 + Ewald/k 空间 + 2 步可极化场迭代 |
| **CACE-LES / CACE-SOG** | ✓ | Kim 多极 + Ewald；SOG 用**可学习高斯和**替代固定 erf 核 |

一句话：

- **DPA4** 把算力花在「短程算得又快又准」；  
- **MACE-POLAR-1 / SOG** 把算力花在「物理长程 + 电响应可解释」。

所以后面谈「谁更强」，必须先问：**比的是 PES 力误差、训练成本，还是 BEC/IR/Raman？**

---

## 02. DPA4 在短程上做了什么？（SeZM 核心）

DPA4 在代码里叫 **SeZM**（Smooth Equivariant **Zone-bridging** Model）。它**没有**接 Ewald，但在 **$r_c$ 以内** 做了一系列很值得 SOG 用户关注的工程与算法升级。

### 2.1 边局部 SO(2) 等变：便宜地保持旋转对称

传统 SO(3) 等变网络靠 Clebsch–Gordan 张量积，角向代价随 $\ell_{\max}$ 涨得很快。

DPA4 的做法：对每条边 $i\to j$ 建**边对齐局部坐标系**，把绕边轴的旋转约化成 **SO(2)**，用块对角线性层完成消息传递，再旋回全局坐标。

直觉：**完整 3D 旋转等变还在，但最贵的角向耦合被「降维」了。**

### 2.2 Envelope-gated Attention：邻居不是一视同仁

DPA4 可选用**截断包络参与 softmax 分子和分母**的注意力聚合邻居信息——cutoff 附近的边在分子、分母里一起衰减，力更光滑。

这和 SOG/LES 强调的 **$C^3$ 光滑截断** 是同一类 MD 工程哲学。

### 2.3 Zone Bridging + ZBL：极短程交给解析式

可选 **ZBL**（Ziegler–Biersack–Littmark）处理 $r\lesssim1$ Å 的**核–核排斥**，并通过距离钳制 + 源门控，防止神经网络在「该由 ZBL 管」的区域抢力。

注意：**ZBL ≠ 长程静电**，它不管 10 Å 外的库仑。

### 2.4 原生 torch.compile：力损失训练真正加速

力损失要算 $\partial^2 E/(\partial \mathbf{r}\,\partial\theta)$。DPA4 用 `make_fx` + Inductor 把**含二阶导的训练图**编译起来——据报道训练效率相对 DPA3 可再提一个数量级，且 Matbench / SPICE 双榜领先。

### 2.5 DPA4 短程小结

| 维度 | DPA4 短程强项 |
|------|----------------|
| 精度–效率 | 参数少、单卡训练成本低，中性/弱极性 PES 极强 |
| 对称性 | SO(3) 等变 + 保守力 |
| MD 可用性 | $C^3$ cutoff、可选 ZBL |
| 刻意不做 | 显式库仑、可变总电荷/自旋、BEC 光谱头 |

**对 SOG 用户的启示：** DPA4 是「短程 backbone 天花板」候选之一——**长程仍应交给 SOG**，而不是指望 DPA4 自己学出 $1/r$。

---

## 03. SOG 长程 vs LES：优势在哪？

Kim 2026 的默认长程是 **LES**：屏蔽 Coulomb 核

$$
\phi(r)=\frac{\mathrm{erf}(r/\sqrt{2\sigma})}{r},
$$

短程–长程分裂清晰，+$U^{\mathrm{sr}}$ 学剩余成键，+$U^{\mathrm{elec}}$ 用 Ewald 算多极静电。

**CACE-SOG** 把长程核换成 **Sum-of-Gaussians（SOG）**：

$$
K(r)=\sum_{\ell=0}^{M-1}\omega_\ell \exp\!\left(-\frac{r^2}{s_\ell^2}\right),
\qquad \omega_\ell,\, s_\ell \ \text{可训练}.
$$

相对 **CACE-LES（erf）**，SOG 在本组实现里的优势可以概括为：

### 3.1 核形状可学习，不锁死在 erf

LES/erf 的分裂宽度 $\sigma$ 固定或弱可调；SOG 的 **12 项（典型）幅值与宽度** 可在训练中逼近体系真实的有效 $1/r$ 与介电屏蔽，对 **周期固体（如 MAPbI$_3$）** 更灵活。

### 3.2 BSA 初值 + 实/倒空间对齐

采用 **双边级数近似（BSA）** 初始化 $\omega_\ell \propto b^{-\ell}$，并修正倒空间乘子

$$
\widetilde K(k)\ \propto\ \sum_\ell \pi^{3/2}\, s_\ell^3\, \omega_\ell\, e^{-s_\ell^2 k^2/4},
$$

使**同一组** `amp`/`bandwidth` 在实空间与 FFT/Ewald 路径自洽——周期体系训练时少踩「实空间一套、倒空间另一套」的坑。

### 3.3 与 Kim 框架即插即用

SOG 通过 `LesWrapper` / `sog.Sog` 接入，**多极层次（`-u`、`-Q`）、诱导电荷（`-iq`）、诱导偶极（`-iu`）** 与 LES 版一致；换核不换物理分解：

$$
U = U^{\mathrm{sr}} + U^{\mathrm{elec(SOG)}} + \sum_i\big(U_i^{\mathrm{iq}}+U_i^{\mathrm{iu}}\big).
$$

### 3.4 工程可控性

- **开源代码库**：[github.com/Dazzlemoon/sog](https://github.com/Dazzlemoon/sog)（BSA 初值、Kim 多极接口、文档与验证脚本）  
- `trainable_kernel=False` 可冻结 SOG 核做 ablation；  
- `r_cut` 与 CACE `rcut` 可对齐；  
- 已有 MAPbI$_3$ / 水体系 **BEC、能量–电荷** 等评测 pipeline。

### 3.5 SOG vs LES 一句话

> **LES** = 物理先验强、实现简单、Kim 论文主 baseline；  
> **SOG** = 在同一 Kim 物理框架下，**把长程核本身也变成可学习自由度**，并用 BSA 保证周期一致性。

---

## 04. 那会不会比 MACE-POLAR-1 更强？

**诚实答案：分场景，没有 universal winner。**

### 4.1 MACE-POLAR-1 的护城河

| 能力 | MACE-POLAR-1 |
|------|----------------|
| 预训练 | OMol25 1 亿结构，分子 foundation |
| 长程 | 自旋 GTO 多极 + **2 步 NSC 场迭代** + Fukui $(Q,S)$ 均衡 |
| 任务 | 非共价、蛋白–配体、分子晶体 lattice energy、氧化还原 |
| 输出 | 电荷/自旋密度、dipole、$E_{\mathrm{elec}}$ 等可解释量 |

它是**端到端 electrostatics foundation model**，不是「短程 + 插件」。

### 4.2 SOG + 短程（如 CACE-SOG）的护城河

| 能力 | Kim + SOG |
|------|-----------|
| 模块化 | **任意**短程（CACE / MACE / NequIP / Allegro / **未来 DPA4**） |
| 长程 | Ewald + **线性一次** induced（$\Delta q=-\kappa\Phi$，$\Delta u=\alpha E$） |
| 电响应 | BEC、极化率、IR/Raman——Kim 论文主战场（水、MAPbI$_3$） |
| 训练标签 | 仅 E/F（+ 可选 stress），无需 MLWC/DFT 电荷监督 |

### 4.3 谁更可能赢？

| 场景 | 更可能占优 |
|------|------------|
| 有机分子、蛋白配体、可变 $Q/S$ 开壳化学 | **MACE-POLAR-1**（专门预训练 + Fukui + 自旋通道） |
| 离子/极性固体、PERovskite、Bulk 水、BEC/光谱 | **SOG + 合适短程** 很有竞争力（Kim 已系统验证） |
| 大规模中性材料 MD、Matbench、训练预算极紧 | **DPA4 短程 alone**；要电响应再 **+ SOG** |
| 「我要换短程 backbone 做 ablation」 | **SOG**（MACE-POLAR 短程与长程绑死） |

所以：**SOG 不会在「分子 foundation 开箱即用」上自动碾压 MACE-POLAR-1**；但在 **Kim 框架已验证的固体/液体电响应** 上，**SOG 的灵活核 + 模块化** 可以做到**同等物理透明度、甚至更贴周期体系**——前提是 short-range 足够强（这正是 DPA4 可以补上的那块）。

---

## 05. SOG 还可以怎么改？

结合 Kim 2026、MACE-POLAR-1 与本组 BSA 实现，下面几条是**性价比高**的改进方向：

### 5.1 多极 Ewald 与 SOG 核全面耦合

**现状（详见 [SOG 多极 k 空间实现说明](https://github.com/Dazzlemoon/sog)）：** `sog` 包在 `compute_multipole_bundle` 中**已实现** Kim 式 (15) 的 $S(\mathbf{k})=S_q+S_u+S_Q$ 与 SOG $\widetilde K(k)$ 的 $|S|^2$ 能量；但 BSA/`kfac` 验证主要针对 monopole 的 $4\pi/k^2$，且当前 `cacesog-uiu` 未接 `kappa_key`/`quad_key`，实际训练以 q+u+iu 为主。

**待做：** 接线 `-uQ`/`-iq`；训练时分解 $|S_q|^2,|S_u|^2$ 贡献；可选阶次分辨 $\widetilde K^{(\ell)}(k)$，避免核参数仅被 monopole 梯度塑造。

### 5.2 轻量 NSC（可选 1–2 步）

Kim 框架默认**一次** induced；MACE-POLAR 用 **2 步**场迭代。可在 `-iu` 之上加可选迭代：**$\Phi,\mathbf{E}$ 含 $\Delta q$ 反馈**，作为 `-iu` 与 full SCF 之间的折中。

### 5.3 总电荷 / 自旋约束

对带电缺陷、离子团簇，引入 **Fukui 式均衡** 或全局 $Q$ embedding（借 MACE-POLAR 思路），补 Kim 框架 tinfoil 中性背景之外的场景。

### 5.4 与短程 cutoff 的联合光滑性

统一 SOG `r_cut`、BSA u-series 过渡区与短程 $C^3$ envelope（DPA4 已示范），减少 **力在 $r_c$ 处的不连续**。

### 5.5 训练栈加速

借鉴 DPA4 **`torch.compile` + 力损失二阶导** 路径，对「短程 + SOG FFT 长程」做分段编译；长程 FFT 部分可能需要 graph break，但短程往往才是训练瓶颈。

### 5.6 预训练与多任务

SOG 头参数少，适合 **「DPA4/MACE 短程预训练 + SOG 头微调」** 两阶段——类似 Kim 论文「任意 backbone + LES」，但 backbone 换成更强的 DPA4。

---

## 06. SOG 接入 DPA4：会有什么提升？

目前 **DPA4 官方未集成 LES/SOG**；下面是**合理预期**（需实验验证，不是已发表结论）：

### 6.1 架构上怎么接

沿用 Kim 分解即可：

$$
U = U^{\mathrm{DPA4}}_{\mathrm{sr}} + U^{\mathrm{SOG}}_{\mathrm{elec}} + \sum_i\big(U_i^{\mathrm{iq}}+U_i^{\mathrm{iu}}\big).
$$

- $U^{\mathrm{DPA4}}_{\mathrm{sr}}$：`DescrptSeZM` + `dpa4_ener`（已有 SO(2)、attention、ZBL）  
- $U^{\mathrm{SOG}}_{\mathrm{elec}}$：`sog.Sog`（[GitHub 仓库](https://github.com/Dazzlemoon/sog)）+ 从 DPA4 节点特征 readout 的 $q,u,Q,\kappa,\alpha$  
- DPA4 的 **ZBL** 与 SOG **库仑** 分工明确：ZBL 管 $r\to0$，SOG 管 $r\gtrsim r_c$

### 6.2 预期增益（按体系）

| 体系类型 | 相对「纯 DPA4」 | 相对「CACE-SOG」 |
|----------|-----------------|------------------|
| 中性分子 PES（SPICE 类） | 增益可能**有限**（DPA4 已极强） | 短程或略优或持平，看 `rcut`/训练预算 |
| 离子晶体、铁电、MAPbI$_3$ | **力误差、介电响应** 有望显著下降 | 短程更强 + 同 SOG 长程 → **PES 可能更好** |
| 大规模 MD 吞吐 | compile 短程 + FFT 长程 → 需 profiling | 若短程更快，**总 wall time 可能更低** |
| BEC / IR / Raman | **从几乎不能到能测** | 与 CACE-SOG 同框架，比纯 DPA4 质变 |

### 6.3 相对 MACE-POLAR-1 的定位

**DPA4 + SOG** 若做成，更像：

> **「材料向超强短程（DPA4）+ Kim/SOG 可极化长程」**  

对 **Matbench、无机固体、高通量 MD** 可能更顺手；  
对 **OMol 分子开箱、自旋分辨开壳化学** 仍要补 Fukui/自旋通道，短期难替代 MACE-POLAR-1 foundation。

**最有希望打出差异化的组合：**

- **DPA4-SOG-uiu** 做 MAPbI$_3$ / 氧化物 / 水 → 对标 Kim 的 CACE-SOG，但 **Matbench 级短程 + BSA-SOG 长程**；  
- **DPA4-SOG-iq** 做 ionic cluster /  charged defect → 补 DPA4 无显式 $(Q)$ 的短板。

### 6.4 实施上的主要工作量

1. 在 DeePMD PyTorch 后端为 DPA4 增加 **SOG output module**（可参考 `MACELES` 或 CACE `LesWrapper` 接口）  
2. 从 SeZM **scalar / equivariant** 节点特征 readout 多极与 $\kappa,\alpha$  
3. 统一 **dtype、compile、neighbor list** 与 SOG 周期 FFT 的 batching  
4. 在 MAPbI$_3$、水、Matbench 子集上做 **ablation**：DPA4 vs DPA4-SOG-r vs DPA4-SOG-uiu

---

## 07. 一句话总结

1. **DPA4** 赢在 **短程**：SO(2) 等变、attention、ZBL、compile——**不含**显式长程静电。  
2. **SOG** 赢在 **长程可学习 + 周期实倒一致**，在 Kim 物理框架里 **plug-and-play**，相对 **LES(erf)** 更灵活。  
3. **MACE-POLAR-1** 赢在 **分子 electrostatics foundation**；SOG 不会在全部场景自动更强，但在 **固体电响应 + 模块化** 上有一战之力。  
4. **SOG + DPA4** 是值得做的「**强短程 × 强长程**」组合：理论上有望在 **离子/极性材料** 上同时拿到 DPA4 的效率与 Kim/SOG 的物理长程——**下一步就是工程对接与 benchmark**。

---

## 08. 延伸阅读（站内）

- [Kim 2026：可极化多极矩长程电静学](/2026/05/16/Kim2026-可极化多极矩长程电静学/)
- [CACE-SOG 阶段进展（water / MAPbI₃）](/2026/05/17/CACE-SOG-阶段进展汇报-water-MAPbI3/)
- [SOG 开源仓库](https://github.com/Dazzlemoon/sog)

---

## 参考

- Kim D, King D S, Park Y, et al. **Polarizable atomic multipoles for learning long-range electrostatics**. arXiv:2605.05746, 2026.  
- Batatia I, et al. **MACE-POLAR-1: A Polarisable Electrostatic Foundation Model for Molecular Chemistry**. arXiv:2602.19411, 2026.  
- DeepMD-kit **DPA4/SeZM** 文档：[DPA4/SeZM 文档](https://github.com/deepmodeling/deepmd-kit)  
- **SOG 长程插件（开源）**：[https://github.com/Dazzlemoon/sog](https://github.com/Dazzlemoon/sog)  
- 深势 DPA4 新闻解读：[医药魔方，2026-05-22](https://bydrug.pharmcube.com/news/detail/2a0dfc94dfc074c0dd0836f6b36eb585)
