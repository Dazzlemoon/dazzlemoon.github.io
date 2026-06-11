---
layout:     post
title:      通用 MLIP 能算声子了吗？Loew 等用约 1 万条 DFT 声子系统评测七款 uMLIP——Loew et al. 2025 解读
subtitle:   Ruhr 大学 Loew 等（npj Comput. Mater. 2025）：基于 MDR 约 1 万非磁半导体 PBE 声子库，对比 M3GNet、CHGNet、MACE-MP-0、SevenNet-0、MatterSim-v1、ORB、eqV2-M；MatterSim-v1 谐波声子/热力学性质 MAE 可逼近 PBE–PBEsol 差异，ORB/eqV2-M 几何极准但声子常虚频。
date:       2026-06-11
author:     qqz
header-img: img/post-bg-desk.jpg
catalog:    true
tags:
    - uMLIP
    - phonons
    - MatterSim
    - benchmark
---

![一图总结](/img/posts/2026-06-11-loew-umlip-phonons-ready/cover.png)

# 通用 MLIP 能算声子了吗？Loew 等用约 1 万条 DFT 声子系统评测七款 uMLIP

> **论文**：Antoine Loew, Dewen Sun, Hai-Chen Wang, Silvana Botti & Miguel A. L. Marques, *Universal machine learning interatomic potentials are ready for phonons*, **npj Comput. Mater.** **11**, 178 (**2025**).  
> **DOI**：[10.1038/s41524-025-01650-1](https://doi.org/10.1038/s41524-025-01650-1)  
> **机构**：Ruhr University Bochum（Research Center Future Energy Materials and Systems / ICAMS）。  
> **数据**：基于 **MDR** 声子数据库（约 1 万非磁半导体）；本文用 **PBE** 重算全库以匹配 uMLIP 训练泛函。

---

## 一、背景：E/F 排行榜很高，声子却没人系统测

过去数年，**通用机器学习原子间势（uMLIP）** 在 Matbench Discovery 等榜单上不断刷新 **能量、力、应力** 精度：M3GNet、CHGNet、MACE-MP-0、SevenNet-0、MatterSim-v1、ORB、eqV2-M 等模型覆盖周期表、可处理多样晶体结构。

但现有评测多集中在 **近平衡几何** 上的 E/F——训练数据（Materials Project、MPtrj、Alexandria 等）也以优化结构或短程 MD 轨迹为主。**声子**来自势能面 **二阶导数（力常数）**，探测的是平衡位附近 **Å 量级小位移** 的曲率；uMLIP 在 E/F 上领先，**未必**意味着谐波声子、热力学性质或动力学稳定性判断同样可靠。

Loew 等的工作即填补这一空白：在 **与训练泛函一致的 PBE 参考声子库** 上，对 **七款主流 uMLIP** 做 **几何弛豫 → 力常数 → 声子性质 → 动力学稳定性** 的全链路 benchmark，并讨论 **非保守力**（力非能量梯度）与 **训练数据规模** 的权衡。

---

## 二、评测对象：七款 uMLIP 与训练规模（表 4）

七模型均出现在 **Matbench Discovery** 榜单（撰文时排名约第 12–1 名）。架构与训练数据差异很大：

![表 4：七款 uMLIP 的训练样本数 $N\_{\mathrm{training}}$、数据来源与参数量 $N\_w$（原文 Table 4）。](/img/posts/2026-06-11-loew-umlip-phonons-ready/table04-models-training-params.png)

**表 4 解读**：

| 模型 | $N\_{\mathrm{training}}$ | 数据源 | $N\_w$ | 要点 |
|------|------------------------|--------|-------|------|
| **M3GNet** | 188 k | MPF | 228 k | 早期 uMLIP 代表；三体项；力由能量自动微分 |
| **MACE-MP-0** | 1.58 M | MPtrj | 4.69 M | ACE 局域描述符；消息传递步数少 |
| **CHGNet** | 1.58 M | MPtrj | 413 k | 参数量最小之一；仍保持竞争力 |
| **SevenNet-0** | 1.58 M | MPtrj | 842 k | 基于 NequIP；并行化消息传递 |
| **MatterSim-v1** | 6 M | MatterSim | 4.5 M | M3GNet 架构 + 主动学习扩数据 |
| **ORB** | 1.58 M | MPtrj | 25.2 M | SOAP + 图网络；**力为网络直接输出** |
| **eqV2-M** | 110 M | Alexandria, OMat | 86 M | 等变 Transformer；**力为网络直接输出** |

**关键区分**：前五款（M3GNet、CHGNet、MACE-MP-0、SevenNet-0、MatterSim-v1）的力 **严格为能量对坐标的梯度**；**ORB** 与 **eqV2-M** 将力作为 **独立输出**，训练更灵活、E/F 可更准，但 **破坏保守性**，对声子（小位移 Hessian）可能致命——后文结果印证这一点。

---

## 三、基准数据集：MDR 声子库与 PBE 重算（图 1–2）

### 3.1 数据规模与化学多样性

基准数据来自 **MDR 数据库**（Machine-learning Dataset for Raman 等声子应用），约 **1 万非磁半导体**，原始声子用 **VASP + PBEsol** 计算。因 **全部 uMLIP 均在 PBE 数据上训练**，作者用 **PBE 泛函重算整个声子库**，避免「参考泛函 ≠ 训练泛函」的歧义；同时保留 **PBE vs PBEsol** 差异作为 **绝对误差标尺**（泛函本身带来的声子偏差）。

![图 1：基准数据集多样性——(a) 元胞元素数分布（三元/四元为主）；(b) 晶系分布（单斜/正交最多）；(c) PBE 带隙分布（0–4 eV 密集）。原文 Fig. 1。](/img/posts/2026-06-11-loew-umlip-phonons-ready/fig01-dataset-diversity.png)

**图 1 解读**：

- **(a)**：约 **5800** 个三元、**2500** 个四元化合物，二元约 1200；高元数体系较少。  
- **(b)**：**单斜**与**正交**各约 2700 结构，三方/四方次之；**三斜**缺失（对称性低、计算成本高）。  
- **(c)**：PBE 带隙在 **0.5–4 eV** 最密，峰约 **2.0–2.2 eV**；覆盖绝缘/半导体为主，与高通量无机库一致。

![图 2：数据集中各元素出现频次（周期表热图；灰色为未出现元素）。原文 Fig. 2。](/img/posts/2026-06-11-loew-umlip-phonons-ready/fig02-periodic-table-frequency.png)

**图 2 解读**：**O** 出现 **5194** 次（氧化物为主），**S、Na、K、F、P** 等亦高频；**Fe、V、Ni** 等 3d 磁性相关元素偏少；**Tc、Eu、Gd** 等缺失（放射性或 VASP 收敛问题）。化学空间偏氧化物，但不影响 **跨模型相对比较**；作者指出偏差主要来自 Materials Project / ICSD 的固有分布。

---

## 四、几何弛豫：E/F 收敛与晶胞体积（表 1、图 3）

声子计算前，各 uMLIP 从 **PBE 参考结构** 出发做 **FIRE 弛豫**（力收敛 **0.005 eV/Å**，ASE + 空间群对称约束）。

![表 1：弛豫收敛失败率、能量 MAE(E)（meV/atom）与体积 MAE(V)（Å³/atom），相对 PBE。原文 Table 1。](/img/posts/2026-06-11-loew-umlip-phonons-ready/table01-relaxation-energy-volume.png)

**表 1 解读**：

| 模型 | Failed (%) | MAE(E) | MAE(V) |
|------|------------|--------|--------|
| CHGNet | **0.09** | 334 | 0.518 |
| MatterSim-v1 | 0.10 | **29** | 0.244 |
| MACE-MP-0 | 0.14 | 31 | 0.392 |
| SevenNet-0 | 0.15 | 31 | 0.283 |
| M3GNet | 0.12 | 33 | 0.516 |
| ORB | 0.82 | 31 | **0.082** |
| eqV2-M | 0.85 | 33 | **0.084** |
| PBEsol（参考） | — | — | 1.283 |

- **收敛**：CHGNet / MatterSim-v1 失败率最低（~0.1%）；**ORB / eqV2-M** 约 **0.8%** 未收敛（非保守力高频噪声或非物理力区域）。  
- **能量**：CHGNet MAE(E)=**334 meV/atom** 偏高（本文未用其训练时常用的能量修正流程）；其余多在 **29–33 meV/atom**。  
- **体积**：**ORB / eqV2-M** 体积误差 **最小**（~0.08 Å³/atom），优于 **PBE–PBEsol 平均差**（1.28 Å³/atom）；MatterSim-v1、SevenNet-0 次之。

![图 3：晶胞体积 per-atom 误差 $\Delta V$ 的小提琴图（相对 PBE）。原文 Fig. 3。](/img/posts/2026-06-11-loew-umlip-phonons-ready/fig03-volume-error-violin.png)

**图 3 解读**：**PBEsol** 体系性 **负偏差**（晶胞收缩，中心约 **−1 Å³/atom**），反映 PBE 欠结合；多数 uMLIP 分布 **集中在 0 附近**，MAE(V) **小于 PBE–PBEsol 差异**。**ORB / eqV2-M** 小提琴最窄，几何优化已可 **替代 DFT** 做晶格常数预测——但这 **不** 代表声子同样可靠。

---

## 五、谐波声子与热力学性质（表 2、图 4–5）

在弛豫结构上，用 **有限位移法（Phonopy）** 求力常数，并计算：

1. **最高声子频率** $\omega\_{\max}$（K；$1\,\mathrm{K} \approx 0.695\,\mathrm{cm}^{-1}$）——检测势阱 **软硬**（虚频前兆）；  
2. **声子 DOS**（剔除 <0.1 states/THz）；  
3. **三支声学支平均声速**；  
4. **300 K** 下 **振动熵 $S$、Helmholtz 自由能 $F$、定容热容 $C\_V$**（Fourier 插值至 20×20×20 q 网格，与 MDR 一致）。

![表 2：六类声子相关性质的 MAE 汇总（原文 Table 2）。](/img/posts/2026-06-11-loew-umlip-phonons-ready/table02-phonon-mae-summary.png)

**表 2 解读（节选）**：

| 模型 | MAE($\omega\_{\max}$) [K] | MAE($F$) [kJ/mol] | MAE($C\_V$) [J/K/mol] | MAE(avg. $v\_s$) |
|------|--------------------------|-------------------|----------------------|-----------------|
| **MatterSim-v1** | **17** | **5** | **3** | **401** |
| SevenNet-0 | 40 | 19 | 9 | 510 |
| MACE-MP-0 | 61 | 24 | 13 | 523 |
| CHGNet | 89 | 45 | 21 | 649 |
| M3GNet | 98 | 56 | 22 | 617 |
| ORB | 291 | 175 | 57 | 1198 |
| eqV2-M | **780** | 241 | 100 | 1240 |
| PBEsol（标尺） | 33 | 10 | 5 | 305 |

**PBE–PBEsol** 差异可视为「泛函噪声」；**MatterSim-v1** 多数 MAE **低于或可比** 该标尺，且误差分布 **中心在零、展宽极小**。**ORB / eqV2-M** 声子 MAE 比几何误差大 **一个数量级以上**。

![图 4：六类声子性质误差小提琴图：(a) $\omega\_{\max}$ (b) 熵 (c) 自由能 (d) 热容 (e) DOS (f) 平均声速。原文 Fig. 4。](/img/posts/2026-06-11-loew-umlip-phonons-ready/fig04-phonon-errors-violin.png)

**图 4 解读**：

- **第一档（差）**：**ORB、eqV2-M**——$\omega\_{\max}$、$F$、声速等 **大幅负/正偏**，分布极宽；力常数常 **非物理**。  
- **第二档（中）**：**M3GNet、CHGNet、MACE-MP-0、SevenNet-0**——系统 **软化** 声子（$\omega\_{\max}$、$F$ 偏低；$S$、$C\_V$ 偏高），MAE 通常 **大于 PBE–PBEsol**；四者训练数据同为 **MPtrj 1.58M**，表现相近，说明 **数据与架构** 对声子同样关键。  
- **最佳**：**MatterSim-v1**——各子图小提琴 **最窄且居中**；基于 M3GNet 但通过 **更大数据（6M）与主动学习** 显著改善 **二阶导数质量**，胜过更复杂的等变大模型（SevenNet-0、eqV2-M）。

![图 5：各模型预测的 $\omega\_{\max}$ 直方图 vs PBE 参考（左：M3GNet/SevenNet/MatterSim/PBEsol；右：MACE/CHGNet/ORB/eqV2-M）。原文 Fig. 5。](/img/posts/2026-06-11-loew-umlip-phonons-ready/fig05-max-frequency-distribution.png)

**图 5 解读**：参考分布主峰在 **500–2000 K**，轻元素体系可达 **~5500 K**。**M3GNet、CHGNet** 明显 **左移（软化）**；**MACE、SevenNet、MatterSim** 与参考 **形状接近**；**eqV2-M** 在 **200–300 K** 出现 **畸形尖峰**（大量虚频/近零频），**ORB** 分布亦严重失真——与图 4 一致。

---

## 六、动力学稳定性判别（表 3）

高通量筛选常用 **动力学稳定**（无虚频，Γ 点允许声学支小虚频阈值 **−50 K**）。以 **PBE 参考** 为真值，统计混淆矩阵（稳定 8189 / 不稳定 1769）：

![表 3：动力学稳定性预测混淆矩阵（TS/FU/TU/FS，%）。原文 Table 3。](/img/posts/2026-06-11-loew-umlip-phonons-ready/table03-dynamical-stability-confusion.png)

**表 3 解读**：

| 模型 | TS（真稳定%） | FU（假不稳定%） | TU（真不稳定%） | FS（假稳定%） |
|------|--------------|----------------|----------------|--------------|
| **MACE-MP-0** | **95** | 5 | 73 | 27 |
| **MatterSim-v1** | **95** | 5 | 75 | 25 |
| SevenNet-0 | 81 | 19 | **80** | 20 |
| M3GNet | 87 | 13 | 73 | 27 |
| CHGNet | 77 | 23 | 73 | 27 |
| ORB | **15** | **85** | 92 | 8 |
| eqV2-M | **7** | **93** | 94 | 6 |
| PBEsol | 97 | 3 | 76 | 24 |

- **MatterSim-v1 / MACE-MP-0**：对 **稳定** 结构识别率 **95%**，适合高通量 **筛掉不稳定相**。  
- **ORB / eqV2-M**：将 **85–93%** 本应稳定的结构判为不稳定（**假不稳定**），因声子 **系统性虚频**；对不稳定结构的 **召回** 高（TU 92–94%），但 **假稳定** 风险仍在。  
- 与 **PBEsol** 对比：泛函切换带来的稳定性误判率 **低于** 多款 uMLIP 的声子误差，再次强调 **PBE 参考 + 保守势** 的重要性。

---

## 七、讨论：几何准 ≠ 声子准；选型要看任务

### 7.1 非保守力的代价

**ORB / eqV2-M** 在 **E/F/体积** 上顶尖，但 **声子常虚频、热力学性质失真**。作者归因于力 **非能量梯度**：省 back-prop、拟合 E/F 更灵活，却在 **Å 级小位移** 上力与 Hessian **不自洽**——与文献 [45] 对非保守 MLIP 的分析一致。增大有限位移步长可部分缓解，但会引入 **非谐污染**。

### 7.2 MatterSim-v1 为何胜出

MatterSim-v1 **延续 M3GNet 简单架构 + 保守力**，靠 **6M 级数据与主动学习** 覆盖更广化学空间；声子 MAE **可比拟 DFT 泛函差异**，计算成本却低数个量级。说明 **训练数据规模与采样策略** 有时比 **更重的等变网络** 对 **响应性质（二阶导）** 更关键。

### 7.3 效率权衡

- **最快**：**M3GNet**（单核 CPU 可快于其他模型 GPU 推理）；  
- **最慢**：**eqV2-M、MACE-MP-0**；  
- 选型需同时看 **精度指标 + 硬件 + 是否需声子/热力学/动力学筛选**。

### 7.4 局限

- 基准限于 **非磁半导体**、**谐波** 声子；金属、磁性、强非谐未覆盖。  
- 仅 **PBE** 训练族 uMLIP；与 **r²SCAN、HSE** 等泛函混用需重新 benchmark。  
- 动力学稳定判据依赖 **Γ 点 −50 K 阈值**，与实验/严格阈值可能不完全一致。

---

## 八、方法摘要

| 环节 | 设置 |
|------|------|
| **DFT 参考** | VASP；PBE；力常数 Phonopy 有限位移；与 MDR 工作流一致 |
| **uMLIP 弛豫** | ASE + FretchetCellFilter 保对称；FIRE；$\lvert F \rvert<0.005$ eV/Å |
| **热力学** | 300 K；DOS Fourier 插值 20³ q 网格 |
| **声速** | Γ 附近小 q 群速度，三支声学支 xx/yy/zz 分量平均 |
| **稳定判据** | 除 Γ 外无虚频；Γ 声学支允许 $>-50$ K |

---

## 九、总结

Loew 等用 **~1 万条 PBE 声子计算** 首次系统表明：

1. **部分 uMLIP 已可用于谐波声子与 300 K 热力学性质**——**MatterSim-v1** 全面领先，**SevenNet-0** 次之。  
2. **E/F 榜单前列 ≠ 声子可靠**——**ORB、eqV2-M** 几何极准但声子 **常失败**；**保守力** 对响应性质至关重要。  
3. **MPtrj 同族四模型** 声子表现相近，**训练数据** 与 **任务指标** 需分开优化。  
4. 公开 **PBE 声子基准** 可推动下一代 uMLIP 把 **声子、动力学稳定性** 纳入训练与评测，而不仅是 E/F RMSE。

对做 **高通量相图、热导、格波谱学、声子稳定性筛选** 的材料计算用户：优先选用 **保守、经声子 benchmark 验证** 的模型（本文推荐 **MatterSim-v1**）；若仅做 **几何优化或短程 MD**，ORB/eqV2-M 仍具吸引力，但 **不宜直接信任其声子或 $C\_V$、$F$**。

---

## 延伸阅读（文献）

- **MDR 声子库**：Grumet et al., 机器学习拉曼/声子数据集（见原文引 [34]）。  
- **MatterSim-v1**：Yang et al., 大规模主动学习通用势。  
- **非保守 MLIP 与声子**：原文引 [45] 及社区对 ORB/eqV2 力输出的讨论。  
- **Matbench Discovery**：[matbench-discovery.materialsproject.org](https://matbench-discovery.materialsproject.org/)（E/F 榜单与本文声子结论需对照阅读）。

## 延伸阅读（站内）

- [机器学习原子间势基础模型的六个开放问题——Creed 等 2026 解读](/2026/06/08/机器学习原子间势基础模型的六个开放问题-Creed-等-2026-解读/)

