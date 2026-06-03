---
layout:     post
title:      恒电势 MD 看见锂枝晶成核：Hu 等用 DP-QEq 把 QEq 长程静电接进 Deep Potential——Hu 等 2025 解读
subtitle:   北大/清华深研院等（Nat Commun 2025）：DP-QEq+ConstP 看见锂枝晶成核；文末讨论与 Gao PQEq 结合路径（PQEq 长程极化 + ConstP 电极边界）。RMSE ConstQ 约 1.3 meV/atom、力 0.16 eV/Å。
date:       2026-06-03
author:     qqz
header-img: img/post-bg-desk.jpg
catalog:    true
tags:
    - 机器学习势
    - 锂金属电池
    - DP-QEq
    - 恒电势MD
    - 枝晶
    - 电池界面
---

![一图总结](/img/posts/2026-06-03-hu-dp-qeq-dendrite-constp/cover.png)

# 恒电势 MD 看见锂枝晶成核：Hu 等用 DP-QEq 把 QEq 长程静电接进 Deep Potential

> **论文**：Taiping Hu, Haichao Huang, Guobing Zhou, Xinyan Wang, Jiaxin Zhu, Zheng Cheng, Fangjia Fu, Xiaoxu Wang, Fuzhi Dai, Kuang Yu & Shenzhen Xu, *Observation of dendrite formation at Li metal-electrolyte interface by a machine-learning enhanced constant potential framework*, **Nature Communications** 16, 7379 (2025).  
> **DOI**：[10.1038/s41467-025-62824-5](https://doi.org/10.1038/s41467-025-62824-5)  
> **机构**：北京大学材料学院、北京科学智能研究院、清华大学深圳国际研究生院、DP Technology、厦门大学、北京科技大学等。

---

## 一、背景：锂金属很诱人，枝晶与恒电势 MD 是瓶颈

**锂金属负极** 理论比容量约 **3860 mAh/g**、密度约 **0.59 g/cm³**，被视为下一代高能量密度电池的关键材料。但充放电循环中 **锂枝晶（dendrite）** 不可控生长会带来库仑效率下降、内短路与安全风险，制约产业化。

实验上可用 SEM/TEM 观察枝晶形貌，却在 **时空分辨率** 上难以捕捉 **成核与早期演化** 的原子尺度过程。MD 能提供原子级图像，但锂的 **电沉积/溶解** 发生在 **电化学接触**（恒电势、电荷转移）而非单纯化学接触条件下——传统 MD 往往 **固定电极电荷** 或缺乏与 ML 势兼容的 **可变原子电荷** 恒电势（**ConstP**）方案，且 AIMD（如 GC-DFT）成本过高；ReaxFF+EChemDID 等经典路线精度又受限。

Hu 等提出 **DP-QEq**：用 **电荷平衡（QEq）** 处理 **长程 Coulomb**，用 **Deep Potential（DP）** 拟合 DFT 总能量减去 QEq 后的 **短程项** $E\_{\mathrm{Short}}$，并在 **ConstQ / ConstP** 约束下做 **并发学习（DP-GEN）** 训练；在 **Li / [EC+LiPF$\_6$] / Li** 双电极界面模型上跑 **循环 ConstP MD**，直接观察 **枝晶成核** 并给出 **SEI 无机相中 Li 局域富集** 的微观机制。

![图 1a–b：DP-QEq 能量分解 $E\_{\mathrm{Total}}=E\_{\mathrm{Short}}+E\_{\mathrm{QEq}}$；并发学习流程（ConstQ/ConstP 探索 → DFT 标注 → 训练 DP_Short）。](/img/posts/2026-06-03-hu-dp-qeq-dendrite-constp/fig01a-dp-qeq-architecture-workflow.png)

![图 1c–d：训练采样用单/双界面模型；枝晶研究用含一对对电极的夹层超胞（真空–电极–电解液–电极–真空），ConstP 下对 Li 施加 $\chi=\chi^0\_{\mathrm{Li}}+\Phi$。](/img/posts/2026-06-03-hu-dp-qeq-dendrite-constp/fig01d-double-interface-sandwich-model.png)

---

## 二、理论框架：总能量分解、QEq 与 ConstQ / ConstP

### 2.1 能量分解（式 1、10）

DFT 总势能写为：

$$
E_{\mathrm{Total}} \approx E_{\mathrm{Short}} + E_{\mathrm{QEq}}
\tag{1}
$$

- **$E\_{\mathrm{QEq}}$**：Gaussian 电荷的 Coulomb（PME + 自相互作用修正）+ 原子电负性 $\chi^0\_i$ 与硬度 $J\_i$ 的二次项（式 2）。  
- **$E\_{\mathrm{Short}}$**：局域成键等 **短程** 相互作用，由 ML 势表示；训练标签为  
  $$E_{\mathrm{Short}} = E_{\mathrm{DFT}} - E_{\mathrm{QEq}}$$
  在 **ConstQ** 下求 QEq 电荷后相减（式 10）。  
- 对比 **CENT、4G-HDNNP、BAMBOO、SpookyNet、SO3LR** 等「长短程拆分」势：它们一般 **不能在 ConstP 下** 处理真实电化学界面。

### 2.2 ConstQ：总电荷约束（式 3–4）

系统总电荷 $Q\_{\mathrm{tot}}$ 固定时，对 $E\_{\mathrm{QEq}}$ 引入 Lagrange 乘子 $\chi\_{\mathrm{eq}}$：

$$
\mathcal{L} = E_{\mathrm{QEq}} - \chi_{\mathrm{eq}}\left(\sum_i Q_i - Q_{\mathrm{tot}}\right)
\tag{3}
$$

对每个构型最小化 $\mathcal{L}$ 得原子电荷 $Q\_i$ 与 $\chi\_{\mathrm{eq}}$（式 4）。这与常规 DFT 中性超胞设定一致，便于 **只在 ConstQ 数据上训练 DP_Short**。

### 2.3 ConstP：电极外加势与电负性移动（式 5–7）

在 **开放电化学** 图像下，对电极 Li 施加外加势 $\phi\_i$，**巨势** 型量为：

$$
\Omega = E_{\mathrm{QEq}} + \sum_i \phi_i Q_i
\tag{5}
$$

$\phi\_i$ 按 Li 的 **Li–Li 配位数 CN** 与体相金属 CN$\_{\mathrm{Metal}}=8$（BCC Li）判定是否属于电极表面；双电极超胞中阳极/阴极分别取 $\phi\_{\mathrm{Li},1}$、$\phi\_{\mathrm{Li},2}$（式 6）。**施加 $\phi\_i$ 在数学上等价于移动该原子的电负性**——还原性增强对应阴极、氧化性增强对应阳极。

周期边界下仍要求 **$Q\_{\mathrm{tot}}=0$**，故 ConstP 的 Lagrange 形式与 ConstQ 相同（式 7）。界面法向（本文 $z$）需加 **偶极修正** $E\_{\mathrm{dipole}}^{\mathrm{corr}}$（式 8–9），纳入 $E\_{\mathrm{QEq}}$。

**设计要点**：训练只在 **ConstQ** 完成，ConstP 通过 **电负性偏移** 实现，避免 Zhou/Chen 等方案那样 **多种偏压分别训练多张势能面** 的高成本。

---

## 三、DP-QEq 训练与验证

### 3.1 数据与并发学习（Fig. 1b、Methods）

| 阶段 | 内容 |
|------|------|
| 初始集 | 体相 BCC Li（Materials Project + AIMD 微扰）、bulk [EC+LiPF$\_6$]（Packmol + ReaxFF 200 ps 采样）、单界面 Li/(EC+LiPF$\_6$)（534 原子，ReaxFF 预采样） |
| 并发学习 | **DP-GEN**：训练 4 个不同种子的 DP_Short（deepmd-kit，$4\times10^5$ 步；嵌入网 25-50-100，拟合网三层各 240 节点；截断 6 Å）→ **DP-QEq** 驱动 ConstQ/ConstP 探索（NVT，200–400 K，20 ps）→ 力偏差 0.1–0.2 eV/Å 的帧做 **ABACUS DFT** 标注 |
| 探索构型 | ConstQ：**548 原子** 单界面；ConstP：**834 原子** 双对电极界面（Fig. 1c 下） |

DFT：**ABACUS 3.4.0**，PBE + DZP NAO，D3(BJ) 色散；界面超胞用大 $\Gamma$ 中心 k 网格。

QEq 电荷求解：**投影梯度 + 自动微分**（SI 1.3），准线性标度，已集成 **DMFF**；与 DeePMD-kit 联用做 MD。

### 3.2 Full DP 与 DP-QEq 精度（Fig. 2）

测试集：**50 ps ConstQ** + **4 轮 × 100 ps ConstP**（Fig. 1c 下界面模型；ConstQ 末态作 ConstP 初态）。

| 对比 | 能量 RMSE | 力 RMSE |
|------|-----------|---------|
| Full DP vs DFT | **3.04 meV/atom** | **0.154 eV/Å** |
| DP-QEq（ConstQ 轨迹）vs DFT | **1.31 meV/atom** | **0.163 eV/Å** |
| DP-QEq（ConstP 轨迹）vs DFT | **10.0 meV/atom** | **0.227 eV/Å** |

ConstP 轨迹误差更大，但作者认为在 400 ps 长测试轨迹上仍 **可接受**（与既有电池 ML 势文献可比）。**枝晶相关 Li 金属原子** 的力 RMSE 全程约 **0.07–0.09 eV/Å**（Fig. S8）。

![图 2a：DFT / Full DP / DP-QEq 能量–力 parity 与 RMSE。b：500 ps ConstQ 下 Li–O、Li–F、C–O 的 g(r) 与 Full DP 一致。c：500 ps 轨迹上势能随时间，DP-QEq 与 Full DP、DFT 重合。](/img/posts/2026-06-03-hu-dp-qeq-dendrite-constp/fig02-dpshort-full-dp-performance.png)

**结构验证**：500 ps ConstQ MD 的 RDF 与 Full DP 几乎重合；500 ps 势能演化三者一致（Fig. 2b–c）。说明 **从总能量中剥离 QEq 再训 DP_Short** 在结构与动力学上可靠。

---

## 四、QEq 电荷与 DFT 后处理对比

### 4.1 玩具双电极模型（Fig. 3a）

两 Li 金属板 + 真空间隙：**ConstQ**（20 Å）电中性；**ConstP** 阴极 $\chi=\chi^0\_{\mathrm{Li}}+6\,\mathrm{V}$、阳极 $\chi=\chi^0\_{\mathrm{Li}}-2\,\mathrm{V}$（总压差 **8 V**）：

- 真空中 **电势线性下降**；  
- 电极表面电荷密度 **0.212 $\lvert e \rvert/\mathrm{nm}^2$**（20 Å 间隙）；间隙 **40 Å** 时密度 **减半**（0.106），符合静电直觉。

### 4.2 真实 Li/[EC+LiPF$\_6$] 界面（Fig. 3b）

QEq 给出的 **Li 原子电荷空间分布** 与 DFT **Hirshfeld、CM5** 定性一致（界面附近负/正电荷层交替）。作者指出 QEq **定量** 可与 DFT 电荷分析有偏差，但 **趋势合理**；CM5 在凝聚相电荷分析上亦存在已知局限。

![图 3：玩具模型 ConstQ/ConstP 电荷与电势；真实界面 Li 电荷 QEq vs Hirshfeld vs CM5。](/img/posts/2026-06-03-hu-dp-qeq-dendrite-constp/fig03-qeq-toy-model-charge-validation.png)

---

## 五、界面演化：从 ConstQ 平衡到 ConstP 循环

### 5.1 模拟参数（Methods 摘要）

| 步骤 | 设定 |
|------|------|
| 预平衡 | **3070 原子** 超晶格，**200 ps NPT ConstQ**（xy 1 bar，z 100 bar），抑制空腔 |
| 双界面胞 | 切割得 **33.00 × 16.07 × 90.00 Å³**，**500 ps NVT ConstQ** |
| ConstP 循环 | **4 轮 × 300 ps NVT**（1 fs，Nosé–Hoover）；每轮末构型作下一轮初态；**每两轮反转** 上下电极 $\phi$（模拟充放电） |
| 电极判据 | Li–Li CN  cutoff **3.5 Å**，CN$\_{\mathrm{Li-Li}}>8$ 则施加 $\phi$；**$\phi\_{\mathrm{Li},1}=-2\,\mathrm{V}$**（阳极侧），**$\phi\_{\mathrm{Li},2}=+6\,\mathrm{V}$**（阴极侧）；电极最外 **4 层 Li 固定** |
| 反应识别 | **ReacNetGenerator** 追踪物种（EC、LiPF$\_6$ 分解产物等） |

偏压 **8 V** 大于真实电池工作电压，用于 **加速** 红ox 与枝晶观察（与 Wu、ReaxFF 电化学 MD 文献类似）；$\phi$ 控制的是 **电化学驱动力**，不必一一对应开路电压。

### 5.2 电荷分布与 DFT 对照（Fig. 4）

两轮 **300 ps ConstP** 后，**Li 电荷沿 $z$ 分布** 显示氧化/还原区交替；**100 ps ConstP** 小胞初末态：QEq（ConstP 快照、ConstQ 求电荷）与 DFT Hirshfeld/CM5 **趋势一致**。

![图 4a：两轮 ConstP 下 Li 电荷着色形貌与 $z$ 向分布。b：小胞 100 ps ConstP 初末，QEq vs Hirshfeld vs CM5。](/img/posts/2026-06-03-hu-dp-qeq-dendrite-constp/fig04-constp-li-charge-vs-dft.png)

---

## 六、核心结果：枝晶成核动力学（Fig. 5）

在 **500 ps ConstQ 末态** 基础上，**4 轮 ConstP**（共 **1.2 ns** 量级 ConstP 时间；每轮 $3\times10^5$ 步）：

1. **第 1 轮**：阳极 Li 氧化，EC / LiPF$\_6$ 分解形成 **SEI**。  
2. **反转偏压后（第 2 轮）**：SEI **无机相** 中出现 **Li 聚集**（低电荷、偏金属性，图中蓝色团簇）→ **阴极侧不均匀 Li 沉积** → **枝晶成核**（第 2 轮右图黑圈）。  
3. **后续轮次**：枝晶变尖；第 4 轮尖端可形成 **~1 nm** 外的孤立 Li 团簇。  
4. **CN cutoff 敏感性**：Li–Li CN cutoff 改为 **3.3 Å** 仍观察到类似成核（Fig. S16）。

**电位监控**：两电极间静电势差与施加的 **电负性偏移差** 一致，说明 ConstP 实现正确（Fig. 5a）。

**SEI 产物（第 4 轮末，Fig. 5c）**：

| 区域 | 物种 |
|------|------|
| 无机 SEI（近电极） | **Li$\_2$CO$\_3$、Li$\_2$O、LiF** |
| 气体 | **CO$\_2$、CO、C$\_2$H$\_4$** 等 |

枝晶成核区与 **无机 SEI 富集区空间重叠**；有机 SEI 分布对成核影响小，文中未重点讨论。机制链条：

> **无定形无机 SEI 中 Li–Li 距离缩短 → Li 离子局域过饱和/聚集 → 阴极表面不均匀沉积 → 枝晶成核**

与 Tan 等通过调控 SEI 无机相促进均匀沉积的实验（*Adv. Mater.* 2024）相呼应。

![图 5a：四轮循环中上下电极势差。b：各轮 Li 电荷态演化与枝晶、不均匀沉积标注。c：末态 Li 电荷与 SEI 无机/气体产物分布。](/img/posts/2026-06-03-hu-dp-qeq-dendrite-constp/fig05-dendrite-nucleation-cyclic-constp.png)

---

## 七、讨论、局限与和 PQEq 路线的关系

### 7.1 作者强调的贡献

- **MLFF 框架下 ConstP + 可变原子电荷**（QEq），效率与精度兼顾；**双电极单胞** 同时模拟阴阳极，Li 氧化还原可仅通过 Li 转移平衡，无需强迫电解液「陪跑」氧化还原（对比部分单电极 ReaxFF 模型）。  
- **准线性标度 QEq 求解** + **DMFF / DeePMD-kit** 生态，可扩展到 **数千原子** 含长程静电的界面 MD。  
- 对 **全固态电池固–固界面**、电催化腐蚀等 **复杂电化学界面** 有推广潜力。

### 7.2 局限（原文 Discussion）

- QEq 电荷与 DFT 分区电荷 **定量** 可不一致；$\chi^0$、$J$ 等可 **反演拟合** DFT 电荷（未来工作）。  
- 外加 $\phi$（电负性移动）提供 **表面 redox 驱动力**，**不能** 直接对应真实电池 **工作电压** 数值。  
- 加速偏压（8 V）下动力学为 **非平衡** 过程，每轮 300 ps 不追求热力学完全平衡。  
- 枝晶模拟对 **CN cutoff** 等参数有敏感性，需验证（文内已做 3.3 Å 测试）。

### 7.3 与 Gao 2025 PQEq 基础势的对比

同 workspace 下 **PQeq 方法**（`post-to-wechat/电池领域思路/PQeq方法/`）对应 **Gao 等 Nat. Commun. 2025** 博客解读，核心分解为：

$$
E_{\mathrm{pot}} = \sum_i E_i^0 + E_{\mathrm{PQEq}} + E_{\mathrm{D3}}
$$

| 维度 | **Gao PQEq 基础势** | **Hu DP-QEq + ConstP** |
|------|---------------------|-------------------------|
| 长程静电 | **PQEq**（core–shell 可极化 Gaussian） | **经典 QEq**（电荷可变、壳层不位移） |
| 短程 | 等变 **GNN** $E\_i^0$（MPtrj 预训练至 Pu） | **DP_Short**（界面体系专用） |
| 色散 | 显式 **DFT-D3** | 并入 DFT 参考（PBE+D3BJ），短程 DP 学残差 |
| 边界条件 | 常规中性/固定电荷 MD | **ConstQ 训练 + ConstP 生产**（双电极 $\phi\_i$） |
| 典型目标 | 泛化势、带电团簇、>5 Å Coulomb、外场极化 | **Li 枝晶成核**、SEI 反应、循环电化学 MD |
| 软件生态 | 自研 PQEq + GNN 训练栈 | **DMFF + DeePMD-kit + DP-GEN + ABACUS** |

两篇 **都没有** 在对方论文里实现合并；但物理上一条长程、一条短程、电化学界面要 **开放边界**，天然存在「能否嫁接」的问题。

### 7.4 PQEq 与 ConstP 能否结合？——机制互补与一条可行研究路径

**结论先行**：**可以结合，而且物理上相当自然**；但目前 **尚无已发表的一体化实现**。更准确的表述是：把 Hu 文的 **ConstP 边界条件** 嫁接到 Gao 文的 **PQEq 长程项**（并保留短程 ML 势 + D3），得到示意性 **「DP（或 GNN）–PQEq–ConstP」** 框架，而不是把两篇论文的代码直接拼在一起就能跑。

#### 7.4.1 为什么「值得结合」

1. **分工正交**  
   - **PQEq** 解决：截断外 Coulomb、分子/离子 **极化**（Gao 图 2b：外场下 H$\_2$O 能量；补充图 5：QEq 非物理电荷漂移 vs PQEq）。  
   - **ConstP** 解决：电极 **开放电化学边界**——电荷随 $\phi\_i$（电负性移动）更新，双电极单胞里阴阳极同时红ox（Hu 式 5–7）。  
   Gao 的 foundation 势在 **中性或固定总电荷** 超胞里很强，但 **不会自动给出**「循环充放电下电极电势差驱动 Li 沉积」这一边界；Hu 的 ConstP 正好补这一块。

2. **直接对应 Hu 文局限**  
   Hu 在 Discussion 承认：经典 **QEq 电荷与 DFT 分区电荷可定量偏差**；电解液分子在界面电场下的 **极化** 用固定 QEq 描述偏弱。  
   **PQEq** 用 shell 位移实现 **原子级极化**，且 Gao 强调 **不学 DFT 分区电荷、而以静电能为目标**——与 Hu「用 QEq 抠出 $E\_{\mathrm{Short}}$ 再训 DP」的 **能量分解哲学一致**，只是把 $E\_{\mathrm{QEq}}$ 升级为 $E\_{\mathrm{PQEq}}$。

3. **与枝晶机制更相关的一阶效应**  
   枝晶链条依赖 **SEI 无机相中 Li 局域环境** 与 **阴极表面不均匀沉积**。界面附近 **EC、PF$\_6^-$ 分解产物、LiF/Li$\_2$O** 在强电场下若可极化，可能影响局域电场与 Li$^+$ 输运；**PQEq 比 QEq 多这一自由度**，ConstP 循环下或能给出与 Hu 不同的（需验证的）中间态图像。

#### 7.4.2 结合时应长什么样（示意方程）

在 Hu 的式 (1) 框架下，将长程项替换并保留 D3 显式色散（与 Gao 对齐）：

$$
E_{\mathrm{Total}} \approx E_{\mathrm{Short}}^{\mathrm{ML}} + E_{\mathrm{PQEq}} + E_{\mathrm{D3}}
$$

- **$E\_{\mathrm{Short}}^{\mathrm{ML}}$**：仍由 **$E\_{\mathrm{DFT}} - E\_{\mathrm{PQEq}} - E\_{\mathrm{D3}}$** 在 **ConstQ** 下定义并训练（对应 Hu 式 (10)，仅把 QEq 换 PQEq）。短程可用 **DP**（Hu 路线）或 **等变 GNN $E\_i^0$**（Gao 路线）；界面反应密、数据少时，Hu 式 **DP-GEN 并发学习** 仍更合适，Gao 的 MPtrj 预训练 $E\_i^0$ 可作 **初始化** 再微调。  
- **ConstP**：在 PQEq 电荷自洽之上，沿用 Hu 的 **巨势型** 约束，示意为  
  $$\Omega = E_{\mathrm{PQEq}} + E_{\mathrm{D3}} + \sum_i \phi_i Q_i$$  
  并对 **电极 Li**（仍可用 CN$\_{\mathrm{Li-Li}}>8$ 判据）施加 $\phi\_{\mathrm{Li},1}$、$\phi\_{\mathrm{Li},2}$；周期超胞仍加 **$Q\_{\mathrm{tot}}=0$** 与 **偶极修正**（Hu 式 8–9）。  
- **训练策略**：与 Hu 相同——**只在 ConstQ 轨迹上标 DFT 并训练短程**；ConstP 通过 **$\phi\_i$ 移动电负性** 在采样与生产 MD 中启用，避免为多偏压重复训练整张势能面。


#### 7.4.3 务实判断：现在能做什么

**现在就能做的（概念与研发规划）**

- 用 **PQEq 基础势** 做 **ConstQ** 下 Li 界面结构/径向分布、外场下溶剂极化的 **对照**（无需 ConstP），检验是否比 Hu 文中 QEq 更接近 DFT。  
- 在 **单电极或双电极** 模型上，手工实现 **$\phi\_i$ 偏移 + PQEq 自洽** 的原型（哪怕体系小于 Hu 的 3000 原子），验证势差–表面电荷密度关系是否仍像 Fig. 3a 那样合理。  
- 短程用 **Gao 预训练 $E\_i^0$** 微调界面数据，长程用 PQEq，减少从头 DP-GEN 的数据量（**迁移学习** 思路）。


#### 7.4.4 小结：一条推荐的「结合」叙事

若用一句话概括二者关系：

> **PQEq 负责「长程静电 + 可极化」是否物理；ConstP 负责「电极电化学边界」是否像电池。** Hu 文用 **QEq + ConstP** 首次把后者用到 **锂枝晶成核** 全轨迹；Gao 文用 **PQEq + GNN** 把前者推到 **foundation 尺度**。二者结合 = **在 ConstP 约束下，用 PQEq 取代 QEq 长程项，并保留能量分解训练短程 ML 势**——这是电池界面 MD 很自然的下一步，但属于 **方法开发课题**

---

## 八、方法摘要

| 项目 | 内容 |
|------|------|
| 软件 | ABACUS（DFT）、DP-GEN + DeePMD-kit（DP_Short）、DMFF（QEq）、ASE（MD）、ReacNetGenerator、VMD |
| 核心方程 | $E\_{\mathrm{Total}}=E\_{\mathrm{Short}}^{\mathrm{DP}}+E\_{\mathrm{QEq}}$；ConstQ（式 3–4）、ConstP（式 5–7） |
| 训练 | 并发学习；探索温度 200–400 K；ConstQ/ConstP 双界面采样 |
| 生产 MD | 3070→双界面 500 ps ConstQ；4×300 ps ConstP，$\Delta\chi=8\,\mathrm{V}$ |
| 主图 | Fig. 1–5（架构、精度、电荷验证、界面、枝晶） |

---

## 九、一句话总结

Hu 等 2025 用 **DP-QEq** 把 **QEq 长程静电** 与 **Deep Potential 短程** 拆开训练，在 **Li / [EC+LiPF$\_6$] / Li** 双电极模型上实现 **循环恒电势 MD**，首次在 ML 加速框架下 **直接看到** SEI 无机相中 **Li 聚集 → 不均匀沉积 → 锂枝晶成核** 的原子过程。**PQEq 基础势**（Gao 2025）与 **ConstP** 机制互补、可原则性结合为「短程 ML + PQEq + D3 + 电极 $\phi\_i$」，但尚无一体化发表实现（见第七节 7.4）。

---

## 参考文献（精选）

- Hu et al., *Nat. Commun.* **16**, 7379 (2025). [DOI](https://doi.org/10.1038/s41467-025-62824-5)  
- Rappe & Goddard, QEq, *J. Phys. Chem.* **95**, 3358 (1991).  
- Zhang et al., Deep Potential, *Phys. Rev. Lett.* **120**, 143001 (2018).  
- Wang et al., DMFF, *J. Chem. Theory Comput.* **19**, 5897 (2023).  
- Gubler et al., 加速 QEq, *J. Chem. Theory Comput.* **20**, 7264 (2024).  
- Tan et al., SEI 无机相调控均匀沉积, *Adv. Mater.* **36**, 2404815 (2024).  
- Gao et al., PQEq foundation MLIP, *Nat. Commun.* **16**, 10484 (2025). [DOI](https://doi.org/10.1038/s41467-025-65496-3)
