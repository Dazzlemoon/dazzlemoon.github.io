---
layout:     post
title:      CACE-SOG 阶段进展汇报（water / MAPbI3）
subtitle:   汇报当前 CACE-SOG 在 water 与 MAPbI3 两个体系的阶段进展，重点包括长程核替换、BEC 指标评估及 epsilon_used 变化解释
date:       2026-05-17
author:     qqz
header-img: img/post-bg-desk.jpg
catalog:    true
tags:
    - MLIP
    - 势函数
    - 谱学
---

本文汇报 CACE-SOG 的阶段性进展，重点回答三件事：

1. 将 LES 长程核替换为 SOG 后，BEC 是否仍可稳定计算；
2. `epsilon_used` 为什么会随体系/模型变化；
3. 当前结果相对 CACE-LES 的改进幅度如何。

---

## 一、方法更新：LES 核到 SOG 核

LES 原始长程核：

$$
\phi(r)=\frac{\mathrm{erf}\left(r/(\sqrt{2}\sigma)\right)}{r}
$$

CACE-SOG 中采用高斯和基展开：

$$
\mathcal{K}_{\mathrm{SOG}}(r)=\sum_{k=1}^{K} a_k \exp(-b_k r^2)
$$

其中 `a_k`、`b_k` 为可训练参数。

阶段结论：CACE-SOG 并未去除长程物理项，而是替换长程核参数化形式；因此在响应量定义不变前提下，BEC 仍可按原流程计算。

---

## 二、关键数学口径：BEC 与 `epsilon_used`

BEC 定义：

$$
Z^*_{i\alpha\beta}=\frac{\partial P_\alpha}{\partial r_{i\beta}}
$$

当前评估中使用到的介电相关量如下：

| 量 | 来源 | 定义/获取方式 | 含义 |
|---|---|---|---|
| `epsilon_inf` | 文献/实验/DFPT | 例如 MAPbI3 取 4.7 | 材料高频介电常数 |
| `epsilon_e` | 论文框架内生 | $\epsilon_e=\epsilon_{\infty}/(1+\chi^{\mathrm{les}})$ | 构型相关有效屏蔽 |
| `epsilon_used` | 本项目后处理 | $\epsilon_{\mathrm{used}}=(\alpha^*)^2$（$\alpha^*$ 为最小二乘对齐系数） | 评估尺度对齐系数 |

说明：`epsilon_used` 不是固定材料常数，而是当前模型在当前数据集上的最优对齐结果。

---

## 三、阶段结果总览（已完成）

### 1) MAPbI3：CACE-SOG 相对 CACE-LES

![MAPbI3体系对比结果](/img/posts/2026-05-17-cacesog-water-mapbi3/MAPbI3体系.png)

### 2) water：CACE-SOG 相对 CACE-LES

![水体系对比结果](/img/posts/2026-05-17-cacesog-water-mapbi3/水体系.png)

---

## 四、`epsilon_used` 变化的解释

SOG 框架下 `epsilon_used` 变化可能原因包括：

1. 长程核函数族发生变化（Ewald 型 -> 高斯和型）；
2. 可训练参数吸收了部分尺度自由度（`a_k,b_k` 与潜变量耦合）；
3. 不同模型分支（`r/u/uiu`）对应不同最优对齐系数。

---

## 五、当前结论

- CACE-SOG 对于 CACE-LES 能量和力的 RMSE 略有提升。
- CACE-SOG 在 BEC 指标上如果采用线性拟合的方式相比于 CACE-LES 更准确，但 CACE-LES 中用的材料高频介电常数由 DFPT（密度泛函理论）得到。

---

## 六、后续计划

- 继续绘制 CACE-LES 文章中对于 water 和 MAPbI3 的红外光谱曲线以及拉曼光谱曲线并与 CACE-LES 观察。
- 学习 DFPT 的计算方式，明确文中 water / MAPbI3 的 `epsilon` 数据获取路径，并解释 SOG 核替换后其变化原因。

