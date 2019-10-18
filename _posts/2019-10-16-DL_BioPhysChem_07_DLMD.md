---
layout:     post
title:      生物大分子模拟中的深度学习
subtitle:   深度学习的生物物理化学原理 第7章
date:       2019-10-16
author:     TablewareBox
header-img: img/Molecular-Dynamics-DL-2.jpg
catalog: true
tags:
    - 分子动力学
    - 多尺度建模与计算
    - 深度学习
    - 统计力学
    - 粗粒化
    - 增强抽样
---

## [深度学习的生物物理化学原理 - Notes Project Overview](https://tablewarebox.github.io/2019/02/16/DL_BioPhysChem_content/)

![knowledge atlas](https://tablewarebox.files.wordpress.com/2018/11/concept-map-81.png)

## 引言

**分子模拟**广泛用于化学、凝聚态物理、材料、生物等学科。在生命过程的研究中，它不但能提供静态的最优结构，还能从原子尺度模拟蛋白和其他生物大分子如何通过构象变化、折叠、配体结合等行使其功能，获得热力学和动力学信息，是蛋白设计、药物设计、机理研究等流程中不可或缺的环节。

受限于计算能力，分子模拟的时长一般很难达到实验可观测的时间尺度，难以对构象变化等**稀有事件**直接进行模拟。为了解决这个问题，理论化学家们发展了**多尺度计算**方法（如获2013年诺贝尔化学奖的QM/MM）、**粗粒化**方法和**增强抽样**方法。近年这些方法都受益于深度学习从而大大拓展了适用范围，同时我们也能从中看出

笔记的这一章节架构如下：

### Part I - 分子模拟的插画简介(Illustrated Molecular Dynamics)

- [x] 体系简介 - scheme 1
- [x] 势能面(力场)和运动方程 - scheme 2
- [ ] 构象空间探索 - scheme 3
- [ ] 平衡、玻尔兹曼分布&宏观热力学、动力学性质统计 - scheme 4
- [ ] 时间尺度分离&多尺度建模 - scheme 5
- [ ] 反应坐标&自由能 - scheme 6
- [ ] 粗粒化 - scheme 7
- [ ] 增强抽样 - scheme 8

### Part II - 监督学习用于粗粒化力场构建

### Part III - 生成模型和强化学习用于求解统计力学

### Part IV - 生成模型和强化学习用于增强抽样

## 7.1 分子模拟的插画简介(Illustrated Molecular Dynamics)

### 7.1.1 体系简介

![1-1 system](https://tablewarebox.files.wordpress.com/2019/10/1-1-system.jpg)

<div align="center">图1  蛋白 1HD0 及其势能面示意图</div>

我们用一个蛋白分子 1HD0 作为生物体系的样例，给它一个简称叫 PHD，为了拟人化一些~ $N$ 个原子的蛋白分子可以用 $3N$ 个笛卡尔坐标 $\mathbf{R}=(\boldsymbol{R_1, R_2, \cdots, R_N})$，正如 PHD 们在生活中有相当多的行动可以选择。因为原子间有化学键等相互作用，实际上任意一个 $\mathbf{R}$ 就对应一个能量值 $U(\mathbf{R})$，实际上给足够时间的话可以通过量子力学原理将它算出来，称为**势能面**。将所有原子坐标的 $3N$ 个自由度 $\mathbf{R}$ 假想地画在一个二维平面上，纵坐标为 $U(\mathbf{R})$，就是势能面的一个示意图，分子结构的变化就相当于在势能面上移动。图上有山谷（能量**局部极小**值，对应一个**亚稳态**）、山峰（能量**局部极大**值）、连接山谷的**鞍点**（对应**过渡态**）等。

### 7.1.2 势能面(力场)和运动方程

然而实际不可能用计算机求解这么大体系的量子力学问题，得到精确的**势能面**。科学家们就假设体系运行遵循经典力学，通过一个有解析表达式的函数**近似表达**蛋白分子的能量，我们称它为**力场**，为简便起见同样记为 $U(\mathbf{R})$，其负梯度为原子的受力，再由牛顿第二定律得运动方程：

$$\boldsymbol{F}_{i}(t)=-\nabla_{\boldsymbol{R}_i} U(\mathbf{R}) $$

$$\frac{\partial^2 \boldsymbol{R}_i}{\partial t^2}=-\frac{\nabla_{\boldsymbol{R}_i} U(\mathbf{R})}{m_i} $$

### 7.1.3 构象空间探索

### 7.1.5 时间尺度分离

正如刚刚 7.1.3 图中所示，生物大分子在势能面上移动时，可以发现它**同时具有几种运动模式**：在一些方向上进行快速的往返运动，如**键的振动、部分溶剂分子的运动**；在另一些方向上运动则非常缓慢，预示着一些**反应、配体结合、构象变化**等稀有事件的发生。这被称为**时间尺度分离(separation of timescales)**。生物大分子典型变化的时间尺度如图。作类比的话，PHD 们平时玩游戏、刷朋友圈

### 7.1.6 反应坐标&自由能

分子模拟需要较好地**表现运动最快的自由度**，因此需要设定**足够小的时间步长**，导致模拟的总时长被快自由度限制。一个自然的想法就是，既然我们关注的**稀有事件**发生时，快自由度中的事件已发生了相当多次，可否将它们平均掉，只作为其他运动时的**噪声**出现？**布朗运动**就是一个极好的例子：将溶剂分子的快速运动平均为**随机力**。生物大分子的各种动力学过程，同样可以选取一些核心的描述整个变化过程的量，称为**集团变量(collective variables, CV)** 或**反应坐标(reaction coordinates, RC)**：

$$
\mathbf{s}(\mathbf{R})=\left(s_{1}(\mathbf{R}), s_{2}(\mathbf{R}), \ldots, s_{d}(\mathbf{R})\right),\quad\quad d \ll 3N
$$

可以看出实际上是一个对高维运动**降维**的过程。对快自由度的能量平均后加入**熵**（多个微观状态 $\mathbf{R}$ 对应同一个反应坐标 $\mathbf{s}$，就有了微观状态数）就得到了**自由能** $F(\mathbf{s})$：

$$
p(\mathbf{s})=\int_{\mathbf{s}=\mathbf{s}(\mathbf{R})} \mathrm{d} \mathbf{R} p(\mathbf{R}),\quad \quad p(\mathbf{R})=\frac{e^{-\beta U(\mathbf{R})}}{Z}
$$

$$
\begin{aligned}
    F(\mathbf{s})&=-\frac{1}{\beta} \ln [p(\mathbf{s})\cdot Z]=-k_\mathrm{B}T \ln [p(\mathbf{s})\cdot Z]\\
    &=-\frac{1}{\beta} \ln \int_{\mathbf{s}=\mathbf{s}(\mathbf{R})} \mathrm{d} \mathbf{R}\cdot e^{-\beta U(\mathbf{R})} \\
    &=-k_\mathrm{B}T  \int_{\mathbf{s}=\mathbf{s}(\mathbf{R})} \mathrm{d} \mathbf{R}\cdot \ln e^{-\beta U(\mathbf{R})}\\
    &\quad\quad-T\cdot \left(-k_\mathrm{B} \int_{\mathbf{s}=\mathbf{s}(\mathbf{R})} \mathrm{d} \mathbf{R}\cdot e^{-\beta U(\mathbf{R})}\ln e^{-\beta U(\mathbf{R})}\right)\\
    &=\lang U(\mathbf{R})\rang _{\mathbf{s}=\mathbf{s}(\mathbf{R})}-TS(\mathbf{s})
\end{aligned}
$$

自由能面(Free Energy Surface, FES)比势能面光滑许多。

### 7.1.7 粗粒化

**粗粒化**可以说就是实现上述过程的一个方法：将数个原子视作一个粗粒，从而完成了对**局部化学键振动**这一快自由度的平均。由此实现降维，自由度大大减少，可采取的时间步长增大。

### 7.1.8 增强抽样

与粗粒化不同，虽然一大类增强抽样方法也依赖于**反应坐标**，但它仍然采用全原子体系，而**在反应坐标的维度上修改势能面**使稀有事件更易发生，计算宏观性质的统计平均时只需根据**重要性采样**原理乘上两个势能面导致的概率分布之差。
