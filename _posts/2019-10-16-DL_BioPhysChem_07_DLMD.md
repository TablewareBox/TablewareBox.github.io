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
- [ ] 构象空间探索、玻尔兹曼分布、遍历性原理 - scheme 3
- [ ] 宏观热力学、动力学性质统计 - scheme 4
- [ ] 时间尺度分离、多尺度建模 - scheme 5
- [ ] 反应坐标、自由能 - scheme 6
- [ ] 粗粒化 - scheme 7
- [ ] 增强抽样 - scheme 8

### Part II - 监督学习用于粗粒化力场构建

### Part III - 生成模型和强化学习用于求解统计力学

### Part IV - 生成模型和强化学习用于增强抽样

- - -

## 7.1 分子模拟的插画简介(Illustrated Molecular Dynamics)

### 7.1.1 体系简介

![1-1 system](https://tablewarebox.files.wordpress.com/2019/10/1-1-system.jpg)

> 我们用一个蛋白分子 1HD0 作为生物体系的样例，为了拟人化一些，给它一个简称叫 PHD（显然是精心选择过的哈哈哈）。

![1-1 forcefield](https://tablewarebox.files.wordpress.com/2019/10/1-1-pes.png)
<div align="center">图1  蛋白的漏斗势能面示意图</div>

$N$ 个原子的蛋白分子可以用 $3N$ 个笛卡尔坐标 $\mathbf{R}=(\boldsymbol{R_1, R_2, \cdots, R_N})$，正如 PHD 们在生活中有相当多的行动可以选择。因为原子间有化学键等相互作用，实际上任意一个 $\mathbf{R}$ 就对应一个能量值 $U(\mathbf{R})$，实际上给足够时间的话可以通过量子力学原理将它算出来，称为**势能面**。将所有原子坐标的 $3N$ 个自由度 $\mathbf{R}$ 假想地画在一个二维平面上，纵坐标为 $U(\mathbf{R})$，就是势能面的一个示意图，分子结构的变化就相当于在势能面上移动。图上有山谷（能量**局部极小**值，对应一个**亚稳态**）、山峰（能量**局部极大**值）、连接山谷的**鞍点**（对应**过渡态**）等。

### 7.1.2 势能面(力场)和运动方程

然而实际不可能用计算机求解这么大体系的量子力学问题，得到精确的**势能面**。科学家们就假设体系运行遵循经典力学，通过一个有解析表达式的函数**近似表达**蛋白分子的能量，我们称它为**力场**，为简便起见同样记为 $U(\mathbf{R})$，其负梯度为原子的受力，再由**牛顿第二定律**得运动方程：

$$\boldsymbol{F}_{i}(t)=-\nabla_{\boldsymbol{R}_i} U(\mathbf{R}) $$

$$\frac{\partial^2 \boldsymbol{R}_i}{\partial t^2}=-\frac{\nabla_{\boldsymbol{R}_i} U(\mathbf{R})}{m_i} $$

![1-2 forcefield](https://tablewarebox.files.wordpress.com/2019/10/1-2-forcefield.png)
<div align="center">图2  典型力场构成：范德华力、静电、键长偏离、键角偏离、二面角偏离</div>

### 7.1.3 构象空间探索、玻尔兹曼分布、遍历性原理

分子模拟可以在不同控制条件(系综)下进行。如控制**温度和体积**不变时，**化学平衡**的原理告诉我们，经过无限长时间，探索过每处 $\mathbf{R}$，概率分布达到平衡后，两点间的概率之比为平衡常数 $K=\exp[-\Delta U/k_\mathrm{B}T]$，即在某一点 $\mathbf{R}$ 处停留的概率正比于 $\exp[-U(\mathbf{R})/k_\mathrm{B}T]$。这样的概率分布在**物理化学**中称为**玻尔兹曼分布**：

$$
p(\mathbf{R})=\frac{e^{-\beta U(\mathbf{R})}}{Z},\quad \quad
Z=\int e^{-\beta U(\mathbf{R})}\mathrm{d} \mathbf{R},\quad \quad \beta=\frac{1}{k_\mathrm{B}T}
$$

由于只能通过平衡知道任意两点间概率的比值，求解**统计力学**的核心任务就是对这些比值进行**归一化**，表达为概率 $p(\mathbf{R})$。归一化常数 $Z$ 称为**配分函数(partition function)**，需要对所有可能的 $\mathbf{R}$ 积分。对**对数配分函数** $\ln Z$ 求导我们就能获得一个体系所有的**平衡态热力学性质**。

![1-3 explore](https://tablewarebox.files.wordpress.com/2019/10/1-3-explore.jpg)

> “我太难了。”

然而这样的高维积分实际上是一个无法计算(intractable)的任务。分子模拟的策略是改**积分计算**为**采样计算**(对样本取平均)：有限时间的模拟，实际上是在探索构象空间，对**玻尔兹曼分布**进行**采样**。然而在**有限时间**的模拟中，不可能遍历整个构象空间，分子会陷在**局部最小值附近的势阱**中，短时间难以越过**高势垒**探索新的构象区域，采样就无法覆盖到一些重要变化，称为**遍历性破缺(broken ergodicity)**。

### 7.1.4 宏观热力学、动力学性质统计

始终应该明确一点，观测到的宏观结果都是许多微观结构、微观反应路径统计平均的结果。

### 7.1.5 时间尺度分离

![1-5 timetable](https://tablewarebox.files.wordpress.com/2019/10/1-5-timetable.jpg)

> PHD 们平时玩游戏、刷朋友圈是快自由度，决定他们未来方向的文献阅读、JC、idea、答辩、论文则是慢自由度。

正如刚刚 7.1.3 图中所示，生物大分子在势能面上移动时，可以发现它**同时具有几种运动模式**：在一些方向上进行快速的往返运动，如**键的振动、部分溶剂分子的运动**；在另一些方向上运动则非常缓慢，预示着一些**反应、配体结合、构象变化**等稀有事件的发生。这被称为**时间尺度分离(separation of timescales)**。生物大分子典型变化的时间尺度如图3。

![1-5 timescale](https://tablewarebox.files.wordpress.com/2019/10/1-5-timescale.png)
<div align="center">图3  生物大分子内各过程的时间尺度</div>

### 7.1.6 反应坐标&自由能

分子模拟需要较好地**表现运动最快的自由度**，因此需要设定**足够小的时间步长**，导致模拟的总时长被快自由度限制。一个自然的想法就是，既然我们关注的**稀有事件**发生时，快自由度中的事件已发生了相当多次，可否将它们平均掉，只作为其他运动时的**噪声**出现？**布朗运动**和**朗之万方程**就是极好的例子：将溶剂分子的快速运动平均为**随机力**。生物大分子的各种动力学过程，同样可以选取一些核心的描述整个变化过程的量，称为**集团变量(collective variables, CV)** 或**反应坐标(reaction coordinates, RC)**：

$$
\mathbf{s}(\mathbf{R})=\left(s_{1}(\mathbf{R}), s_{2}(\mathbf{R}), \ldots, s_{d}(\mathbf{R})\right),\quad\quad d \ll 3N
$$

一个简单例子是选为 $\boldsymbol{R}_i$ 的线性组合。可以看出实际上是一个**对高维运动降维**的过程。对快自由度的能量平均后加入**熵**（多个微观状态 $\mathbf{R}$ 对应同一个反应坐标 $\mathbf{s}$，就有了微观状态数）就得到了**自由能** $F(\mathbf{s})$：

$$
p(\mathbf{s})=\int_{\mathbf{s}=\mathbf{s}(\mathbf{R})}  p(\mathbf{R})\mathrm{d} \mathbf{R},\quad \quad p(\mathbf{R})=\frac{e^{-\beta U(\mathbf{R})}}{Z},\quad \quad
$$

$$
Z=\int e^{-\beta U(\mathbf{R})}\mathrm{d} \mathbf{R},\quad \quad \beta=\frac{1}{k_\mathrm{B}T}
$$

$$
\begin{aligned}
    F(\mathbf{s})&=-\frac{1}{\beta} \ln [p(\mathbf{s})\cdot Z]=-k_\mathrm{B}T \ln [p(\mathbf{s})\cdot Z]\\
    &=-\frac{1}{\beta} \ln \int_{\mathbf{s}=\mathbf{s}(\mathbf{R})} e^{-\beta U(\mathbf{R})}\mathrm{d} \mathbf{R} \\
    &=-k_\mathrm{B}T  \int_{\mathbf{s}=\mathbf{s}(\mathbf{R})} p(\mathbf{R})\ln e^{-\beta U(\mathbf{R})}\mathrm{d} \mathbf{R}\\
    &\quad\quad-T\cdot \left(-k_\mathrm{B} \int_{\mathbf{s}=\mathbf{s}(\mathbf{R})} p(\mathbf{R})\ln p(\mathbf{R}) \mathrm{d} \mathbf{R}\right)\\
    &=\mathbb{E}_{\mathbf{R},\mathbf{s}(\mathbf{R})=\mathbf{s}} [U(\mathbf{R})]-T\cdot -k_\mathrm{B} \mathbb{E}_{\mathbf{R},\mathbf{s}(\mathbf{R})=\mathbf{s}} [\ln p(\mathbf{R})]\\
    &=\mathbb{E}_{\mathbf{R},\mathbf{s}(\mathbf{R})=\mathbf{s}} [U(\mathbf{R})]-T\cdot S(\mathbf{s})
\end{aligned}
$$

自由能面(Free Energy Surface, FES)比势能面光滑许多。

![1-6 rccv](https://tablewarebox.files.wordpress.com/2019/10/1-6-rccv.jpg)

### 7.1.7 粗粒化

![1-7 cg](https://tablewarebox.files.wordpress.com/2019/10/1-7-cg.jpg)

> 两耳不闻窗外事，一心只读圣贤书。<br><br>
> “我变秃了，也变强了。”

**粗粒化**可认为是实现上述过程的一个方法：将数个原子视作一个粗粒，从而完成了对**局部化学键振动**这一快自由度的平均。由此实现降维，自由度大大减少，可采取的时间步长增大。粗粒化方法大致分为 **bottom-up coarse-graining**(拟合全原子模拟的受力或热力学参数) 和 **top-down coarse-graning**(拟合实验数据)。

### 7.1.8 增强抽样

![1-8 enhanced-sampling](https://tablewarebox.files.wordpress.com/2019/10/1-8-enhanced-sampling.jpg)

> 月下柳梢星沉湾，伫马听浪入梦难。学海无涯苦作船。<br>
> 春蚕吐丝千斤缆，蜡炬点灯万丈光。征帆不落桨声欢。<br><br>
> “老师真是世界上最好的导师，我一个人可能100年都发不出的文章，在Ta指点下1个月就完成了。”

与粗粒化不同，虽然一大类增强抽样方法也依赖于**反应坐标**，但它仍然采用全原子体系，而**在反应坐标的维度上修改势能面**使稀有事件更易发生，计算宏观性质的统计平均时只需根据**重要性采样(importance sampling)** 原理乘上两个势能面导致的概率分布之差。**退火(annealing, tempering)** 类方法可视作将 $U(\mathbf{R})$ 作为反应坐标。

$$
\mathbb{E}_{\mathbf{s}\sim p(\mathbf{s})}[f(\mathbf{s})]=\int f(\mathbf{s}) p(\mathbf{s}) \mathrm{d}\mathbf{s} = \int \frac{f(\mathbf{s}) p(\mathbf{s})}{q(\mathbf{s})}\cdot q(\mathbf{s})\mathrm{d}\mathbf{s} = \mathbb{E}_{\mathbf{s}\sim q(\mathbf{s})} \left[\frac{f(\mathbf{s}) p(\mathbf{s})}{q(\mathbf{s})} \right]
$$

$$
p(\mathbf{s})=\frac{e^{-\beta F(\mathbf{s})}}{Z_p},\quad \quad
q(\mathbf{s})=\frac{e^{-\beta [F(\mathbf{s})+V(\mathbf{s})]}}{Z_q}
$$

增强抽样的一个代表性方法是 Metadynamics，它通过不断在当前位置加**偏置势(bias potential)** 跃出势阱。

![1-8 metadynamics](https://tablewarebox.files.wordpress.com/2019/10/1-8-metadynamics.png)
<div align="center">图4  Metadynamics 增强抽样</div>

## 7.2 监督学习用于粗粒化力场构建

### Highlights

* 特征选取为粗粒化模型的**内坐标**，保证三维旋转、平移对称性
* 三种模型架构考虑的不同**相互作用方式**

## 7.3 生成模型和强化学习用于求解统计力学(Boltzmann Generator)

### Highlights

* $\mathbf{x}$ 和 $\mathbf{z}$ 互相作为对方的“反应坐标”
* 神经网络（具体为深度生成模型中的**流模型(flow model)**）实现**相互作用体系**到**无相互作用体系**(如理想气体)的**可逆变换**
* 平行采样，计算自由能和自由能变
* 目标 AlphaFold Zero，只用物理知识预测大分子的结构系综和动力学
* 统计力学启发了 Boltzmann Machine 一类基于能量的生成模型，演化为如今的 Boltzmann Generator，反哺统计力学

## 7.4 生成模型和强化学习用于增强抽样

### Discussion Highlights

* 核心是**反应坐标**的选取，这不是简单的对**高维数据**的降维，而是对**高维运动**（有了时间影响）。
* 神经网络虽然是**通用函数拟合器**，但需要人为设计架构用以保证**泛化能力**，不然就是**人工智障**。保证泛化能力的关键是抓住体系/数据的共性和结构，称为**诱导偏置(inductive bias)**，如**对称性**：
  * 三维平移、旋转对称性（通过**内坐标**等方法实现）
  * 同类原子交换对称性（极难满足）
  * Hamilton 力学的辛结构（较难满足）
* 统计力学（生物物理化学）与深度学习的 Cross-fertilization