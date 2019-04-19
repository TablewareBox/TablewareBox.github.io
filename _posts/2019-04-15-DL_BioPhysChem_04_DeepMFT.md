---
layout:     post
title:      深度平均场与统计神经动力学
subtitle:   深度学习的生物物理化学原理 第4章
date:       2019-04-15
author:     TablewareBox
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - DL & BioPhysChem
    - 深度学习理论
    - 平均场理论
    - 生物物理化学
    - 统计力学
    - 无序系统
---

## [深度学习的生物物理化学原理 - Notes Project Overview](https://tablewarebox.github.io/2019/02/16/DL_BioPhysChem_content/)

![knowledge atlas](https://tablewarebox.files.wordpress.com/2018/11/concept-map-81.png)

## 引言

**深度平均场理论**是近年来 Google Brain 研究人员提出的，用于解释深度神经网络的**表达能力、训练技巧**和**模型架构**的理论框架，其根源可追溯到日本学者**甘利俊一(Shun-ichi Amari)** 上世纪70年代提出的**统计神经动力学**[^1]。在80年代 **Hopfield 网络**提出之后，H. Sompolinsky, A. Crisanti 和 H. J. Sommers 等人将其发展为**自旋玻璃模型**所衍生**无向神经网络**的**动态平均场理论**[^2]，研究网络运行时一般性的动力学性质，并描述了**有序-混沌相变**。由于无向神经网络的**时间演化、动态问题**类似于有向神经网络的**层间传递、深度问题**，S. S. Schoenholz, S. Ganguli, J. Pennington 和 J. Sohl-Dickstein 等人重新以现代的**有向（前馈）随机神经网络**结构进行了推导，他们的一系列理论工作已能一定程度上为网络结构和训练技巧的设计提供指导。

笔记的这一章节架构如下：

### Part I - 理论基础

- [x] 动态平均场理论回顾[^2]
- [x] 深度平均场：理论假设与高斯过程视角[^3]
- [ ] 深度网络的指数级表达能力[^4]
- [ ] 深度信息传播：训练中的有序-混沌相变[^5]

### Part II - 训练技巧

- [ ] 训练技巧 I: 正交初始化 - 动态等距与普适类[^6][^7]
- [ ] 训练技巧 II: 残差网络的运行原理[^8]
- [ ] 训练技巧 III: 层宽度变化、层方差变化的运行原理[^9]
- [ ] 训练技巧 IV: 批归一化的运行原理[^10]

### Part III - 网络结构

- [ ] CNN 的平均场理论[^11]
- [ ] RNN, LSTM, GRU 的平均场理论[^12][^13]
- [ ] 图网络的平均场理论[^14]

## 4.1 动态平均场理论回顾

### 4.1.1 模型建立

甘利俊一(Shun-ichi Amari) 建立的模型中，$N$ 个**神经元**由连续变量 $\{s_i(t)\in[-1,1]\},\,\,i=1,...,N$ 描述（对应自旋），“**突触矩阵**” $\mathbf{W}$（对应自旋的耦合常数）表达它们的相互作用，阈值 $-\mathbf{b}$ 相当于外场。$W _ {ij}\sim\mathcal{N}(w_0/N,\sigma _ w^2/N),b _ i\sim\mathcal{N}(b_0,\sigma _ b^2)$ 选为一组独立的高斯随机变量。每一时刻 $t$ 神经元的状态 $x_i(t)$ 由**局域场 $h_i(t)$** 决定：

$$x_{i}(t)=\phi(h_{i}(t))$$

其中 $\phi(h)$ 为**非线性激活函数**，可选为任意 S型函数，如 $\phi(h)=\tanh (gh)$，需满足 $\phi(\pm\infty)=\pm 1, \phi(-h)=-\phi(h), \phi'(h)>0.$ 

$g$ 为**非线性指数**，可以类比 $\beta=1/k_\mathrm{B}T$：$g\to 0, T\to\infty$ 时 $\phi(h)\sim gh$；$g\to \infty,T\to 0$ 时 $\phi(h)\to \pm 1.$

时间演化的**动力学方程**为

$$
\partial_{t} h_{i}=-h_{i}+\sum_{j=1}^{N} W_{i j} x_{j}+b_i=-h_{i}+\sum_{j=1}^{N} W_{i j} \phi(h_{j})+b_i
$$

若将其移项，可以看出它和前馈神经网络的关系：

$$
\begin{cases}
    h_{i}+\partial_{t} h_{i}=\sum_{j=1}^{N} W_{i j} x_{j}+b_i \\
    x_i=\phi(h_{i})
\end{cases}
\longleftrightarrow
\begin{cases}
    \mathbf{h}^{l+1}=\mathbf{W}^{l+1} \mathbf{x}^{l}+\mathbf{b}^{l+1} \\
    \mathbf{x}^{l}=\phi(\mathbf{h}^{l})
\end{cases}
$$

实际上这也是**无向神经网络**与**有向神经网络**的差别：无向神经网络的**时间演化、动态问题**类似于有向神经网络的**层间传递、深度问题**。从这一点看自旋玻璃类**无向神经网络**可以理解成**权重相同的无限层有向神经网络**。

我们知道平衡态时**序参量**有定值。因此考察动态（或深度）问题的一个角度是将平衡态作为**序参量演化**的一个**稳定不动点**。

> 事实上动态问题比静态（平衡态）问题复杂许多。如仅当 $J _ {ij}=J _ {ji}$ 时动力学才代表向平衡态的弛豫过程；且在上一章中提到，由于**各态历经破缺**，有时无法弛豫到平衡态，动力学相变温度常高于热力学相变温度，详细可参考上一章自旋玻璃理论以及**动态平均场理论**。

**深度平均场理论**也采用了这种思路，关注前向传播中**序参量**的变化。在**自旋玻璃理论**[^15]中，序参量是**重叠度(overlap)矩阵**：

$$
q_{aa}=\frac{1}{N} \sum_{i=1}^{N}\sigma_{i,a}^{2},\quad q_{ab}=\frac{1}{N} \sum_{i=1}^{N}\sigma_{i,a}\sigma_{i,b}
$$

$$
q_{\alpha \alpha}=\frac{1}{N} \sum_{i=1}^{N}\langle\sigma_{i}\rangle_{\alpha}^{2},\quad q_{\alpha \beta}=\frac{1}{N} \sum_{i=1}^{N}\langle\sigma_{i}\rangle_{\alpha}\langle\sigma_{i}\rangle_{\beta}
$$

**自重叠度(self-overlap)** $q _ {aa}, q _ {\alpha \alpha}$ 衡量**构型、复本 $a$** 或**态 $\alpha$** 的**大小**，交叉项 $q _ {ab}, q _ {\alpha \beta}$ 衡量**构型、复本 $a,b$** 或**态 $\alpha,\beta$** 间的**相似度**。自旋玻璃热力学的**复本方法(replica method)** 中，对随机的相互作用分布做平均时，解耦了不同自旋，但关联了不同复本。由此在不同温度下有**复本对称(replica symmetric, RS)平均场解**（交叉项 $q _ {ab}$ 全部相等，但不同于 $q _ {aa}$）、**一阶复本对称破缺(1RSB)平均场解**（交叉项 $q _ {ab}$ 有两个取值 $q _ 0,q _ 1$，类似于动力系统的“**双稳**”）等。

$$ 
q_{a b}=\left[
    \begin{array}{ccccccc}
    1     & q_{1} & q_{1} &       &       &       & \\
    q_{1} & 1     & q_{1} &       & q_{0} &       & \cdots \\
    q_{1} & q_{1} & 1     &       &       &       & \\
          &       &       & 1     & q_{1} & q_{1} & \\
          & q_{0} &       & q_{1} & 1     & q_{1} & \\
          &       &       & q_{1} & q_{1} & 1     & \\
          &\vdots &       &       &       &       & \ddots
    \end{array}
\right]
$$

<div align="center">图1  1RSB 平均场解的典型<b>重叠度矩阵</b>。详见自旋玻璃章节</div>

类似地，网络的不同输入可以类比自旋玻璃的不同**复本(replicas)** ，对网络的每一层有

$$ 
q_{aa}^{l}=\frac{1}{N_{l}} \sum_{i=1}^{N_{l}}[\mathbf{h}_{i}^{l}(\mathbf{x}^{0, a})]^{2},\,\,\,\,\,q_{a b}^{l}=\frac{1}{N_{l}} \sum_{i=1}^{N_{l}} \mathbf{h}_{i}^{l}(\mathbf{x}^{0, a}) \mathbf{h}_{i}^{l}(\mathbf{x}^{0, b}) \quad a, b \in\{1,2\}
$$

我们将在 4.2 节讨论**深度平均场理论**时继续展开。

### 4.1.2 Amari 解

Amari 最初提出的解是**朴素平均场**近似的结果，只能描述简单的动态性质。

* **假设1(平均场近似)**：$N$ 足够大时，由于受前一时刻大量自旋的影响，由**中心极限定理**，所有 $h _ {i}(t)$ 相互独立且满足高斯分布：$h _ {i}(t)\sim\mathcal{N}(m(t),q(t))$。

由此所有 $h _ i(t)$ 的函数 $f(h)$ 对应的宏观量可通过对 $h _ i(t)$ 高斯分布的积分得到：

$$
\begin{aligned}
    F & =\int_{-\infty}^{\infty} \frac{1}{\sqrt{2 \pi q(t)}} f(h) \exp \left\{-\frac{(h-m(t))^{2}}{2 q(t)}\right\} \mathrm{d} h \\
    &=\int \mathcal{D}z \cdot f\left[\sqrt{q(t)}z+m(t)\right],\quad \mathcal{D}z =\frac{\mathrm{d} z}{\sqrt{2 \pi}} e^{-z^{2}/2}
\end{aligned}

$$

对时间演化方程积分可得，

$$
h_{i}(t)=\int_{0}^{t}\left\{\sum_j w_{i j} x_{j}(\tau)+b_{i}\right\} e^{\tau-t} \mathrm{d} \tau+h_{i}(0) e^{-t}
$$

时间足够长时，最后一项可以忽略，定义

$$
\tilde{x}_{j}(t)=\int_{0}^{t} x_{j}(\tau) e^{t-t} d \tau \quad\Longrightarrow\quad h_{i}(t)=\frac{1}{N} \sum_{j} w_{i j} \tilde{x}_{j}(t)+b_{i}
$$

* **假设2(时间相关假设)**：对足够大的 $N,t$，

$$
\frac{1}{N} \sum_{i=1}^{N} x_i(t)\tilde{x}_{i}(t) \simeq \frac{1}{N} \sum_{i=1}^{N} x_{i}^{2}(t)=\int \mathcal{D}z \cdot \phi^2\left[\sqrt{q(t)}z+m(t)\right]
$$

对时间演化方程做平均可以得到**序参量的演化方程**：

$$
\begin{aligned}
    \partial_t m(t)& =\frac{1}{N}\sum_i \partial_t h_i(t)\\
    &=-\left(\frac{1}{N}\sum_i h_{i}(t)\right)+\sum_j \left(\frac{1}{N}\sum_i w_{ij}\right)x_j+\left(\frac{1}{N}\sum_i b_{i}\right) \\
    &=-m(t)+w_0\cdot\frac{1}{N}\sum_j x_j+b_0 \\
    &=-m(t)+w_0\cdot\int \mathcal{D}z \, \phi\left[\sqrt{q(t)}z+m(t)\right]+b_0\\
    &\equiv -m(t)+\mathcal{U}(m(t),q(t)|w_0,b_0)\\
    & \\
    \frac{1}{2}\partial_t q(t)& =\frac{1}{N}\sum_i [h_i(t)-m(t)][\partial_t h_i(t)-\partial_t m(t)]=\operatorname{Cov}[h_i(t),\partial_t h_i(t)]\\
    &=\sum_j x_j\operatorname{Cov}[h_i,w_{ij}] + \operatorname{Cov}[h_i,b_i]-q(t) \\
    &=\left[\sum_j x_j\left(\sum_i\tilde{x}_i \frac{\sigma_w^2}{N}\delta_{ij}+\sum_j\operatorname{Cov}[b_i,w_{ij}]\right)\right] \\
    &\quad\quad\quad+ \left[ \sum_{j} \tilde{x}_{j} \operatorname{Cov}[w_{ij}, b_i]+\operatorname{Cov}(b_i,b_i) \right]  -q(t) \\
    &=\sigma_w^2 \cdot \frac{1}{N}\sum_j x_j \tilde{x}_j + \sigma_b^2-q(t)\\
    &=-q(t)+\sigma_w^2 \int \mathcal{D}z \cdot \phi^2\left[\sqrt{q(t)}z+m(t)\right] +\sigma_b^2\\
    &\equiv -q(t)+\mathcal{V}(m(t),q(t)|\sigma_w^2,\sigma_b^2)
\end{aligned}
$$

不动点处有

$$
\left\{
\begin{aligned}
    m^*&=\mathcal{U}(m^*,q^*|w_0,b_0)=w_0\int \mathcal{D}z \cdot \phi(\sqrt{q^*}z+ m^*)+b_0\\
    q^*&=\mathcal{V}(m^*,q^*|\sigma_w^2,\sigma_b^2)=\sigma_w^2 \int \mathcal{D}z \cdot \phi^2(\sqrt{q^*}z+m^*) +\sigma_b^2
\end{aligned}
\right.
$$

这已经是后文**深度平均场理论**的主要方法之一。不同的是，深度平均场理论中，随机初始化一般会使 $w _ 0=b _ 0=0$，即将表达式简化为

$$
q^*=\sigma_w^2 \int \mathcal{D}z \cdot \phi^2(\sqrt{q^*}z)+\sigma_b^2
$$

这样自动满足了

$$
\left.\frac{\partial \mathcal{V}(q|\sigma_w^2,\sigma_b^2)}{\partial q}\right|_{q=q^*} <1
$$

已经解决了单一输入（没有复本）时**不动点稳定性**的问题，要解决的是**两输入的相关问题**([4.2.2](#422-两个输入与迭代相关映射-mathcalcc-_-12q-_-11q-_-12))。Amari 认为，网络的稳定性首先应由单一输入（没有复本）时，不动点的稳定性表示。

### 4.1.3 动态平均场方程推导

> 太长不看版：Amari 解已经能描述简单的动态性质，**深度平均场理论**也主要使用了类似她的推导方法，若本节太难可直接跳至 [4.2](#42-深度平均场理论假设与高斯过程视角)。**动态平均场理论**对时间相关的推导更严密一些，并且考虑了复本（不同输入）间的相关性随时间的演化。

方程中包含的随机性来源于随机的耦合常数 $J_{ij}$。**动态问题的统计性质**也就是随机的 $J_{ij}$ 下运行路径 $\mathbf{h}(t)$ 的概率分布。然而概率分布往往无法直接计算，只能通过近似方法计算它的**矩(moment**)。关于**生成泛函（路径积分）方法**的简介可以参考[^16]：

* **随机变量**概率分布的**矩**可以通过**矩生成函数(对应配分函数**)对**共轭变量**不断求导得到，
* 类似地，**随机过程**概率分布的**矩**可以通过**生成泛函**对**共轭变量**不断求导得到。

生成泛函需要对 $J_{ij}$ 及所有可能的**路径** $\mathbf{h}(t)$ 做**泛函积分**，也就是说 $D \mathbf{h}=D[\mathbf{h}(t)].$ 引入 $\hat{\mathbf{h}}(t)$ 通过 $\delta$ 函数的傅里叶变换以满足时间演化方程，引入共轭场 $\mathbf{l}(t)$ 和 $\hat{\mathbf{l}}(t)$，定义**生成泛函**：

$$
\begin{aligned}
    Z_{\mathbf{J}}[\mathbf{l}(t),\mathbf{\hat{l}}(t)] &
    =\int D \mathbf{h} D \mathbf{\hat{h}} \,\,p(\mathbf{h, \hat{h}})\exp \left\{\sum_{i, t}\left(l_{i}(t) h_{i}(t)+i \hat{l}_{i}(t) i \hat{h}_{i}(t)\right)\right\} \\
    & =\int D \mathbf{h} D \mathbf{\hat{h}} \exp \left\{\sum_{i, t}\left(l_{i}(t) h_{i}(t)+i \hat{l}_{i}(t) i \hat{h}_{i}(t)\right)+L[\mathbf{h, \hat{h}}]\right\} \\
\end{aligned}

$$

$$
L[h, \hat{h}]=\sum_{i, t}-i \hat{h}_{i}(t)\left[(1+\partial_{t}) h_{i}(t)-\sum_{j} J_{i j} s_{j}(t)\right], \,\,Z_{\mathbf{J}}[\mathbf{0},\mathbf{0}]=1
$$

它的作用相当于平衡态统计物理中的**配分函数**，包含了系统演化的所有信息。动态问题中最重要的矩是**时间关联函数**和**响应函数**：

$$ 
C(t, t^{\prime})=\left.\frac{\partial Z}{\partial \mathbf{\hat{l}}(t^{\prime}) \partial \mathbf{\hat{l}}(t)}\right|_{\mathbf{l}(t)=\mathbf{\hat{l}}(t)=0},\,\,\,
R(t, t^{\prime})=\left.\frac{\partial Z}{\partial \mathbf{\hat{l}}(t^{\prime})\partial \mathbf{l}(t)}\right|_{\mathbf{l}(t)=\mathbf{\hat{l}}(t)=0}
$$

### 4.1.4 有序—混沌相变

## 4.2 深度平均场：理论假设与高斯过程视角

回到对**前馈神经网络**的讨论。

* 网络共有 $D+1$ 层**神经元** $\mathbf{x}^0,...,\mathbf{x}^D$，第 $l$ 层的**宽度**为 $N_l$，
* $D$ 层**权重** $\mathbf{W}^1,...,\mathbf{W}^D$ 和**偏置** $\mathbf{b}^1,...,\mathbf{b}^D$。$\mathbf{x}^l, \mathbf{b}^l \in\mathbb{R}^{N_l},\mathbf{W}^l \in\mathbb{R}^{N_l\times N_{l-1}}.$
* 对于**随机初始化**的神经网络，$\mathbf{W} _ {ij}^l,\mathbf{b} _ {i}^l$ 为独立的零均值高斯随机变量，方差设定使得 $l-1$ 层神经元对 $l$ 层神经元场的贡献为 $\mathcal{O}(1)$，且选定后不再变化：

$$\mathbf{W}_{ij}^l \sim \mathcal{N}(0,\sigma_{w}^{2} / N_{l-1}),\quad\mathbf{b}_{i}^l \sim \mathcal{N}(0,\sigma_{b}^{2})
$$

* 前向传播的动力学为

$$
\mathbf{h}^{l}=\mathbf{W}^{l} \mathbf{x}^{l-1}+\mathbf{b}^{l},\quad
\mathbf{x}^{l}=\phi(\mathbf{h}^{l})
$$

**深度平均场理论**直接关注了前向传播中**序参量**的变化。我们前面提到，网络的不同输入类似于自旋玻璃的不同**复本(replicas)** ，网络的每一层的“**序参量**”为

$$ 
q_{aa}^{l}=\frac{1}{N_{l}} \sum_{i=1}^{N_{l}}(\mathbf{h}_{i}^{l})^{2},\quad q_{a b}^{l}=\frac{1}{N_{l}} \sum_{i=1}^{N_{l}} \mathbf{h}_{i}^{l}(\mathbf{x}^{0, a}) \mathbf{h}_{i}^{l}(\mathbf{x}^{0, b}) \quad a, b \in\{1,2\}
$$

### 4.2.1 单一输入与迭代长度映射 $\mathcal{V}(q_{aa})$

**平均场近似**认为，$N _ {l-1}$ 很大时，$\mathbf{h} _ {i}^{l}=\sum _ {j} \mathbf{W} _ {i j}^{l} \phi(\mathbf{h} _ {j}^{l-1})+\mathbf{b} _ {i}^{l}$ 是许多独立随机变量的和，由**中心极限定理**，服从**高斯分布**，可用对高斯随机变量 $z$ 的平均取代对 $N_{l-1}$ 个神经元的平均。方差随着前向传播而传递：

$$
\begin{aligned}
    q^{l}=\langle(\mathbf{h}_{i}^{l})^{2}\rangle &=\left\langle[\mathbf{W}_{ij}^{l} \cdot \phi(\mathbf{h}_{j}^{l-1})]^{2}\right\rangle+\langle(\mathbf{b}_{i}^{l})^{2}\rangle \\
    &=\sigma_{w}^{2} \frac{1}{N_{l-1}} \sum_{j=1}^{N_{l-1}} \phi(\mathbf{h}_{j}^{l-1})^{2}+\sigma_{b}^{2} \\
    &=\mathcal{V}(q^{l-1} | \sigma_{w}, \sigma_{b}) \\
    &\equiv \sigma_{w}^{2} \int \mathcal{D} z\,\, \phi\left(\sqrt{q^{l-1}} z\right)^{2}+\sigma_{b}^{2}
\end{aligned}
$$

$$ 
\mathcal{D} z=\frac{\mathrm{d} z}{\sqrt{2 \pi}} e^{-z^{2}/2},\quad
q^{0}=\frac{1}{N_{0}} \mathbf{x}^{0} \cdot \mathbf{x}^{0},\quad
q^{1}=\sigma_{w}^{2} q^{0}+\sigma_{b}^{2}
$$

函数 $\mathcal{V}(q)$ 为迭代的长度映射，对单调的非线性激活函数 $\phi$ 是**单调递增的凹函数**。$q^{l}=\mathcal{V}(q^{l-1}\vert\sigma _ w,\sigma _ b)$ 与对角线 $q^{l}=q^{l-1}$ 相交于不动点 $q^*(\sigma _ w,\sigma _ b).$ 不动点有以下几种情形：

* $\sigma _ b =0, \sigma _ w <1$ 时，唯一不动点为 $q^*=0$，前向传播中 $q$ 衰减为0.
* $\sigma _ b =0, \sigma _ w >1$ 时，$q^*=0$ 为不稳定不动点，同时有另一稳定不动点 $q^ * >0.$
* $\sigma _ b >0$ 时，总有稳定不动点 $q^ * >0.$

![deepmft_1-1](https://tablewarebox.files.wordpress.com/2019/04/deepmft_1-1.png)
<div align="center">图1  $\phi(h)=\tanh(h)$，宽度 $N_l=1000$ 的网络中 $q^l$ 的动力学。</div>

<div align="center">(A)(B) $\sigma_b=0.3,\sigma_w=1.3,2.5,4.0$ 时的迭代长度映射 $\mathcal{V}$ 和迭代中向不动点（五角星）的收敛情况</div>

<div align="center">(C)不动点 $q^*$ 作为 $\sigma _ w, \sigma _ b$ 的函数 (D)距不动点误差小于1%所需迭代次数。蓝绿红三点为(A)(B)图中三个样本的位置</div>

### 4.2.2 两个输入与迭代相关映射 $\mathcal{C}(c _ {12},q _ {11},q _ {12})$

当有两个输入 $\mathbf{x}^{0,1}, \mathbf{x}^{0,2}$ 时，$2\times2$ 的重叠矩阵

$$ 
q_{a b}^{l}=\frac{1}{N_{l}} \sum_{i=1}^{N_{l}} \mathbf{h}_{i}^{l}(\mathbf{x}^{0, a}) \mathbf{h}_{i}^{l}(\mathbf{x}^{0, b}) \quad a, b \in\{1,2\}
$$

随前向传播变化，**平均场近似**下，$N _ {l-1}$ 很大时，$\mathbf{h} _ {i}^{l}(\mathbf{x}^{0, a})$ 和 $\mathbf{h} _ {i}^{l}(\mathbf{x}^{0, b})$ 的联合分布是许多独立随机变量的和，由**中心极限定理**，服从协方差为 $q _ {ab}^l$ 的**二维高斯分布**，协方差矩阵随着前向传播而传递：

$$
\begin{aligned}
    q_{12}^{l}&=\mathcal{C}(c_{12}^{l-1}, q_{11}^{l-1}, q_{22}^{l-1} | \sigma_{w}, \sigma_{b}) \\
    &\equiv \sigma_{w}^{2} \int \mathcal{D} z_{1} \mathcal{D} z_{2} \phi(u_{1}) \phi(u_{2})+\sigma_{b}^{2}
\end{aligned}
$$

$$ 
u_{1}=\sqrt{q_{11}^{l-1}} z_{1}, \quad u_{2}=\sqrt{q_{22}^{l-1}}\left[c_{12}^{l-1} z_{1}+\sqrt{1-(c_{12}^{l-1})^{2}} z_{2}\right]
$$

$$ 
\langle u_{a} u_{b}\rangle= q_{a b}^{l-1},\quad c_{12}^{l}=\frac{q_{12}^{l}}{\sqrt{q_{11}^{l}q_{22}^{l}}}
$$

两个输入点在前向传播中的变化可以通过**相关系数** $c _ {12}^l$ 跟踪，$c _ {12}^l$ 在前向传播中逐渐收敛到**不动点** $c^ * (\sigma _ w,\sigma _ b).$ 由于 $q _ {11}, q _ {22}$ 迅速收敛到不动点 $q^ * (\sigma _ w,\sigma _ b)$，故可在相关系数的前向传播迭代中用 $q^ *$ 代替 $q _ {11}, q _ {22}$：

$$
c_{12}^{l}=\frac{1}{q^{*}} \mathcal{C}(c_{12}^{l-1}, q^{*}, q^{*} | \sigma_{w}, \sigma_{b})
$$

容易验证至少有一不动点 $c^ * (\sigma _ w,\sigma _ b)=1.$ 不动点的稳定性可通过计算函数 $\mathcal{C}$ 在 $c^ *$ 的斜率 $\chi _ 1$：

$$
\begin{aligned}
    \chi_{1} & \equiv\left.\frac{\partial c_{12}^{l}}{\partial c_{12}^{l-1}}\right|_{c=1} \\
    &=\left.\sigma_{w}^{2} \int \mathcal{D} z_{1} \mathcal{D} z_{2} \phi^{\prime}(u_{1}) \phi^{\prime}(u_{2})\right|_{c=1}\\
    & =\sigma_{w}^{2} \int \mathcal{D} z\left[\phi^{\prime}(\sqrt{q^{*} z})\right]^{2}
\end{aligned}
$$

推导利用了高斯随机变量的分部积分性质：

$$
\int \mathcal{D} z F(z) z=\int \mathcal{D} z F^{\prime}(z)
$$

* $\chi _ 1<1$，则函数 $\mathcal{C}$ 在对角线上方，$c^ * =1$ 为稳定不动点，两个输入在前向传播过程中**越来越相似**；
* $\chi _ 1>1$，则函数 $\mathcal{C}$ 在 $c=1$ 附近在对角线下方，$c^ * =1$ 为不稳定不动点，两个输入在前向传播过程中**逐渐分开**。

由此 $\chi _ 1$ 可以被理解为“伸缩系数”。定量计算方法是考虑已达不动点的 $\mathbf{h}^l$ 对 $\mathbf{h}^{l-1}$ 的**雅各比矩阵** $\mathbf{J} _ {i j}^{l}=\partial \mathbf{h} _ i^l / \partial \mathbf{h} _ j^{l-1}=\mathbf{W} _ {i j}^{l} \phi^{\prime}(\mathbf{h} _ {j}^{l-1})$，前向传播中不动点附近的微扰 $\mathbf{h}^{l-1}+\mathbf{u}$ 会变为 $\mathbf{h}^{l}+\mathbf{J} \mathbf{u}$. 微扰放大的倍数 $\lVert\mathbf{J u}\rVert _ {2}^{2} / \lVert\mathbf{u}\rVert _ {2}^{2}$ 对微扰 $\mathbf{u}$，随机矩阵 $\mathbf{W}$，和在 $i=1,...,N_l$ 近似高斯分布的 $\mathbf{h} _ i^l$ 平均后即为 $\chi _ 1$。运用**雅各比矩阵**的传递性质，可以推广为之后的**动态等距**概念。

![deepmft_2](https://tablewarebox.files.wordpress.com/2019/04/deepmft_2.png)
<div align="center">图2  $\phi(h)=\tanh(h)$，宽度 $N _ l=1000$ 的网络中 $q _ {ab}^l$ 的动力学。</div>

<div align="center">(A)(B) $\sigma_b=0.3,\sigma_w=1.3,2.5,4.0$ 时的迭代相关映射 $\mathcal{C}$ 和迭代中向不动点（五角星）的收敛情况</div>

<div align="center">(C)不动点 $c^*$ 作为 $\sigma _ w, \sigma _ b$ 的函数 (D)不动点导数 $\chi _ 1$ 作为 $\sigma _ w, \sigma _ b$ 的函数。蓝绿红三点为(A)(B)图中三个样本的位置</div>

### 4.2.3 有序—混沌相变

## 参考文献

[^1]: S. I. Amari. **Characteristics of Random Nets of Analog Neuron-Like Elements.** *IEEE Trans. Syst. Man Cybern.* **2**, 643 (1972).

[^2]: H. Sompolinsky, A. Crisanti, and H. J. Sommers. **Chaos in random neural networks.** *Physical Review Letters*, 61(3): 259, 1988.

[^3]: Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S. Schoenholz, Jeffrey Pennington, and Jascha Sohl-Dickstein. **Deep neural networks as Gaussian processes.** *International Conference on Learning Representations*, 2018.

[^4]: Ben Poole, Subhaneil Lahiri, Maithreyi Raghu, Jascha Sohl-Dickstein, and Surya Ganguli. **Exponential expressivity in deep neural networks through transient chaos.** In *Advances In Neural Information Processing Systems*, pages 3360–3368, 2016.

[^5]: Samuel S. Schoenholz, Justin Gilmer, Surya Ganguli, and Jascha Sohl-Dickstein. **Deep Information Propagation.** *International Conference on Learning Representations*, 2017.

[^6]: Jeffrey Pennington, Samuel S. Schoenholz, and Surya Ganguli. **Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice.** *Advances in Neural Information Processing Systems*, 2017.

[^7]: Jeffrey Pennington, Samuel S. Schoenholz, and Surya Ganguli. **The emergence of spectral universality in deep networks.** *International Conference on Artificial Intelligence and Statistics (AISTATS)*, 2018.

[^8]: Greg Yang, and Samuel S. Schoenholz. **Mean field residual networks: On the edge of chaos.** In *Advances in Neural Information Processing Systems*, 2017.

[^9]: Greg Yang and Samuel S. Schoenholz. **Deep mean field theory: Layerwise variance and width variation as methods to control gradient explosion.** *International Conference on Learning Representations*, 2018.

[^10]: Greg Yang, Jeffrey Pennington, Vinay Rao, Jascha Sohl-Dickstein, and Samuel S. Schoenholz. **A mean field theory of batch normalization.** *International Conference on Learning Representations*, 2019.

[^11]: Lechao Xiao, Yasaman Bahri, Jascha Sohl-Dickstein, Samuel S. Schoenholz, and Jeffrey Pennington. **Dynamical isometry and a mean field theory of CNNs: How to train 10,000-layer vanilla convolutional neural networks.** *International Conference on Learning Representations*, 2018.

[^12]: Minmin Chen, Jeffrey Pennington, and Samuel S. Schoenholz. **Dynamical isometry and a mean field theory of RNNs: Gating enables signal propagation in recurrent neural networks.** *International Conference on Learning Representations*, 2018.

[^13]: Dar Gilboa, Bo Chang, Minmin Chen, Greg Yang, Samuel S. Schoenholz, Ed H. Chi, and Jeffrey Pennington. **Dynamical isometry and a mean field theory of LSTMs and GRUs.** *arXiv preprint arXiv:1901.08987*, 2019.

[^14]: Tatsuro Kawamoto, and Masashi Tsubaki. **Mean-field theory of graph neural networks in graph partitioning.** *arXiv preprint arXiv:1810.11908*, 2018.

[^15]: Tommaso Castellani, and Andrea Cavagna. **Spin-glass theory for pedestrians.** *J. Stat. Phys.* (2005) P05012. DOI: 10.1088/1742-5468/2005/05/P05012.

[^16]: Chow C, Buice M (2015) **Path integral methods for stochastic differential equations.** *The Journal of Mathematical Neuroscience* 5(1):8, DOI: 10.1186/s13408-015-0018-5.