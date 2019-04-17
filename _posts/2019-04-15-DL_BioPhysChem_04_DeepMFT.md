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

**深度平均场理论**是近年来 Google Brain 研究人员提出的，用于解释深度神经网络的**表达能力、训练技巧**和**模型架构**的理论框架，其根源可追溯到日本学者**甘利俊一(Shun-ichi Amari)** 上世纪70年代提出的**统计神经动力学**[^1]。在80年代 **Hopfield 网络**提出之后，H. Sompolinsky, A. Crisanti 和 H. J. Sommers 等人将其发展为**自旋玻璃模型**所衍生**无向神经网络**的**动态平均场理论**[^2]，研究网络运行时一般性的动力学性质，并描述了**有序-混沌相变**。由于无向神经网络的**时间演化、动态问题**类似于有向神经网络的**层间传递、深度问题**，S. S. Schoenholz, S. Ganguli, J. Pennington 和 J. Sohl-Dickstein 等人重新以现代的**有向（前馈）神经网络**结构进行了推导，他们的一系列理论工作已能一定程度上为网络结构和训练技巧的设计提供指导。

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

甘利俊一(Shun-ichi Amari) 建立的模型中，$N$ 个**神经元**由连续变量 $\{s_i(t)\in[-1,1]\},\,\,i=1,...,N$ 描述（对应自旋），“**突触矩阵**” $\mathbf{J}$ （对应自旋的耦合常数）表达它们的相互作用。每一时刻 $t$ 神经元的状态 $s_i(t)$ 由**场 $h_i(t)$** 决定：

$$s_{i}(t)=\phi(g h_{i}(t))$$

其中 $\phi(x)$ 为**非线性激活函数**，可选为任意 S型函数，如 $\phi(x)=\tanh (x)$，需满足 $\phi(\pm\infty)=\pm 1, \phi(-x)=-\phi(x), \phi'(x)>0.$ 

$g$ 为**非线性指数**，可以类比 $\beta=1/k_\mathrm{B}T$：$g\sim 0, T\to\infty$ 时 $\phi(x)\sim x$；$g\to \infty,T\to 0$ 时 $\phi(x)\to \pm 1.$

时间演化的**动力学方程**为

$$
\partial_{t} h_{i}=-h_{i}+\sum_{j=1}^{N} J_{i j} s_{j}+\theta_i=-h_{i}+\sum_{j=1}^{N} J_{i j} \phi(g h_{j})+\theta_i
$$

若将其移项，可以看出它和前馈神经网络的关系：

$$
\begin{cases}
    h_{i}+\partial_{t} h_{i}=\sum_{j=1}^{N} J_{i j} s_{j}+\theta_i \\
    s_i=\phi(g h_{i})
\end{cases}
\longleftrightarrow
\begin{cases}
    \mathbf{h}^{l+1}=\mathbf{W}^{l+1} \mathbf{x}^{l}+\mathbf{b}^{l+1} \\
    \mathbf{x}^{l}=\phi(\mathbf{h}^{l})
\end{cases}
$$

实际上这也是**无向神经网络**与**有向神经网络**的差别：无向神经网络的**时间演化、动态问题**类似于有向神经网络的**层间传递、深度问题**。从这一点看自旋玻璃类**无向神经网络**可以理解成**权重相同的无限层有向神经网络**。

### 4.1.2 方程推导

方程中包含的随机性来源于随机的耦合常数 $J_{ij}$。**动态问题的统计性质**也就是随机的 $J_{ij}$ 下运行路径 $\mathbf{h}(t)$ 的概率分布。然而概率分布往往无法直接计算，只能通过近似方法计算它的**矩(moment**)。**随机变量**概率分布的**矩**可以通过**矩生成函数(对应配分函数**)对**共轭变量**不断求导得到，类似地，**随机过程**概率分布的**矩**可以通过**生成泛函**对**共轭变量**不断求导得到。关于**生成泛函（路径积分）方法**的简介可以参考[^15]。

生成泛函需要对 $J_{ij}$ 及所有可能的**路径** $\mathbf{h}(t)$ 做**泛函积分**，也就是说 $D \mathbf{h}=D[\mathbf{h}(t)].$ 引入 $\hat{\mathbf{h}}(t)$ 以满足时间演化方程，引入共轭场 $\mathbf{l}(t)$ 和 $\hat{\mathbf{l}}(t)$，定义**生成泛函**：

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

### 4.1.3 有序—混沌相变

## 4.2 深度平均场：理论假设与高斯过程视角

回到对**前馈神经网络**的讨论。

* 网络共有 $D+1$ 层**神经元** $\mathbf{x}^0,...,\mathbf{x}^D$，第 $l$ 层的**宽度**为 $N_l$，
* $D$ 层**权重** $\mathbf{W}^1,...,\mathbf{W}^D$ 和**偏置** $\mathbf{b}^1,...,\mathbf{b}^D$。$\mathbf{x}^l, \mathbf{b}^l \in\mathbb{R}^{N_l},\mathbf{W}^l \in\mathbb{R}^{N_l\times N_{l-1}}.$
* 对于**随机初始化**的神经网络，$\mathbf{W} _ {ij}^l,\mathbf{b} _ {i}^l$ 为独立的零均值高斯随机变量，方差设定使得 $l-1$ 层神经元对 $l$ 层神经元场的贡献为 $\mathcal{O}(1)$：

$$\mathbf{W}_{ij}^l \sim \mathcal{N}(0,\sigma_{w}^{2} / N_{l-1}),\,\,\,\,\,\mathbf{b}_{i}^l \sim \mathcal{N}(0,\sigma_{b}^{2})
$$

* 前向传播的动力学为

$$ 
\mathbf{h}^{l}=\mathbf{W}^{l} \mathbf{x}^{l-1}+\mathbf{b}^{l},\,\,\,\,\,
\mathbf{x}^{l}=\phi(\mathbf{h}^{l})
 $$

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

[^15]: Chow C, Buice M (2015) **Path integral methods for stochastic differential equations.** *The Journal of Mathematical Neuroscience* 5(1):8, DOI: 10.1186/s13408-015-0018-5