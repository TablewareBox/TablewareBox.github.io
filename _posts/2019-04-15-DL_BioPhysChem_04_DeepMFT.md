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

**深度平均场理论**是近年来 Google Brain 研究人员提出的，用于解释深度神经网络的**表达能力、训练技巧**和**模型架构**的理论框架，其根源可追溯到日本学者**甘利俊(Shun-ichi Amari)** 上世纪70年代提出的**统计神经动力学**[^1]。在80年代 **Hopfield 网络**提出之后，H. Sompolinsky, A. Crisanti and H. J. Sommers 等人将其发展为**自旋玻璃模型**所衍生神经网络的**动态平均场理论**[^2]，研究网络运行时一般性的动力学性质，并描述了**有序-混沌相变**。S. S. Schoenholz, S. Ganguli, J. Pennington 和 J. Sohl-Dickstein 等人重新以现代神经网络结构进行了推导，他们的一系列理论工作已能一定程度上为网络结构和训练技巧的设计提供指导。

笔记的这一章节架构如下：

- [ ] 动态平均场理论回顾[^2]
- [x] 深度平均场：理论假设与高斯过程视角[^3]
- [ ] 深度网络的指数级表达能力[^4]
- [ ] 深度信息传递：训练中的有序-混沌相变[^5]
- [ ] 训练技巧 I: 残差网络的运行原理[^6]
- [ ] 训练技巧 II: 层宽度变化的运行原理[^7]
- [ ] 训练技巧 III: 批归一化的运行原理[^8]
- [ ] CNN 的平均场理论[^9]
- [ ] RNN, LSTM, GRU 的平均场理论[^10][^11]
- [ ] 图网络的平均场理论[^12]

## 深度平均场：理论假设与高斯过程视角



## 参考文献

[^1]: S. I. Amari. **Characteristics of Random Nets of Analog Neuron-Like Elements.** *IEEE Trans. Syst. Man Cybern.* **2**, 643 (1972).

[^2]: H. Sompolinsky, A. Crisanti, and H. J. Sommers. **Chaos in random neural networks.** *Physical Review Letters*, 61(3): 259, 1988.

[^3]: Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S. Schoenholz,
Jeffrey Pennington, and Jascha Sohl-Dickstein. **Deep neural networks as Gaussian processes.** *International Conference on Learning Representations*, 2018.

[^4]: Ben Poole, Subhaneil Lahiri, Maithreyi Raghu, Jascha Sohl-Dickstein, and Surya Ganguli. **Exponential expressivity in deep neural networks through transient chaos.** In *Advances In Neural Information Processing Systems*, pages 3360–3368, 2016.

[^5]: Samuel S. Schoenholz, Justin Gilmer, Surya Ganguli, and Jascha Sohl-Dickstein. **Deep Information Propagation.** *International
Conference on Learning Representations*, 2017.

[^6]: Greg Yang, and Samuel S. Schoenholz. **Mean field residual networks: On the edge of chaos.** In *Advances in Neural Information Processing Systems*, 2017.

[^7]: Greg Yang and Samuel S. Schoenholz. **Deep mean field theory: Layerwise variance and width variation as methods to control gradient explosion.** *International Conference on Learning Representations*, 2018.

[^8]: Greg Yang, Jeffrey Pennington, Vinay Rao, Jascha Sohl-Dickstein, and Samuel S. Schoenholz. **A mean field theory of batch normalization.** *International Conference on Learning Representations*, 2019.

[^9]: Lechao Xiao, Yasaman Bahri, Jascha Sohl-Dickstein, Samuel S. Schoenholz, and Jeffrey Pennington. **Dynamical isometry and a mean field theory of CNNs: How to train 10,000-layer vanilla convolutional neural networks.** *International Conference on Learning Representations*, 2018.

[^10]: Minmin Chen, Jeffrey Pennington, and Samuel S. Schoenholz. **Dynamical isometry and a mean field theory of RNNs: Gating enables signal propagation in recurrent neural networks.** *International Conference on Learning Representations*, 2018.

[^11]: Dar Gilboa, Bo Chang, Minmin Chen, Greg Yang, Samuel S. Schoenholz, Ed H. Chi, and Jeffrey Pennington. **Dynamical isometry and a mean field theory of LSTMs and GRUs.** *arXiv preprint arXiv:1901.08987*, 2019.

[^12]: Tatsuro Kawamoto, and Masashi Tsubaki. **Mean-field theory of graph neural networks in graph partitioning.** *arXiv preprint arXiv:1810.11908*, 2018.