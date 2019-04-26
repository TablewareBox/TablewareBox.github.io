---
layout:     post
title:      引言 | 复杂无序系统的崎岖势能面和多尺度现象
subtitle:   生物大分子、深度前馈网络的自旋玻璃建模
date:       2019-04-24
author:     TablewareBox
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - DL & BioPhysChem
    - 深度学习
    - 生物物理化学
    - 统计力学
    - 无序系统
---

## [深度学习的生物物理化学原理 - Notes Project Overview](https://tablewarebox.github.io/2019/02/16/DL_BioPhysChem_content/)

![knowledge atlas](https://tablewarebox.files.wordpress.com/2018/11/concept-map-81.png)

深度学习和蛋白质、染色质的折叠和相分离有什么共同的底层结构，又有什么不同？

让我们从蛋白质折叠的一个**早期模型**[^1][^2][^3]，以及那时的神经网络第二次浪潮——**Hopfield联想记忆神经网络**[^4]谈起。



而 **Hopfield 联想记忆神经网络**的核心思想同样是，将目标数据存储在神经元间的**相互作用强度(权重)** 中，让它成为总能量的极小值点。这样通过类似的数据 **“唤醒”记忆**，通过神经元状态的**时间演化**一步步修正，**联想**出目标数据的过程，就对应着势能面上的**能量极小化**过程[^5]。

![hopfield](https://tablewarebox.files.wordpress.com/2019/04/intro_1_hopfield.png)

<div align="center">图1  25×25的 Hopfield 网络权重矩阵中存储了4个25像素的<b>目标数据(模式)</b>。网络运行的时间演化中，任意的输入都会向目标数据(模式)靠近</div>

蛋白折叠更多地表现出**漏斗状势能面**[^6]，或是有不同构象有不同功能的蛋白质表现为 **“多漏斗”势能面**[^7]；而在不同种类的自旋玻璃模型中，往往有更多能量相近的**亚稳态**，并不一定有明显的“漏斗”。[^8]

![rugged](https://tablewarebox.files.wordpress.com/2019/04/intro_2_rugged.png)
<div align="center">图2  典型的<b>粗粒化蛋白质</b>和<b>自旋玻璃模型</b>势能面。</div>

![multifunnel](https://tablewarebox.files.wordpress.com/2019/04/intro_3_multifunnel.png)
<div align="center">图3  某多功能泛素蛋白的多漏斗势能面，用不连通图(disconnectivity graph)表示。</div>

现在我们回望整个发展历程，会发现**自旋玻璃模型**，不仅完全精确地描述了早期的**无向神经网络**结构和性质，同时是一个简单描述基本的相互作用的**粗粒化的物理模型**，又是一个原始的**统计势函数**——给许多已知的结构赋予更低的能量（更高的**打分**），从而统计性地预测未知分子的结构。

在今天，早期的**自旋玻璃类无向神经网络**（笔记第3章），现代结构中已经演变为拥有不同架构的**有向神经网络**，但是基本的组成——拥有不同状态的**神经元**、神经元间的传递**权重和偏置**、引发非线性和概率性质的**激活函数**没有变，从而可以和早期神经网络找到对应：无向神经网络的**时间演化、动态问题**类似于有向神经网络的**层间传递、深度问题**。从而我们仍可以用发展成熟的自旋玻璃**静态、动态**理论，研究深度神经网络前向传播、反向传播中的问题（[笔记第4章](https://tablewarebox.github.io/2019/04/15/DL_BioPhysChem_04_DeepMFT/)）；或者直接建模网络的**损失曲面**[^9]，通过曲面上极小值的分布，给出为什么深度网络**较易训练**，梯度下降收敛到**局部极小值**也接近**全局最小值**（回忆上图自旋玻璃模型的势能面结构），同时还有一定**泛化能力**的原因。此外，随机性更强的**概率图模型**和**贝叶斯神经网络**中，统计力学的理论和算法也发挥着更大的作用（笔记第3章、第6章）。



## 参考文献

[^1]: Joseph D. Bryngelson, and Peter G. Wolynes. **Spin glasses and the statistical mechanics of protein folding.** *Proc. Nati. Acad. Sci. USA* **1987**, *84(21)*, 7524-7528. DOI: 10.1073/pnas.84.21.7524

[^2]: Joseph D. Bryngelson, and Peter G. Wolynes. **Intermediates and barrier crossing in a random energy model (with applications to protein folding).** *J. Phys. Chem.* **1989**, *93*, 6902-6915. DOI: 10.1021/j100356a007

[^3]: Richard A. Goldstein, Zaida A. Luthey-Schulten, and Peter G. Wolynes. **Optimal protein-folding codes from spin-glass theory.** *Proc. Nati. Acad. Sci. USA* **1992**, *89*, 4918-4922. DOI: 10.1073/pnas.89.11.4918

[^4]: J. J. Hopfield. **Neural networks and physical systems with emergent collective computational abilities.** *Proc. Nati. Acad. Sci. USA* **1982**, *79*, 2554-2558. DOI: 10.1073/pnas.79.8.2554

[^5]: David J.C. MacKay. **Information Theory, Inference, and Learning Algorithms (ITILA).**

[^6]: José Nelson Onuchic, Zaida Luthey-Schulten and Peter G. Wolynes. **Theory of protein folding: the energy landscape perspective.** *Annu. Rev. Phys. Chem.*  **1997**, *48*, 545–600. DOI: 10.1146/annurev.physchem.48.1.545

[^7]: Konstantin Röder, Jerelle A. Joseph, Brooke E. Husic, and David J. Wales. **Energy Landscapes for Proteins: From Single Funnels to Multifunctional Systems.** *Adv. Theory Simul.* **2019**, 1800175. DOI: 10.1002/adts.201800175

[^8]: Wolfhard Janke (Ed.) **Rugged Free Energy Landscapes: Common Computational Approaches to Spin Glasses, Structural Glasses and Biological Macromolecules.** *Lect. Notes Phys.* 736 (Springer, Berlin Heidelberg 2008), DOI: 10.1007/978-3-540-74029-2

[^9]: Choromanska, Anna, Henaff, Mikael, Mathieu, Michaël, Arous, Gérard Ben, and LeCun, Yann. **The loss surface of multilayer networks.** *JMLR*, 38, 2015.