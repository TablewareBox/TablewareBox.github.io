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

而 **Hopfield 联想记忆神经网络**的核心思想同样是，将目标数据存储在神经元间的**相互作用强度(权重)** 中，让它成为总能量的极小值点。这样通过类似的数据 **“唤醒”记忆**，通过一步步修正，**联想**出目标数据的过程，就对应着势能面上的**能量极小化**过程。[^5]

![hopfield](https://tablewarebox.files.wordpress.com/2019/04/intro_1_hopfield.png)

<div align="center">图1  25×25的 Hopfield 网络权重矩阵中存储了4个25像素的<b>目标数据(模式)</b>。网络运行的时间演化中，任意的输入都会向目标数据(模式)靠近</div>

蛋白折叠更多地表现出**漏斗状势能面**，而在不同种类的自旋玻璃模型中，往往有更多能量相近的**亚稳态**，

![rugged](https://tablewarebox.files.wordpress.com/2019/04/intro_2_rugged.png)
<div align="center">图2  25×25的 Hopfield 网络权重矩阵中存储了4个25像素的<b>目标数据(模式)</b>。网络运行的时间演化中，任意的输入都会向目标数据(模式)靠近</div>

## 参考文献

[^1]: Joseph D. Bryngelson, and Peter G. Wolynes. **Spin glasses and the statistical mechanics of protein folding.** *Proc. Nati. Acad. Sci. USA* **1987**, *84(21)*, 7524-7528. DOI: 10.1073/pnas.84.21.7524

[^2]: Joseph D. Bryngelson, and Peter G. Wolynes. **Intermediates and barrier crossing in a random energy model (with applications to protein folding).** *J. Phys. Chem.* **1989**, *93*, 6902-6915. DOI: 10.1021/j100356a007

[^3]: Richard A. Goldstein, Zaida A. Luthey-Schulten, and Peter G. Wolynes. **Optimal protein-folding codes from spin-glass theory.** *Proc. Nati. Acad. Sci. USA* **1992**, *89*, 4918-4922. DOI: 10.1073/pnas.89.11.4918

[^4]: J. J. Hopfield. **Neural networks and physical systems with emergent collective computational abilities.** *Proc. Nati. Acad. Sci. USA* **1982**, *79*, 2554-2558. DOI: 10.1073/pnas.79.8.2554

[^5]: David J.C. MacKay. **Information Theory, Inference, and Learning Algorithms (ITILA).**