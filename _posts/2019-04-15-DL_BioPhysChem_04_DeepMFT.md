---
layout:     post
title:      第4章 深度平均场理论与统计神经动力学
subtitle:   
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

## 引言

> More is different. ——P. W. Anderson，1977年诺贝尔物理学奖得主

**深度平均场理论**是近年来 Google Brain 研究人员提出的，用于解释深度神经网络的**表达能力、训练技巧**和**模型架构**的理论框架，其根源可追溯到日本学者**甘利俊(Shun-ichi Amari)** 上世纪70年代提出的**统计神经动力学**^[S. I. Amari. Characteristics of Random Nets of Analog Neuron-Like Elements. In *IEEE Trans. Syst. Man Cybern.* **2**, 643 (1972).]。在80年代 **Hopfield 网络**提出之后，H. Sompolinsky, A. Crisanti and H. J. Sommers 等人将其发展为**自旋玻璃模型**所衍生神经网络的**动态平均场理论**^[2]，研究网络运行时一般性的动力学性质，并描述了**有序-混沌相变**。

![knowledge atlas](https://tablewarebox.files.wordpress.com/2018/11/concept-map-81.png)
<div align="center">图2 深度学习与物理化学</div>

## 目录预告

### 引言：复杂无序系统的崎岖势能面和多尺度现象

* 蛋白质折叠、生物进化、结构玻璃和自旋玻璃
* 信息论与编码、压缩感知、组合优化、神经网络

### 深度学习简介

* 物理视角下的深度学习历史回顾
* 有监督学习、深度神经网络简介
  * 计算图
  * 反向传播算法
  * 正则化方法
  * 卷积、循环和图神经网络
* 基于能量的模型
* 概率图模型简介
  * 有向图模型（贝叶斯网络）
  * 无向图模型（马尔可夫随机场）
  * 因子图
  * 图模型中的推断
  * 概率图模型与深度神经网络的关系
* 无监督学习和生成模型简介
  * 联想记忆：Hopfield 网络
  * 玻尔兹曼机
  * 变分自编码器（VAE）
  * 生成对抗网络（GAN）
  * 深度生成模型的概率图模型和能量视角

### 自旋玻璃与消息传递：Hopfield 网络到 RBM

* Hopfield 网络与自旋玻璃模型
* 热力学（静态）性质
  * 平均场、变分法和微扰法
  * 复本对称平均场解与信念传播：Hopfield 网络存储容量相变
  * 一阶复本对称破缺（1RSB）与概观传播
* 动力学（动态）性质
  * Langevin 方程
  * 响应函数、关联函数、涨落-耗散定理
  * 生成泛函（路径积分）方法
  * 动力学相变
* 深度前馈网络的自旋玻璃建模
* 生物大分子和进化动力学的自旋玻璃建模

### 深度平均场理论与统计神经动力学

* 动力学性质回顾与基本假设
* 理解 DNN 的指数级表达能力和损失曲面
* 理解训练过程：有序-混沌相变
* 理解正则化和训练方法：Batch Normalization 和残差网络
* 理解 CNN, RNN 和 GNN

### 重正化群、临界现象和信息瓶颈理论

* RBM 与变分重正化群的对应关系
* 信息瓶颈原理与重正化群

### 高斯过程与贝叶斯神经网络

* DNN 作为高斯过程
* 贝叶斯神经网络
* 扩散概率模型与非平衡统计力学

## 参考文献