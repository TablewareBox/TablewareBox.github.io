---
layout:     post
title:      朴素平均场到 TAP 方程
subtitle:   《Advanced Mean Field Theory》读书笔记 第2章
date:       2019-04-15
author:     TablewareBox
header-img: img/post-bg-ios9-web.jpg
catalog: true
mathjax: true
tags:
    - Adv_MFT
    - 概率图模型
    - 统计力学
    - 无序系统
---

## 2.1 引言

平均场方法通过保留随机变量间的相互作用但忽略特定的相关性，从而近似计算高维分布的求和或积分。本章介绍三种方法：

* **变分平均场（朴素平均场）**：采用**变量分离近似**，变分求解
* **场论方法**：引入辅助变量，在复平面用**鞍点近似**积分求解
* **Thouless-Anderson-Palmer (TAP) 方法**，从**空穴法**和 **Plefka 展开**两个角度理解

> 以下以 **Ising 模型**（对应机器学习中的 **Boltzmann 机**）为例。

* 随机向量 $\mathbf{s}=(s_1,\ldots,s_N),s_i=\pm 1$ 表示自旋。
* 哈密顿量

$$H(\mathbf{s})=-\sum_{i<j}s_i J_{ij} s_j - \sum_i \theta_i s_i=-\frac{1}{2} \mathbf{s}^T \mathbf{J} \mathbf{s}-\theta^T \mathbf{s}$$

* 配分函数

$$Z=\sum_\mathbf{s} e^{-\beta H(\mathbf{s})}$$

* 真实分布

$$p(\mathbf{s})=\frac{e^{-\beta H(\mathbf{s})}}{Z}$$

## 2.2 变分平均场（朴素平均场）

变分方法以分布 $q(\mathbf{s})$ 近似真实分布 $p(\mathbf{s})$。$q(\mathbf{s})$ 通常从一族易处理的分布中选取，可用 **KL散度** 衡量它与真实分布的距离：

$$ KL(q||p)=\sum_\mathbf{s} q(\mathbf{s}) \ln\frac{q(\mathbf{s})}{p(\mathbf{s})}=\mathbb{E}_q\left[\ln \frac{q}{p}\right]
$$

将 $p(\mathbf{s})$ 表达式代入可得

$$ KL(q||p)=\ln Z+\beta E(q)-S(q)=\ln Z+\beta F(q)
$$

其中：

* **变分能量**

$$E(q)=\mathbb{E}_q [H(\mathbf{s})]=\sum_\mathbf{s} q(\mathbf{s})H(\mathbf{s})$$

* **分布 $q$ 的熵**

$$S(q)=-\mathbb{E}_q [\ln q(\mathbf{s})]=-\sum_\mathbf{s} q(\mathbf{s})\ln q(\mathbf{s})$$

* **变分自由能**

$$F(q)=E(q)-\frac{1}{\beta}S(q)=\frac{1}{\beta}\left[KL(q||p)-\ln Z\right]\geq -\frac{1}{\beta}\ln Z=F(p), \,\beta=\frac{1}{k_\mathrm{B}T}$$

等号仅当 $q(\mathbf{s})=p(\mathbf{s})$ 时成立。故 $F(q)$ 取最小值时近似最好，可通过变分实现：

$$\frac{\delta F(q)}{\delta q}=0
$$

> 这一方法在机器学习的**变分推断(Variational Inference)** 中被广泛用于对目标分布的近似，**变分自由能**在机器学习中被称为**变分下界(Variational Lower Bound, V-LB)** 或**证据下界(Evidence Lower Bound, ELBO)**。

$$ 
\begin{aligned} \log Z=\log p(\mathbf{x}) &=\log \int_{\mathbf{h}} p(\mathbf{x,h}) \\ 
&=\log \int_{\mathbf{h}} p(\mathbf{x,h}) \frac{q(\mathbf{h})}{q(\mathbf{h})} \\ 
&=\log \left(\mathbb{E}_{q}\left[\frac{p(\mathbf{x,h})}{q(\mathbf{h})}\right]\right) \\ 
& \geq \mathbb{E}_{q}[\log p(\mathbf{x,h})]-\mathbb{E}_{q}[\log q(\mathbf{h})] \\
&=-\mathbb{E}_{q}[H(\mathbf{x,h})]+S(q)=-E(q)+S(q)\end{aligned}
$$

一种最简单的 $q$ 选法是忽略不同自旋间的统计相关性，即**朴素平均场**近似：

$$q(\mathbf{s})=q(s_1,\ldots,s_N)=\prod_i q_i(s_i) \\
\mathbb{E}_q[s_i s_j] = \mathbb{E}_q[s_i]\cdot \mathbb{E}_q[s_j]=m_i m_j
$$

其中 $q_i(s_i)$ 随 $s_i$ 的期望 $m_i = \mathbb{E}_q [s_i]$ 在$[0,1]$ 间线性变化：

$$q_i(s_i;m_i) = \frac{1+s_i m_i}{2}
$$

将 $q(\mathbf{s})$ 的表达式代入得

$$ E(q) = -\sum_{i<j}m_i J_{ij} m_j -\sum_i \theta_i m_i \\
S(q) = -\sum_i \left( \frac{1+m_i}{2}\ln\frac{1+m_i}{2} + \frac{1-m_i}{2}\ln\frac{1-m_i}{2} \right)
$$

由变分条件最终推出 $N$ 个方程组成的**自洽场方程组**：

$$\frac{\delta F(q)}{\delta q}=0 \,\,\,\Longrightarrow\,\,\,\frac{\partial F(q(\mathbf{m}))}{\partial m_i}=0 \,\,\,\Longrightarrow\,\,\,m_i=\tanh\left( \beta\left[\sum_j J_{ij}m_j+\theta_j\right] \right), \,\,(i,j=1,...,N)
$$

方程组可在 $N$ 的多项式时间内得出数值解。

也可以求出实际分布时 $s_i$ 的期望遵循的方程组（以下 $\mathbf{s}\backslash s_i$ 表示除 $s_i$ 外 $N-1$ 个自旋）：

$$\begin{aligned}
    \mathbb{E}_p [s_i] & = \dfrac{1}{Z}\cdot \sum_\mathbf{s}s_i e^{-\beta H(\mathbf{s})}\\
    & = \dfrac{1}{Z}\cdot \sum_{\mathbf{s}\backslash s_i}
    \sum_{s_i} s_i e^{-\beta H(s_i,\mathbf{s}\backslash s_i)}\\
    & = \dfrac{1}{Z}\cdot \sum_{\mathbf{s}\backslash s_i}
    \sum_{s_i} e^{-\beta H(s_i,\mathbf{s}\backslash s_i)}
    \cdot \dfrac{\sum_{\sigma_i} \sigma_i e^{-\beta H(\sigma_i,\mathbf{s}\backslash s_i)}}
    {\sum_{\sigma_i} e^{-\beta H(\sigma_i,\mathbf{s}\backslash s_i)}}\\
    & = \dfrac{1}{Z}\cdot \sum_\mathbf{s} e^{-\beta H(\mathbf{s})}
    \cdot \tanh\left( \beta\left[\sum_j J_{ij}s_j+\theta_j\right] \right)\\
    & = \mathbb{E}_p\left[\tanh\left( \beta\left[\sum_j J_{ij}s_j+\theta_i\right] \right)\right]
    =\mathbb{E}_p\left[\tanh\left( \beta h_i\right)\right] \\
    & \\
    \mathbb{E}_q [s_i] & = \tanh\left( \beta\left[\sum_j J_{ij}\mathbb{E}_q[s_j]+\theta_i\right] \right)
    = \tanh\left( \beta \mathbb{E}_q[h_i]\right)\\
\end{aligned}
$$

可以看出平均场近似将**涨落的场** $h_i=\sum_j J_{ij}s_j+\theta_i$ 用“**平均场**” $\mathbb{E}_q[h_i]=\sum_jJ_{ij}\mathbb{E}_q[s_j]+\theta_i$ 代替了。

## 2.3 线性响应校正

虽然变量分离的 $q(\mathbf{s})$ 完全忽略了随机变量间的相关性，但仍可近似计算它们的协方差：

$$\mathbb{E}_p [s_i] = \frac{1}{Z}\sum_\mathbf{s}s_i e^{-\beta H(\mathbf{s})}
$$

两侧对 $\theta_j$ 求导得

$$\begin{aligned}
    \dfrac{\partial \mathbb{E}_p [s_i]}{\partial \theta_j} & = -\dfrac{1}{Z^2}\cdot \sum_\mathbf{s}s_i e^{-\beta H(\mathbf{s})}\cdot \dfrac{\partial Z}{\partial \theta_j}
    + \dfrac{1}{Z}\cdot \sum_\mathbf{s}s_i \dfrac{\partial e^{-\beta H(\mathbf{s})}}{\partial \theta_j}\\
    & = \beta\cdot\left\{
    -\dfrac{1}{Z^2}\cdot \sum_\mathbf{s}s_i e^{-\beta H(\mathbf{s})}\cdot \sum_\mathbf{s'}s_j' e^{-\beta H(\mathbf{s'})}
    + \dfrac{1}{Z}\cdot \sum_\mathbf{s}s_i s_j e^{-\beta H(\mathbf{s})}
    \right\}\\
    & =\beta\cdot\left\{ -\mathbb{E}_p[s_i]\cdot\mathbb{E}_p[s_j] + \mathbb{E}_p[s_i s_j] \right\}
\end{aligned}
$$

若平均场近似足够合理，则可近似得到协方差：

$$
\mathbb{E}_p[s_i s_j] -\mathbb{E}_p[s_i]\cdot\mathbb{E}_p[s_j] =
\dfrac{1}{\beta}\dfrac{\partial \mathbb{E}_p [s_i]}{\partial \theta_j} \approx \dfrac{1}{\beta}\dfrac{\partial \mathbb{E}_q [s_i]}{\partial \theta_j}
$$

> 这种对协方差的近似已用于 **Boltzmann 机学习** 和 **ICA（独立成分分析）**。

## 2.4 场论方法

另一个思路是，**积分的近似**比**求和的近似**更容易计算准确。可将对 $s_i$ 求期望转变为对**辅助变量**求积分，再采用 **Laplace 近似**或**鞍点近似**。对高斯形式的函数主要方法有两种，视指数的正负号而定。证明可通过配方（略）：

* 用**高斯函数的傅里叶变换**进行转化（也称为 **Hubbard–Stratonovich 变换**）：

$$
\exp \left[-\frac{a}{2} y^{2}\right]=\frac{1}{\sqrt{2 \pi a}} \int_{-\infty}^{\infty} \exp \left[-\frac{x^{2}}{2 a}\pm i x y\right] \mathrm{d} x,
\,\,\,(a>0)
$$

$$
\exp \left[-\frac{1}{2} \mathbf{s^T J s}\right] = \frac{1}{(2 \pi)^{N / 2} \sqrt{\det(\mathbf{J})}} \int_{-\infty}^{\infty} \mathrm{d}\mathbf{x}\, \exp\left[-\frac{1}{2} \mathbf{x^T J^{-1} x}\pm i\mathbf{x^T s}\right],
\,\,\,(\det(\mathbf{J})>0)
$$

* 用**高斯变换**进行转化（也称为 **Weierstrass 变换**）：

$$
\exp \left[\frac{a}{2} y^{2}\right]=\frac{1}{\sqrt{2 \pi a}} \int_{-\infty}^{\infty} \exp \left[-\frac{x^{2}}{2 a}\pm x y\right] \mathrm{d} x,
\,\,\,(a>0)
$$

$$
\exp \left[\frac{1}{2} \mathbf{s^T J s}\right] = \frac{1}{(2 \pi)^{N / 2} \sqrt{\det(\mathbf{J})}} \int_{-\infty}^{\infty} \mathrm{d}\mathbf{x}\, \exp\left[-\frac{1}{2} \mathbf{x^T J^{-1} x}\pm\mathbf{x^T s}\right],
\,\,\,(\det(\mathbf{J})>0)
$$

配分函数对应第二种情形，变换后结果为：

$$
\begin{aligned} Z &=\sum_{\mathbf{s}} \exp\left[ \frac{1}{2} \mathbf{s^T(\beta J)s+\beta\theta^Ts} \right] \\
&=\frac{1}{(2 \pi)^{N / 2} \sqrt{\det(\mathbf{\beta J})}}\sum_{\mathbf{s}} \int_{-\infty}^{\infty} \mathrm{d}\mathbf{x}\, \exp\left[-\frac{1}{2} \mathbf{x^T (\beta J)^{-1} x}+\mathbf{(x+\beta\theta)^T s}\right] \\
&\propto\int \mathrm{d}\mathbf{x}\,\exp\left[-\frac{1}{2} \mathbf{x^T (\beta J)^{-1} x}\right] \sum_{\mathbf{s}} \exp\left[\mathbf{(x+\beta\theta)^T s}\right] \\
&\propto\int \mathrm{d}\mathbf{x}\,\exp\left[-\frac{1}{2} \mathbf{x^T (\beta J)^{-1} x}\right] \prod_{j}\left\{\sum_{s_{j}} e^{s_{j}\left(x_{j}+\beta\theta_{j}\right)}\right\} = \int \mathrm{d}\mathbf{x}\, e^{\Phi(\mathbf{x})}
\end{aligned}
$$

$$
\Phi(\mathbf{x})=-\frac{1}{2} \mathbf{x^T (\beta J)^{-1} x}+\sum_{j} \ln \left[2 \cosh \left(x_{j}+\beta\theta_{j}\right)\right]
$$

由此高维求和转化为高维的非高斯积分，可采用 **Laplace 近似**：积分主要由 $\Phi(\mathbf{x})$ **最大值附近的部分**贡献。

$$
Z \approx e^{\Phi(\mathbf{x}^{0})},\,\mathbf{x}^{0}=\arg \max \Phi(\mathbf{x}) \\
\nabla_{\mathbf{x}} \Phi(\mathbf{x}^0)=0 \,\,\,\Longrightarrow\,\,\,\sum_{j}\left(\frac{1}{\beta}\mathbf{J}^{-1}\right)_{i j} x_{j}^{0}=\tanh \left(x_{i}^{0}+\beta\theta_{i}\right),\, \frac{1}{\beta}\mathbf{J^{-1}x^0}=\tanh(\mathbf{x^0}+\beta\theta)
$$

与前式对比可知，
$$ 
x_{i}^{0} \equiv \beta\sum_{j} J_{i j} m_{j},\, \mathbf{x^0}\equiv\beta\mathbf{Jm}
$$

这里的鞍点近似（Laplace 近似）在对 $\mathbf{x}$ 的积分中将 $\mathbf{x}$ 设为常数，得到了与之前**朴素平均场**一样的结果。配分函数和概率分布也表示为**变量分离**的形式：

$$
q(\mathbf{s}) \propto \exp[\mathbf{(x^0+\beta\theta)^T s}] = \prod_{i} e^{s_{i}\left(x_{i}^{0}+\beta\theta_{i}\right)} \propto \prod_{i} q_i(s_i)
$$

乍一看我们并没有得到新的发现。然而当相互作用变得比二次型**复杂许多**时，场论方法仍能简洁地分离变量，且没有变分平均场方法与之对应。常用方法是 **Hubbard-Stratonovich 变换**和 **δ-函数变换**，都采用了傅里叶积分的形式。

$$ 
1=\int \mathrm{d} h \delta(h-x)=\int \frac{\mathrm{d} h \mathrm{d} \hat{h}}{2 \pi} e^{i \hat{h}(h-x)}
$$

$$
\begin{aligned} Z &=\sum_{\mathbf{s}} \prod_{j}f\left(\sum_{k} J_{j k} s_{k}\right) \\ 
&=\sum_{\mathbf{s}} \int \prod_{j}\left\{\mathrm{d} h_{j} f(h_{j}) \delta\left(h_{j}-\sum_{k} J_{j k} s_{k}\right)\right\} \\ 
&=\int \prod_{j}\left(\frac{\mathrm{d} h_{j} \mathrm{d} \hat{h}_{j}}{2 \pi}\right)f(h_{j}) e^{-i \sum_{j} \hat{h}_{j} h_{j}} \prod_{k}\left\{\sum_{s_{k}} e^{i s_{k}\sum_{j} J_{j k} \hat{h}_{j}} \right\}\end{aligned}
$$

此时可用**鞍点近似**求解。场论方法还有其他优点，如可将 $\Phi(\mathbf{x})$ 在稳定点附近展开，从而系统性地提高近似的准确度；也还常用于**生成泛函（或路径积分）法**，处理动态问题。

## 2.5 平均场近似何时准确？

回忆 **Callen 方程**，朴素平均场近似忽略了**局域场 $h_i$ 的涨落**：

$$h_{i}=\sum_{j} J_{i j} s_{j}+\theta_i\\
\begin{array}{rl}
    \mathbb{E}_p [s_i] &
    =\mathbb{E}_p\left[\tanh\left( \beta h_i\right)\right] \\
    & \\
    \mathbb{E}_q [s_i] &
    = \tanh\left( \beta \mathbb{E}_q[h_i]\right)\\
\end{array}
$$

$h_i$ 是一系列随机变量的和，故当随机变量数量很大时，$h_i$ 的涨落很小，平均场近似准确。主要有两种极端情况下成立：

### Case I

所有 $J_{ij}$ 均相等且大于0。为使局域场 $h_i$ 随 $N$ 增加以 $\mathcal{O}(1)$ 增长，$J_{ij}=J_0/ N>0.$

考虑此时单个自旋的局域场 $h_i$，自旋 $s_j$ 的贡献为 $J_{ij}s_j$，它的涨落

$$
\operatorname{Var}\left(J_{i j} S_{j}\right)=J_{0}^{2}\left(1- \mathbb{E}_p[S_{j}]^{2}\right) / N^{2}
$$

故 $N$ 个自旋对局域场的总贡献的涨落在 $\mathcal{O}(1/N)$ 量级，$N\to\infty$ 时可以忽略。

也可通过场论方法作更严格的证明：

$$
\begin{aligned} 
\exp\left[ \frac{1}{2} \mathbf{s^T(\beta J)s} \right]& =\exp \left[\frac{\beta J_{0}}{2} N \sum_{i j} s_{i} s_{j}\right]=\exp \left[\frac{\beta J_{0}}{2 N}\left(\sum_{i} s_{i}\right)^{2}\right]\\
& \propto \int \mathrm{d} x \exp\left[-\frac{N}{2 \beta J_{0}} x^{2} +x \sum_{i} s_{i}\right]\\
\\
Z &=\sum_{\mathbf{s}} \exp\left[ \frac{1}{2} \mathbf{s^T(\beta J)s+\beta\theta^Ts} \right] \\

&\propto\int \mathrm{d} x \exp\left[-\frac{N}{2 \beta J_{0}} x^{2} \right] \sum_{\mathbf{s}} \exp\left[ x \sum_{i} s_{i} \right]\exp\left[\mathbf{\beta\theta^T s}\right] \\
&\propto\int \mathrm{d} x\,\exp\left[-\frac{N}{2 \beta J_{0}} x^{2}\right] \prod_{j}\left\{\sum_{s_{j}} e^{s_{j}\left(x+\beta\theta_{j}\right)}\right\} = \int \mathrm{d} x\, e^{\Phi(x)}
\end{aligned}
$$

$$
\Phi(x)=-\frac{N}{2 \beta J_{0}} x^{2}+\sum_{j} \ln \left[2 \cosh \left(x+\beta\theta_{j}\right)\right]
$$

然而实际应用中多数情况是 $J_{ij}$ 与观测数据有关，会存在强烈变化。因此考虑 $J_{ij}$ 为随机变量的情形就显得重要。

### Case II

$J_{ij}$ 选为一组**零均值、互相独立的随机变量**。为简单起见令外场 $\theta_i=0$，则局域场 $h_i$ 是 $N$ 个正负出现概率大致相等的随机变量之和，为使局域场 $h_i$ 随 $N$ 增加以 $\mathcal{O}(1)$ 增长，$\sigma(J_{ij})=J_0/ \sqrt{N}>0.$

此时若忽略自旋间的统计相关性，$\operatorname{Var}[h_i] \sim N\sigma^2(J_{ij}) \sim \mathcal{O}(1)$，朴素平均场近似不准确。下一节中介绍的 **TAP 平均场理论** 通过加入修正项，可以使此情形下，$N\to\infty$ 时平均场解准确。

事实上只要满足**相互作用 $J_{ij}$ 范围无限**或**无穷维**，都可以构建出精确的平均场理论。这两个概念都和空间有关。**范围无限**指的是空间距离 

$$|| i-j || \to\infty$$ 

时 $J_{ij}$ 不衰减到0；**无穷维**下相互作用范围有限时也能和**无穷个**近邻自旋作用。这两种情形下随机变量（自旋）间的相关性都足够弱，可用平均场理论精确处理。

## 2.6 TAP 方程 I - 空穴法

**TAP 平均场方程**根据对 **Sherrington-Kirkpatrick (SK) 自旋玻璃模型**推导出平均场理论的三位科学家 D.J. Thouless, P.W. Anderson 和 R.G. Palmer 命名。

Under construction...

## 2.7 TAP 方程 II - Plefka 展开

Under construction...

## 2.8 TAP 方程 III - 模型以外

Under construction...

## 2.9 展望

Under construction...