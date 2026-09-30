---
date: '2026-09-29T18:01:04+08:00'
title: '有限马尔可夫决策过程的三种求解方法：DP、MC 与 TD'
description: '梳理动态规划、蒙特卡洛与时序差分三种强化学习核心方法的核心思想、推导与对比。'
categories: ["RL"]
tags: ["强化学习", "动态规划", "蒙特卡洛", "时序差分"]
math: true
---

解决有限马尔可夫决策过程的三种方法：

- **动态规划（DP）**：在数学上得到了很好的发展，但需要一个完整而准确的环境模型。
- **蒙特卡洛方法（MC）**：不需要模型，并且在概念上很简单，但不适合逐步增量计算。
- **时序差分方法（TD）**：不需要模型，完全是增量的，但分析起来更复杂。

这三种方法在效率和收敛速度方面也有不同的表现。

参考教材：

- 《动手强化学习》
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — [书籍主页](http://incompleteideas.net/book/the-book-2nd.html)

## 1. 前情提要

### 1.1 马尔可夫过程

在**马尔可夫奖励过程**（Markov reward process, MRP）中，一个状态的期望回报（即从这个状态出发的未来累积奖励的期望）被称为这个状态的**价值**（value）。所有状态的价值就组成了**价值函数**（value function）。

在马尔可夫过程的基础上加入奖励函数和折扣因子，就可以得到马尔可夫奖励过程 $\langle S, P, r, \gamma \rangle$。通过贝尔曼方程展开可以得到：

$$
V(s) = r(s) + \gamma \sum_{s' \in S} p(s' \mid s) V(s')
$$

写成矩阵形式：

$$
\begin{bmatrix}
V(s_1) \\
V(s_2) \\
\vdots \\
V(s_n)
\end{bmatrix} =
\begin{bmatrix}
r(s_1) \\
r(s_2) \\
\vdots \\
r(s_n)
\end{bmatrix} +
\gamma
\begin{bmatrix}
p(s_1 \mid s_1) & p(s_2 \mid s_1) & \cdots & p(s_n \mid s_1) \\
p(s_1 \mid s_2) & p(s_2 \mid s_2) & \cdots & p(s_n \mid s_2) \\
\vdots & \vdots & \ddots & \vdots \\
p(s_1 \mid s_n) & p(s_2 \mid s_n) & \cdots & p(s_n \mid s_n)
\end{bmatrix}
\begin{bmatrix}
V(s_1) \\
V(s_2) \\
\vdots \\
V(s_n)
\end{bmatrix}
$$

即：

$$
\mathcal{V} = \mathcal{R} + \gamma P \mathcal{V}
$$

移项求解：

$$
\begin{aligned}
(I - \gamma P)\mathcal{V} &= \mathcal{R} \\
\mathcal{V} &= (I - \gamma P)^{-1}\mathcal{R}
\end{aligned}
$$

MRP 解析解的计算复杂度为 $O(n^3)$。

为了求解大规模的有限马尔可夫奖励过程中的价值函数 $V^\pi(s)$，引出了下面三种方法：动态规划（dynamic programming）、蒙特卡洛方法（Monte-Carlo method）和时序差分（temporal difference）。

### 1.2 贝尔曼最优方程

$$
\begin{aligned}
v_*(s)
&= \max_a \mathbb{E}\left[
R_{t+1} + \gamma v_*(S_{t+1})
\mid S_t = s,\ A_t = a
\right] \\
&= \max_a \sum_{s', r} p(s', r \mid s, a)
\left[ r + \gamma v_*(s') \right].
\end{aligned}
$$

$$
\begin{aligned}
q_*(s, a)
&= \mathbb{E}\left[
R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a')
\mid S_t = s,\ A_t = a
\right] \\
&= \sum_{s', r} p(s', r \mid s, a)
\left[ r + \gamma \max_{a'} q_*(s', a') \right].
\end{aligned}
$$

## 2. 动态规划（Dynamic Programming）

> The term **dynamic programming** (DP) refers to a collection of algorithms that can be used to compute optimal policies given a perfect model of the environment as a Markov decision process (MDP).

**核心思想**：DP 以及一般意义上的强化学习的核心思想，是使用价值函数来组织并结构化对好策略的搜索。

DP 需要知道环境的**状态转移函数**和**奖励函数**。基于动态规划的强化学习算法主要有两类：

- **策略迭代（policy iteration）**
  - **策略评估（policy evaluation）**：使用贝尔曼期望方程来得到一个策略的状态价值函数。
  - **策略改进（policy improvement）**
- **价值迭代（value iteration）**：直接使用贝尔曼最优方程来进行动态规划，得到最终的最优状态价值。

### 2.1 策略评估（Policy Evaluation / Prediction）

**定义**：对于一个任意策略，如何计算其状态价值函数。

$$
\begin{aligned}
v_{\pi}(s)
&\doteq \mathbb{E}_{\pi}\left[ G_t \mid S_t = s \right] \\
&= \mathbb{E}_{\pi}\left[ R_{t+1} + \gamma G_{t+1} \mid S_t = s \right] \\
&= \mathbb{E}_{\pi}\left[ R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s \right] \\
&= \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)
\left[ r + \gamma v_{\pi}(s') \right].
\end{aligned}
$$

**迭代策略评估**（iterative policy evaluation）：

$$
\begin{aligned}
v_{k+1}(s)
&\doteq \mathbb{E}_{\pi}\left[
R_{t+1} + \gamma v_k(S_{t+1}) \mid S_t = s
\right] \\
&= \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)
\left[ r + \gamma v_k(s') \right].
\end{aligned}
$$

这里可以只使用一个数组就地更新，仍然能收敛到价值函数 $V$。

**证明**：固定策略 $\pi$，设 $0 \le \gamma < 1$。由贝尔曼期望方程：

$$
v_\pi = T_\pi v_\pi, \qquad v_{k+1} = T_\pi v_k.
$$

记最大误差为 $E_k = \max_s |v_k(s) - v_\pi(s)|$，则：

$$
\begin{aligned}
|v_{k+1}(s) - v_\pi(s)|
&= \gamma \left|
\sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)
\left[ v_k(s') - v_\pi(s') \right]
\right| \\
&\le \gamma \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) E_k \\
&= \gamma E_k.
\end{aligned}
$$

因此：

$$
E_{k+1} \le \gamma E_k
\quad \Longrightarrow \quad
E_k \le \gamma^k E_0 \xrightarrow{k \to \infty} 0
\quad \Longrightarrow \quad
v_k \to v_\pi.
$$

原地更新也成立：一轮开始时若最大误差为 $E$，每次更新所读取的新旧值误差均不超过 $E$，更新后的状态误差不超过 $\gamma E$。完整遍历后，所有状态的误差均不超过 $\gamma E$，故仍然收敛。

### 2.2 策略改进（Policy Improvement）

策略评估计算一个给定策略下的价值函数，是为了寻找更好的策略。我们假设在每个状态 $s$ 下采取一个更优的动作：

$$
\begin{aligned}
v_{\pi}(s)
&\le q_{\pi}\bigl(s, \pi'(s)\bigr) \\
&= \mathbb{E}\left[
R_{t+1} + \gamma v_{\pi}(S_{t+1})
\mid S_t = s,\ A_t = \pi'(s)
\right] \\
&= \mathbb{E}_{\pi'}\left[
R_{t+1} + \gamma v_{\pi}(S_{t+1})
\mid S_t = s
\right] \\
&\le \mathbb{E}_{\pi'}\left[
R_{t+1} + \gamma q_{\pi}\bigl(S_{t+1}, \pi'(S_{t+1})\bigr)
\mid S_t = s
\right] \\
&= \mathbb{E}_{\pi'}\left[
R_{t+1} + \gamma \mathbb{E}\left[
R_{t+2} + \gamma v_{\pi}(S_{t+2})
\mid S_{t+1},\ A_{t+1} = \pi'(S_{t+1})
\right]
\mid S_t = s
\right] \\
&= \mathbb{E}_{\pi'}\left[
R_{t+1} + \gamma R_{t+2} + \gamma^2 v_{\pi}(S_{t+2})
\mid S_t = s
\right] \\
&\le \mathbb{E}_{\pi'}\left[
R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 v_{\pi}(S_{t+3})
\mid S_t = s
\right] \\
&\ \vdots \\
&\le \mathbb{E}_{\pi'}\left[
R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \cdots
\mid S_t = s
\right] \\
&= v_{\pi'}(s).
\end{aligned}
$$

存在一个贪心策略：

$$
\begin{aligned}
\pi'(s)
&\doteq \underset{a}{\arg\max}\; q_{\pi}(s, a) \\
&= \underset{a}{\arg\max}\;
\mathbb{E}\left[
R_{t+1} + \gamma v_{\pi}(S_{t+1})
\mid S_t = s,\ A_t = a
\right] \\
&= \underset{a}{\arg\max}\;
\sum_{s', r} p(s', r \mid s, a)
\left[ r + \gamma v_{\pi}(s') \right].
\end{aligned}
$$

### 2.3 策略迭代与价值迭代

**策略迭代（Policy Iteration）**：交替进行策略评估与策略改进，直至策略稳定。

**价值迭代（Value Iteration）**：当策略评估只进行一轮扫描（每个状态更新一次）时，就退化为价值迭代：

$$
\begin{aligned}
v_{k+1}(s)
&\doteq \max_a \mathbb{E}\left[
R_{t+1} + \gamma v_k(S_{t+1})
\mid S_t = s,\ A_t = a
\right] \\
&= \max_a \sum_{s', r} p(s', r \mid s, a)
\left[ r + \gamma v_k(s') \right].
\end{aligned}
$$

### 2.4 广义策略迭代（Generalized Policy Iteration）

GPI 描述了策略评估与策略改进相互作用的通用框架，最终收敛到最优价值函数和最优策略。

## 3. 蒙特卡洛方法（Monte Carlo Methods）

蒙特卡洛方法是一类**统计模拟方法**。

> **定义**：Monte Carlo methods are a broad class of computational algorithms based on repeated random sampling for obtaining numerical results.

例如，使用蒙特卡洛方法估算 $\pi$ 值：放置 30000 个随机点后，$\pi$ 的估算值与真实值相差 0.07%。

$$
\pi \approx 4 \times \frac{\text{圆内点数}}{\text{总点数}}
$$

$$
V^\pi(s)
= \mathbb{E}_\pi\left[ G_t \mid S_t = s \right]
\approx \frac{1}{N} \sum_{i=1}^{N} G_t^{(i)}
$$

**与 DP 的区别**：MC 不假设完全了解环境，只需要**经验（experience）**。

**MC prediction** 的增量更新形式：

$$
V(s) \leftarrow V(s) + \alpha \left[ G_t - V(s) \right].
$$

**MC control**：近似最优策略。利用 GPI 的思想，蒙特卡洛通过统计模拟、重复随机采样的方式得到价值函数，再通过贪心策略选择最佳动作来进行策略迭代。

### 3.1 两个假设及其解决

**问题**：无法遍历所有的 (action, state) 对？

假设当前 $\pi$ 是一个确定性的策略，就会出现这种情况，此时蒙特卡洛采样无法根据经验改善动作价值函数。类似多臂老虎机问题（MAB），我们使用 $\epsilon$-贪心算法确保会采取新的动作；此外还可以使用 **ES（Exploring Starts，试探性出发）**。

即只考虑在每个状态下所有动作都有非零概率被选中的策略。

**蒙特卡洛算法的收敛基于两个假设：**

1. the episodes have exploring starts（试探性出发假设）
2. policy evaluation could be done with an infinite number of episodes（有无限多 episodes）

**解决第二个假设：**

- 逼近误差幅度小的时候提前终止
- 减少 policy evaluation 次数（例如 value iteration），在每个 episode 间交替进行 predict/evaluation（即多次采样）和 improvement

**解决第一个假设：**

- **on-policy**：MC ES + $\epsilon$-贪心算法
- **off-policy**

### 3.2 重要性采样（Importance Sampling）

**定义**：Estimating expected values under one distribution given samples from another.

**覆盖性假设（Assumption of coverage）**：every action taken under $\pi$ is also taken, at least occasionally, under $b$. That is, we require that $\pi(a \mid s) > 0$ implies $b(a \mid s) > 0$.

$$
\begin{aligned}
&\Pr\left\{
A_t, S_{t+1}, A_{t+1}, \ldots, S_T
\mid S_t,\ A_{t:T-1} \sim \pi
\right\} \\
&\quad =
\pi(A_t \mid S_t) p(S_{t+1} \mid S_t, A_t)
\pi(A_{t+1} \mid S_{t+1}) \cdots
p(S_T \mid S_{T-1}, A_{T-1}) \\
&\quad =
\prod_{k=t}^{T-1}
\pi(A_k \mid S_k) p(S_{k+1} \mid S_k, A_k).
\end{aligned}
$$

重要性采样比：

$$
\rho_{t:T-1}
\doteq
\frac{
\displaystyle\prod_{k=t}^{T-1}
\pi(A_k \mid S_k) p(S_{k+1} \mid S_k, A_k)
}{
\displaystyle\prod_{k=t}^{T-1}
b(A_k \mid S_k) p(S_{k+1} \mid S_k, A_k)
} =
\prod_{k=t}^{T-1}
\frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}.
$$

**为什么使用重要性采样的时候，不宜两个分布差异过大？**

**答**：定性分析上，在采样次数足够小的时候，我们从 $q$ 分布下采样得到的预计期望是正值，但实际上应该是负值。公式推导上，期望虽然相同，但是**方差不一样**，因此采样次数不够的时候两种期望估计会有较大差异。

## 4. 时序差分学习（Temporal-Difference Learning）

TD 学习引入了**自举（bootstrap）**的思想，即用当前的估计值来更新估计值。

**问题**：What if the trajectory never ends?（如果轨迹永不终止怎么办？）

- **TD Prediction**
- **TD Control**：我们转而使用 TD 预测方法来解决控制问题。
- **Sarsa**：On-policy TD Control
- **Q-Learning**：Off-policy TD Control

Sarsa 和 Q-Learning 的区别在于对动作价值函数进行更新的时候，Q-Learning 可以使用任意策略，只需要使未来的动作价值函数预测值最大。

## 5. DP、MC、TD 的对比

三种方法在做的内容是一致的：

- estimating value functions（估计价值函数）
- discovering optimal policies（发现最优策略）

$$
\underbrace{
\mathbb{E}_{\pi}\left[ r + \gamma V(s') \mid s \right]
}_{\text{贝尔曼期望}}
\xrightarrow{\text{模型已知}}
\underbrace{
\mathrm{target}_{\mathrm{DP}}
}_{\text{展开计算}}
\xrightarrow{\text{模型未知}}
\underbrace{
G_t
}_{\text{MC：完整采样}}
\xrightarrow{\text{不等完整未来}}
\underbrace{
r + \gamma V_k(s')
}_{\text{TD：一步自举}}
$$

**动态规划（DP）**：

$$
V_{k+1}(s) \leftarrow
\sum_a \pi(a \mid s)
\left[
R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V_k(s')
\right]
$$

**蒙特卡洛（MC）**：

$$
V(s) \leftarrow V(s) + \alpha \left[ G_t - V(s) \right]
$$

**时序差分（TD）**：

$$
V(s) \leftarrow V(s) + \alpha \left[ r + \gamma V(s') - V(s) \right]
$$

## 6. 延伸：DQN 与 LLM 中的 Q-Learning

### 6.1 DQN

用神经网络 $Q_\theta(s, a)$ 代替表格化的 $Q(s, a)$。

关键创新包括：

- **经验重放缓冲区（experience replay）**：异策略数据复用
- **目标网络（target network）**：提升稳定性
- **$\epsilon$-贪心探索**

DQN 损失函数：最小化从重放缓冲区采样的 mini-batch 上的均方误差来训练。

$$
\mathcal{L}(\theta) =
\mathbb{E}_{(s, a, r, s') \sim \mathcal{B}}
\left[
\left(
r + \gamma \max_{a'} Q_{\bar{\theta}}(s', a') - Q_{\theta}(s, a)
\right)^2
\right].
$$

### 6.2 为什么 Q-Learning 不适用于 LLM？

生成中的动作空间是整个词表（$|A| = 32\text{K} \sim 128\text{K}$），状态空间是所有可能的 token 序列（无限）。在每个 token 位置对 128K 个动作计算 $\max_a Q(s, a)$ 是不可行的。这就是 LLM RL 使用基于策略方法（PPO、GRPO）的原因。

### 6.3 Q-Learning 能不能用于 LLM 中的 advantage 估计？

理论上是可以的，类似下面的 GAE，可以作为一种估计的方式。
