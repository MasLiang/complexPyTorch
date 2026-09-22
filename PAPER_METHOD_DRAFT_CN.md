# 面向 FPGA LUT6 的可学习复数神经元网络

## 论文式技术方案初稿

> 文档状态：当前实现与实验结果的技术归纳稿。本文区分“核心方法”“数据集特定扩展”和“尚待验证的假设”，避免把历史探索路线写成最终方法。

## 摘要

复数神经网络能够同时建模幅度和相位信息，但标准复数乘法、浮点激活以及复数卷积在 FPGA 上会带来较高的乘法器和存储开销。本文提出一种面向 FPGA LUT6 的可学习复数神经元方法。与把 LUT 作为单个复数乘法近似器的早期路线不同，本文将 LUT 视为局部神经元：每个局部单元接收两个二值复数激活，直接输出一个二值复数结果。实部和虚部输出通过四分类 categorical 参数化联合学习，并最终映射为两张共享输入的单输出查找表。

方法首先将已训练的复数 BiReal 卷积解析编译为 categorical LUT4，从而获得硬件一致的局部真值表。随后，每个复数激活增加一个相位细化 bit，将两复数输入从4 bit扩展到6 bit。原 LUT4 真值表被无损复制到四个相位组合，并在其上学习零均值 categorical residual。该构造保证 LUT6 初始前向与 LUT4 严格等价，同时允许网络逐步学习相位条件下的新局部布尔映射。训练阶段采用 hard argmax 前向和 softmax 直通估计反向；部署时将 base、residual、释放系数和可选 correction 合并为固定真值表，因此最终硬件不包含 softmax、浮点 logit、residual 分支或运行时缩放。

在 CIFAR-10 上，categorical LUT4 达到82.53%，核心 residual LUT6 经续训达到84.56%；可选 LUT4 common correction 的最佳结果为85.09%。在 San Francisco AIRSAR 空间划分实验中，LUT4 为93.55% OA，核心 residual LUT6 为93.74% OA，而 common correction 未带来收益。结果表明，严格 warm start 和受控 residual 能够提升 LUT6 的可训练性，但额外相位信息的收益依赖数据分布；当前主要限制包括少数类泛化、深层地址分布集中以及训练协议差异。

## 1. 研究问题

### 1.1 目标

本文希望构造一种满足以下条件的复数神经网络局部算子：

1. 前向传播在训练时就与最终硬件的离散行为一致；
2. 每个局部算子可由 FPGA LUT 直接实现；
3. 实部和虚部作为一个复数状态联合优化，而不是两张完全无关的真值表；
4. 能从已有复数 BiReal 模型稳定迁移，避免随机 hard LUT 从头训练困难；
5. 能利用 LUT6 相比 LUT4 增加的输入引脚表达额外相位信息；
6. 训练期增加的连续参数不增加最终部署资源。

### 1.2 从“LUT as operator”到“LUT as neuron”

早期路线把 LUT 当作复数乘法操作的替代品，并尝试联合优化卷积权重和 LUT operation。该路线存在两个问题：第一，LUT 真值表跨越离散决策边界时会改变局部 operation，导致优化曲面剧烈变化；第二，卷积权重与 LUT operation 同时学习时存在冗余和不稳定性。

当前路线改为 **LUT as neuron**。对一个卷积窗口中的若干二值复数输入，不再显式保留对应的复数权重乘法，而是直接学习从局部输入 bits 到局部输出复数 bits 的映射。对本文主配置，每个神经元接收两个复数激活：

\[
(x_0,x_1)\longrightarrow (y_r,y_i).
\]

每个输入复数初始由实部、虚部两个符号 bit 表示，因此基础单元是4输入、2输出的局部布尔映射。物理上使用两张共享相同输入的 LUT：一张输出实部 bit，另一张输出虚部 bit。

## 2. 网络骨架

### 2.1 复数 BiReal 残差结构

网络沿用复数 BiReal 主干。一个残差 block 可写为：

\[
q=Q(x),
\]

\[
u=\mathcal{O}(q),
\]

\[
y=\operatorname{CBN}(u)+\mathcal{S}(x),
\]

其中：

- \(Q\) 是复数激活量化；
- \(\mathcal{O}\) 是当前阶段的复数卷积、二值复数卷积或 LUT 神经元卷积；
- \(\operatorname{CBN}\) 是卷积主分支后的 complex BatchNorm；
- \(\mathcal{S}\) 是 identity 或尺寸匹配 projection shortcut。

当前默认 shortcut 在分辨率或通道变化时使用：

\[
\operatorname{AvgPool}\rightarrow
\operatorname{ComplexConv}_{1\times1}\rightarrow
\operatorname{CBN}.
\]

我们测试了无参数 option-A shortcut，即隔点下采样与通道补零，但其 LUT6 增益没有超过默认 FP shortcut，因此主方法保留默认 projection。

### 2.2 分阶段训练

当前方法可描述为以下阶段：

- **Phase 1：FP complex network。** 激活和复数卷积均为全精度。
- **Phase 2：complex BiReal network。** 激活实部、虚部二值化，卷积权重实部、虚部二值化。
- **Phase 3：categorical LUT4 network。** 用局部4输入、2输出 LUT 神经元替换 residual 主分支二值复数卷积。
- **Phase 4：categorical residual LUT6 network。** 每个复数增加一个相位 bit，以 LUT4 为精确起点学习 LUT6 residual。
- **可选扩展：LUT4 common correction。** 在已训练 LUT6 residual 上释放四个相位 slice 共享的基础修正。该扩展只在 CIFAR-10 上观察到收益，在 San Francisco 上无收益，因此不是默认主方法。

Stem、projection、complex BN、residual addition 和 classifier 在当前实现中不被 LUT 神经元替换。

## 3. 二值复数表示

### 3.1 实部与虚部符号 bit

对量化前复数激活：

\[
x=x_r+jx_i,
\]

BiReal 激活分别产生符号值：

\[
\hat{x}_r=\operatorname{sign}(x_r),\qquad
\hat{x}_i=\operatorname{sign}(x_i),
\]

其中 \(\hat{x}_r,\hat{x}_i\in\{-1,+1\}\)。进入 LUT 前编码为：

\[
s_r=\frac{\hat{x}_r+1}{2},\qquad
s_i=\frac{\hat{x}_i+1}{2},
\]

因此 LUT 的实际输入值域严格为 \(\{0,1\}\)。

### 3.2 相位细化 bit

为了利用 LUT6 的额外输入引脚，每个复数激活增加一个相位细化 bit：

\[
\delta(x)=
\operatorname{sign}(x_i)x_r-
\operatorname{sign}(x_r)x_i,
\]

\[
p=\mathbb{I}[\delta(x)\ge 0].
\]

该比较器结合当前象限改变比较方向，可将每个由 \((s_r,s_i)\) 表示的象限进一步划分为两个区域。它不是简单地在所有象限统一执行 \(|x_r|\ge |x_i|\)，而是一个与符号位共同构成相位 Gray 语义的条件 bit。

当前最佳配置使用 stopped-gradient comparator：

\[
\frac{\partial p}{\partial x}=0.
\]

原因是此前实验中，额外 comparator STE 梯度会扰动已训练的符号特征，而 hard comparator 本身已能作为条件输入被后续 LUT residual 利用。

## 4. 局部输入组织

### 4.1 Channel-major 展开

对输入通道数 \(C_{in}\)、卷积核大小 \(K\times K\)，每个输出位置包含：

\[
L=C_{in}K^2
\]

个复数逻辑位置。实现按以下顺序展开：

\[
\text{channel}\rightarrow\text{kernel row}\rightarrow\text{kernel column}.
\]

相邻两个复数逻辑位置组成一个 LUT 神经元组。由于同一通道的 \(K^2\) 个空间位置连续排列，因此大多数组由同一输入通道内的相邻 kernel taps 构成。

组数为：

\[
G=\left\lceil\frac{C_{in}K^2}{2}\right\rceil.
\]

### 4.2 LUT4 地址

对一组中的两个二值复数：

\[
q_0=(s_{r0},s_{i0}),\qquad
q_1=(s_{r1},s_{i1}),
\]

LUT4 地址定义为：

\[
a_4=(s_{r0},s_{i0},s_{r1},s_{i1}),
\]

整数地址为：

\[
A_4=8s_{r0}+4s_{i0}+2s_{r1}+s_{i1}.
\]

### 4.3 LUT6 地址

加入两个相位 bit 后：

\[
a_6=(s_{r0},s_{i0},p_0,s_{r1},s_{i1},p_1),
\]

整数地址为：

\[
A_6=32s_{r0}+16s_{i0}+8p_0+4s_{r1}+2s_{i1}+p_1.
\]

去除相位位后对应的基础 LUT4 地址映射为：

\[
\pi(A_6)=8s_{r0}+4s_{i0}+2s_{r1}+s_{i1}.
\]

## 5. Categorical LUT4 神经元

### 5.1 联合复数输出码本

一个二值复数输出有四种状态：

\[
\mathcal{C}={(0,0),(0,1),(1,0),(1,1)\}.
\]

定义固定码本：

\[
E=
\begin{bmatrix}
0&0\\
0&1\\
1&0\\
1&1
\end{bmatrix}.
\]

对每个输出通道 \(o\)、输入组 \(g\) 和 LUT4 地址 \(a\)，维护四分类 logits：

\[
\Theta^{(4)}_{o,g,a}\in\mathbb{R}^{4}.
\]

与分别维护实部、虚部两张独立 logit 表相比，categorical 参数化显式把 \((y_r,y_i)\) 作为一个合法复数状态联合选择，使两路输出在训练参数化层面保持关联。

### 5.2 从 Phase 2 解析初始化

Phase 2 中每个二值复数权重为：

\[
w_m=w_{r,m}+jw_{i,m},qquad
w_{r,m},w_{i,m}\in\{-1,+1\}.
\]

对每个两复数输入组，遍历全部16个输入状态，并计算局部复数乘加：

\[
u_r=\sum_{m=0}^{1}
(x_{r,m}w_{r,m}-x_{i,m}w_{i,m}),
\]

\[
u_i=\sum_{m=0}^{1}
(x_{i,m}w_{r,m}+x_{r,m}w_{i,m}).
\]

将局部输出比较为两个 bit：

\[
b_r=\mathbb{I}[u_r\ge0],\qquad
b_i=\mathbb{I}[u_i\ge0].
\]

目标类别为：

\[
c^*=2b_r+b_i.
\]

由此把 Phase 2 的二值复数权重解析编译为 categorical LUT4 初始表。该初始化不是要求 Phase 3 永远保持复数乘法，而是给可学习局部神经元提供一个已训练网络附近的离散起点。

## 6. Hard-forward / soft-backward 训练

### 6.1 前向硬选择

给定四分类 logits \(Z\)，先计算：

\[
P=\operatorname{softmax}(Z/\tau).
\]

硬类别为：

\[
h=\operatorname{onehot}(\arg\max_c P_c).
\]

训练张量使用 straight-through 组合：

\[
\tilde{h}=P+\operatorname{stopgrad}(h-P).
\]

因此：

- forward 中 \(\tilde{h}=h\)，输出严格 one-hot；
- backward 中 \(\partial\tilde{h}/\partial Z=\partial P/\partial Z\)。

通过码本得到实部、虚部表项：

\[
(t_r,t_i)=\tilde{h}^{\mathsf T}E.
\]

当前 categorical temperature 为 \(\tau=1\)。由于前向始终 hard，训练时查表行为与部署真值表一致。

### 6.2 查表与组累加

对输出通道 \(o\) 和组 \(g\)：

\[
t_{r,o,g}=T_{r,o,g}[A],\qquad
 t_{i,o,g}=T_{i,o,g}[A],
\]

其中 \(A\) 是 LUT4 或 LUT6 地址。卷积位置的局部输出为：

\[
y_{r,o}=\sum_{g=1}^{G}t_{r,o,g},\qquad
 y_{i,o}=\sum_{g=1}^{G}t_{i,o,g}.
\]

因此每张 LUT 输出1 bit，多个 LUT 神经元通过计数累加形成复数卷积输出。输出随后经过 complex BN，BN 同时承担尺度恢复和 \(\{0,1\}\) 累加带来的直流偏置校准，再与 shortcut 相加。

## 7. LUT4 到 LUT6 的严格 warm start

### 7.1 无损复制

LUT6 比 LUT4 多出 \((p_0,p_1)\) 两个输入。初始化时忽略这两个 bit：

\[
T_6(A_6)=T_4(\pi(A_6)).
\]

等价地，每个 LUT4 entry 被复制到四个相位组合：

\[
(p_0,p_1)\in\{00,01,10,11\}.
\]

因此在 residual 为零时，对任意输入：

\[
f_{LUT6}(x)=f_{LUT4}(x).
\]

这避免了把成熟 LUT4 operation 突然替换为随机64-entry布尔函数所产生的量化冲击。

### 7.2 零均值 categorical residual

对 LUT6 地址 \(s=A_6\)，记其 LUT4 基础地址为 \(a=\pi(s)\)。冻结的基础 logits 为：

\[
B_{o,g,a}\in\mathbb{R}^{4}.
\]

维护可学习 LUT6 residual：

\[
R_{o,g,s}\in\mathbb{R}^{4}.
\]

对同一基础地址的四个相位 slice 做中心化：

\[
\bar{R}_{o,g,s}
=
R_{o,g,s}
-
\frac{1}{4}
\sum_{s':\pi(s')=a}R_{o,g,s'}.
\]

核心方法的有效 logits 为：

\[
Z_{o,g,s}=B_{o,g,a}+\alpha\bar{R}_{o,g,s}.
\]

零均值约束使 residual 主要表达“相位条件差异”，而不是四个 slice 一起漂移并重复学习 LUT4 base。

### 7.3 Residual 释放系数

训练中将 \(\alpha\) 从较小值逐渐增加：

\[
\alpha_e=alpha_{start}+
\min\left(\frac{e}{E_{ramp}},1\right)
(\alpha_{end}-\alpha_{start}).
\]

已有主实验使用 \(0.1\rightarrow1\)。但 \(\alpha=1\) 不是硬件要求。对任意固定 \(\alpha^*>0\)，最终真值表都可由：

\[
T(A_6)=\arg\max_c
\left(B+\alpha^*\bar{R}\right)_c
\]

离线生成。因此 \(\alpha\) 是优化和正则化超参数，不是部署时乘法。San Francisco 的训练轨迹提示较小终值可能更稳，后续应测试 \(0.35\) 和 \(0.5\)。

## 8. 可选 LUT4 common correction

为了允许四个相位 slice 共同修正基础 operation，引入：

\[
C_{o,g,a}\in\mathbb{R}^{4},
\]

并在类别维中心化：

\[
\bar{C}_{o,g,a}
=C_{o,g,a}-\frac{1}{4}\sum_{c=0}^{3}C_{o,g,a,c}.
\]

完整扩展 logits 为：

\[
Z_{o,g,s}
=
B_{o,g,a}
+
\beta\bar{C}_{o,g,a}
+
\alpha\bar{R}_{o,g,s}.
\]

CIFAR-10 上顺序释放该项曾把最佳结果从84.56%提高到85.09%，但 San Francisco 上从93.74%降到93.59%。因此论文中应将其作为数据集相关扩展和消融，而不是核心模块。核心方法使用 \(\beta=0\)。

## 9. 硬件映射

### 9.1 单层资源

对输出通道数 \(C_{out}\)，每个输出通道有 \(G\) 个输入组。每组部署：

- 一张64-entry、1-bit输出的实部 LUT6；
- 一张64-entry、1-bit输出的虚部 LUT6。

因此 LUT6 数量为：

\[
N_{LUT6}=2C_{out}
\left\lceil\frac{C_{in}K^2}{2}\right\rceil.
\]

当前实现只使用每张 LUT6 的 O6 输出，不使用 O5。O5/O6 联合方案已经实验，但没有超过 O6-only 主路线。

### 9.2 训练参数不进入硬件

训练结束时计算：

\[
c^*(s)=\arg\max_c Z_s(c),
\]

并由码本拆成：

\[
T_r(s)=E_{c^*(s),0},\qquad
T_i(s)=E_{c^*(s),1}.
\]

部署只保存 \(T_r,T_i\) 两张硬真值表。以下对象不会进入硬件：

- softmax 和 temperature；
- categorical 四分类 logits；
- LUT4 base、LUT6 residual 和 common correction 的分解；
- \(\alpha\) 和 \(\beta\) 的运行时乘法；
- STE 或任何反向传播结构。

训练参数化只改变如何找到最终布尔映射，不改变部署 LUT 数量。

### 9.3 LUT 后的聚合

每个 LUT 输出为 \(\{0,1\}\)，同一输出通道内分别对实部、虚部做整数累加。硬件还需要：

1. LUT6 阵列；
2. 实部和虚部各自的加法树或计数树；
3. 后续 BN/阈值等效变换；
4. residual shortcut 所需的数据通路；
5. 当前 FP stem 和 projection 的对应实现。

因此“LUT 核心完全离散”不等于整个网络已经全 LUT 化。当前论文应明确区分 LUT 主算子资源与网络其余部分资源。

## 10. 训练算法

### 10.1 核心流程

**输入：** 已训练 Phase 2 complex BiReal checkpoint。

1. 读取共享的 stem、BN、projection、classifier 参数；
2. 对每个 residual 主卷积，按 channel-major 顺序把两个复数逻辑位置组成一组；
3. 遍历每组16个输入状态，将 Phase 2 二值复权重编译为 categorical LUT4；
4. 训练 LUT4 网络，使局部 LUT 神经元与共享网络参数共同适配；
5. 将每个 LUT4 entry 复制到四个 \((p_0,p_1)\) slice；
6. 冻结 LUT4 base，初始化 LUT6 residual 为零；
7. 从较小 \(\alpha\) 开始，逐渐释放零均值 residual；
8. 在固定终值 \(\alpha\) 下继续训练并选择 checkpoint；
9. 合并 logits，逐地址 hard argmax，导出两张 LUT6 真值表。

### 10.2 优化器分组

训练中区分：

- 普通网络参数：stem、BN、projection、classifier 等；
- LUT6 residual 参数；
- 可选 LUT4 common correction 参数。

三组使用独立学习率。经验上 LUT residual 的学习率可高于普通网络参数，但必须从严格 warm start 出发；随机 hard LUT 配合较大学习率容易导致训练崩溃。

### 10.3 Checkpoint 选择

CIFAR-10 历史实验沿用 train/test 协议并按 test best 报告，与部分旧论文协议一致，但不属于严格独立测试。San Francisco 使用 train/validation/test 空间划分，只按 validation 选择 checkpoint并在训练结束后评估一次 test。

对类别不平衡数据，仅按 validation OA 选择会偏向多数类。后续正式论文实验应至少同时保存：

- validation OA best；
- validation AA best；
- 预先定义的 OA/AA 综合指标 best。

最终论文必须在实验开始前固定主选择指标，避免根据 test 结果选择 checkpoint。

## 11. 实验结果

### 11.1 CIFAR-10

当前主要历史结果如下：

| 模型 | Test accuracy | 说明 |
|---|---:|---|
| FP Phase 1 | 89.48% | 全精度复数网络 |
| Complex BiReal Phase 2 | 约85.05% | 二值激活与二值复权重 |
| Categorical LUT4 | 82.53% | 两复数4 bit输入、联合2 bit输出 |
| Residual LUT6 主训练 | 84.28% | LUT4 strict warm start，逐渐释放 residual |
| Residual LUT6 续训 | 84.56% | 核心方法最佳 |
| LUT6 + common correction | 85.09% | 可选扩展最佳，final为84.34% |

核心 residual LUT6 相比 LUT4 提高：

\[
84.56\%-82.53\%=2.03\text{ pp}.
\]

可选 common correction 的 best 相比 LUT4 提高2.56 pp，但其后期回落说明存在 drift。由于 CIFAR-10 历史流程没有独立 validation，该85.09%应作为当前工程最佳结果，而不是最终论文的无偏泛化估计。

### 11.2 San Francisco AIRSAR

数据输入为6个 complex64 C3 上三角通道，patch 大小为 \(6\times11\times11\)，共5类。默认使用空间 block 划分：train 53,987，validation 68,320，test 594,094。

默认 FP-shortcut 主流程结果：

| 模型 | Test OA | Test AA |
|---|---:|---:|
| 独立 FP CNN baseline | 94.67% | 81.77% |
| BiReal-topology FP | 92.92% | 81.36% |
| Complex BiReal | 94.35% | 79.99% |
| Categorical LUT4 | 93.55% | 78.87% |
| Residual LUT6 | 93.74% | 78.97% |
| LUT6 + common correction | 93.59% | 78.37% |

Residual LUT6 相比 LUT4只提高0.19 pp OA和0.09 pp AA，仍低于 BiReal 0.60 pp OA，也低于独立 FP baseline。

### 11.3 类别不平衡

San Francisco test split 的类别占比为：

| 类别 | 样本数 | 占比 |
|---|---:|---:|
| Class 1 | 7,185 | 1.21% |
| Class 2 | 41,799 | 7.04% |
| Class 3 | 250,343 | 42.14% |
| Class 4 | 258,190 | 43.46% |
| Class 5 | 36,577 | 6.16% |

Class 3/4 合计占85.60%，且准确率长期接近97%至99%，因此 OA 对少数类变化不敏感。默认 shortcut 下，LUT4到LUT6的逐类变化为：

\[
(-4.89,-0.58,-0.00,-0.19,+6.11)\text{ pp}.
\]

因此 LUT6 的主要正收益来自 Class 5，同时牺牲了 Class 1。额外相位 bit 当前更像对特定少数类边界的条件修正，而不是所有类别共享的普遍增强。

## 12. 消融实验与经验结论

### 12.1 有效设计

- categorical 联合复数输出优于实部、虚部完全独立参数化；
- LUT4 到 LUT6 的严格复制可避免随机 LUT6 的优化崩溃；
- 对四个相位 slice 做零均值 residual 能让额外 bit 真正改变部分真值表，同时保持基础 operation；
- hard-forward/soft-backward 保证训练前向与最终硬件映射一致；
- 小学习率续训能进一步提升已经 fully released 的 residual LUT6。

### 12.2 未获得稳定收益的设计

- 随机 categorical residual 从头训练：CIFAR-10 仅约57.76%；
- TripleLUT6 将三个复数压缩为一个复数：约76.94%；
- Walsh residual：约83.53%，低于 direct residual；
- O5/O6 双输出扩展：best 84.81%，低于 O6-only 85.09%；
- 无参数 option-A shortcut：San Francisco LUT6为93.49%，低于默认 shortcut 93.74%；
- LUT4 common correction：CIFAR-10 有短期收益，但 San Francisco 无收益，不具备稳定迁移性。

### 12.3 地址利用率

在 San Francisco 的12,800个训练 patch 上统计真实地址访问：

| Stage | \(H(p)\) | \(p=1\) | LUT6有效状态 | 归一化地址熵 | 未访问entry |
|---|---:|---:|---:|---:|---:|
| Stage 2 | 0.993 | 0.451 | 30.83/64 | 0.814 | 0.0% |
| Stage 3 | 0.976 | 0.411 | 26.47/64 | 0.776 | 0.0% |
| Stage 4 | 0.903 | 0.323 | 17.43/64 | 0.668 | 2.6% |

这说明额外 bit 没有整体坍缩，绝大多数 entry 也至少被访问过；真正问题是深层访问频率高度集中，导致很多 residual entry 获得的有效训练信号较弱。

## 13. 方法贡献的建议表述

论文中可以把贡献概括为：

1. **复数 LUT 神经元。** 提出两复数输入、一复数输出的局部 LUT 神经元，将复数卷积局部计算直接转化为硬件真值表学习问题。
2. **联合 categorical 输出。** 用四分类码本联合参数化实部和虚部输出，保持复数状态耦合，同时最终仍映射为两张单 bit LUT。
3. **严格等价的 LUT4-to-LUT6 warm start。** 将成熟 LUT4 operation 无损复制到相位扩展空间，避免离散函数随机重置。
4. **零均值条件 residual。** 在不改变共享 LUT4 基础分量的约束下，学习由额外相位 bits 引起的条件修正。
5. **训练部署一致性。** 使用 hard argmax 前向和 softmax STE 反向，训练后可将所有连续参数合并为固定 LUT6 truth table，不增加部署资源。
6. **跨自然图像和 PolSAR 的验证。** 在 CIFAR-10 和 San Francisco AIRSAR 上分析方法收益、地址利用率、类别不平衡和硬件友好性。

应避免声称“显著超过 BNN”或“普遍提高所有类别”。当前证据更准确地支持：该方法能够在严格硬件约束下学习不同于 LUT4 的相位条件布尔映射，并在 CIFAR-10 上恢复大部分 LUT4 量化损失，但其跨数据集增益仍有限。

## 14. 当前局限

1. CIFAR-10 历史主结果使用 test-best 协议，缺少独立 validation；
2. San Francisco 空间划分存在强类别不平衡和明显 domain shift；
3. LUT4 替换仍是 San Francisco 上主要精度损失来源；
4. 相位 bit 使用 stopped gradient，前级特征不会主动为该 bit 重排；
5. 深层 LUT6 地址分布集中，有效状态远少于64；
6. Stem、projection、BN、加法树和 classifier 尚未完全 LUT 化；
7. 当前只验证单 seed，尚缺少均值、标准差和显著性检验；
8. 当前资源结论主要来自结构计数，仍需综合、时序、功耗和吞吐量报告。

## 15. 下一步实验优先级

### 15.1 必须完成

1. 用固定 train/validation/test 协议重跑 CIFAR-10，并报告至少3个 seeds；
2. 同时保存 validation OA best 与 validation AA best；
3. 对 San Francisco 测试 \(\alpha_{end}=0.35\) 和0.5，固定后继续训练；
4. 测试温和类别平衡损失，例如平方根逆频率或 effective-number weighting；
5. 导出最终 hard LUT，并做软件推理与真值表推理逐元素一致性验证；
6. 给出 LUT 数量、FF、BRAM、DSP、Fmax、延迟、吞吐量和功耗。

### 15.2 可选研究方向

1. 设计与 PolSAR C3 相位统计更匹配的第三 bit，而不是沿用当前 comparator；
2. 对低频 LUT 地址使用 occupancy-aware 正则或采样；
3. 研究 layer-wise alpha，使深层 residual 释放更保守；
4. 在不增加最终 LUT 数量的条件下研究类别蒸馏或 LUT4 teacher regularization；
5. 将 FP stem/projection 逐步替换为硬件友好结构，但必须重新训练完整流程。

## 16. 推荐论文结构

1. **Introduction**：复数网络优势、FPGA 实现困难、LUT as operator 的优化问题；
2. **Related Work**：BNN、complex BNN、LUTNet、QAT、FPGA LUT6 映射；
3. **Method**：复数 LUT neuron、categorical LUT4、相位 bit、residual LUT6、STE；
4. **Hardware Mapping**：地址、真值表导出、资源公式、累加与 residual path；
5. **Training Strategy**：Phase 1/2/3/4、strict warm start、alpha schedule；
6. **Experiments**：CIFAR-10、San Francisco、评价协议与硬件平台；
7. **Ablation**：independent/categorical、warm start、residual、alpha、shortcut、O5/O6；
8. **Limitations and Discussion**：类别不平衡、地址集中、非全 LUT 化和协议限制；
9. **Conclusion**：总结硬件一致的可学习复数 LUT 神经元方法。

## 17. 一句话方法定义

> 我们将两个二值复数激活编码为一个共享输入的局部 LUT 神经元，以四分类 categorical 参数化联合生成实部和虚部输出，并通过严格等价的 LUT4 warm start 与零均值相位条件 residual 学习 LUT6 真值表；训练后的全部连续参数被离线折叠为两张硬 LUT6，从而在不增加运行时算术的条件下实现可学习的复数局部布尔映射。
