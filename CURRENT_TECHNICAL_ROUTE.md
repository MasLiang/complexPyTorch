# 当前技术路线：复数 Bi-Real 网络到双 LUTConv 网络

## 1. 路线目标

这条路线的目标是在保留复数神经网络结构和复数运算语义的前提下，将二值复数卷积替换为适合 FPGA 实现的 LUT 卷积。

整个过程分为三个阶段：

1. 使用全精度复数网络建立基础模型。
2. 将网络训练为复数 Bi-Real 二值网络。
3. 保持网络拓扑不变，仅将二值复数卷积替换为可学习的 LUTConv。

这条路线不再把 LUT 设计成额外的复杂算子，也不改变残差网络结构。LUTConv 直接承担原二值卷积的功能。

## 2. Phase 1：全精度复数网络

Phase 1 使用全精度复数激活和全精度复数卷积。

对于复数输入和权重

\[
x=x_r+jx_i, \qquad w=w_r+jw_i,
\]

复数卷积为

\[
y_r=\operatorname{Conv}_{w_r}(x_r)-\operatorname{Conv}_{w_i}(x_i),
\]

\[
y_i=\operatorname{Conv}_{w_r}(x_i)+\operatorname{Conv}_{w_i}(x_r).
\]

因此，一个复数卷积只有两套可学习的实数卷积参数：实部权重和虚部权重。两套权重分别在实部、虚部交叉计算中复用。

Phase 1 的作用是学习完整的复数特征表达，并为后续二值化提供基础。

## 3. Phase 2：复数 Bi-Real 二值网络

Phase 2 保留 Phase 1 的网络拓扑，将残差主分支中的激活和卷积权重二值化。

复数激活被表示为两个二值分量：

\[
x_b=\operatorname{sign}(x_r)+j\operatorname{sign}(x_i).
\]

复数权重同样由二值实部和二值虚部组成。前向传播继续使用标准复数卷积公式，因此仍然只有两套二值卷积权重，并在四个实数卷积项之间交叉复用。

Bi-Real 的代理梯度用于绕过二值符号函数，使二值网络能够通过反向传播训练。除二值激活和二值卷积外，残差连接、归一化、下采样和分类器结构保持不变。

Phase 2 建立了 LUTConv 需要替代的直接基线：一个能够正常训练的二值复数网络。

## 4. Phase 3：可配置分组复数 LUT 卷积

Phase3 保持 Phase2 的 Bi-Real activation、残差、复数 BatchNorm、stem、projection
和 classifier，只替换主分支卷积。Activation 的实部和虚部先各自二值化，再从
`{-1,+1}` 编码为严格 `{0,1}`。

通过 `lut_inputs=k` 配置单张 LUT 的输入 bit 数，当前支持 `k in {4,6}`。每个
二值复数占两个 bit，所以一组包含

\[
n=k/2
\]

个复数 activation：

- `k=4`：两个复数输入，两张 LUT4 分别输出实部、虚部 bit；
- `k=6`：三个复数输入，两张 LUT6 分别输出实部、虚部 bit。

卷积窗口通过 `F.unfold` 按

```text
channel -> kernel_row -> kernel_col
```

展开为 `Cin*kernel²` 个复数位置，再按每 `n` 个连续复数分组。每组地址顺序始终为

```text
[x0_real,x0_imag,x1_real,x1_imag,...]
```

组数为

\[
G=\left\lceil\frac{C_{in}kernel^2}{n}\right\rceil.
\]

尾组不足时，从当前卷积窗口的开头复制输入位置补齐；Phase2 解析初始化将这些补齐
位置的数学权重设为 0。每组的两路 LUT 输出分别沿 group 维累加，得到实部、虚部
`[0,G]` 计数，再进入 post complex-BN 与残差路径。

### Phase2 checkpoint 初始化

提供 Phase2 checkpoint 时，对每组对应的 `n` 个二值复权重遍历全部 `2^k` 个输入
状态，计算

\[
z=\sum_{t=0}^{n-1}x_tw_t,
\]

并以 `real(z)>=0`、`imag(z)>=0` 生成初始双 LUT 表。该解析初始化只是一个可复现
起点，不意味着局部截断运算与原 Phase2 整层累加等价。共享 BN、projection、stem
和 classifier 继续从 Phase2 checkpoint 加载。

## 5. LUT 参数化与硬件映射

两种训练参数化均支持 `k=4/6`，默认仍是 `k=4 + independent`。

### Independent（默认）

维护

```text
[2*Cout, G, 2^k]
```

个 logits；实部和虚部表独立 hard threshold，反向使用 identity STE。

### Categorical complex output（可选）

维护

```text
[Cout, G, 2^k, 4]
```

个 logits。每个地址的四个类别共同表示一个合法二值复数输出：

```text
0 -> 00 -> -1-j
1 -> 01 -> -1+j
2 -> 10 ->  1-j
3 -> 11 ->  1+j
```

forward 使用 hard argmax，乘固定 `[4,2]` 码本后生成两张严格 `{0,1}` 的
`2^k`-entry LUT；backward 使用 `softmax(logits/tau)` 作为 argmax 代理，当前
`tau=1.0`。Phase2 初始化时目标类别 logit 为 0.25，其余类别为 0；from-scratch
时使用 `N(0,0.01)`。

### CUDA 与 FPGA

- `k=4`：将两个 dummy 0 bit 追加到地址，并把 16-entry 表复制扩展成 64-entry，
  调用现有 LUT6 backend；
- `k=6`：64-entry 表和六位地址直接调用现有 LUT6 backend；
- 两种 categorical 配置最终都只导出两张硬表，训练期增加的四分类 logits 不进入
  FPGA，因此不会额外增加推理逻辑。

`complexPyTorch/lut_backend.py` 和 CUDA 源文件均不需要修改。

### 已完成 LUT4 categorical 基线

`runs/phase3_pair_lut4_categorical_phase2init_sf11_lr002` 已完成 256 epochs：

- optimizer Adam，initial LR `0.02`，multistep schedule；
- best test accuracy `82.53%`，epoch 248；
- final test accuracy `81.80%`；
- 全程 256 epochs finite，无 NaN/Inf。

这是本次三复数 LUT6 实验的直接比较基线。

### LUT6 categorical 命令

```bash
GPU_ID=1 \
CHECKPOINT=bi_workdir/chkpts/Bestmodel_phase2.pt \
START_FILTER=11 \
NUM_BLOCKS=3 \
PHASE3_OPERATOR=pair_lut4 \
PAIR_LUT_PARAMETERIZATION=categorical \
LUT_INPUTS=6 \
LR=0.02 \
SCHEDULE=multistep \
WORKDIR=runs/phase3_lut6_categorical_phase2init_sf11_lr002 \
./run_phase3.sh
```

batch 中间日志默认关闭，每个 epoch 打印一次汇总。

## 6. 已完成扩宽实验

此前 same-spatial/channel-priority PairLUT4 的 256-epoch from-scratch 实验：

- best test：`81.10%`，epoch 116；
- final test：`77.83%`；
- Phase2 baseline：`85.05%`。

随后测试了 TripleLUT6：同一 spatial tap 内每三个复数输入六个 bit，两张 LUT6
分别输出实部和虚部。该实验同样训练 256 epoch：

- best test：`76.94%`，epoch 120；
- final test：`67.42%`；
- 相对 PairLUT4 best 下降 `4.16 pp`；
- 完整模型参数约从 `2.03M` 增至 `5.63M`；
- 对 \(C_{in}=11,k=3\)，物理 LUT site 估算从 当前 channel-major PairLUT4 的 50 增至
  TripleLUT6 的 72，约增加 44%。

恢复后的 channel-major PairLUT4 尚未用当前 Phase2 解析初始化重新训练，因此上述 81.10% 仅作为历史布局结果，不能视为当前配置成绩。

因此 TripleLUT6 没有带来精度收益，且后期回落更严重、资源更多，不再作为默认
路线。实现保留用于复现：

```bash
PHASE3_OPERATOR=triple_lut6 ./run_phase3.sh
```

旧 shared-LUT6 complex-convolution 对照也继续保留：

```bash
PHASE3_OPERATOR=shared_lut6 ./run_phase3.sh
```

## 7. 路线总结

\[
\text{全精度复数网络}
\rightarrow
\text{复数 Bi-Real 网络}
\rightarrow
\text{成对 LUT4 复数网络}.
\]

当前结论不是 PairLUT4 已超过 Phase2，而是它在已测试的 LUT-neuron 粒度中优于
TripleLUT6，并使用更少资源，因此恢复为后续优化的基础版本。


## 8. 并行 dominance-bit LUT6 分支

该分支不替换默认 PairLUT4，而是增加一个独立对照。每个量化前复数激活并行产生：

\[
s_r=1[x_r>0],\quad s_i=1[x_i>0],\quad p=1[|x_r|>|x_i|].
\]

每两个复数形成一个六位地址：

```text
[sr0, si0, p0, sr1, si1, p1]
```

一组 categorical logits 的 shape 为 `[Cout, group_num, 64, 4]`，四个类别映射为
复数输出 bit `00/01/10/11`。forward 使用 hard argmax，backward 使用 softmax STE。
`p` 的 forward 是严格比较器。默认 `dominance_grad_mode=stop`，因此 LUT 输出不会
经由 `p` 回传到量化前实部和虚部，但 categorical LUT slice 仍正常学习。旧的
clipped-linear STE 代理保留为 `dominance_grad_mode=ste`，只用于复现实验。

dominance LUT6 的主初始化方式是读取训练完成的 categorical LUT4 Phase3 checkpoint。
对新地址 `[r,i,p,r',i',p']`，按 `[r,i,r',i']` 计算旧地址并复制完整四类
logits，因此每个旧 entry 在两个 dominance 维度上扩展为四个语义等价 entry。
由于 bit 排列交错，旧地址 `0000` 对应的新整数地址为 `0,1,8,9`，不能直接按
连续内存 `repeat_interleave(4)`。

该 warm start 在初始 hard forward 和 softmax proxy 上都与 LUT4 严格等价，随后
`p0/p1` 对应的 logits slice 独立接收梯度并允许分化。Phase2 checkpoint 与
from-scratch 随机 LUT6 仍保留为对照，但不再是推荐起点。默认
`standard + LUT4` 路线保持不变。

运行配置：

```bash
CHECKPOINT=runs/phase3_pair_lut4_categorical_phase2init_sf11_lr002/chkpts/Bestmodel_phase3.pt \
PAIR_LUT_PARAMETERIZATION=categorical \
LUT_INPUTS=6 \
PAIR_LUT_ENCODING=dominance \
DOMINANCE_GRAD_MODE=stop \
DOMINANCE_STE_MARGIN=1.0 \
./run_phase3.sh
```
## 9. 共享 LUT4 基表的 dominance residual

无约束 dominance LUT6 会把每个 LUT4 地址立即拆成四个低样本 slice。新实验模式
`PAIR_LUT_PARAMETERIZATION=categorical_residual` 改为：

[
Z_6(a,p)=B_4(a)+alpha(D_6(a,p)-mean_p D_6(a,p)).
]

- `B4` 从 categorical LUT4 checkpoint 读取并冻结；
- `D6` 为可学习 64-entry residual，初始化为零；
- 同一个旧地址的四个 dominance residual 强制零均值；
- `alpha` 从较小值逐渐增加，控制 LUT slice 的分化速度；
- residual 使用独立学习率，普通网络参数使用更小的 continuation LR；
- forward 从始至终物化为 hard 64-entry LUT6，最终硬件没有额外结构。

推荐实验：

```bash
GPU_ID=0 \
CHECKPOINT=runs/phase3_pair_lut4_categorical_phase2init_sf11_lr002/chkpts/Bestmodel_phase3.pt \
START_FILTER=11 NUM_BLOCKS=3 NUM_EPOCHS=200 \
PHASE3_OPERATOR=pair_lut4 \
PAIR_LUT_PARAMETERIZATION=categorical_residual \
LUT_INPUTS=6 PAIR_LUT_ENCODING=dominance \
DOMINANCE_GRAD_MODE=stop \
LR=0.0002 SCHEDULE=constant \
DOMINANCE_RESIDUAL_LR=0.002 \
DOMINANCE_RESIDUAL_ALPHA_START=0.1 \
DOMINANCE_RESIDUAL_ALPHA_END=1.0 \
DOMINANCE_RESIDUAL_RAMP_EPOCHS=120 \
WORKDIR=runs/phase3_lut6_dominance_sharedres_lr2e4_reslr2e3 \
./run_phase3.sh
```

主训练最佳 test accuracy 为 84.28%。随后从其 best checkpoint 以普通参数 LR
`0.0001`、residual LR `0.001`、`alpha=1` 继续训练 100 epochs，得到当前最佳：

- best test accuracy：**84.56%**（continuation epoch 77）；
- LUT4 categorical warm-start：82.53%；
- dominance residual 相对 LUT4 提升：2.03 个百分点；
- p0/p1 hard category sensitivity：17.04% / 17.02%，说明两个新增 bit 已被 LUT6 使用；
- 最终硬件仍是两张 64-entry、单 bit 输出的 LUT6，不保留训练期 logits、base 或 residual 结构。

当前推荐的续训方式：

```bash
GPU_ID=0 \
CHECKPOINT=runs/phase3_lut6_dominance_sharedres_lr2e4_reslr2e3/chkpts/Bestmodel_phase3.pt \
START_FILTER=11 NUM_BLOCKS=3 NUM_EPOCHS=100 \
PHASE3_OPERATOR=pair_lut4 \
PAIR_LUT_PARAMETERIZATION=categorical_residual \
LUT_INPUTS=6 PAIR_LUT_ENCODING=dominance DOMINANCE_GRAD_MODE=stop \
LR=0.0001 SCHEDULE=constant DOMINANCE_RESIDUAL_LR=0.001 \
DOMINANCE_RESIDUAL_ALPHA_START=1.0 DOMINANCE_RESIDUAL_ALPHA_END=1.0 \
DOMINANCE_RESIDUAL_RAMP_EPOCHS=1 \
WORKDIR=runs/phase3_lut6_dominance_sharedres_continue_lr1e4 \
./run_phase3.sh
```

Walsh/ANOVA 参数化的 clean 对照只有 83.53%，低于 direct categorical residual。
因此 Walsh 已从可执行代码、CLI、测试和当前分析脚本中移除；失败结果仍保留在
`EXPERIMENT_LOG.md` 作为实验历史。


## 10. 可控 LUT4 common correction

在当前 84.56% categorical residual 路线上，增加与 dominance residual 正交的
LUT4 common correction：

\[
Z(a,p)=B(a)+\beta\,C_c(a)+\alpha\,R_c(a,p).
\]

其中 `B[O,G,16,4]` 是冻结的 LUT4 checkpoint logits；
`C[O,G,16,4]` 是新增可学习 common correction，并在四个输出类别上减均值得到
`C_c`；`R[O,G,64,4]` 继续在同一个 LUT4 地址的四个 dominance slice 上减均值
得到 `R_c`。因此 `C_c` 只修改四个 slice 共同的 LUT4 operation，`R_c` 只修改
slice 间差异，两者没有不可辨识的重叠。

该功能默认关闭：`DOMINANCE_BASE_LR=0`、base alpha 为 0。旧 84.56% checkpoint
没有 correction 参数时会自动零初始化，加载后的 hard forward 完全不变。最终仍将总 logits
argmax 后物化为两张 64-entry LUT6，不增加 FPGA 资源。

建议从当前 best checkpoint 做第一组联合续训：

```bash
GPU_ID=0 \
CHECKPOINT=runs/phase3_lut6_dominance_sharedres_continue_lr1e4/chkpts/Bestmodel_phase3.pt \
WORKDIR=runs/phase3_lut6_commonbase_lr1e4_reslr5e4 \
START_FILTER=11 NUM_BLOCKS=3 NUM_EPOCHS=100 \
PHASE3_OPERATOR=pair_lut4 \
PAIR_LUT_PARAMETERIZATION=categorical_residual \
LUT_INPUTS=6 PAIR_LUT_ENCODING=dominance DOMINANCE_GRAD_MODE=stop \
LR=0.00005 SCHEDULE=constant \
DOMINANCE_RESIDUAL_LR=0.0005 \
DOMINANCE_RESIDUAL_ALPHA_START=1 DOMINANCE_RESIDUAL_ALPHA_END=1 \
DOMINANCE_RESIDUAL_RAMP_EPOCHS=1 \
DOMINANCE_BASE_LR=0.0001 \
DOMINANCE_BASE_ALPHA_START=0 DOMINANCE_BASE_ALPHA_END=1 \
DOMINANCE_BASE_RAMP_EPOCHS=20 \
./run_phase3.sh
```


### From-scratch 对照

为检验 LUT4/LUT6 warm-start 是否限制在旧局部最优，增加一组除初始化外完全相同的
对照。`categorical_residual` 无 checkpoint 时，16-entry base 和 64-entry residual
均使用 `N(0,0.01)`，破除 hard argmax 的类别对称；common correction 仍从 0 开始。
提供 LUT4 或旧 residual checkpoint 时，加载逻辑会覆盖随机值并恢复严格等价起点。

```bash
GPU_ID=3 CHECKPOINT= \
WORKDIR=runs/phase3_lut6_commonbase_fromscratch_samecfg \
START_FILTER=11 NUM_BLOCKS=3 NUM_EPOCHS=100 \
PHASE3_OPERATOR=pair_lut4 \
PAIR_LUT_PARAMETERIZATION=categorical_residual \
LUT_INPUTS=6 PAIR_LUT_ENCODING=dominance DOMINANCE_GRAD_MODE=stop \
LR=0.00005 SCHEDULE=constant \
DOMINANCE_RESIDUAL_LR=0.0005 \
DOMINANCE_RESIDUAL_ALPHA_START=1 DOMINANCE_RESIDUAL_ALPHA_END=1 \
DOMINANCE_RESIDUAL_RAMP_EPOCHS=1 \
DOMINANCE_BASE_LR=0.0001 \
DOMINANCE_BASE_ALPHA_START=0 DOMINANCE_BASE_ALPHA_END=1 \
DOMINANCE_BASE_RAMP_EPOCHS=20 \
./run_phase3.sh
```

该命令刻意保留 continuation 的低普通参数 LR，以形成单变量初始化消融。若 scratch
组失败，结论应限定为“当前低 LR schedule 依赖 warm start”，不能直接推出随机 LUT6
无法从头训练。


### Common correction 首轮结果与联合释放消融

从已训练 LUT6 residual 的 84.56% checkpoint 再释放 LUT4 common correction，100
epochs 完成：best test 85.09%（epoch 39），final 84.34%。这说明 common correction
有约 0.53 个百分点收益，但后期存在 drift，必须按 best checkpoint 选择。完全 scratch
同配置在 epoch 60 停止，best 57.76%（epoch 57），说明 continuation 低 LR 明显依赖
预训练起点。

下一组从 82.53% categorical LUT4 checkpoint 开始。加载时冻结 base 精确复制旧 LUT4，
64-entry LUT6 residual 与 16-entry LUT4 common correction 同时为 0；两者都在前 120
epochs 由 alpha 0.1 增至 1，从同一个起点联合释放：

```bash
GPU_ID=3 \
CHECKPOINT=runs/phase3_pair_lut4_categorical_phase2init_sf11_lr002/chkpts/Bestmodel_phase3.pt \
WORKDIR=runs/phase3_lut4_joint_base_and_lut6_residual \
START_FILTER=11 NUM_BLOCKS=3 NUM_EPOCHS=200 \
PHASE3_OPERATOR=pair_lut4 \
PAIR_LUT_PARAMETERIZATION=categorical_residual \
LUT_INPUTS=6 PAIR_LUT_ENCODING=dominance DOMINANCE_GRAD_MODE=stop \
LR=0.0002 SCHEDULE=constant \
DOMINANCE_RESIDUAL_LR=0.002 \
DOMINANCE_RESIDUAL_ALPHA_START=0.1 DOMINANCE_RESIDUAL_ALPHA_END=1 \
DOMINANCE_RESIDUAL_RAMP_EPOCHS=120 \
DOMINANCE_BASE_LR=0.0002 \
DOMINANCE_BASE_ALPHA_START=0.1 DOMINANCE_BASE_ALPHA_END=1 \
DOMINANCE_BASE_RAMP_EPOCHS=120 \
./run_phase3.sh
```

该实验与原 84.28% LUT4->LUT6 residual 主训练相比，唯一新增自由度是同步释放
LUT4 common correction，因此可以直接判断“先训 LUT6 再解冻 LUT4”与“从 LUT4
同时联合优化”哪种更好。
