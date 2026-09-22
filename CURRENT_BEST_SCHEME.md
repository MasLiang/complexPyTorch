# 当前最佳方案：O6-only categorical residual LUT6

## 1. 最终结论

当前最佳结果是 **85.09% test accuracy**，checkpoint 为：

```text
runs/phase3_lut6_commonbase_lr1e4_reslr5e4/chkpts/Bestmodel_phase3.pt
```

该模型来自顺序训练：

```text
Phase 2 complex Bi-Real
  -> categorical LUT4
  -> dominance categorical-residual LUT6
  -> LUT6 residual continuation
  -> 小幅释放 LUT4 common correction
```

最终硬件只使用实部、虚部各一张 LUT6 的 **O6** 输出；不使用 O5，不保留训练期
softmax、base/residual 分解或浮点 logits。

## 2. 网络与 LUT 输入

Phase 3 保持 Phase 2 的 Bi-Real 拓扑、残差连接、complex BatchNorm、stem、projection
和 classifier，只替换残差主分支中的二值复数卷积。

每个量化前复数激活产生三个 bit：

```text
[sr, si, p]
```

- `sr`、`si` 是实部和虚部符号经 `{-1,+1} -> {0,1}` 编码后的 bit；
- `p` 是相位/Gray comparator bit：

  \[
  p=1[\operatorname{sign}(x_i)x_r-\operatorname{sign}(x_r)x_i\ge 0].
  \]

  它等价于按象限调整方向的 `|x_r|`/`|x_i|` 比较，使八个相位区域按 Gray 语义编码；
- 当前 `DOMINANCE_GRAD_MODE=stop`，`p` 不向量化前激活反传梯度。

卷积窗口按 `channel -> kernel_row -> kernel_col` 展开，相邻两个复数位置组成一组：

```text
[sr0, si0, p0, sr1, si1, p1]
```

因此每组使用两张 64-entry LUT6：一张输出实部 bit，一张输出虚部 bit。组数为：

\[
G=\left\lceil\frac{C_{in}K^2}{2}\right\rceil.
\]

各组实部/虚部 bit 分别累加，再进入原有 post complex-BN 和残差连接。

## 3. 最佳 categorical residual 参数化

最终每个 LUT6 地址维护四分类 logits，对应复数输出：

```text
0 -> 00
1 -> 01
2 -> 10
3 -> 11
```

有效 logits 为：

\[
Z(a,p)=B(a)+\beta C_c(a)+\alpha R_c(a,p).
\]

- `B[O,G,16,4]`：从 categorical LUT4 checkpoint 复制并冻结；
- `C_c[O,G,16,4]`：四个 dominance slice 共享的 LUT4 common correction，在类别维减均值；
- `R_c[O,G,64,4]`：LUT6 dominance residual，在同一 LUT4 地址的四个 slice 上减均值；
- `alpha`、`beta`：训练期释放系数，最终均为 1。

forward 始终执行 hard argmax，并将类别映射为两张严格 `{0,1}` LUT6；backward 使用
`softmax(logits/tau)` 代理梯度，当前 `tau=1`。因此训练和部署的前向逻辑始终一致。

## 4. 最佳训练链路

### 4.1 Categorical LUT4 起点

```text
runs/phase3_pair_lut4_categorical_phase2init_sf11_lr002/chkpts/Bestmodel_phase3.pt
```

- 从 Phase 2 checkpoint 解析初始化；
- `LUT_INPUTS=4`，`PAIR_LUT_PARAMETERIZATION=categorical`；
- Adam，LR `0.02`，multistep，256 epochs；
- best test：82.53%。

### 4.2 LUT4 扩展为 dominance LUT6

每个 LUT4 entry 按 `p0,p1` 复制为四个 LUT6 slice，初始 hard forward 与 LUT4 严格
等价；随后只学习零均值 LUT6 residual。

```bash
GPU_ID=0 \
CHECKPOINT=runs/phase3_pair_lut4_categorical_phase2init_sf11_lr002/chkpts/Bestmodel_phase3.pt \
WORKDIR=runs/phase3_lut6_dominance_sharedres_lr2e4_reslr2e3 \
START_FILTER=11 NUM_BLOCKS=3 NUM_EPOCHS=200 \
PHASE3_OPERATOR=pair_lut4 \
PAIR_LUT_PARAMETERIZATION=categorical_residual \
LUT_INPUTS=6 PAIR_LUT_ENCODING=dominance DOMINANCE_GRAD_MODE=stop \
LR=0.0002 SCHEDULE=constant \
DOMINANCE_RESIDUAL_LR=0.002 \
DOMINANCE_RESIDUAL_ALPHA_START=0.1 \
DOMINANCE_RESIDUAL_ALPHA_END=1 \
DOMINANCE_RESIDUAL_RAMP_EPOCHS=120 \
./run_phase3.sh
```

主训练 best test：84.28%。

### 4.3 LUT6 residual 续训

```bash
GPU_ID=0 \
CHECKPOINT=runs/phase3_lut6_dominance_sharedres_lr2e4_reslr2e3/chkpts/Bestmodel_phase3.pt \
WORKDIR=runs/phase3_lut6_dominance_sharedres_continue_lr1e4 \
START_FILTER=11 NUM_BLOCKS=3 NUM_EPOCHS=100 \
PHASE3_OPERATOR=pair_lut4 \
PAIR_LUT_PARAMETERIZATION=categorical_residual \
LUT_INPUTS=6 PAIR_LUT_ENCODING=dominance DOMINANCE_GRAD_MODE=stop \
LR=0.0001 SCHEDULE=constant \
DOMINANCE_RESIDUAL_LR=0.001 \
DOMINANCE_RESIDUAL_ALPHA_START=1 \
DOMINANCE_RESIDUAL_ALPHA_END=1 \
DOMINANCE_RESIDUAL_RAMP_EPOCHS=1 \
./run_phase3.sh
```

续训 best test：84.56%。

### 4.4 小幅释放 LUT4 common correction

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
DOMINANCE_RESIDUAL_ALPHA_START=1 \
DOMINANCE_RESIDUAL_ALPHA_END=1 \
DOMINANCE_RESIDUAL_RAMP_EPOCHS=1 \
DOMINANCE_BASE_LR=0.0001 \
DOMINANCE_BASE_ALPHA_START=0 \
DOMINANCE_BASE_ALPHA_END=1 \
DOMINANCE_BASE_RAMP_EPOCHS=20 \
./run_phase3.sh
```

- best test：**85.09%**，epoch 39；
- final test：84.34%；
- 必须使用 `Bestmodel_phase3.pt`，不能使用 last checkpoint。

## 5. 硬件结果

对每个输出 channel 和每个输入分组，最终导出：

- 一张 64-entry、1-bit 输出的实部 LUT6；
- 一张 64-entry、1-bit 输出的虚部 LUT6；
- 每张 LUT 只使用 O6；
- 两个 LUT 输出沿 group 维分别计数累加；
- 训练期 categorical logits、LUT4 base、common correction 和 residual 在导出前合并为
  唯一 hard truth table。

## 6. 已确认的消融结论

- sequential release 最好：85.09%；
- 从 LUT4 同时释放 residual/common：84.61%；
- 完全 scratch：57.76%，已停止；
- Walsh 参数化：83.53%；
- O5/O6 双输出：best 84.81%，已回退；
- TripleLUT6 三复数压缩：76.94%；
- 当前不应继续增加 LUT 数量或 common correction 强度。

## 7. 当前限制与下一步

- Phase 2 best 约 85.05%，当前 85.09% 只略高 0.04 pp；主要价值是获得可部署、可学习的
  dominance LUT6，而不是显著超越 BNN。
- 当前使用 train/test、无 validation，并按 test best 保存 checkpoint；该数字与旧论文协议
  对齐，但不能当作严格独立验证结果。
- 85.09% 后回落到 84.34%，当前首选下一步是冻结 hard LUT，只恢复 BN、projection、
  stem、classifier 和其他非 LUT 参数，而不是继续提高 LUT 学习率。
