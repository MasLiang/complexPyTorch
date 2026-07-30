<!-- experiment-entry:route-reset-phase12-20260729 -->
## 2026-07-29 - Phase 1/2 route reset and legacy-route archive

### 1. 本次路线切换的目标与结论

本次不是在旧实验树上继续修补，而是对活动代码做一次基线重置：

- 活动训练路线只保留 Phase 1 和 Phase 2。
- Phase 1 是 full-precision complex Bi-Real ResNet。
- Phase 2 是 binary complex activation + binary complex convolution weight。
- Phase 2 默认从 Phase 1 checkpoint 初始化，也可显式使用 `--train-from-scratch`。
- Phase 2.1、Phase 3/3.1/3.2/3.5/3.6、Phase 4/4.2/5，以及 LUT5、LUT6、C8、magnitude、dominance、MLP/NIN、fixed-analytic 等旧路线全部退出活动入口。
- 按用户要求，`complexPyTorch/complexLayers.py`、`complexPyTorch/lut_backend.py` 和 `lut_cuda/` 中的 LUT complex layer 与后端实现继续保留，但 Phase 1/2 模型不会实例化 LUT module。
- 旧代码、日志、报告、checkpoint 和运行结果均归档，不做删除。

### 2. 归档前仍在运行的任务

清理开始时发现 fixed-analytic dominance 任务仍在运行：

- Workdir: `runs/fixed_analytic_dominance_from_phase1_cmpbeta1_e200`
- 计划 epoch: 200
- epoch 174 时先写入 `route_reset_snapshot.md`
- 终止信号生效前，正在执行的 epoch 175 完整写入日志
- 最终状态：175/200，incomplete
- 最佳 validation accuracy：0.6250，epoch 173
- epoch 173 对应 test accuracy：0.6217
- 独立最佳 test accuracy：0.6242，epoch 166
- epoch 175：train 0.6031，validation 0.6164，test 0.6187
- epoch 174 快照统计：8/8 code occupied，dominance ratio 0.4982，occupancy entropy 0.9857，三个 bit 的记录梯度均非零

任务已通过进程组 SIGTERM 停止，并确认没有旧 trainer/launcher 进程继续运行。最终摘要由
`scripts/analyze_training_run.py` 生成到
`backup/route_reset_phase12_20260729/fixed_analytic_final_summary.md`。

### 3. 主归档内容

归档根目录：

`backup/route_reset_phase12_20260729/`

机器可读清单：

`backup/route_reset_phase12_20260729/archive_manifest.json`

旧目录树：

`backup/route_reset_phase12_20260729/legacy_tree/`

主归档工具移动了以下 42 个 repository-relative path，并在 manifest 中记录目标路径、文件数量、总字节数；小于 16 MiB 的单文件还记录 SHA-256：

1. `training.py`
2. `training.py.orig`
3. `training_direct_lut5.py`
4. `training_dual_lut6.py`
5. `training_fixed_analytic.py`
6. `training_mlp_flow.py`
7. `complexPyTorch/complexBinaryResNet.py`
8. `complexPyTorch/complexBinaryResNet.py.orig`
9. `complexPyTorch/complexLayers.py.orig`
10. `complexPyTorch/directLut5Flow.py`
11. `complexPyTorch/dualLut6Flow.py`
12. `complexPyTorch/fixedAnalyticDominanceFlow.py`
13. `complexPyTorch/mlpFlow.py`
14. `run_all_phases.py`
15. `run_direct_lut5.sh`
16. `run_dual_lut6.sh`
17. `run_fixed_analytic.sh`
18. `run_magnitude_lut5.sh`
19. `run_mlp_flow.sh`
20. `run_phase2p1.sh`
21. `run_phase3p1.sh`
22. `run_phase3p5.sh`
23. `run_phase3p6.sh`
24. `run_phase4.sh`
25. `run_phase5.sh`
26. `PHASE4_LUT_EXPERIMENT_LOG.md`
27. `scripts/`：34 个旧脚本
28. `tests/`：37 个旧测试文件
29. `reports/`：22 个旧报告文件
30. `runs/`：655 个旧运行文件
31. `logs/`：2 个根日志文件
32. `chkpts/`：20 个根 checkpoint 文件
33. `bi_workdir/`：旧完整工作目录
34. `train_acc.txt`
35. `train_loss.txt`
36. `val_acc.txt`
37. `val_loss.txt`
38. `test_acc.txt`
39. `test_loss.txt`
40. `pixel_mean.pt`
41. `__pycache__/`
42. `complexPyTorch/__pycache__/`

主归档完成后的静态扫描又发现 `build/` 是旧源码的 generated package 副本，仍含旧 Phase 3/4/5 内容。因此：

- `archive_legacy_routes.py` 增加可复用的 `--supplement` 机制。
- `build/` 被作为 supplement 移至 `legacy_tree/build/`。
- supplement 同样写入 `archive_manifest.json`。
- 活动树不再存在这份陈旧 generated source。

目录型记录内部的每个文件均按原相对路径原样移动；完整逐文件定位以 manifest 和 `legacy_tree/` 为准。

### 4. 归档支持文件

- `backup/route_reset_phase12_20260729/archive_legacy_routes.py`
  - dry-run-first 的归档工具。
  - 支持首次归档和后续 `--supplement`。
  - 新增路径规范化，拒绝绝对路径和 `..` 越界。
- `backup/route_reset_phase12_20260729/archive_manifest.json`
  - 记录 42 个主路径、retained snapshots、恢复的 Phase 1/2 资产和 `build/` supplement。
- `backup/route_reset_phase12_20260729/README.md`
  - 说明归档结构、旧任务最终状态、重要路径和恢复方法。
- `backup/route_reset_phase12_20260729/restore_legacy_path.py`
  - 可选择单文件或目录恢复。
  - 默认 dry run。
  - 只复制，不删除归档内容。
  - 目标存在时拒绝覆盖。
- `backup/route_reset_phase12_20260729/fixed_analytic_final_summary.md`
  - 由统一日志分析脚本生成的中断任务最终摘要。
- `backup/route_reset_phase12_20260729/legacy_tree/runs/fixed_analytic_dominance_from_phase1_cmpbeta1_e200/route_reset_snapshot.md`
  - epoch 174 时的停止前现场记录。
- `retained_snapshots/`
  - 保存重置前的 `README.md`、`CHANGELOG.md`、`run_training.sh`、`complexLayers.py` 和 `lut_backend.py` 快照。

此前已有的 `backup/magnitude_lut5_20260714/` 保持原样，没有覆盖或搬动。

### 5. 保留并恢复的 Phase 1/2 资产

从归档的旧 `bi_workdir/` 中复制回活动树：

- `bi_workdir/chkpts/Bestmodel_phase1.pt`
- `bi_workdir/chkpts/Bestmodel_phase2.pt`
- `bi_workdir/pixel_mean.pt`
- `bi_workdir/phase_metrics.json`

`phase_metrics.json` 已过滤为只含 `phase1` 和 `phase2`。

保留基线：

- Phase 1：best validation 0.9092，test 0.8948，epoch 170
- Phase 2：best validation 0.8310，test 0.8221，epoch 126

### 6. 活动代码逐文件变更

- `complexPyTorch/complexBinaryResNet.py`
  - 从旧多路线 router 重建为 Phase 1/2-only model。
  - `ACTIVE_PHASES=(1, 2)`。
  - Phase 1 block 使用 `ComplexReLU + ComplexConv2d`。
  - Phase 2 block 使用 `BinaryComplexActivation + BinaryComplexConv2d`。
  - Phase 1/2 保持相同 state-dict key 和 tensor shape。
  - Phase 3+ 构造会直接报错。
  - 删除活动模型对 C8、MLP、LUT5/LUT6 和实验 flow module 的依赖。

- `training.py`
  - 从旧 4800+ 行多路线 trainer 重建为 Phase 1/2-only trainer。
  - CLI 的 `--phase` choices 只有 1 和 2。
  - 删除 LUT/C8/MLP/Phase 3+ 参数与调度入口。
  - 删除 `bireal_tune` 隐式参数覆盖。
  - `batch_size`、`num_epochs`、`lr`、optimizer 和 schedule 始终服从外部参数。
  - schedule 只保留 `bireal`、`cosine`、`constant`。
  - warmup 只增长到用户设置的 base LR，不会放大到另一个固定值。
  - cosine 对数值误差做上限钳制，严格不超过 base LR。
  - Phase 2 默认要求 Phase 1 checkpoint；支持显式 `--checkpoint` 和 `--train-from-scratch`。
  - checkpoint loader 去除 `module.`/`_orig_mod.` wrapper，按 key 和 shape 映射，并要求所有 trainable parameters 成功加载。
  - 每个 epoch 原子保存 `Lastmodel_phaseN.pt`。
  - best 保存为 `Bestmodel_phaseN.pt`，不会再使用跨 phase 的共享名字。
  - history 文件也增加 phase 前缀。
  - DDP rank 0 使用 unwrapped model 做验证，避免单 rank 进入 DDP forward 同步。
  - compile 请求仍支持；complex operator 下将 inductor 路由到 `aot_eager`。
  - CIFAR10/CIFAR100/SVHN、validation split、no-validation、pixel mean 和 augmentation 基线保留。

- `run_training.sh`
  - 保留为通用 Phase 1/2 launcher。
  - 默认 dataset/workdir 改为 repository-relative 明确目录。
  - DDP 统一使用选定 `PYTHON_BIN -m torch.distributed.run`。
  - compile 默认关闭。
  - 使用 `exec` 让退出码和信号正确传递。

- `run_phase1.sh`
  - 新增 Phase 1 专用入口。
  - 支持 `GPU_ID`、`WORKDIR`、`DATADIR`、`NUM_EPOCHS`、`BATCH_SIZE`、`LR`、`SCHEDULE`、`WEIGHT_DECAY` 等环境变量。
  - 可选显式 Phase 1 warm-start checkpoint。

- `run_phase2.sh`
  - 新增 Phase 2 专用入口。
  - 默认 checkpoint 为 `bi_workdir/chkpts/Bestmodel_phase1.pt`。
  - 支持 `CHECKPOINT` 覆盖和 `TRAIN_FROM_SCRATCH=1`。
  - 支持 binary weight scale 配置。
  - 默认 workdir 与 Phase 1 分离。

- `scripts/analyze_training_run.py`
  - 新增统一训练日志分析器。
  - 解析 epoch train/validation/test 指标、配置、完成状态、best validation、best test 和 last。
  - 同一个日志含多次 invocation 时，只分析最后一次运行。
  - 支持 Markdown/JSON 和 `--output`。

- `scripts/append_experiment_log.py`
  - 新增统一日志追加工具。
  - 支持 body/body-file/stdin、日期和 dedupe key。
  - 本条记录即通过该脚本写入。

- `scripts/audit_active_route.py`
  - 新增活动路线审计。
  - 检查训练 phase 集合、活动 phase launcher、实验 module 是否退出活动树。
  - 检查 Phase 1/2 model 不实例化 LUT。
  - 同时检查保留的 LUT layer 仍可导入。

- `scripts/__init__.py`
  - 将新维护脚本目录定义为可测试的 Python package。

- `tests/test_phase12_route.py`
  - 覆盖 Phase choices、实验 CLI 清理、model layer 类型、state key 一致性、checkpoint 全量映射、Phase 1/2 前后向、LR 上限和 LUT layer 保留。

- `tests/test_log_tools.py`
  - 覆盖最新 invocation 日志解析和实验日志 dedupe。

- `tests/__init__.py`
  - 定义当前测试 package。

- `README.md`
  - 首页增加当前 Phase 1/2 路线、运行命令、归档入口和 LUT layer 保留说明。

- `CURRENT_TECHNICAL_ROUTE.md`
  - 新增当前技术路线契约。
  - 记录 phase 定义、转换规则、默认参数、输出文件、并发 workdir 约束、基线、归档和验证命令。

- `EXPERIMENT_LOG.md`
  - 新建活动路线实验总日志。
  - 旧完整历史保存在归档的 `PHASE4_LUT_EXPERIMENT_LOG.md`，不再继续向旧日志追加。

### 7. 明确保留且本次未改写的实现

以下文件在进入清理前已有用户/前序实验改动。本次没有回退、没有重写，只按要求继续保留：

- `complexPyTorch/complexLayers.py`
- `complexPyTorch/lut_backend.py`
- `lut_cuda/` 下的 source、binary 和 build artifacts

这些文件仍可能包含 LUT4/LUT5/LUT6/C8 等实现，但它们现在只是未接入活动 Phase 1/2 的 building blocks。活动 router、trainer 和 launcher 均没有相应入口。

导入 LUT backend 时会出现 PyTorch 关于 `torch.cuda.amp.custom_fwd/custom_bwd` 的 FutureWarning。该警告来自保留代码，不影响本次 Phase 1/2 路线审计，因此本次未顺手改动。

### 8. 验证结果

在 `lut_net` conda 环境中完成：

- `python -m py_compile`：新的 trainer、model、分析脚本、审计脚本、归档与恢复脚本全部通过。
- `bash -n`：`run_training.sh`、`run_phase1.sh`、`run_phase2.sh` 全部通过。
- `python -m unittest discover -s tests -v`：10/10 tests passed。
- Phase 1 和 Phase 2 model state-dict key 集合完全相同。
- retained canonical Phase 1 checkpoint 含 258 个 tensor；258/258 全部映射到新的 Phase 2 model，0 unmatched、0 shape mismatch、0 missing buffer。
- canonical model Phase 1/2 参数量均为 261,719。
- 小模型 Phase 1/2 forward 和 backward 均成功。
- route audit：
  - active route exposes only Phase 1/2
  - Phase 1/2 models instantiate no LUT modules
  - retained LUT complex layers remain importable
- selective restore tool dry-run 成功，未写入目标。
- reusable analyzer 成功读取旧中断任务并识别 incomplete。
- 进程扫描确认没有 retired experiment trainer/launcher 继续运行。
- 验证过程生成的活动 `__pycache__` 已清除；历史 cache 仍保存在 archive 中。

本次没有重新执行 200 epoch 的完整 Phase 1/2 训练；完整训练成本较高，兼容性由 canonical checkpoint 全量映射、前后向测试和已有基线指标覆盖。下一次真正训练后，应先运行 `scripts/analyze_training_run.py`，再基于生成摘要追加实验结论。

### 9. 新路线的运行与分析规则

Phase 1：

```bash
GPU_ID=0 WORKDIR=runs/phase1 ./run_phase1.sh
```

Phase 2：

```bash
GPU_ID=1 \
WORKDIR=runs/phase2 \
CHECKPOINT=bi_workdir/chkpts/Bestmodel_phase1.pt \
./run_phase2.sh
```

并发实验必须使用不同 `WORKDIR`。Phase-specific 文件名可避免 Phase 1 与 Phase 2 互相覆盖，但两个相同 phase 的并发任务如果共用同一 workdir，仍会覆盖各自的 best/last checkpoint。

后续任何新技术路线应作为新的、显式命名的 flow 加入，并先与当前 Phase 1/2 baseline 对比；不要重新把旧实验分支塞回 Phase 1/2 router。每次修改文件均继续追加本日志，每次分析新 run 先调用统一分析脚本。

<!-- experiment-entry:route-reset-phase12-final-verification -->
## 2026-07-29 - Route-reset final verification supplement

### 补充文件变更

- `.gitignore`
  - 新增 Python `__pycache__`、`*.pyc`、pytest/coverage cache 和 `*.egg-info` 忽略规则。
  - 新增仅针对 repository root 的 `/build/`、`/runs/`、`/logs/`、`/chkpts/` 忽略规则。
  - 不忽略 `backup/`、`bi_workdir/`、`*.pt`、LUT CUDA source 或现有二进制，因此归档与保留基线仍清晰可见。
  - 目的：完成本次提交后，运行测试和训练不会再次用 generated cache/build output 污染活动源码状态。

- `backup/route_reset_phase12_20260729/archive_legacy_routes.py` 与
  `restore_legacy_path.py`
  - 已设置 executable bit；所有 root launcher 和三个维护脚本也均确认可执行。

### 最终补充验证

- Phase 1 launcher 使用 `PYTHON_BIN=echo` dry-run：
  - `GPU_ID=7`
  - `NUM_EPOCHS=3`
  - `BATCH_SIZE=16`
  - `LR=0.007`
  - `SCHEDULE=constant`
  - 所有值原样进入 `training.py`，未发生覆盖。
- Phase 2 launcher 使用 `PYTHON_BIN=echo` dry-run：
  - `GPU_ID=6`
  - `CHECKPOINT=bi_workdir/chkpts/Bestmodel_phase1.pt`
  - `NUM_EPOCHS=4`
  - `BATCH_SIZE=32`
  - `LR=0.009`
  - `SCHEDULE=cosine`
  - phase、checkpoint 与全部数值原样进入 `training.py`。
- 最终 manifest JSON 可解析：
  - primary moved paths：42
  - supplement：`build/`，6 files，81,857 bytes
  - restored Phase 1/2 assets：4
- 活动 entrypoint retired-route 关键词扫描无结果。
- 除 `legacy_tree/` 中有意保留的历史 cache 外，活动树没有 `__pycache__`。
- 本次修改文件的 `git diff --check` 通过。
- 全 worktree `git diff --check` 仍报告
  `lut_cuda/lut_conv_binary_cuda_backend.cu` 第 290、327 行的既有 trailing whitespace；该文件属于用户此前保留的 LUT CUDA 改动，本次没有修改或回退。
- 最终进程扫描无旧训练任务。

<!-- experiment-entry:pair-lut-neuron-phase3-implementation -->
## 2026-07-29 - LUT-as-neuron Phase 3 implementation

### 1. 路线目标

本次在保留 Phase 1/2 和旧 LUT building blocks 的前提下，新增全新的 Phase 3：从“LUT as operator + spatial weight 协同优化”切换为“LUT as neuron”。

每个输出 channel 的 Phase 2 二值复数卷积核按 `[input_channel, ky, kx]` flatten，每两个复数权重组成一个 neuron。该 neuron 接收两个二值复数 activation，即 4 个输入 bit `[x0_r, x0_i, x1_r, x1_i]`，并通过共享输入的两张 4-LUT 分别输出 real/imag 1 bit。每个 output channel 和每个 flatten pair 都有独立、可学习的 real/imag truth table。

### 2. Phase 2 到 Phase 3 初始化

对每一对固定的 Phase 2 二值复数权重，遍历全部 16 种输入状态，直接计算：

```text
sum_r = x0_r*w0_r - x0_i*w0_i + x1_r*w1_r - x1_i*w1_i
sum_i = x0_r*w0_i + x0_i*w0_r + x1_r*w1_i + x1_i*w1_r
lut_r = 1[sum_r >= 0]
lut_i = 1[sum_i >= 0]
```

Phase 2 latent weight 先按原 `complex_binary_weight` 语义取 sign；其 complex magnitude mean alpha 作为 Phase 3 固定 `output_scale` buffer。硬 LUT 输出按 `(count - pair_count/2) * 4 * alpha` 恢复为对称累加尺度。

当 `C * K * K` 为奇数时，最后一组的第二个复数输入连接固定-low dummy bit，初始化时第二个数学权重为 0。因此这一尾组只有 4 个可达输入状态，不额外引入 spatial weight。

canonical `bi_workdir/chkpts/Bestmodel_phase2.pt` 转换结果：

- 18 个 residual main-path conv 全部转换成功。
- 1934 个 flatten pair LUT neuron。
- 7 层存在 odd tail。
- 2,022,592 个 real+imag LUT entries。
- source weight exact-zero component 为 0。
- 初始化后 sign diff 为 0。
- Phase 3 主干中没有 `conv_r/conv_i` spatial weight 参数。
- Phase 3 总参数量为 2,031,663；增加来自独立 LUT logits，不是保留旧主卷积权重。

### 3. 可微 LUT 与退火

LUT logits 通过 `(tanh(logit * tau) + 1) / 2` 得到 soft entry。渐进 hard 使用：

```text
forward = soft + hard_ratio * stop_gradient(hard - soft)
hard = 1[logit >= 0]
```

因此 forward 从 soft 连续混合到 hard，而 backward 在包括 `hard_ratio=1` 的 fully-hard 阶段仍沿 soft derivative 更新 LUT。输入 activation 在 forward 使用二值 bit，backward 使用 STE 回传到前层。

默认 200 epoch：

- epoch 1-160：`hard_ratio=0`，tau 从 0.5 几何退火到 10。
- epoch 161-200：tau 固定 10，`hard_ratio` 从 1/40 逐步增加到 1。
- epoch 200 是第一个 fully-hard training/evaluation epoch。
- Phase 3 的 epoch 数若不足以到达 fully hard，会在训练开始前拒绝运行。
- 只有 fully-hard epoch 可以更新 `Bestmodel_phase3.pt`；`Lastmodel_phase3.pt` 每个 epoch 正常保存。
- checkpoint 额外保存显式 uint8 hard real/imag truth tables、初始化转换报告和 LUT sign-diff。
- 每 10 epoch及 hard 阶段记录 tau、hard ratio、real/imag sign diff。

LUT 参数使用独立 optimizer group，无 weight decay。默认 network LR 为 0.001 cosine，LUT LR 为 0.01 constant；两者均可从外部配置。

### 4. 文件修改总结

- `complexPyTorch/complexLayers.py`
  - 新增 `annealed_binary_table`。
  - 新增 `PairLUTNeuronConv2d`，包含 Phase2 truth-table 初始化、pair connection、odd-tail dummy、CPU reference、CUDA K=4 backend、soft/hard STE、sign diff 和 hard table 导出。
  - 保留原有 `ComplexLUTConv2d`、`LUTAwareComplexBinaryConv2d` 等旧 building blocks。

- `complexPyTorch/complexBinaryResNet.py`
  - active phases 更新为 1/2/3。
  - Phase 3 residual main conv 路由到 `PairLUTNeuronConv2d`。
  - Phase 1/2 路由保持 full precision / binary；直接构造模型时默认根据 phase 推导 `is_binary`。
  - 增加 LUT logit/tau 构造参数，不恢复旧实验 flow 参数。

- `training.py`
  - 新增严格的 Phase2-to-Phase3 converter。
  - 新增 LUT optimizer group、独立 LR schedule、tau/hard-ratio scheduler、sign-diff 聚合和 hard table capture。
  - Phase 3 禁止 from scratch，要求 Phase 2 checkpoint。
  - 只允许 fully-hard 指标选择 best checkpoint。
  - 增加全部 Phase 3 CLI 参数及启动前校验。

- `run_phase3.sh`
  - 新增可执行 launcher。
  - 默认 canonical Phase2 checkpoint、200 epochs、160+40 annealing、network cosine LR 与 LUT constant LR。
  - 所有关键配置均可由环境变量覆盖。

- `tests/test_phase12_route.py`
  - active route 断言更新为 Phase 1/2/3。
  - 保留 Phase1/2 state-key 和 checkpoint 兼容测试。
  - 新增 Phase3 类型及“无 spatial weight”断言。

- `tests/test_pair_lut_phase3.py`
  - 新增 16 状态 truth-table exact test。
  - 新增 odd-tail dummy independence test。
  - 新增 soft 与 fully-hard STE 下输入/LUT 双向梯度测试。
  - 新增 Phase2 checkpoint 全层转换和默认退火终点测试。

- `scripts/audit_active_route.py`
  - 路线审计更新为仅暴露 Phase 1/2/3。
  - 验证 Phase1/2 无 LUT，Phase3 仅使用 pair-LUT 且无主干 spatial weight，旧 LUT building blocks 仍可导入。

- `README.md`
  - 增加 Phase3 路线说明与启动命令。

- `CURRENT_TECHNICAL_ROUTE.md`
  - 详细记录 pair-LUT 数学定义、flatten 顺序、odd-tail、尺度、退火、checkpoint 和命令契约。

- `EXPERIMENT_LOG.md`
  - 通过 `scripts/append_experiment_log.py` 追加本条，作为后续 Phase3 实验分析基线。

### 5. 验证结果

- Python 语法检查通过。
- `run_phase3.sh`、`run_phase2.sh`、`run_training.sh` 的 `bash -n` 通过。
- launcher dry-run 确认所有默认参数原样进入 `training.py`。
- exhaustive 16-state truth table 与直接两次复数乘加+阈值结果完全一致，最大误差 0。
- canonical full model 在 GPU 上 forward/backward 成功，输出 shape `(2, 10)` 且 finite。
- 36 个 real/imag LUT parameter tensor 全部获得梯度，LUT gradient sum 约 1071.24。
- CPU reference 与 CUDA K=4 kernel 对比：最大绝对误差约 `5.86e-6`，平均绝对误差约 `9.02e-7`。
- 完整单元测试 15/15 passed。
- active route audit passed。

本次没有启动 200 epoch CIFAR-10 正式训练，因此尚无 Phase 3 accuracy 结论。第一组正式结果出来后，应先运行：

```bash
python scripts/analyze_training_run.py runs/phase3_pair_lut
```

再基于本条中的 sign diff、soft/hard transition 和 hard checkpoint 指标分析 LUT neuron 是否真正改变 truth table并带来收益。

<!-- experiment-entry:pair-lut-phase3-random-scratch -->
## 2026-07-29 - Phase 3 random initialization from scratch

### 目标与行为

按新实验要求，Phase 3 现在允许完全不读取 Phase 1/2 checkpoint，直接随机初始化整网训练。默认 checkpoint truth-table 初始化仍保留，只有显式设置 `--train-from-scratch` 或 launcher 环境变量 `TRAIN_FROM_SCRATCH=1` 时才使用新路径。

scratch 模式下：

- stem、projection、BN affine、classifier 等普通参数使用 PyTorch 原生随机初始化。
- 每个 pair-LUT neuron 的 real/imag entry logits 独立采样自 `Normal(0, LUT_LOGIT_INIT)`。
- 默认 `LUT_LOGIT_INIT=1.0`，此时它表示随机分布标准差，不是 Phase2 truth-table 模式下固定的绝对 logit magnitude。
- 初始 hard truth table 由随机 logit sign 决定，理论上 0/1 各约 50%。
- 随机初始 sign 被保存为 sign-diff baseline，因此 epoch 0 的 sign diff 为 0，后续统计仍表示相对本次随机起点发生的真值表变化。
- 没有 Phase2 alpha 时，固定 `output_scale=1.0`。
- soft-to-hard 的 160+40 默认退火、独立 LUT LR、hard-only best checkpoint 规则不变。
- checkpoint 中使用 `phase3_initialization.mode=random_normal` 记录该实验来源，不会与 Phase2 truth-table 初始化混淆。

### 文件修改

- `complexPyTorch/complexLayers.py`
  - `PairLUTNeuronConv2d` 新增 `initialize_random(logit_std)`。
  - 随机初始化同步设置 logits、初始 sign、output scale 和 initialized buffer。
  - Phase2 初始化报告增加 `mode=phase2_truth_table`，使两条路径可区分。

- `training.py`
  - 新增 `initialize_phase3_random`，遍历并初始化全部 pair-LUT 层。
  - Phase3 的 `--train-from-scratch` 不再被禁止。
  - 无 checkpoint 时必须显式初始化 LUT，避免零 logits 或未初始化 layer。
  - checkpoint metadata 统一改为 `phase3_initialization`。

- `run_phase3.sh`
  - 新增 `TRAIN_FROM_SCRATCH=1` 分支。
  - scratch 分支只传 `--train-from-scratch`，不再传 `--checkpoint`。
  - 启动信息打印 random-normal 标准差，避免误认实验来源。

- `tests/test_pair_lut_phase3.py`
  - 新增随机分布均值/标准差、初始化状态、sign-diff baseline 和前后向测试。
  - 新增 Phase3 scratch checkpoint resolver 测试。

- `README.md`
  - 增加随机从头训练的最短命令。

- `CURRENT_TECHNICAL_ROUTE.md`
  - 将 Phase3 定义更新为 Phase2 truth-table 与 random-normal 两种显式初始化模式。
  - 记录 scratch 参数和 output scale 语义。

- `EXPERIMENT_LOG.md`
  - 使用 `scripts/append_experiment_log.py` 追加本条。

### 建议首轮命令

```bash
GPU_ID=0 \
TRAIN_FROM_SCRATCH=1 \
LUT_LOGIT_INIT=1.0 \
LUT_LR=0.01 \
LR=0.001 \
WORKDIR=runs/phase3_pair_lut_scratch_std1 \
./run_phase3.sh
```

这是结构变化较大的实验：不再继承 Phase2 的 82% 左右起点，早期准确率明显更低属于预期。判断价值时应重点观察训练是否持续上升、LUT sign diff 是否增长、以及最终 fully-hard epoch 是否恢复，而不能用最初几个 epoch 直接与 checkpoint fine-tuning 比较。

<!-- experiment-entry:pair-lut-phase3-random-scratch-verification -->
## 2026-07-29 - Phase 3 random scratch verification supplement

### LR 补充

完整从头训练不再沿用 checkpoint fine-tuning 的 network LR。run_phase3.sh 现在按模式选择默认值：

- TRAIN_FROM_SCRATCH=1：network LR=0.01。
- Phase2 checkpoint 初始化：network LR=0.001。
- 两种模式的 LUT LR 默认均为 0.01 constant。
- 外部显式设置 LR 时始终覆盖上述默认值。

因此 scratch 首轮正式命令以 LR=0.01 为准；上一条记录命令中的 LR=0.001 已被本补充替代。

### 最终验证

- canonical-size 随机 Phase3：18 层、1934 个 pair-LUT neuron 全部成功初始化。
- GPU full-model batch-2 forward/backward 成功，输出 shape (2, 10) 且全部 finite。
- LUT gradient sum 约 281.15。
- 完整测试 17/17 passed。
- active route audit passed。
- scratch launcher dry-run 确认参数包含 --train-from-scratch，不包含 --checkpoint，network LR 为 0.01，LUT LR 为 0.01。
- 本次相关文件 git diff --check 通过。

最终建议命令：

```bash
GPU_ID=0 \
TRAIN_FROM_SCRATCH=1 \
LUT_LOGIT_INIT=1.0 \
LUT_LR=0.01 \
LR=0.01 \
WORKDIR=runs/phase3_pair_lut_scratch_std1 \
./run_phase3.sh
```

<!-- experiment-entry:real-versus-complex-pair-lut-comparison -->
## 2026-07-29 - Real LUT-BiReal versus complex pair-LUT comparison

### 对比对象

读取并对照了实数工程 ~/workspace/LutNet/cifar10_bireal 中的 train.py、lut_birealnet.py 和实际存在的 lut_layer_main_compile.py，并使用新脚本解析 log_2x.txt、log_2x_baseline.txt、log_2x_baseline_3values.txt、log_lr0p01.txt。用户提到的 lut_layer_main_compiler.py 实际文件名是 lut_layer_main_compile.py。

### 实数 LUT-BiReal 的关键训练机制

- 每个 residual block 使用 BinaryActivation -> lut_conv_group -> BatchNorm -> residual add。
- activation forward 是 0/1 hard bit，backward 是 Bi-Real piecewise surrogate。
- 每个 LUT 默认 6 输入、1 输出，64-entry；一个 3x3 patch flatten 后每 6 个 scalar bit 分组，各 LUT 输出求和。
- LUT logits 使用双峰初始化：50% 来自 Normal(-1, 0.2)，50% 来自 Normal(+1, 0.1)。这使初始 hard table 约 50/50，但 entry 几乎都远离 0。
- BasicBlock 构造后立即 hard.fill_(1)，所以从第一个 epoch 开始 forward 就是物理 0/1 truth table。
- hard LUT backward 使用 (hard - logits).detach() + logits，对 logit 的代理导数恒等于 1；没有 tanh 饱和。
- optimizer 是 Adam，LUT 和其他参数共用 LR；默认无 weight decay。
- 没有 clip_grad_norm 或 clip_grad_value。
- LR 使用线性衰减；训练 256 epoch。
- 使用全部 50k CIFAR-10 train、标准 mean/std normalize、RandomCrop、flip、CIFAR10 AutoAugment 和 label smoothing 0.1。

### 历史曲线

可复用脚本 scripts/analyze_real_lut_bireal_log.py 的结果：

- log_2x.txt（LUT）：epoch 1/5/10/20/50/100/150/256 为 23.02/34.68/41.76/48.83/69.34/76.59/80.39/83.47%，best 83.47%。
- log_lr0p01.txt（LUT，未完整结束）：epoch 1/5/10/20/50/100 为 23.17/34.60/46.63/55.46/74.79/85.06%，epoch 132 best 87.50%。
- log_2x_baseline.txt：best 81.45%。

当前复数 random scratch run runs/phase3_pair_lut_scratch_std1 在 epoch 20：train 31.78%，val 34.66%，test 33.76%。它并非完全没有学习，但明显慢于实数 LUT 同期 48.83%-55.46%。

### 最关键差异与诊断

1. 优化器不一致。
   实数 LUT 使用 Adam；当前复数 run 使用 SGD。LUT 有 2,022,592 个独立 entry，单 batch 对 entry 的更新具有稀疏、尺度不均特性，Adam 的 per-entry 自适应步长比 SGD 更匹配。

2. 全局 gradient clipping 严重压缩 LUT 更新。
   当前复数训练默认 clipnorm=1、clipval=1，实数版本完全不裁剪。canonical-size batch-16 诊断得到：LUT raw L2=1.769，其他参数 L2=23.878，全局 L2=23.944，因此 clip coefficient 仅 0.04176。LUT mean |grad|=3.28e-4，经 clipping 和 SGD LR=0.01 后，典型单步更新约 1.37e-7。这个量级无法让 Normal(0,1) logits 有效跨 0。

3. 当前 LUT 的离散 truth table 几乎没有学习。
   epoch 1 sign diff 29/2,022,592；epoch 11 仅 278/2,022,592，即 0.0137%。准确率上升主要可能来自 stem、projection、BN 和 classifier 对近似随机 LUT feature 的适配，而不是 LUT operation 本身改变。

4. hard/soft 和代理梯度完全不同。
   实数版本从 epoch 1 就 hard-forward，部署目标与训练 forward 一致，同时 logit backward 恒等于 1。当前复数版本前 160 epoch soft-forward，之后才渐进 hard，并且即使 fully hard，backward 仍是 tanh soft derivative。tau 增大后，远离 0 的 logits 会梯度饱和。这会重现此前“LUT sign 不动”的问题。

5. 初始化不同。
   实数版本是稳定的 bimodal ±1 初始化；当前复数是 Normal(0,1)，约 38% entry 落在 |logit|<0.5。若切 hard identity STE，后者可能产生过多早期翻转；若保持 soft+tanh，又会形成长期连续 surrogate 与最终 hard table 的目标错位。

6. 数据和训练 recipe 也更强。
   实数版本有 Adam、256 epoch、全 50k train、AutoAugment、标准差 normalize、label smoothing；当前复数是 45k train、200 epoch、较弱 augmentation、无 label smoothing。这能解释部分 generalization gap，但不能解释 sign diff 几乎为 0；LUT 更新机制仍是第一优先级。

### 判断与下一步优先级

当前 run 可以继续用于证明 soft-SGD 路径的上限，但不适合作为“随机 LUT 能否从头训练”的公平结论。最优先应做一个 real-compatible controlled branch，而不是先改复数数学结构：

1. LUT 和普通参数改用 Adam，先用统一 LR=0.001 或对 LUT 单独 0.001。
2. 关闭 global clipnorm/clipval，或至少将 LUT 排除在普通参数的全局裁剪之外。
3. 增加 bimodal random init：Normal(-1,0.2)/Normal(+1,0.1)。
4. 增加 hard-forward + identity-logit-STE 模式，第一 epoch 就使用 hard table，不做 tau annealing。
5. 保持当前 4-input/2-output complex pair-LUT 结构不变，先隔离训练 recipe 的影响。
6. 记录每 epoch LUT grad norm、update norm、sign diff；若这条 real-compatible 模式仍失败，再检查复数双输出、ComplexBN 或 pair grouping 的结构影响。

建议下一组实验与当前 run 并行，而不是立刻覆盖：一组严格复现实数 recipe（Adam + no clipping + bimodal + hard identity STE），一组只改 Adam + no clipping，以做消融。

### 文件修改

- scripts/analyze_real_lut_bireal_log.py：新增可复用实数 LUT 日志解析器，输出 epoch 数、best/last、LR 首尾和固定 milestone accuracy。
- EXPERIMENT_LOG.md：使用 scripts/append_experiment_log.py 追加本次对比分析。

本次只做分析和日志工具，没有修改当前正在运行的 Phase3 数学或训练行为，也没有停止现有 run。

<!-- experiment-entry:phase3-real-compatible-20260729 -->
## 2026-07-29 - Phase3 real-compatible 训练机制对照分支

### 动机与可检验假设

对照 ~/workspace/LutNet/cifar10_bireal 后，当前复数 pair-LUT 从头训练较慢不能直接归因于复数 4-input/2-output 结构。现有 run 同时存在 SGD、全局 clipnorm=1、Normal(0,1)、长 soft annealing 等配方差异。一次 canonical-size 梯度诊断中，LUT raw L2=1.769、其他参数 L2=23.878、全局 L2=23.944，因此全局裁剪系数约为 0.04176；SGD LR=0.01 下典型 LUT entry 更新只有约 1.37e-7。epoch 11 的 sign diff 也只有 278/2,022,592（0.0137%）。

本次新增隔离的 real-compatible 分支。假设是：如果训练不动主要来自优化配方，那么 hard-forward identity STE + Adam + no clipping 应显著增加 LUT sign diff，并加快前 20-50 epoch 的准确率增长；如果 sign diff 已明显增加但准确率仍低，下一步才应优先检查复数双输出、pair grouping、ComplexBN 或累加尺度等结构差异。

### 对照模式的数学行为

- 复数结构保持不变：每个 pair-LUT neuron 仍接收两个二值复数 activation，即 4 bit，共享地址并分别输出 real/imag 两个 bit。
- LUT 前向从 epoch 1 起严格 hard：entry 为 1[logit >= 0]。
- LUT 反向使用 identity-logit STE：(hard - logit).detach() + logit，因此 LUT entry 对自身 logit 的局部导数恒为 1，不再经过 tanh 饱和。
- real-compatible 模式不使用 tau annealing 或 hard transition；所有 epoch 都满足硬件可部署 forward，可参与 best checkpoint 选择。
- scratch LUT 初始化为 50/50 双峰混合：N(-1,0.2) 与 N(+1,0.1)。原 Normal(0,std) 与原 annealing 模式保留为默认，不受影响。

### 专用训练配方

run_phase3_real_compatible.sh 默认参数：

- train from scratch，256 epochs，单卡全局 batch size 256；
- Adam；普通参数 LR=0.01，LUT LR=0.01；
- 两组 LR 都使用从 base LR 开始、逐 epoch 线性下降的 schedule；
- weight decay=0，clipnorm=0，clipval=0；
- CIFAR-10 RandomCrop + horizontal flip + AutoAugment，标准 mean/std normalize；
- label smoothing=0.1；
- 使用全部 50k train，不设 validation，以复现实数代码的数据协议。

命令：

```bash
GPU_ID=1 \
WORKDIR=runs/phase3_pair_lut_real_compatible \
./run_phase3_real_compatible.sh
```

注意：no-validation 模式使用 test accuracy 选择 best checkpoint，只适合训练机制复现实验，不能作为无偏最终 test 报告。真正架构比较仍应另跑保留 validation 的版本。

### 观察指标

1. 每 10 epoch 的 Pair-LUT Sign Diff 数量与比例，必须明显高于 annealing+SGD run 的 0.0137%（epoch 11）。
2. epoch 10/20/50 accuracy 与实数 LUT 日志 milestone 46.63%/55.46%/74.79% 对照，但只作为优化速度参考，因为网络拓扑和复数双输出不同。
3. hard forward 从 epoch 1 开始，因此不存在 soft accuracy 与部署 hard accuracy 的转换落差。
4. 若 sign diff 仍接近 0，检查 Adam 实际 update norm 和地址 occupancy；若 sign diff 大量变化但 accuracy 剧烈波动，优先降低 LUT_LR，而不是重新引入全局 clipping。

### 文件修改

- complexPyTorch/complexLayers.py：为 PairLUTNeuronConv2d 增加 real_compatible hard identity STE；增加可配置双峰初始化；保留原 annealed_binary_table 默认行为。
- complexPyTorch/complexBinaryResNet.py：将 lut_training_mode 从模型入口传递到所有 Phase3 pair-LUT 层。
- training.py：增加 LUT init/training mode 参数、双峰配置、linear 普通/LUT schedule、label smoothing、real_lut CIFAR-10 augmentation；real-compatible 从 epoch 1 标记 fully hard；scratch initializer 记录具体初始化元数据。
- run_phase3.sh：暴露 optimizer、clipping、augmentation、label smoothing、LUT training/init mode 和双峰参数；支持 NO_VALIDATION。
- run_phase3_real_compatible.sh：新增一键对齐实数 LUT-BiReal 配方的独立启动器。
- tests/test_pair_lut_phase3.py：新增 hard 0/1 forward、identity gradient、双峰统计、首轮 fully-hard 和 linear LR 测试。
- tests/test_phase12_route.py：将 linear schedule 纳入不超过 base LR 的回归检查。
- scripts/audit_active_route.py：将新的合法 Phase3 recipe wrapper 纳入 active-route 白名单。
- README.md：增加对照模式命令、行为和 test-selection 风险说明。
- CURRENT_TECHNICAL_ROUTE.md：记录 normal/bimodal 初始化、anneal/real-compatible 两条 Phase3 训练路径及完整配方。
- EXPERIMENT_LOG.md：通过 scripts/append_experiment_log.py 追加本条详细记录。

### 验证

- conda run -n lut_net python -m unittest discover -s tests -v：21/21 通过。
- conda run -n lut_net python scripts/audit_active_route.py：通过。
- bash -n run_phase3.sh run_phase3_real_compatible.sh：通过。
- 使用 PYTHON_BIN=/bin/echo dry run 核对，完整参数包含 Adam、LR/LUT_LR=0.01、linear、no clipping、bimodal、real_lut、label smoothing=0.1、no-validation 和 train-from-scratch。
- git diff --check 的唯一告警来自用户此前保留的 lut_cuda/lut_conv_binary_cuda_backend.cu 两处尾随空格，本次没有修改该文件。

<!-- experiment-entry:git-route-binary-cuda-alignment-20260729 -->
## 2026-07-29 - Git 路线分支与 binary CUDA 全面对齐

### 路线分支

本次将两条技术路线正式分离，Git 中只提交代码、启动脚本、测试和必要文档，不提交 checkpoint、数据集、runs、训练日志或 CUDA 编译产物。

- 旧 LUT-as-operator 路线恢复到 `archive/lut-operator-legacy`，提交为 `bab0ade`，并已推送到 origin。恢复源来自 `backup/route_reset_phase12_20260729`；其中旧版 complexBinaryResNet 支持 Phase 1/2/2.1/3/3.1/3.5/3.6/4/5。
- 用户保留的 `complexLayers_lut_as_op_backup.py` 与 `lut_backend_lut_as_op_backup.py` 和路线重置快照完全一致。名为 `complexBinaryResNet_lut_as_op_backup.py` 的文件实际只允许 Phase 1/2，因此旧分支采用备份目录中真正支持旧 phase 的版本。
- 当前 LUT-as-neuron 路线位于 `route/lut-neuron-real-aligned`。它保留 Phase 1/2，并以两个二值复数输入、一个二值复数输出的 4-input/2-output pair-LUT 作为 Phase 3。

旧分支验证：Python 编译通过，全部 shell 启动器语法通过，旧路线单元测试 106/106 通过。提交前检查确认没有 `.pt`、`.pth`、`.so`、`.o`、`.pyc`、runs、data 或 chkpts 文件进入 Git。

### 与实数 LUT-BiReal 的 kernel 对齐

实数参考实现的真实调用链是 `train.py -> lut_birealnet.BasicBlock -> lut_conv_group -> lut_conv_func_group -> lut_kernel_group -> LUT6Function.apply -> torch.ops.mylib.lut_forward -> lut_cuda_grouped_binary.forward/backward`。其前向把浮点形式的 0/1 输入按 `>0.5` 硬索引；反向对 LUT 返回被选 entry 的梯度，对输入返回相邻两个 entry 的差值梯度。

当前复数 pair-LUT 的 real-compatible 路径现已使用同构的 packed binary CUDA lookup：

1. 两个复数 activation 形成 `[x0_r,x0_i,x1_r,x1_i]` 四个地址 bit，real/imag 使用相同地址、两张独立 16-entry table。
2. `BinaryComplexActivation` 先产生 -1/+1，再用 `(x+1)/2` 转成 0/1 probability 编码。binary kernel 使用 `>0.5`，避免把 0 错当成 high。
3. Bi-Real signed activation 的代理导数为 `2(1-|x|)`，随后的除以 2 恰好得到 `1-|x|`，与实数参考实现的 0/1 Bi-Real 反向尺度一致。
4. 输入 bit 按 channel 每 32 位打包，offset 覆盖 kernel 空间和输入 channel，shift 指定字内 bit。输出通道也由 CUDA kernel 分 tile 处理。
5. annealing 路线仍使用 floating multilinear kernel；`real_compatible` 默认使用 binary kernel。可用 `--lut-kernel-mode auto|floating|binary` 显式控制，且 binary 模式只允许 hard real-compatible forward。
6. CUDA 回归测试使用 18 个复数输入 channel 和 35 个输出 channel，覆盖多 input packed word 和多 output tile；binary/floating 在 hard Boolean corner 的 forward、activation gradient、real LUT gradient、imag LUT gradient逐项完全相等。

### 当前路线文件修改

- `complexPyTorch/lut_backend.py`：binary autograd wrapper 增加 signed/probability 输入编码；旧调用默认 signed，pair-LUT 使用 probability。
- `complexPyTorch/complexLayers.py`：PairLUTNeuronConv2d 增加 auto/floating/binary kernel 选择、packed offsets/shifts 构造和 binary CUDA 调用；阻止 annealing 错用 binary backend。
- `complexPyTorch/complexBinaryResNet.py`：把 LUT kernel mode 从模型入口传到所有 Phase3 block。
- `training.py`：增加 `--lut-kernel-mode` 并传入模型。
- `run_phase3.sh`：暴露、打印并传递 `LUT_KERNEL_MODE`。
- `run_phase3_real_compatible.sh`：默认 `LUT_KERNEL_MODE=binary`。
- `tests/test_pair_lut_phase3.py`：增加非法模式检查以及跨 packing/tile 边界的 CUDA forward/backward 精确等价测试。
- `README.md`：说明 real-compatible 默认 binary CUDA 与 0/1 梯度约定。
- `CURRENT_TECHNICAL_ROUTE.md`：记录 backend 选择、数学梯度关系和验证范围。
- `EXPERIMENT_LOG.md`：使用可复用的 `scripts/append_experiment_log.py` 写入本条记录。

### 下一次实验

使用 `run_phase3_real_compatible.sh` 从头训练，默认 Adam、普通参数/LUT LR 都为 0.01、linear 256 epochs、batch size 256、bimodal LUT logits、hard identity STE、binary CUDA、无全局梯度裁剪、50k train/no-validation。首先观察前 10-20 epoch 的准确率、LUT sign diff、LUT update norm 和 GPU 吞吐，再判断训练不动是否仍来自 complex pair-LUT 结构本身。

<!-- experiment-entry:binary-cuda-validation-handoff-20260729 -->
## 2026-07-29 - binary CUDA 回归验证与运行交接

### 验证结果

- `CUDA_VISIBLE_DEVICES=0 conda run -n lut_net python -m unittest discover -s tests -v`：23/23 通过。
- CUDA binary/floating 等价测试覆盖 18 complex input channels 和 35 output channels，forward 与全部输入/LUT gradient 完全一致。
- `conda run -n lut_net python scripts/audit_active_route.py`：通过，活动路线只暴露 Phase 1/2/3。
- `bash -n run_phase1.sh run_phase2.sh run_phase3.sh run_phase3_real_compatible.sh`：通过。
- real-compatible dry run 确认参数包含 `real_compatible`、`binary`、Adam、LR/LUT_LR=0.01、linear、bimodal、no clipping、batch size 256、no-validation 和 train-from-scratch。
- 1 epoch 前台诊断成功：训练进入 binary CUDA forward/backward，第 1 epoch test accuracy 25.84%，LUT sign diff 1350/2,022,592，说明 LUT entry 已发生更新；该诊断输出与 checkpoint 仅位于本地 `runs/debug_phase3_real_binary`，不进入 Git。
- 曾启动的 GPU 0 tmux 正式任务已按用户要求停止，GPU 由用户自行启动；推荐独立 workdir 为 `runs/phase3_pair_lut_real_aligned_binary_clean`。

### 用户运行命令

```bash
GPU_ID=0 \
WORKDIR=runs/phase3_pair_lut_real_aligned_binary_clean \
./run_phase3_real_compatible.sh
```

### 文件记录

本条仅通过 `scripts/append_experiment_log.py` 更新 `EXPERIMENT_LOG.md`，记录验证和运行交接；没有修改模型数学、训练配置或 CUDA 实现。

<!-- experiment-entry:phase3-pair-lut-stall-reference-diff-20260730 -->
## 2026-07-30 - Phase3 pair-LUT 109 epoch 停滞与实数参考差异复核

### 当前 run 的事实

使用 `scripts/analyze_training_run.py` 预处理 `runs/phase3_pair_lut_real_aligned_binary_clean`。分析时任务仍在运行，已完成 109/256 epoch：

- best test 49.12%（epoch 92）；
- epoch 109：train 40.40%，test 49.04%；
- epoch 20/50/75/100 test 分别为 47.86%/43.91%/48.07%/46.69%，已经形成长期平台，不是单纯收敛较慢；
- LUT sign diff 从 epoch 1 的 0.0603% 增长到 epoch 20 的 8.2916%、epoch 50 的 15.0882%、epoch 100 的 20.0208%、epoch 104 的 20.2569%。

因此“训不动”不是 LUT 参数没有更新。约五分之一 truth-table entry 已越过零点，但这些大量布尔变化没有形成更好的表示；继续降低 LR 的剩余 epoch 不太可能从 49% 自动跃迁到 80% 以上。

### 0.02 的来源复核

旧路线中表现较好的复数实验 `phase2p1_c8_octants_semantic_phase_beta2_bireal_lr01` 的真实 invocation 是：

```
--optimizer sgd --lr 0.1 --schedule bireal --batch-size 128
```

旧 bireal schedule 使用 5 epoch warmup，所以 epoch 1 打印 0.02，之后为 0.04、0.06、0.08，并从 epoch 5 起达到配置的 base LR 0.1。故记忆中的“初始 LR=0.02”是 SGD base LR=0.1 的首个 warmup 值，不是 Adam 固定 LR、LUT LR，也不是 bimodal logit 初始化尺度。

实数 LUT-BiReal 的最好现存日志 `log_lr0p01.txt` 明确从 Adam LR=0.01 开始：epoch 1/5/10/20/50/100 为 23.17%/34.60%/46.63%/55.46%/74.79%/85.06%，best 87.50%。当前 run 已经正确复现了这个 0.01 Adam 配方。因此直接将当前 Adam LR 和 LUT LR 都翻倍到 0.02 不是“修正遗漏”，只是新的超参数实验；鉴于当前已有 20% sign flip，它也可能加剧无序翻转。

### 已经对齐的部分

- Adam，betas 0.9/0.999；
- network/LUT 初始 LR 0.01，256 epoch 线性下降；
- weight decay 0、无 gradient clipping；
- batch size 256，对应每 epoch 196 batch；
- CIFAR-10 RandomCrop、horizontal flip、AutoAugment、标准 mean/std；
- label smoothing 0.1，全部 50k train、test selection；
- bimodal logits：50/50 的 N(-1,0.2) 和 N(+1,0.1)；
- hard forward + identity logit STE；
- binary CUDA 对 LUT entry 和 activation 使用 neighboring-entry difference gradient。

scheduler 只有 epoch 编号上的一个极小边界差异，不足以解释 35 个百分点的差距。weight decay 为 0 时，两边参数分组差异也不产生实际影响。

### 尚未对齐且可能重要的部分

1. **LUT 函数阶数不同**：实数实现每个 LUT 是 6 independent scalar bits -> 1 bit，有 64-entry 任意布尔函数；当前是两个完整复数 activation，即 4 correlated scalar bits -> real/imag 各 1 bit，每张表只有 16 entries。当前模型只能在固定两复数组内表达四变量交互，无法复现实数 LUT6 的六变量高阶交互。这是最大的表达能力差异。
2. **Block 顺序不同**：实数 block 是 `BinaryActivation -> LUT -> BatchNorm -> residual add`；当前是 `ComplexBatchNorm(pre) -> BinaryComplexActivation -> pair-LUT -> ComplexBatchNorm(post) -> residual add`。额外 pre-BN 会改变每层 truth-table 地址分布。
3. **BN 数学不同**：实数使用普通标量 BatchNorm（eps 1e-5，gamma 初始 1）；当前 post-BN 做 real/imag 2x2 协方差白化和交叉仿射，eps 1e-4，仿射对角初始 sqrt(2)。随机 hard truth table 不断翻转时，复数协方差统计和 LUT 同时移动，优化面明显更复杂。
4. **下采样 shortcut 不同**：实数为 AvgPool2d(stride=2) 后接 stride-1 1x1 conv；当前直接使用 stride-2 complex 1x1 conv。两者不是等价采样。
5. **输入和 stem 不同**：实数 RGB 直接进入实数 conv；当前先由 LearnImagBlock 生成虚部，再经全精度 complex stem。最后分类器也拼接 real/imag。这个结构保留了复数模型本身，但不能称为完全复制实数网络。
6. **尾部 padding 不同**：实数 LUT6 分组不足时重复输入 bit；当前奇数个 complex spatial position 时使用固定 low dummy bit。影响范围较小，但仍是差异。
7. **优化历史不同**：旧成功复数模型通常从 Phase1 checkpoint 开始，用 SGD+bireal schedule、batch 128；当前直接从随机 hard pair-LUT 开始，用 Adam+linear、batch 256。当前既不是旧复数训练路径，也只对齐了实数实现的训练 recipe，而没有对齐实数 LUT6 拓扑。

### 判断与下一步优先级

当前 run 可以停止；继续到 256 epoch 的信息增益很低。下一步不要先把所有 LR 机械改成 0.02，而应先回答“结构有能力、只是 scratch 难，还是 pair-LUT 压缩本身损失太大”：

1. **首选诊断**：从最佳 Phase2 checkpoint 枚举初始化 pair-LUT，第一 epoch 就 hard-forward，然后观察转换后的 epoch-0/1 accuracy。如果一开始接近 Phase2，说明 pair-LUT 表达能力够，问题主要在随机 scratch 优化；如果立即降到很低，说明两复数压成一复数的局部表示本身是瓶颈。
2. **优化器消融**：若 checkpoint 初始化可行，再给当前 K4 pair-LUT 增加与旧复数训练完全一致的 LUT-follow-base bireal schedule，即 SGD base LR 0.1、前五 epoch 0.02->0.1、batch 128。当前代码的 LUT schedule 尚不支持 bireal/follow-base，不能只靠现有环境变量严格复现。
3. **结构消融**：保持 K4/2-output 硬件约束，单独提供 real-style block：去掉 bn_pre、post 改为 real/imag 独立 BN、shortcut 使用 AvgPool+1x1。与当前 covariance-BN block 并行比较，避免把 LR 和拓扑同时改变。
4. Adam LR=0.02 可以作为低优先级对照，但不应作为主实验；当前 sign flip 已经足够多，主要矛盾不是 entry 无法越零。

<!-- experiment-entry:lut6-compression-prebn-correction-20260730 -->
## 2026-07-30 - LUT6 压缩与 pre-BN 来源的结论修正

### 对“6 输入压成 1 bit”的修正

上一条记录把 LUT6 的六变量高阶交互能力直接表述成比当前 LUT4 更强，容易误导。正确比较必须拆成两个维度：

1. **输出带宽/压缩率**：实数局部 LUT6 是 6 scalar input bits -> 1 bit，压缩率确实高；当前 pair-LUT 是两个复数，即 4 scalar input bits -> 2 bits，局部输出带宽更高、压缩更弱。从这个角度，不能说当前结构天然具有更严重的信息瓶颈。
2. **单个布尔函数的交互阶数**：一个 LUT6 的单输出表有 64 entries，可表达任意六变量布尔函数；当前每个 real/imag LUT4 只有 16 entries，各自表达任意四变量函数。LUT6 可以直接建模六个输入之间的高阶交互，而当前只能在固定“两复数一组”内建模四变量交互。
3. **层级输出并非最终 1 bit**：实数卷积 patch 被分成多个六输入组，每组各输出 1 bit，随后这些局部 bits 求和并进入 BN；不是整个 patch 被一个 LUT 压成 1 bit。当前也对多个 pair 的 real/imag bits 分别求和。因此两者都是“分组布尔函数 + 多组整数累加”，只是在组大小和每组输出数上不同。
4. **结论**：LUT6 有更强的单组非线性交互但更窄的局部输出；LUT4 pair 有更宽的局部输出但更低的单组交互阶数。仅凭 6->1 与 4->2 不能判断哪个总体表达瓶颈更大，更不能把当前 49% 平台直接归因于表达能力。后续应优先用 Phase2 truth-table 初始化测试结构上限，再区分表达瓶颈与随机 hard-LUT 优化失败。

### pre-BN 的历史来源

Git 历史显示：

- `complexPyTorch/complexBinaryResNet.py` 首次出现在 commit `228fb43 add binary`，从第一版起，单卷积 `BiRealComplexResidualBlock` 就是 `ComplexBN(pre) -> Sign -> BinaryComplexConv -> ComplexBN(post) -> residual add`。
- 原始同期的全精度 `ComplexResidualBlock` 是两卷积 pre-activation 结构：`BN1 -> ReLU -> Conv1 -> BN2 -> ReLU -> Conv2 -> residual add`。它确实在每个卷积前使用 BN，但第二个卷积后、残差相加前没有额外 post-BN。
- 二值改写把每个传统两卷积 block 展开成两个单卷积 Bi-Real blocks，同时既保留了每个单卷积前的 pre-BN，又新增 post-BN 来控制二值累加尺度。因此当前每个单卷积的双 BN 不是原始全精度 complex block 的严格等价展开，而是二值路线的主动设计。
- 当前 Phase1/2/3 共用这个 block；Phase1 只把 Sign/BinaryConv 换成 ReLU/ComplexConv，仍保留 pre+post BN。旧 Phase1/2 的高精度已经证明 pre-BN 并不会让普通 complex/BNN 必然训崩。
- 实数 LUT-BiReal block 没有 pre-BN，只有 `BinaryActivation -> LUT -> BatchNorm -> residual add`。因此 pre-BN 仍是 scratch pair-LUT 与实数参考的真实差异，可能通过持续重排 truth-table 地址和 complex covariance statistics 增加联合优化难度，但不能在没有消融实验时认定它是唯一原因。
- pre-BN 对 Bi-Real surrogate 也有潜在正面作用：它把 sign 前输入拉回约单位尺度，使 `2(1-|x|)` 的非零梯度窗口得到更多覆盖。直接删除 pre-BN 可能减少梯度，必须作为独立开关消融，不能无条件删除。

### 文件记录

本条仅通过 `scripts/append_experiment_log.py` 更新 `EXPERIMENT_LOG.md`；没有修改模型、训练器或启动命令。

<!-- experiment-entry:pre-post-complex-bn-hardware-20260730 -->
## 2026-07-30 - pre/post ComplexBN 的硬件代价与消融设计

### 当前实现确认

`BiRealComplexResidualBlock` 中：

- `bn_pre = ComplexBatchNorm2d(in_channels, eps=1e-4)`
- `bn_post = ComplexBatchNorm2d(out_channels, eps=1e-4)`
- projection 分支也使用 `ComplexBatchNorm2d`

这里的 ComplexBatchNorm2d 不是 real/imag 独立的 NaiveComplexBatchNorm2d，而是计算 real/imag 的 2x2 covariance whitening，并带有交叉仿射参数。推理阶段 running statistics 固定后，每个复数 channel 可合并为：

```
y_r = A00*x_r + A01*x_i + c0
y_i = A10*x_r + A11*x_i + c1
```

其中 A 同时包含 covariance whitening 和 affine matrix。也就是说，当前 pre/post BN 都需要 real/imag 交叉通路。

### 硬件友好性

**pre-BN 相对更容易折叠**。因为它后面立刻接 sign，实际不需要物化浮点 BN 输出，只需计算：

```
bit_r = 1[A00*x_r + A01*x_i + c0 >= 0]
bit_i = 1[A10*x_r + A11*x_i + c1 >= 0]
```

这可以实现成两个固定系数二维比较器。它仍比普通 threshold 昂贵，因为存在 real/imag cross term，但不需要保留归一化后的幅值。

若 pre-BN 改成 `NaiveComplexBatchNorm2d`，real/imag 完全独立，则 sign(BN(x)) 可严格折叠为每通道两个阈值比较（外加 scale 为负时的极性翻转），基本不需要乘法器。这是最硬件友好的形式。

**post-BN 更难折叠**。它的输出先与 residual 相加，幅值和尺度都会影响后续层，不能只保留 sign。当前结构实际需要：

```
out = A * [lut_count_r, lut_count_i] + c + residual
```

即使 A 和 c 都是推理期常数，也仍需在 LUT 整数累加结果上执行定点常系数乘加；cross term 还会让 real LUT count 参与 imag 输出、imag count 参与 real 输出。可以用 DSP、shift-add 或加权 adder tree 实现，但不能像 pre-BN 那样简单退化为阈值。

post-BN 也不能直接与下一层 pre-BN 完全相消，因为二者之间夹着 residual add；代数组合后仍需要分别缩放 LUT branch 和 identity branch。

### 训练角度

硬件更友好不等于训练中更不重要：

- pre-BN 将 sign 输入拉回单位尺度，有助于 Bi-Real surrogate `2(1-|x|)` 保持非零，但也会随 running/affine 参数变化持续改变 LUT 地址分布。
- post-BN 负责归一化大量局部 0/1 LUT outputs 的整数累加，并校准 residual branch 尺度。实数 LUT-BiReal 保留的正是 post-BN，因此完全删除 post-BN 的训练风险高于删除 pre-BN。
- 当前日志不能判断是哪一个 BN 导致 49% 平台；旧 Phase1/2 同时使用二者仍能训练，说明二者都不是普遍错误，但随机 hard-LUT scratch 下的联合动态可能不同。

### 推荐消融顺序

不要第一步同时删除两个 BN。建议固定优化器、seed、LUT 初始化和其他结构，按以下顺序：

1. baseline：covariance pre + covariance post（当前）。
2. no-pre：无 pre-BN + covariance post。只回答额外 pre-BN 是否妨碍 hard-LUT scratch，并更接近实数 block。
3. naive-pre：naive/diagonal pre + covariance post。若它优于 baseline，说明问题主要来自 pre-BN 的 real/imag covariance cross coupling，而不是 pre-normalization 本身。
4. covariance pre + naive/diagonal post。测试 post-BN 的 cross coupling，同时保留 LUT count 的尺度归一化。
5. no-post 只作为最后的高风险实验；若测试，应同时加入简单可学习/定点友好的 branch scale，否则 LUT count 与 residual 尺度失配会混淆结论。

从硬件与训练的折中看，最有希望的最终形式是 **naive pre-BN（折叠成阈值）+ naive post-BN（两个独立定点 scale/offset）**；是否需要 pre-BN 本身由 no-pre 对照决定。

### 文件记录

本条只通过 `scripts/append_experiment_log.py` 更新 `EXPERIMENT_LOG.md`，没有修改模型代码。

<!-- experiment-entry:bn-mode-ablation-20260730 -->
## 2026-07-30 - Configurable residual pre/post BatchNorm ablation

### 目标与实验假设

当前 complex Bi-Real residual block 同时包含 convolution 前的 pre-BN 和 convolution 后的 post-BN；二者默认均为 real/imag covariance whitening。为区分训练不动究竟来自额外 pre-normalization、real/imag cross coupling，还是 post-BN 本身，本次将 pre/post normalization 解耦为独立开关。

每个位置支持三种模式：

- covariance：原有 ComplexBatchNorm2d，执行 real/imag 2x2 covariance whitening 和交叉 affine。
- naive：NaiveComplexBatchNorm2d，real/imag 分别执行普通 BatchNorm，不发生交叉耦合。
- none：nn.Identity，完全移除该位置的归一化。

默认仍为 pre=covariance、post=covariance，因此不带新参数的旧命令、当前 GPU0 基线以及旧 checkpoint 架构均保持原行为。post 模式同时作用于 main path 的 bn_post 和 projection shortcut 的 BN，确保每个 post-BN 消融配置定义完整。stem 的 bn1 暂时保持 covariance，不混入本轮 residual-block 消融。

### 修改文件

- complexPyTorch/complexBinaryResNet.py：新增统一 BN factory；BiRealComplexResidualBlock 和 BinaryComplexResNet 接收 pre_bn_mode/post_bn_mode；所有 stage block 透传开关；projection BN 跟随 post 模式。
- training.py：新增 --pre-bn-mode 与 --post-bn-mode CLI，choices 为 covariance/naive/none；build_model 透传到网络。
- run_phase3.sh：新增 PRE_BN_MODE/POST_BN_MODE 环境变量、CLI 透传和启动时配置打印。
- run_phase3_real_compatible.sh：显式保留 covariance/covariance 默认，同时允许外部环境变量覆盖。
- tests/test_phase12_route.py：覆盖默认值、全部九种结构组合、projection 路由、naive/none 前后向以及非法模式拒绝。
- README.md：加入用户入口、模式说明和示例。
- CURRENT_TECHNICAL_ROUTE.md：记录开关语义、projection 约束和 stem 不参与本轮切换。
- EXPERIMENT_LOG.md：由 scripts/append_experiment_log.py 追加本条结构化记录。

### 验证

- python -m py_compile complexPyTorch/complexBinaryResNet.py training.py tests/test_phase12_route.py：通过。
- bash -n run_phase3.sh run_phase3_real_compatible.sh：通过。
- git diff --check：通过。
- CUDA_VISIBLE_DEVICES=8 conda run -n lut_net python -m unittest discover -s tests -v：25 tests 全部通过，包括 binary/floating LUT CUDA forward/backward equality。
- 默认 start_filter=11、num_blocks=3 的 Phase 3 参数量：cov/cov 2,031,663；none/cov 2,029,518；naive/cov 2,031,234；cov/naive 2,031,124；naive/naive 2,030,695；cov/none 2,028,968。

### 并行实验矩阵

GPU0 已有的 phase3_pair_lut_real_aligned_binary_clean 是 covariance/covariance 基线，继续保留。

- GPU1：none/covariance。回答额外 pre-BN 是否阻碍 hard pair-LUT scratch。
- GPU2：naive/covariance。与 GPU1 联合判断 pre-BN 的作用来自归一化本身还是 covariance cross coupling。
- GPU3：covariance/naive。单独检验 post-BN cross coupling，同时保留累加尺度归一化。
- GPU4：naive/naive。硬件友好的完整候选，pre 可折叠为独立阈值，post 为独立定点 scale/offset。
- covariance/none 风险最高：post-BN 负责 LUT 累加和 residual branch 的尺度校准，因此暂不列入第一批，除非前四组提示 post-BN 本身有害。

所有实验必须使用唯一 WORKDIR，防止 checkpoint 和日志覆盖；其余 real-compatible 参数与 GPU0 基线完全相同，使用相同默认 seed，便于单变量比较。

<!-- experiment-entry:bn-ablation-results-20260730 -->
## 2026-07-30 - Residual BN ablation results and remaining real-LUT differences

### 已完成的 BN 消融结果

使用 scripts/analyze_training_run.py 统一解析完成实验。四组均采用 real-compatible hard-forward identity-STE、Adam、普通参数与 LUT LR=0.01 线性衰减、batch size 256、50k train、相同 seed 和独立 workdir。

| pre-BN | post-BN | best test | best epoch | final test | best train | final LUT sign diff |
|---|---:|---:|---:|---:|---:|---:|
| none | covariance | 51.33% | 205 | 50.12% | 41.78% | 23.25% |
| naive | covariance | 49.77% | 219 | 48.10% | 39.46% | 22.85% |
| covariance | naive | 51.11% | 251 | 50.87% | 41.13% | 22.57% |
| naive | naive | 48.85% | 239 | 47.36% | 39.04% | 22.64% |

旧 covariance/covariance clean run 在 epoch 130 被提前停止，已达到 best 50.03%，不能作为完整 256-epoch final，但平台与四组一致。

结论：

1. 删除 pre-BN、将 covariance whitening 改为 real/imag 独立 naive BN，均只能产生约 1-2 个百分点波动，无法解释与实数 LUT 网络约 30-35 个百分点的差距。
2. 四组最终都有约 22.5%-23.2% LUT entry sign flip，说明 LUT 参数并非不更新。问题是大量离散变化只形成约 39%-42% augmented train accuracy，属于表示/优化方向不足，而不是 LUT 冻结。
3. covariance/none 仍作为最后一个高风险 BN 对照，命令为：
   GPU_ID=0 PRE_BN_MODE=covariance POST_BN_MODE=none WORKDIR=runs/phase3_bn_covpre_nopost ./run_phase3_real_compatible.sh

### 与实数 LUT-BiReal 已确认对齐的项目

- 数据：CIFAR-10 50k train、RandomCrop、HorizontalFlip、CIFAR10 AutoAugment、相同 mean/std、label smoothing 0.1。
- optimizer：Adam，无 weight decay，无 gradient clipping。
- schedule：初始 LR 0.01，256 epoch linear decay；LUT 与普通参数使用同一 LR。
- LUT logit：50/50 bimodal Normal(-1,0.2) / Normal(+1,0.1)。
- LUT forward：从 epoch 1 使用 hard 0/1 entry，logit 使用 identity STE。
- activation 总梯度：实数 BinaryActivation 的 0/1 surrogate 导数为 1-|x|；复数路径先以 +/-1 输出产生 2(1-|x|)，进入 pair-LUT probability 编码时乘 1/2，合成后同样为 1-|x|。
- binary CUDA 与 floating multilinear backend 已有 exact forward/backward corner regression test，因此当前 binary packing 不是首要嫌疑。

### 仍未对齐且优先级更高的差异

1. LUT fan-in 与局部函数阶数。实数网络每个 neuron 使用 LUT6：6 个 scalar activation bits 到 1 bit，单表 64 entries。当前 pair-LUT 使用两个复数 activation，也就是 4 bits，并分别输出 real/imag bit：两张 LUT4 合计只有 32 entries。当前每个输出分量只能学习 4-bit interaction，不能复现实数 LUT6 的 6-bit interaction。
2. Boolean 参数容量。实数 BiRealNet20 的 LUT entries 约 2,850,816；当前 start_filter=11 pair-LUT 为 2,022,592，只有约 71%。更关键的是，当前每个物理双输出 LUT 只使用 32 个 INIT 自由度，而 LUT6 单输出使用 64 个。
3. 物理利用率。按当前 stage 宽度估算，pair-LUT 需要约 63,206 个 4-input/2-output physical neurons；实数 16-channel LUT6 网络约 44,544 个。当前方案可能消耗更多 physical LUT，却因为只使用 4 个输入而获得更低的局部阶数。
4. 网络前端。实数版是 real Conv stem；当前先用 LearnImagBlock 从 RGB 学习 imaginary，再走 ComplexConv + covariance stem BN。
5. projection。实数下采样 shortcut 为 AvgPool -> 1x1 Conv -> BN；当前是 stride-2 ComplexConv -> selected post-BN。
6. 宽度。实数网络 start width=16 real channels；当前为 11 complex channels，不能只按 channel 数直接等价比较。
7. 随机训练能力尚无控制组。现有 canonical Phase 2 的 82.21% 来自 Phase 1 checkpoint，不是同配方 scratch。必须先判断 complex BNN 本身能否用当前 scratch recipe 训练。

### 下一步控制实验

与 no-post 并行跑 Phase 2 scratch，保持 no-pre/cov-post 和实数训练配方：

GPU_ID=1 TRAIN_FROM_SCRATCH=1 NUM_EPOCHS=256 BATCH_SIZE=256 LR=0.01 SCHEDULE=linear WEIGHT_DECAY=0 WORKDIR=runs/phase2_scratch_real_recipe_nopre_covpost ./run_phase2.sh --optimizer adam --clipnorm 0 --clipval 0 --augmentation real_lut --label-smoothing 0.1 --no-validation --pre-bn-mode none --post-bn-mode covariance

判据：

- 若 Phase 2 scratch 也约 50%，先解决 complex backbone scratch optimization，不能继续归因于 pair-LUT。
- 若 Phase 2 scratch 达到约 80%，则主要瓶颈就是 LUT4 pair grouping。下一版优先考虑将全部 real/imag scalar bits flatten 后每 5 bit 一组，用一块 LUT6_2 实现两个独立 LUT5 outputs。这样每个物理单元拥有 32+32=64 entries，与实数 LUT6 的 64 个布尔自由度相当，同时 physical neuron 数比当前 4-bit grouping 更少；输出仍可解释为一个复数的 real/imag 两个 bit。

<!-- experiment-entry:phase2-scratch-control-20260730 -->
## 2026-07-30 - Phase 2 scratch control reaches 80.96 percent

### 完成结果

使用 scripts/analyze_training_run.py 解析 runs/phase2_scratch_real_recipe_nopre_covpost：

- 状态：complete，256 epochs。
- best test accuracy：80.96%，epoch 255。
- final test accuracy：80.18%。
- best epoch augmented train accuracy：75.26%。
- 配置：Phase 2、train from scratch、pre-BN=none、post-BN=covariance、Adam、LR=0.01 linear decay、batch size 256、weight decay=0、无 gradient clipping、real_lut augmentation、label smoothing 0.1、50k train/no-validation。

### 归因更新

该控制组与随机 hard pair-LUT 的最佳 51.33% 相差 29.63 个百分点，而 backbone、数据、optimizer、schedule、activation surrogate、stem/projection 和 BN 模式保持一致。由此可以排除以下主因：

- complex backbone 无法从头训练；
- no-pre/cov-post 结构本身错误；
- Adam 0.01 linear 配方不适用于复数网络；
- 数据增强或 label smoothing 导致约 50% 平台。

问题已经收敛到 BinaryComplexConv2d 被随机 4-input/2-output PairLUTNeuronConv2d 替换后的表示与优化：

1. 随机 hard LUT sign 在前几十个 epoch 快速大规模翻转，缺少二值卷积提供的连续 latent spatial-weight 参数化和结构先验。
2. 每个输出分量只有 4-bit interaction；实数成功路线是 6-bit interaction。
3. 当前每个物理双输出单元只有两张 LUT4，共 32 个 entry bits；实数 LUT6 为 64。
4. 当前 pair grouping 的 truth tables 完全独立，参数空间维度高但缺少相邻 spatial position 或 complex multiplication 的共享约束，因此 identity STE 的离散搜索难度大。

### 下一步实验

先使用本次最佳 Phase 2 checkpoint 初始化 Phase 3 pair-LUT，而不是随机 bimodal：

GPU_ID=2 TRAIN_FROM_SCRATCH=0 CHECKPOINT=runs/phase2_scratch_real_recipe_nopre_covpost/chkpts/Bestmodel_phase2.pt PRE_BN_MODE=none POST_BN_MODE=covariance WORKDIR=runs/phase3_pair_from_phase2_scratch80_hard ./run_phase3_real_compatible.sh

该实验保持 hard-forward identity STE，但初始真值表来自已达到 80.96% 的二值卷积。它回答两个问题：

- pairwise local threshold 编译本身造成多少即时 accuracy 损失；
- 有结构先验的 LUT 初始化能否避开随机 hard LUT 的 50% 平台。

若初始化后立即显著高于随机版本并能继续恢复，短期路线采用 Phase2 -> Phase3。若仍迅速降到约 50%，则 4-bit pairwise local function 本身是主要表示瓶颈，应转向 flatten scalar bits 后的 5-input/2-output LUT6_2 neuron。
