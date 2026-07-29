#!/usr/bin/env python3
"""Generate the standalone octant-phase LUT5 design document."""

import argparse
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT / "reports" / "lut5_octant_phase_quantization_design.md"
)

HIGH = math.cos(math.pi / 8.0)
LOW = math.sin(math.pi / 8.0)

# Ordered by activation phase: 22.5, 67.5, ..., 337.5 degrees.
# The document convention is d=0 for real dominance and d=1 for imag dominance.
ACTIVATION_STATES = (
    (+1, +1, 0),
    (+1, +1, 1),
    (-1, +1, 1),
    (-1, +1, 0),
    (-1, -1, 0),
    (-1, -1, 1),
    (+1, -1, 1),
    (+1, -1, 0),
)

# Ordered by binary-weight phase: 45, 135, 225, 315 degrees.
WEIGHT_STATES = (
    (+1, +1),
    (-1, +1),
    (-1, -1),
    (+1, -1),
)


def decode_activation(sign_r, sign_i, dominance):
    magnitude_r = HIGH if dominance == 0 else LOW
    magnitude_i = LOW if dominance == 0 else HIGH
    return sign_r * magnitude_r, sign_i * magnitude_i


def local_product(sign_r, sign_i, dominance, weight_r, weight_i):
    quant_r, quant_i = decode_activation(sign_r, sign_i, dominance)
    output_r = quant_r * weight_r - quant_i * weight_i
    output_i = quant_r * weight_i + quant_i * weight_r
    return output_r, output_i


def output_bit(value):
    if abs(value) < 1e-12:
        raise RuntimeError("octant phase table unexpectedly contains zero")
    return int(value > 0.0)


def phase_degrees(real, imag):
    return math.degrees(math.atan2(imag, real)) % 360.0


def truth_table_markdown():
    headers = ["激活相位", "\(s_r\)", "\(s_i\)", "\(d\)"]
    for weight_r, weight_i in WEIGHT_STATES:
        headers.append(
            "\((t_r,t_i)=({:+d},{:+d})\)".format(weight_r, weight_i)
        )
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---:"] * len(headers)) + " |")

    minimum_component = float("inf")
    zero_components = 0
    for sign_r, sign_i, dominance in ACTIVATION_STATES:
        quant_r, quant_i = decode_activation(sign_r, sign_i, dominance)
        row = [
            "| {:.1f} deg | {:+d} | {:+d} | {} |".format(
                phase_degrees(quant_r, quant_i),
                sign_r,
                sign_i,
                dominance,
            )
        ]
        for weight_r, weight_i in WEIGHT_STATES:
            output_r, output_i = local_product(
                sign_r,
                sign_i,
                dominance,
                weight_r,
                weight_i,
            )
            minimum_component = min(
                minimum_component, abs(output_r), abs(output_i)
            )
            zero_components += int(abs(output_r) < 1e-12)
            zero_components += int(abs(output_i) < 1e-12)
            row.append(
                " **{}{}** |".format(
                    output_bit(output_r),
                    output_bit(output_i),
                )
            )
        lines.append("".join(row))

    if zero_components != 0:
        raise RuntimeError("truth table contains {} zero components".format(
            zero_components
        ))
    if not math.isclose(minimum_component, HIGH - LOW, abs_tol=1e-12):
        raise RuntimeError(
            "unexpected minimum component magnitude: {}".format(
                minimum_component
            )
        )
    return "\n".join(lines), minimum_component, zero_components


DOCUMENT_TEMPLATE = r"""# LUT5 八扇区中心相位量化：5 输入到双 1-bit 输出

> 本文档由 `scripts/generate_lut5_octant_phase_doc.py` 自动生成，用于独立整理
> 相位型 LUT5 方案的数学定义、真值表、硬件映射、训练梯度、代码实现和历史实验结果。

## 中文主文

### 1. 核心结论

你的判断是正确的，但适用对象必须是位于
`pi/8, 3pi/8, ..., 15pi/8` 的八个扇区中心，而不是包含坐标轴的
`0, pi/4, pi/2, ...`。

```text
激活 3 bit：[sign(real), sign(imag), dominance]
权重 2 bit：[sign(weight_real), sign(weight_imag)]
                         |
                         v
固定的量化复数乘法
                         |
                         v
局部输出 2 bit：[output_real_bit, output_imag_bit]
```

所以它确实形成一个确定性的：

```text
5 input bits -> 1 real output bit + 1 imag output bit
```

实部和虚部分别是一张 32-entry LUT5。量化复乘结果不会落在实轴或虚轴上，
因此局部实部、虚部都没有 `0` 抵消态。部署时，局部二值比较已经被编入
真值表，不需要在 LUT 后增加比较器。

### 2. 两类 C8 的区别

| codebook | 激活相位 | 包含坐标轴 | 复乘后保证无 0 |
| --- | --- | --- | --- |
| `roots` | `0, pi/4, pi/2, ...` | 是 | 否 |
| `octants` | `pi/8, 3pi/8, ...` | 否 | 是 |

例如 roots 中 `45 deg + 45 deg = 90 deg`，结果实部就是 0。真正满足
“复乘结果没有边界值 0”的版本是 **octant-centered C8**，也可以称为
**offset 8-PSK** 或 **八扇区中心相位量化**。

### 3. 第三个激活 bit 与极性

令激活为 `a=a_r+j*a_i`：

```text
s_r = sign(a_r)
s_i = sign(a_i)

d = 0, |a_r| >= |a_i|   实部占优
d = 1, |a_i| >  |a_r|   虚部占优
```

这与用户图中的定义一致。物理 sign bit 约定为 `1 -> +1`、
`0 -> -1`。

当前仓库代码使用互补极性：

```text
d_impl = 1[|a_r| >= |a_i|] = 1 - d_doc
```

两者的 operation 完全相同，只是交换 LUT 中两个 16-entry 地址半区。
本文表格采用 `d_doc`，即 `0=实部占优`。导出或比较真值表时必须先统一
这个约定。

`|a_r|=|a_i|`、单个分量为 0 或原点仍是输入 quantizer 的边界，需要
确定性规则。本文在相等时选择 `d=0`。这与下面被消除的“局部复乘结果
等于 0”是两件事。

### 4. 八扇区中心解码

定义：

```text
H = cos(pi/8) = {{HIGH}}
L = sin(pi/8) = {{LOW}}
```

按本文 `d` 极性：

```text
m_r(d) = (1-d)*H + d*L
m_i(d) = (1-d)*L + d*H

q_r = s_r*m_r(d)
q_i = s_i*m_i(d)
q   = q_r + j*q_i
```

即：

```text
d=0: q = s_r*H + j*s_i*L
d=1: q = s_r*L + j*s_i*H
```

由于 `H^2+L^2=1`，八个码字的模长都是 1，相位为：

```text
theta_k = (2*k+1)*pi/8, k=0,...,7
        = 22.5, 67.5, ..., 337.5 deg
```

硬件不需要计算 `atan2`、三角函数或归一化。两个 sign comparator 加一个
dominance comparator 已经唯一确定八个相位状态。

### 5. 二值复权重和无 0 证明

二值复权重为：

```text
t_r = sign(w_r) in {-1,+1}
t_i = sign(w_i) in {-1,+1}
w_q = alpha*(t_r+j*t_i), alpha > 0
```

`alpha` 是正尺度，不改变输出 sign。局部复乘为：

```text
y_r = q_r*t_r - q_i*t_i
y_i = q_r*t_i + q_i*t_r
```

权重相位属于：

```text
{pi/4, 3pi/4, 5pi/4, 7pi/4}
= (4*m+2)*pi/8
```

激活相位是 `pi/8` 的奇数倍，权重相位是 `pi/8` 的偶数倍，因此输出
仍是奇数倍：

```text
angle(q*w_q) = odd*pi/8 + even*pi/8 = odd*pi/8
```

坐标轴是 `pi/8` 的 4 的整数倍，所以输出不可能落在坐标轴上。笛卡尔
分量的绝对值只可能为：

```text
|y_component| in {H+L, H-L}
              = {{{STRONG}}, {{WEAK}}}
```

离 0 最近仍有 `H-L={{MIN_COMPONENT}}`。脚本穷举全部 32 个状态、64 个
实虚分量，零分量数量为 `{{ZERO_COMPONENTS}}`。

### 6. 完整 5-to-2 真值表

```text
b_r = 1[y_r > 0]
b_i = 1[y_i > 0]
```

由于合法状态中没有 `y=0`，所以 `>0` 和 `>=0` 完全等价，不再需要
为复乘结果设计 tie rule。下面每个单元格是 `b_r b_i`：

{{TRUTH_TABLE}}

表格输入输出顺序为：

```text
[s_r, s_i, d_doc, t_r, t_i] -> [b_r, b_i]
```

它直接验证输入严格为 5 bit，输出严格为两个 1-bit，而且第五位会改变一部分
地址的输出，并非冗余输入。

### 7. 硬件实现

逻辑上是两张共享五个输入的 LUT5：

```text
LUT_real: [s_r,s_i,d,t_r,t_i] -> b_r
LUT_imag: [s_r,s_i,d,t_r,t_i] -> b_i
```

在 Xilinx 风格的 `LUT6_2` 上，两张共享输入的任意 LUT5 通常可以打包到
一个双输出 primitive，分别通过 `O5/O6` 输出。理想资源映射为：

```text
1 physical LUT6_2 -> 2 logical LUT5 outputs
```

是否成功 packing 仍需查看综合报告。保守逻辑计数是两张 LUT5。

LUT 输出已经是 comparator 结果。卷积只需分别累计实部、虚部 bit：

```text
sum_k (2*b_k-1) = 2*popcount(b)-N_valid
```

当前 fixed analytic 实现用
`2*alpha*(2*count-N_valid)` 恢复训练尺度。zero padding 位置不执行
局部 LUT，也不生成伪造输出。

### 8. 训练梯度

部署端可查 LUT，但训练端应保留解析复乘的梯度几何：

```text
原始复激活
 -> hard sign/sign/dominance，backward 用 STE
 -> 固定 H/L 解码
 -> 与二值复权重做解析复乘
 -> 数值 forward 使用 hard local sign
 -> backward 使用 comparator-aware 平滑代理
 -> 累加、BN、残差网络
```

按本文极性，dominance 代理为：

```text
p_d = sigmoid(beta_d*(|a_i|-|a_r|))
d_forward  = 1[|a_i|>|a_r|]
d_backward = p_d
```

代码使用其互补形式。关键是后级梯度不仅到达第五个 bit，还能通过
`|a_r|-|a_i|` 继续回传给原始 `a_r/a_i`。两个 sign bit 使用各自的
Bi-Real STE。

局部 comparator 可使用：

```text
z_soft = tanh(beta_c*z/scale)
output = hard(z).detach() - z_soft.detach() + z_soft
```

这保证训练的数值 forward 从第一个 epoch 起就是硬件 bit，但 backward 来自
解析复乘边界。它不同于 LUT 地址多线性插值。相同 hard truth table 并不意味着
相同 Jacobian 或相同训练难度。

### 9. 历史实验结论

| 阶段 | 局部行为 | best val | 对应 test |
| --- | --- | ---: | ---: |
| Phase2 C4 | 2-bit 激活，普通二值复卷积 | 83.10% | 82.21% |
| Phase2.1 C8 semantic-STE beta2 | 3-bit 相位激活，保留 H/L 两档后再累加 | **84.50%** | **84.16%** |
| Phase3 LUT4-aware | C4 局部复乘压成双 sign bit | 82.42% | 81.54% |
| Phase3.1 C8 progressive-zero | C8 局部复乘压成双 sign bit | 80.12% | 79.22% |

Phase2.1 证明第三个 semantic bit 在梯度正确时确实有价值，并超过了 Phase2。
但 Phase2.1 尚未压成 2-bit 输出，它仍保留 `H+L=1.3066` 与
`H-L=0.5412` 两档局部幅值。

Phase3.1 才是本文的最终 5-to-2 operation。它虽然消除了 0 边界，却把
strong/weak 两档都压成相同正负幅值，所以：

```text
没有 0 边界 != 没有信息损失
```

这解释了 Phase2.1 到 Phase3.1 仍下降约 4 到 5 个百分点。历史 Direct
LUT5 从 Phase1 直接 hard lookup 的 dominance 分支 best val/test 约为
`70.48%/70.59%`，而且少量 shared LUT entry 翻转会造成剧烈波动。当前
fixed analytic 分支因此固定 operation，并改用 comparator-aware backward。

### 10. Gray Code 与直接 semantic 地址

Gray code 只是八个相位状态的双射重排，其三个 bit 不逐位等于
`[sign_r,sign_i,dominance]`。若物理输入本来就来自两个 sign comparator 和
一个 dominance comparator，直接 semantic 地址更合适：

- 不需要额外 Gray encoder；
- 每个 pin 有明确物理含义；
- 三个位可拥有各自的训练代理；
- 需要 Gray 表时仍可离线置换地址，无需重训。

### 11. 当前代码与严谨边界

- `complexPyTorch/complexLayers.py::C8ComplexActivation`：
  roots/octants、Gray/semantic 编码和 STE。
- `complexPyTorch/complexLayers.py::C8LUTAwareComplexBinaryConv2d`：
  历史 Phase3.1 local-comparator endpoint。
- `complexPyTorch/fixedAnalyticDominanceFlow.py`：
  当前 fixed analytic comparator-aware 分支及 LUT5 离线导出。
- `run_phase2p1.sh`、`run_phase3p1.sh`、`run_fixed_analytic.sh`：
  对应实验入口。

当前实现已验证 32 状态 bit-exact、训练/硬件数值前向一致、三个激活 bit
均有梯度、dominance 梯度可继续进入原始实虚部，并且训练图中没有 LUT lookup
或可训练 LUT 参数。新分支尚无完整 accuracy，因此“严格无 0”是数学与实现
结论，不应提前写成精度提升结论。

严谨结论是：

> 八扇区中心 C8 激活与对角二值复权重相乘后，输出相位仍是
> `pi/8` 的奇数倍，实部和虚部均不会为 0。它们的 sign 构成两个确定的
> 5 输入布尔函数，可以实现为两张共享输入的 LUT5，并可尝试打包到一个
> 双输出 LUT6_2。

限制包括：roots 不满足无 0 保证；输入 quantizer 自己仍有 tie；双 1-bit
输出会丢弃 strong/weak 类别；zero-free 不自动保证更高 accuracy；LUT6_2
packing 必须由综合报告确认。

---

# English Technical Appendix

## 1. Short Answer

Yes. With the **octant-centered** activation codebook, the local operation is:

```text
3 activation bits: [sign(real), sign(imag), dominance]
2 binary-weight bits: [sign(weight_real), sign(weight_imag)]
                         |
                         v
fixed quantized complex multiply
                         |
                         v
2 local sign bits: [output_real_bit, output_imag_bit]
```

It is therefore a deterministic **5-input to 2-output Boolean operation**.
The real and imaginary functions are two 32-entry LUT5 truth tables. Because
the quantized product never lands on either coordinate axis, neither output
has a zero/tie state after the complex multiply. The local comparison can be
absorbed into the truth table at deployment; it does not require a separate
hardware comparator.

This statement is true for the codebook at angles
`pi/8, 3pi/8, ..., 15pi/8`. It is not true for the historical `roots`
codebook at multiples of `pi/4`, which includes coordinate-axis phases.

## 2. Bit Convention

Let the pre-quantized activation be
`a = a_r + j a_i`. This document follows the convention in the design
sketch:

```text
s_r = sign(a_r)
s_i = sign(a_i)

d = 0, if |a_r| >= |a_i|   (real component dominates)
d = 1, if |a_i| >  |a_r|   (imag component dominates)
```

The sign bits are represented as `1 -> +1` and `0 -> -1` when encoded as
physical Boolean pins.

### Repository convention

The current semantic implementation uses the complementary address polarity:

```text
d_impl = 1[|a_r| >= |a_i|] = 1 - d_doc
```

This changes only the ordering of the two 16-entry halves of each LUT. It does
not change the quantized complex values or the operation. A truth table must
state which convention it uses; no learned model conversion is needed beyond
swapping the fifth-bit address slices. The table in this document uses
`d_doc`, namely `0 = real-dominant`.

Input ties still need a deterministic quantizer rule. Here
`|a_r|=|a_i|` selects `d=0`. Likewise, zero-valued sign inputs need a
fixed sign rule. These are **input quantization boundaries**. They are
different from the local-product zero state eliminated below.

## 3. Octant-Centered Activation Quantization

Define:

```text
H = cos(pi/8) = {{HIGH}}
L = sin(pi/8) = {{LOW}}
```

For the document convention:

```text
m_r(d) = (1-d)H + dL
m_i(d) = (1-d)L + dH

q_r = s_r * m_r(d)
q_i = s_i * m_i(d)
q   = q_r + j q_i
```

Thus:

- `d=0`: `q = s_r H + j s_i L`;
- `d=1`: `q = s_r L + j s_i H`.

Because `H^2+L^2=1`, every codeword has unit magnitude. The eight possible
phases are:

```text
theta_k = (2k+1)pi/8, k=0,...,7
        = 22.5, 67.5, 112.5, ..., 337.5 degrees
```

This is best described as **octant-centered C8**, **offset 8-PSK**, or
**eight-sector phase quantization**. Calling the sectors “eight quadrants” is
informal; geometrically they are eight octants/sectors.

No `atan2`, sine, cosine, normalization, or Gray-code encoder is required in
hardware. The signs and the dominance comparator directly select one of the
eight fixed semantic states.

## 4. Binary Complex Weight and Local Product

Let the binary complex weight be:

```text
t_r = sign(w_r) in {-1,+1}
t_i = sign(w_i) in {-1,+1}
w_q = alpha * (t_r + j t_i), alpha > 0
```

The positive per-output-channel scale `alpha` affects only magnitude, not
the truth-table bits. Ignoring `alpha`, the local product is:

```text
y_r = q_r*t_r - q_i*t_i
y_i = q_r*t_i + q_i*t_r
```

In polar form, the activation phase is an odd multiple of `pi/8`, while a
binary complex weight has phase:

```text
phi_m in {pi/4, 3pi/4, 5pi/4, 7pi/4}
      = (4m+2)pi/8
```

Therefore:

```text
angle(q*w_q) = odd*pi/8 + even*pi/8 = odd*pi/8  (mod 2pi)
```

The coordinate axes are multiples of `4pi/8`, so the product cannot land on
an axis. Algebraically, every real and imaginary component has magnitude:

```text
|y_component| in {H+L, H-L}
              = {{{STRONG}}, {{WEAK}}}
```

The smallest distance from a local component to zero is therefore
`H-L={{MIN_COMPONENT}}`, and exhaustive enumeration finds
`{{ZERO_COMPONENTS}}` zero components among all `32*2=64` outputs.

## 5. Two 1-Bit Outputs

Define the local output bits:

```text
b_r = 1[y_r > 0]
b_i = 1[y_i > 0]
```

Since `y_r` and `y_i` are never zero, `>0` and `>=0` are equivalent
for every valid quantized state. The complete 32-state truth table is shown
compactly below. Each cell is `b_r b_i`; the four columns enumerate the four
binary complex-weight states.

{{TRUTH_TABLE}}

The input order of this table is:

```text
[s_r, s_i, d_doc, t_r, t_i] -> [b_r, b_i]
```

This table demonstrates three facts directly:

1. there are exactly five Boolean inputs;
2. both outputs are exactly one bit;
3. changing the dominance bit changes the operation for a nonzero subset of
   addresses, so the fifth input is not redundant.

## 6. Hardware Mapping

Logically, each local complex product requires:

```text
LUT_real: 5 shared inputs -> b_r
LUT_imag: 5 shared inputs -> b_i
```

On a Xilinx-style dual-output `LUT6_2`, two arbitrary LUT5 functions sharing
the same five inputs can normally be packed into one physical primitive:
`O5` implements one 5-input table and `O6` implements the other while the
sixth input selects the independent half. Final packing must still be checked
in synthesis because placement and pin constraints can prevent ideal packing.
Without dual-output packing, the conservative accounting is two logical LUT5
functions per local complex product.

There is no post-LUT local comparator. The comparator result is already the
LUT output bit. Across a convolution receptive field, each output path uses a
bit count or bipolar sum:

```text
sum_k (2*b_k-1) = 2*popcount(b)-N_valid
```

A positive layer/channel scale can then restore the training-domain
normalization. The current fixed analytic implementation uses
`2*alpha*(2*count-N_valid)`, equivalently
`4*alpha*(count-N_valid/2)`. Padding positions are skipped and do not
generate fictitious LUT outputs.

## 7. Training Without LUT Lookup

The deployable truth table is fixed, but training should preserve useful
gradients. The recommended computation graph is:

```text
complex activation
  -> hard sign/sign/dominance bits with STE
  -> fixed H/L octant decode
  -> analytic complex multiply with binary weight
  -> hard local real/imag sign in the numerical forward
  -> comparator-aware smooth surrogate in backward
  -> accumulation, BN, residual network
```

For this document's polarity, a dominance proxy is:

```text
p_d = sigmoid(beta_d * (|a_i|-|a_r|))
d   = hard(d_doc) in forward, p_d in backward
```

The repository uses the complement
`sigmoid(beta_d*(|a_r|-|a_i|))` for `d_impl`. Both send the downstream
fifth-bit gradient back through the dominance formula into both `a_r` and
`a_i`. The two sign bits use their own STE paths.

For local products, the fixed analytic branch uses hard outputs numerically
and a smooth comparator such as:

```text
z_soft = tanh(beta_c * z / scale)
output = hard(z).detach() - z_soft.detach() + z_soft
```

This is deliberately different from differentiable LUT-address interpolation.
Two methods may have the same 32-entry hard table but very different
Jacobians. The analytic surrogate follows the complex-product/comparator
geometry and keeps the operation fixed during training.

## 8. Relation to Historical Phases

The same octant representation appeared in two different historical stages:

| Stage | Local behavior | Best val | Test at best val |
| --- | --- | ---: | ---: |
| Phase2 C4 baseline | 2-bit activation, ordinary binary complex convolution | 83.10% | 82.21% |
| Phase2.1 C8 `semantic_ste,beta=2` | 3-bit octant activation, **continuous H/L local product**, then accumulation | **84.50%** | **84.16%** |
| Phase3 LUT4-aware | C4 local product compressed to real/imag sign bits | 82.42% | 81.54% |
| Phase3.1 C8 progressive-zero | octant local product compressed to real/imag sign bits | 80.12% | 79.22% |

Phase2.1 proves that the third semantic bit can improve the network when its
gradient is correct. However, Phase2.1 is **not yet** the final 5-to-2
hardware operation: it retains the two local magnitudes `H+L` and `H-L`
before accumulation.

Phase3.1 is the exact local 5-to-2 endpoint described in this document. Its
accuracy drop is not caused by a zero/tie ambiguity; octant quantization
successfully removes that ambiguity. The drop comes mainly from compressing
both strong and weak local components to the same one-bit magnitude, plus
training/BN/surrogate adaptation. In other words:

```text
no zero boundary != no information loss
```

The current fixed analytic dominance flow revisits this endpoint from the
Phase1 checkpoint with an explicitly comparator-aware backward path. Its
hard forward is exactly the fixed 32-state operation, and its two tables are
only materialized at checkpoint/export time. A completed accuracy result for
this new run is not yet recorded, so mathematical correctness should not be
presented as an accuracy claim.

## 9. Gray Code Versus Semantic Pins

Earlier experiments also represented eight phase states with a cyclic 3-bit
Gray code. Gray coding is a bijective address relabeling, but its three
physical bits do not individually mean sign-real, sign-imag, and dominance.
If the hardware already exposes the two sign comparators and one dominance
comparator, direct semantic addressing is preferable:

```text
[sign_r, sign_i, dominance]
```

It requires no Gray encoder and gives each training bit a clear surrogate
gradient. A Gray-address LUT remains functionally possible by permuting the
eight activation address groups offline.

## 10. Precise Claim and Limitations

The defensible claim is:

> Octant-centered C8 activation quantization combined with diagonal binary
> complex weights forms a closed, zero-free local phase product. Its real and
> imaginary signs define two deterministic 5-input Boolean functions that can
> be implemented as two shared-input LUT5 tables.

Important limits:

- only the `octants` codebook has the zero-free guarantee; `roots` does not;
- input quantizer ties still require deterministic rules;
- the proof assumes a positive weight scale; the Boolean table itself uses
  only weight signs;
- two 1-bit outputs discard the strong/weak magnitude class;
- zero-free local products guarantee an unambiguous truth table, not higher
  network accuracy;
- ideal packing into one `LUT6_2` must be confirmed by synthesis reports.

## 11. Repository References

- `complexPyTorch/complexLayers.py::C8ComplexActivation`: roots/octants
  codebooks, semantic bits, Gray bits, and STE variants.
- `complexPyTorch/complexLayers.py::C8LUTAwareComplexBinaryConv2d`: historical
  Phase3.1 analytic local-comparator endpoint.
- `complexPyTorch/fixedAnalyticDominanceFlow.py`: isolated fixed analytic
  comparator-aware flow and offline LUT5 export.
- `run_phase2p1.sh`, `run_phase3p1.sh`, and `run_fixed_analytic.sh`:
  corresponding experiment entry points.
- `PHASE4_LUT_EXPERIMENT_LOG.md`: full chronological experiments and
  diagnostics.

## 12. Reproducible Verification

Regenerate this document:

```bash
conda run -n lut_net python scripts/generate_lut5_octant_phase_doc.py
```

Check that the committed document matches the formulas:

```bash
conda run -n lut_net python scripts/generate_lut5_octant_phase_doc.py \
  --check --verify-repository
```
"""


def generate_document():
    table, minimum_component, zero_components = truth_table_markdown()
    replacements = {
        "{{HIGH}}": "{:.10f}".format(HIGH),
        "{{LOW}}": "{:.10f}".format(LOW),
        "{{STRONG}}": "{:.10f}".format(HIGH + LOW),
        "{{WEAK}}": "{:.10f}".format(HIGH - LOW),
        "{{MIN_COMPONENT}}": "{:.10f}".format(minimum_component),
        "{{ZERO_COMPONENTS}}": str(zero_components),
        "{{TRUTH_TABLE}}": table,
    }
    document = DOCUMENT_TEMPLATE
    for marker, value in replacements.items():
        document = document.replace(marker, value)
    return document.rstrip() + "\n"


def verify_repository_export():
    """Check the document-polarity table against the implementation export."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from complexPyTorch.fixedAnalyticDominanceFlow import (
        export_fixed_dominance_lut5,
    )

    tables = export_fixed_dominance_lut5()
    checked_outputs = 0
    for sign_r, sign_i, dominance_doc in ACTIVATION_STATES:
        dominance_impl = 1 - dominance_doc
        for weight_r, weight_i in WEIGHT_STATES:
            address = (
                int(sign_r > 0) * 16
                + int(sign_i > 0) * 8
                + dominance_impl * 4
                + int(weight_r > 0) * 2
                + int(weight_i > 0)
            )
            output_r, output_i = local_product(
                sign_r,
                sign_i,
                dominance_doc,
                weight_r,
                weight_i,
            )
            expected = (output_bit(output_r), output_bit(output_i))
            actual = (
                int(tables["real"][address].item()),
                int(tables["imag"][address].item()),
            )
            if actual != expected:
                raise RuntimeError(
                    "repository export mismatch at address {}: {} != {}".format(
                        address, actual, expected
                    )
                )
            checked_outputs += 2
    return checked_outputs


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the existing output differs from regenerated content.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the generated document instead of writing it.",
    )
    parser.add_argument(
        "--verify-repository",
        action="store_true",
        help="Compare all outputs with the repository's complementary-d export.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    document = generate_document()
    checked_outputs = (
        verify_repository_export() if args.verify_repository else None
    )
    if args.stdout:
        sys.stdout.write(document)
        if checked_outputs is not None:
            print(
                "Verified {} repository outputs".format(checked_outputs),
                file=sys.stderr,
            )
        return
    if args.check:
        if not args.output.exists():
            raise SystemExit("Missing generated document: {}".format(args.output))
        existing = args.output.read_text(encoding="utf-8")
        if existing != document:
            raise SystemExit(
                "Generated document is stale: {}".format(args.output)
            )
        print("Document is up to date: {}".format(args.output))
        if checked_outputs is not None:
            print(
                "Repository export matches all {} outputs".format(
                    checked_outputs
                )
            )
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print("Wrote {}".format(args.output))
    if checked_outputs is not None:
        print(
            "Repository export matches all {} outputs".format(checked_outputs)
        )


if __name__ == "__main__":
    main()
