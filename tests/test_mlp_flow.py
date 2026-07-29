import math
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from complexPyTorch.complexFunctions import binary_scale_weight_complex, binary_sign
from complexPyTorch.mlpFlow import (
    MLPOperationComplexConv2d,
    clamp_binary_mlp_latents,
    initialize_phase4_luts,
    operation_diagnostics,
)
from training_mlp_flow import phase4_lut_state


def _explicit_complex_conv(inp, weight_r, weight_i, padding=0):
    output_r = F.conv2d(inp.real, weight_r, padding=padding)
    output_r = output_r - F.conv2d(inp.imag, weight_i, padding=padding)
    output_i = F.conv2d(inp.real, weight_i, padding=padding)
    output_i = output_i + F.conv2d(inp.imag, weight_r, padding=padding)
    return torch.complex(output_r, output_i)


def test_phase1p2_initial_operator_is_exact_float_complex_multiply():
    torch.manual_seed(11)
    layer = MLPOperationComplexConv2d(2, 3, kernel_size=3, padding=1, phase=1.2)
    inp = torch.complex(torch.randn(2, 2, 5, 5), torch.randn(2, 2, 5, 5))

    actual = layer(inp, phase_source=inp)
    expected = _explicit_complex_conv(
        inp,
        layer.weight_r,
        layer.weight_i,
        padding=1,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_binary_mlp_initial_scores_are_exact_and_robust():
    layer = MLPOperationComplexConv2d(
        1,
        1,
        kernel_size=1,
        padding=0,
        phase=2.2,
    )
    score_r, score_i = layer.binary_mlp_scores()
    scores = torch.stack((score_r, score_i), dim=-1)

    torch.testing.assert_close(scores, layer.initial_operation_scores)
    assert set(scores.unique().tolist()) == {-2.0, 0.0, 2.0}


def test_phase2p2_initial_operator_is_exact_binary_complex_multiply():
    torch.manual_seed(12)
    layer = MLPOperationComplexConv2d(
        2,
        3,
        kernel_size=1,
        padding=0,
        phase=2.2,
    )
    source = torch.complex(torch.randn(2, 2, 5, 5), torch.randn(2, 2, 5, 5))
    inp = torch.complex(
        binary_sign(source.real, grad_mode="bireal"),
        binary_sign(source.imag, grad_mode="bireal"),
    )

    actual = layer(inp, phase_source=source)
    signed_weight_r = binary_sign(layer.weight_r)
    signed_weight_i = binary_sign(layer.weight_i)
    expected = _explicit_complex_conv(inp, signed_weight_r, signed_weight_i)
    alpha = binary_scale_weight_complex(
        torch.complex(layer.weight_r, layer.weight_i),
        per_channel=True,
    ).reshape(1, 3, 1, 1)
    expected = expected * alpha

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_binary_mlp_fifth_input_latents_receive_gradient():
    torch.manual_seed(13)
    layer = MLPOperationComplexConv2d(
        1,
        1,
        kernel_size=1,
        padding=0,
        phase=2.2,
    )
    score_r, score_i = layer.binary_mlp_scores()
    loss = score_r[0, 31] + 0.37 * score_i[0, 23]
    loss.backward()

    d_gradient = layer.binary_mlp_hidden_weight.grad[..., 2]
    assert torch.isfinite(d_gradient).all()
    assert d_gradient.abs().sum().item() > 0.0
    assert layer.binary_mlp_output_weight.grad.abs().sum().item() > 0.0


def test_layer_and_channel_operation_allocations_have_expected_ids():
    layer = MLPOperationComplexConv2d(
        2,
        4,
        kernel_size=1,
        phase=2.2,
        operation_allocation="layer",
        operation_sets=2,
    )
    assert layer.operation_sets == 2
    assert layer.operation_set_ids.tolist() == [[0, 1, 0, 1], [0, 1, 0, 1]]

    channel = MLPOperationComplexConv2d(
        2,
        3,
        kernel_size=1,
        phase=2.2,
        operation_allocation="channel",
        operation_sets_per_channel=2,
    )
    assert channel.operation_sets == 6
    assert channel.operation_set_ids.tolist() == [[0, 2, 4], [1, 3, 5]]


def test_phase3p2_locally_binarizes_each_complex_product():
    layer = MLPOperationComplexConv2d(1, 1, kernel_size=1, padding=0, phase=3.2)
    with torch.no_grad():
        layer.weight_r.fill_(1.0)
        layer.weight_i.fill_(1.0)
    inp = torch.complex(torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1))

    output = layer(inp, phase_source=inp)
    alpha = math.sqrt(2.0)
    expected = torch.complex(
        torch.full((1, 1, 1, 1), 2.0 * alpha),
        torch.full((1, 1, 1, 1), 2.0 * alpha),
    )
    torch.testing.assert_close(output, expected)


def test_phase3p2_table_is_only_local_truncation_of_phase2p2_scores():
    phase2 = MLPOperationComplexConv2d(1, 2, kernel_size=1, phase=2.2)
    phase3 = MLPOperationComplexConv2d(1, 2, kernel_size=1, phase=3.2)
    phase3.load_state_dict(phase2.state_dict(), strict=False)

    score_r, score_i = phase2.enumerated_operation_scores()
    table_r, table_i = phase3.hard_lut_tables()
    assert torch.equal(table_r, (score_r >= 0.0).to(torch.uint8))
    assert torch.equal(table_i, (score_i >= 0.0).to(torch.uint8))
    torch.testing.assert_close(phase2.weight_r, phase3.weight_r)
    torch.testing.assert_close(phase2.weight_i, phase3.weight_i)


def test_phase3p2_score_surrogate_keeps_hard_forward_and_phase2_gradient():
    torch.manual_seed(17)
    phase2 = MLPOperationComplexConv2d(
        1,
        2,
        kernel_size=1,
        padding=0,
        phase=2.2,
    )
    hard_phase3 = MLPOperationComplexConv2d(
        1,
        2,
        kernel_size=1,
        padding=0,
        phase=3.2,
    )
    surrogate_phase3 = MLPOperationComplexConv2d(
        1,
        2,
        kernel_size=1,
        padding=0,
        phase=3.2,
        phase3p2_score_surrogate=True,
    )
    hard_phase3.load_state_dict(phase2.state_dict(), strict=False)
    surrogate_phase3.load_state_dict(phase2.state_dict(), strict=False)

    inp_r = torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]])
    inp_i = torch.tensor([[[[-1.0, 1.0], [1.0, -1.0]]]])
    source_r = torch.tensor([[[[0.7, -0.2], [-0.8, 0.4]]]])
    source_i = torch.tensor([[[[-0.1, 0.9], [0.3, -0.6]]]])
    inp = torch.complex(inp_r, inp_i)
    source = torch.complex(source_r, source_i)

    hard_output = hard_phase3(inp, phase_source=source)
    surrogate_output = surrogate_phase3(inp, phase_source=source)
    torch.testing.assert_close(surrogate_output, hard_output)

    def backward_result(layer):
        local_inp_r = inp_r.clone().requires_grad_(True)
        local_inp_i = inp_i.clone().requires_grad_(True)
        local_source_r = source_r.clone().requires_grad_(True)
        local_source_i = source_i.clone().requires_grad_(True)
        output = layer(
            torch.complex(local_inp_r, local_inp_i),
            phase_source=torch.complex(local_source_r, local_source_i),
        )
        (output.real.sum() + 0.37 * output.imag.sum()).backward()
        gradients = {
            "inp_r": local_inp_r.grad,
            "inp_i": local_inp_i.grad,
            "source_r": local_source_r.grad,
            "source_i": local_source_i.grad,
            "weight_r": layer.weight_r.grad,
            "weight_i": layer.weight_i.grad,
            "hidden": layer.binary_mlp_hidden_weight.grad,
            "output": layer.binary_mlp_output_weight.grad,
        }
        return gradients

    phase2_gradients = backward_result(phase2)
    surrogate_gradients = backward_result(surrogate_phase3)
    for name in phase2_gradients:
        torch.testing.assert_close(
            surrogate_gradients[name],
            phase2_gradients[name],
            rtol=1e-5,
            atol=1e-6,
            msg="gradient mismatch for {}".format(name),
        )


def test_phase3p2_score_surrogate_rejects_other_phases():
    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "only valid for Phase 3.2",
    ):
        MLPOperationComplexConv2d(
            1,
            1,
            kernel_size=1,
            phase=2.2,
            phase3p2_score_surrogate=True,
        )


def test_phase4p2_compilation_preserves_phase3p2_hard_table():
    torch.manual_seed(14)
    source = MLPOperationComplexConv2d(1, 2, kernel_size=1, phase=3.2)
    with torch.no_grad():
        source.binary_mlp_output_weight[0, 0, 0].mul_(-1.0)
    target = MLPOperationComplexConv2d(1, 2, kernel_size=1, phase=4.2)
    target.load_state_dict(source.state_dict(), strict=False)
    initialize_phase4_luts(target)

    source_r, source_i = source.hard_lut_tables()
    target_r, target_i = target.hard_lut_tables()
    assert torch.equal(source_r, target_r)
    assert torch.equal(source_i, target_i)


def test_operation_diagnostics_detect_fifth_bit_sensitivity():
    layer = MLPOperationComplexConv2d(1, 1, kernel_size=1, phase=3.2)
    copies_per_state = layer.binary_mlp_hidden // 32
    with torch.no_grad():
        for state in range(32):
            start = state * copies_per_state
            end = start + copies_per_state
            value = 0.5 if state & 0b00100 else -0.5
            layer.binary_mlp_output_weight[0, 0, start:end] = value
    diagnostics = operation_diagnostics(layer)

    assert diagnostics["d_sensitive_ratio"] == 0.5
    assert diagnostics["d_coefficient_abs_mean"] > 0.0
    assert diagnostics["coefficient_delta_l2"] > 0.0


def test_binary_mlp_latents_are_clamped_to_ste_window():
    layer = MLPOperationComplexConv2d(1, 1, kernel_size=1, phase=2.2)
    with torch.no_grad():
        layer.binary_mlp_hidden_weight.fill_(2.0)
        layer.binary_mlp_output_weight.fill_(-3.0)
    count = clamp_binary_mlp_latents(layer)

    assert count == (
        layer.binary_mlp_hidden_weight.numel()
        + layer.binary_mlp_output_weight.numel()
    )
    assert layer.binary_mlp_hidden_weight.max().item() == 1.0
    assert layer.binary_mlp_output_weight.min().item() == -1.0


def test_phase4_schedule_reaches_hard_mode():
    class Args:
        lut_tau_min = 1.0
        lut_tau_max = 10.0
        lut_soft_warmup_epochs = 2
        lut_anneal_epochs = 3
        lut_hard_transition_epochs = 2

    assert phase4_lut_state(0, Args()) == (1.0, 0.0)
    tau, hard_ratio = phase4_lut_state(5, Args())
    assert tau == 10.0
    assert hard_ratio == 0.5
    assert phase4_lut_state(7, Args()) == (10.0, 1.0)


class MLPFlowTest(unittest.TestCase):
    def test_phase1p2_exact_float_start(self):
        test_phase1p2_initial_operator_is_exact_float_complex_multiply()

    def test_binary_mlp_initial_scores(self):
        test_binary_mlp_initial_scores_are_exact_and_robust()

    def test_phase2p2_exact_binary_start(self):
        test_phase2p2_initial_operator_is_exact_binary_complex_multiply()

    def test_fifth_input_gradient(self):
        test_binary_mlp_fifth_input_latents_receive_gradient()

    def test_operation_allocations(self):
        test_layer_and_channel_operation_allocations_have_expected_ids()

    def test_local_output_binarization(self):
        test_phase3p2_locally_binarizes_each_complex_product()

    def test_phase2p2_to_phase3p2_boundary(self):
        test_phase3p2_table_is_only_local_truncation_of_phase2p2_scores()

    def test_phase3p2_score_surrogate(self):
        test_phase3p2_score_surrogate_keeps_hard_forward_and_phase2_gradient()

    def test_phase3p2_score_surrogate_validation(self):
        test_phase3p2_score_surrogate_rejects_other_phases()

    def test_hard_table_compilation(self):
        test_phase4p2_compilation_preserves_phase3p2_hard_table()

    def test_fifth_bit_diagnostics(self):
        test_operation_diagnostics_detect_fifth_bit_sensitivity()

    def test_latent_clamp(self):
        test_binary_mlp_latents_are_clamped_to_ste_window()

    def test_lut_schedule(self):
        test_phase4_schedule_reaches_hard_mode()

    def test_runner_uses_best_checkpoint_and_supports_recovery_phase(self):
        runner = (
            Path(__file__).resolve().parents[1] / "run_mlp_flow.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'TRANSITION_CHECKPOINT_KIND="${TRANSITION_CHECKPOINT_KIND:-best}"',
            runner,
        )
        self.assertIn('START_FLOW_PHASE="${START_FLOW_PHASE:-2.2}"', runner)
        self.assertIn('END_FLOW_PHASE="${END_FLOW_PHASE:-4.2}"', runner)
        self.assertIn('PHASE2P2_CHECKPOINT="${PHASE2P2_CHECKPOINT:-}"', runner)
        self.assertIn("--phase3p2-score-surrogate", runner)
        self.assertIn('run_phase 2.2 "${PHASE1_CHECKPOINT}"', runner)
        self.assertNotIn("phase1p2.pt", runner)
        self.assertIn('${TRANSITION_PREFIX}_phase2p2.pt', runner)
        self.assertIn('${TRANSITION_PREFIX}_phase3p2.pt', runner)
