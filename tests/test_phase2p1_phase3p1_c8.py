import itertools
import math
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn

from complexPyTorch.complexBinaryResNet import BinaryComplexResNet
from complexPyTorch.complexLayers import (
    BinaryComplexConv2d,
    C8ComplexActivation,
    C8LUTAwareComplexBinaryConv2d,
    LUTFloatingConvFunction,
)
from training import (
    C8CodeOccupancyTracker,
    load_phase_checkpoint,
    parse_args,
    phase_checkpoint_filename,
    phase_initialization_sources,
    phase3p1_local_transition_state_for_epoch,
    set_fixed_c8_qat_state,
    set_phase3p1_local_compare_progress,
)


class Phase2p1Phase3p1C8Test(unittest.TestCase):
    def test_c8_activation_is_hard_unit_circle_with_qat_gradients(self):
        activation = C8ComplexActivation(beta=2.0)
        codebook = activation.c8_codebook
        source = torch.complex(
            codebook[:, 0],
            codebook[:, 1],
        ).reshape(1, 1, 1, 8)

        output = activation(source)
        phase_indices = activation.hard_phase_indices(source)
        self.assertEqual(phase_indices.flatten().tolist(), list(range(8)))
        self.assertTrue(torch.equal(output.real.flatten(), codebook[:, 0]))
        self.assertTrue(torch.equal(output.imag.flatten(), codebook[:, 1]))
        self.assertTrue(
            torch.allclose(output.abs(), torch.ones_like(output.abs()), atol=1e-6)
        )

        torch.manual_seed(41)
        real = torch.randn(2, 3, 4, 4, requires_grad=True)
        imag = torch.randn(2, 3, 4, 4, requires_grad=True)
        qat_output = activation(torch.complex(real, imag))
        loss = (0.7 * qat_output.real + 0.3 * qat_output.imag).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(real.grad).all())
        self.assertTrue(torch.isfinite(imag.grad).all())
        self.assertGreater(real.grad.abs().sum().item(), 0.0)
        self.assertGreater(imag.grad.abs().sum().item(), 0.0)

    def test_c8_bireal_ste_keeps_hard_forward_and_matches_old_gradient(self):
        activation = C8ComplexActivation(
            beta=2.0,
            codebook_mode="roots",
            grad_mode="bireal_ste",
        )
        real = torch.tensor([0.2, -0.4, 0.7], requires_grad=True)
        imag = torch.tensor([-0.3, 0.6, -0.8], requires_grad=True)
        grad_r = torch.tensor([0.5, -0.7, 1.1])
        grad_i = torch.tensor([-0.2, 0.9, 0.4])

        source = torch.complex(real, imag)
        output = activation(source)
        hard_index = activation.hard_phase_indices(source)
        hard_value = activation.c8_codebook[hard_index]
        self.assertTrue(torch.equal(output.real, hard_value[:, 0]))
        self.assertTrue(torch.equal(output.imag, hard_value[:, 1]))

        (output.real * grad_r + output.imag * grad_i).sum().backward()
        expected_real = grad_r * (2.0 * (1.0 - real.detach().abs())).clamp(
            min=0.0
        )
        expected_imag = grad_i * (2.0 * (1.0 - imag.detach().abs())).clamp(
            min=0.0
        )
        self.assertTrue(torch.allclose(real.grad, expected_real))
        self.assertTrue(torch.allclose(imag.grad, expected_imag))

    def test_c8_semantic_ste_routes_sign_and_dominance_gradients(self):
        beta = 2.0
        activation = C8ComplexActivation(
            beta=beta,
            codebook_mode="octants",
            grad_mode="semantic_ste",
        )
        source = torch.tensor([0.8, 0.7], requires_grad=True)

        def decode(values):
            output = activation(torch.complex(values[0], values[1]))
            return torch.stack((output.real, output.imag))

        output = decode(source)
        high = math.cos(math.pi / 8.0)
        low = math.sin(math.pi / 8.0)
        self.assertTrue(
            torch.allclose(
                output,
                torch.tensor([high, low]),
                atol=1e-7,
            )
        )

        jacobian = torch.autograd.functional.jacobian(decode, source)
        probability = torch.sigmoid(torch.tensor(beta * (0.8 - 0.7)))
        dominance_grad = (
            (high - low) * beta * probability * (1.0 - probability)
        )
        expected = torch.tensor(
            [
                [high * 0.4 + dominance_grad, -dominance_grad],
                [-dominance_grad, low * 0.6 + dominance_grad],
            ]
        )
        self.assertTrue(torch.allclose(jacobian, expected, atol=1e-6))
        self.assertLess(jacobian[0, 1].item(), 0.0)
        self.assertLess(jacobian[1, 0].item(), 0.0)

        with self.assertRaisesRegex(ValueError, "requires.*octants"):
            C8ComplexActivation(
                codebook_mode="roots",
                grad_mode="semantic_ste",
            )

    def test_c8_semantic_phase_ste_is_scale_invariant_and_tangential(self):
        beta = 2.0
        activation = C8ComplexActivation(
            beta=beta,
            codebook_mode="octants",
            grad_mode="semantic_phase_ste",
        )
        legacy = C8ComplexActivation(
            beta=beta,
            codebook_mode="octants",
            grad_mode="semantic_ste",
        )

        def decode(values):
            output = activation(torch.complex(values[0], values[1]))
            return torch.stack((output.real, output.imag))

        source = torch.tensor([2.0, 1.5], dtype=torch.float64)
        scale = 3.0
        output = decode(source)
        scaled_output = decode(source * scale)
        legacy_output = legacy(torch.complex(source[0], source[1]))
        self.assertTrue(torch.equal(output, scaled_output))
        self.assertTrue(
            torch.equal(
                output,
                torch.stack((legacy_output.real, legacy_output.imag)),
            )
        )

        jacobian = torch.autograd.functional.jacobian(decode, source)
        scaled_jacobian = torch.autograd.functional.jacobian(
            decode,
            source * scale,
        )
        self.assertTrue(
            torch.allclose(
                jacobian,
                scaled_jacobian * scale,
                atol=1e-7,
                rtol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                jacobian @ source,
                torch.zeros(2, dtype=source.dtype),
                atol=1e-7,
            )
        )
        self.assertGreater(jacobian[0, 0].abs().item(), 0.0)
        self.assertGreater(jacobian[0, 1].abs().item(), 0.0)

        with self.assertRaisesRegex(ValueError, "requires.*octants"):
            C8ComplexActivation(
                codebook_mode="roots",
                grad_mode="semantic_phase_ste",
            )

    def test_octant_codebook_preserves_signs_and_adds_dominance(self):
        activation = C8ComplexActivation(
            codebook_mode="octants",
            grad_mode="bireal_ste",
        )
        real = torch.tensor([-2.0, -2.0, -0.5, 0.5, 2.0, 2.0, 0.5, -0.5])
        imag = torch.tensor([-0.5, 0.5, 2.0, 2.0, 0.5, -0.5, -2.0, -2.0])
        source = torch.complex(real, imag)

        output = activation(source)
        self.assertEqual(
            activation.hard_phase_indices(source).tolist(),
            list(range(8)),
        )
        self.assertTrue(torch.equal(output.real.sign(), real.sign()))
        self.assertTrue(torch.equal(output.imag.sign(), imag.sign()))
        self.assertTrue(
            torch.equal(
                output.real.abs() >= output.imag.abs(),
                real.abs() >= imag.abs(),
            )
        )
        self.assertTrue(
            torch.allclose(output.abs(), torch.ones_like(output.abs()), atol=1e-6)
        )
        self.assertGreater(output.real.abs().min().item(), 0.38)
        self.assertGreater(output.imag.abs().min().item(), 0.38)

    def test_phase2p1_uses_ordinary_binary_complex_convolution(self):
        activation = C8ComplexActivation(beta=2.0)
        phase2p1_conv = BinaryComplexConv2d(
            1,
            1,
            kernel_size=1,
            bias=False,
        )
        phase3p1_conv = C8LUTAwareComplexBinaryConv2d(
            1,
            1,
            kernel_size=1,
            padding=0,
        )
        for layer in (phase2p1_conv, phase3p1_conv):
            layer.conv_r.weight.data.fill_(1.0)
            layer.conv_i.weight.data.fill_(1.0)

        source = torch.complex(
            torch.ones(1, 1, 1, 1),
            torch.zeros(1, 1, 1, 1),
        )
        quantized = activation(source)
        ordinary = phase2p1_conv(quantized)
        local_compare = phase3p1_conv(quantized)
        alpha = math.sqrt(2.0)

        self.assertTrue(
            torch.allclose(
                ordinary,
                torch.complex(
                    torch.full_like(ordinary.real, alpha),
                    torch.full_like(ordinary.imag, alpha),
                ),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                local_compare,
                torch.complex(
                    torch.full_like(local_compare.real, 2.0 * alpha),
                    torch.full_like(local_compare.imag, 2.0 * alpha),
                ),
                atol=1e-6,
            )
        )

    def test_phase3p1_rho_zero_matches_phase2p1_forward_and_gradients(self):
        torch.manual_seed(29)
        phase2p1_conv = BinaryComplexConv2d(
            2,
            3,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        phase3p1_conv = C8LUTAwareComplexBinaryConv2d(
            2,
            3,
            kernel_size=3,
            padding=1,
            c8_codebook_mode="octants",
            phase3p1_padding_mode="low_code",
        )
        phase3p1_conv.conv_r.weight.data.copy_(
            phase2p1_conv.conv_r.weight.data
        )
        phase3p1_conv.conv_i.weight.data.copy_(
            phase2p1_conv.conv_i.weight.data
        )
        phase3p1_conv.set_local_compare_progress(0.0)

        source_r = torch.randn(2, 2, 5, 5)
        source_i = torch.randn(2, 2, 5, 5)
        source2_r = source_r.clone().requires_grad_(True)
        source2_i = source_i.clone().requires_grad_(True)
        source3_r = source_r.clone().requires_grad_(True)
        source3_i = source_i.clone().requires_grad_(True)
        activation2 = C8ComplexActivation(
            beta=2.0,
            codebook_mode="octants",
            grad_mode="semantic_ste",
        )
        activation3 = C8ComplexActivation(
            beta=2.0,
            codebook_mode="octants",
            grad_mode="semantic_ste",
        )

        output2 = phase2p1_conv(
            activation2(torch.complex(source2_r, source2_i))
        )
        output3 = phase3p1_conv(
            activation3(torch.complex(source3_r, source3_i))
        )
        loss2 = output2.real.square().mean() + output2.imag.square().mean()
        loss3 = output3.real.square().mean() + output3.imag.square().mean()
        loss2.backward()
        loss3.backward()

        self.assertTrue(torch.allclose(output2, output3, atol=1e-6))
        self.assertTrue(
            torch.allclose(source2_r.grad, source3_r.grad, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(source2_i.grad, source3_i.grad, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                phase2p1_conv.conv_r.weight.grad,
                phase3p1_conv.conv_r.weight.grad,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                phase2p1_conv.conv_i.weight.grad,
                phase3p1_conv.conv_i.weight.grad,
                atol=1e-6,
            )
        )

    def test_phase3p1_progress_interpolates_forward_endpoints(self):
        torch.manual_seed(31)
        activation = C8ComplexActivation(
            beta=2.0,
            codebook_mode="octants",
            grad_mode="semantic_ste",
        )
        layer = C8LUTAwareComplexBinaryConv2d(
            2,
            3,
            kernel_size=3,
            padding=1,
            c8_codebook_mode="octants",
            phase3p1_padding_mode="zero",
        )
        source = torch.complex(
            torch.randn(2, 2, 5, 5),
            torch.randn(2, 2, 5, 5),
        )
        quantized = activation(source)

        layer.set_local_compare_progress(0.0)
        phase2_endpoint = layer(quantized)
        layer.set_local_compare_progress(1.0)
        hard_endpoint = layer(quantized)
        layer.set_local_compare_progress(0.35)
        intermediate = layer(quantized)

        self.assertTrue(
            torch.allclose(
                intermediate,
                torch.lerp(phase2_endpoint, hard_endpoint, 0.35),
                atol=1e-5,
            )
        )

    def test_phase3p1_zero_padding_skips_out_of_bounds_products(self):
        activation = C8ComplexActivation(
            beta=2.0,
            codebook_mode="octants",
        )
        code = activation.c8_codebook[4]
        quantized = torch.complex(code[0], code[1]).reshape(1, 1, 1, 1)
        outputs = {}
        for padding_mode in ("low_code", "zero"):
            layer = C8LUTAwareComplexBinaryConv2d(
                1,
                1,
                kernel_size=3,
                padding=1,
                c8_codebook_mode="octants",
                phase3p1_padding_mode=padding_mode,
            )
            layer.conv_r.weight.data.fill_(1.0)
            layer.conv_i.weight.data.fill_(1.0)
            layer.set_local_compare_progress(1.0)
            outputs[padding_mode] = layer(quantized)

        alpha = math.sqrt(2.0)
        expected_zero = torch.complex(
            torch.tensor([[[[2.0 * alpha]]]]),
            torch.tensor([[[[2.0 * alpha]]]]),
        )
        expected_low_code = torch.complex(
            torch.tensor([[[[-14.0 * alpha]]]]),
            torch.tensor([[[[-14.0 * alpha]]]]),
        )
        self.assertTrue(
            torch.allclose(outputs["zero"], expected_zero, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                outputs["low_code"],
                expected_low_code,
                atol=1e-6,
            )
        )

    def test_phase3p1_transition_schedule_and_model_setter(self):
        start = phase3p1_local_transition_state_for_epoch(
            0,
            100,
            "cosine",
        )
        end = phase3p1_local_transition_state_for_epoch(
            99,
            100,
            "cosine",
        )
        after = phase3p1_local_transition_state_for_epoch(
            150,
            100,
            "cosine",
        )
        legacy = phase3p1_local_transition_state_for_epoch(
            0,
            0,
            "cosine",
        )
        self.assertEqual(start["local_progress"], 0.0)
        self.assertFalse(start["hard_local_compare_ready"])
        self.assertAlmostEqual(end["local_progress"], 1.0)
        self.assertTrue(end["hard_local_compare_ready"])
        self.assertAlmostEqual(after["local_progress"], 1.0)
        self.assertTrue(legacy["hard_local_compare_ready"])

        model = BinaryComplexResNet(
            num_blocks=1,
            start_filters=2,
            num_classes=10,
            is_sar_input=False,
            phase=3.1,
            lut_inputs=5,
            phase3p1_padding_mode="zero",
        )
        count = set_phase3p1_local_compare_progress(model, 0.25)
        self.assertEqual(count, 6)
        for module in model.modules():
            if isinstance(module, C8LUTAwareComplexBinaryConv2d):
                self.assertEqual(module.phase3p1_padding_mode, "zero")
                self.assertAlmostEqual(
                    float(module.c8_local_compare_progress.item()),
                    0.25,
                )

    def test_phase3p1_all_32_states_match_local_multiply_and_compare(self):
        for codebook_mode in ("roots", "octants"):
            with self.subTest(codebook_mode=codebook_mode):
                activation = C8ComplexActivation(
                    beta=2.0,
                    codebook_mode=codebook_mode,
                )
                layer = C8LUTAwareComplexBinaryConv2d(
                    1,
                    1,
                    kernel_size=1,
                    padding=0,
                    beta=2.0,
                    c8_codebook_mode=codebook_mode,
                )
                codebook = activation.c8_codebook
                source = torch.complex(
                    codebook[:, 0],
                    codebook[:, 1],
                ).reshape(1, 1, 1, 8)
                quantized = activation(source)

                for weight_r, weight_i in itertools.product(
                    (-1.0, 1.0), repeat=2
                ):
                    layer.conv_r.weight.data.fill_(weight_r)
                    layer.conv_i.weight.data.fill_(weight_i)
                    hard_r, hard_i = layer.hard_local_sums(quantized)
                    expected_r = (
                        codebook[:, 0] * weight_r
                        - codebook[:, 1] * weight_i
                        >= 0
                    ).to(torch.float32)
                    expected_i = (
                        codebook[:, 0] * weight_i
                        + codebook[:, 1] * weight_r
                        >= 0
                    ).to(torch.float32)
                    self.assertTrue(torch.equal(hard_r.flatten(), expected_r))
                    self.assertTrue(torch.equal(hard_i.flatten(), expected_i))

    def test_phase3p1_avoids_lut_backend_and_backpropagates(self):
        torch.manual_seed(19)
        activation = C8ComplexActivation(beta=2.0)
        layer = C8LUTAwareComplexBinaryConv2d(
            2,
            3,
            kernel_size=3,
            padding=1,
            beta=2.0,
        )
        real = torch.randn(2, 2, 5, 5, requires_grad=True)
        imag = torch.randn(2, 2, 5, 5, requires_grad=True)

        with mock.patch.object(
            LUTFloatingConvFunction,
            "apply",
            side_effect=AssertionError("LUT backend must not be called"),
        ):
            output = layer(activation(torch.complex(real, imag)))
            output.abs().mean().backward()

        self.assertTrue(torch.isfinite(output).all())
        self.assertGreater(real.grad.abs().sum().item(), 0.0)
        self.assertGreater(imag.grad.abs().sum().item(), 0.0)
        self.assertGreater(layer.conv_r.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(layer.conv_i.weight.grad.abs().sum().item(), 0.0)

    def test_resnet_assigns_c8_activation_and_expected_convolution(self):
        common = {
            "num_blocks": 1,
            "start_filters": 2,
            "num_classes": 10,
            "is_sar_input": False,
            "is_binary": True,
            "lut_inputs": 5,
            "c8_beta": 2.0,
            "c8_codebook": "octants",
            "c8_grad_mode": "bireal_ste",
        }
        phase2p1 = BinaryComplexResNet(phase=2.1, **common)
        phase3p1 = BinaryComplexResNet(phase=3.1, **common)

        for block in list(phase2p1.stage2) + list(phase2p1.stage3) + list(phase2p1.stage4):
            self.assertIsInstance(block.act, C8ComplexActivation)
            self.assertIsInstance(block.conv, BinaryComplexConv2d)
            self.assertEqual(block.act.c8_codebook_mode, "octants")
            self.assertEqual(block.act.c8_grad_mode, "bireal_ste")
        for block in list(phase3p1.stage2) + list(phase3p1.stage3) + list(phase3p1.stage4):
            self.assertIsInstance(block.act, C8ComplexActivation)
            self.assertIsInstance(block.conv, C8LUTAwareComplexBinaryConv2d)
            self.assertEqual(block.act.c8_codebook_mode, "octants")
            self.assertEqual(block.act.c8_grad_mode, "bireal_ste")
            self.assertEqual(block.conv.c8_codebook_mode, "octants")

    def test_phase2p1_checkpoint_maps_all_parameters_into_phase3p1(self):
        torch.manual_seed(23)
        common = {
            "num_blocks": 1,
            "start_filters": 2,
            "num_classes": 10,
            "is_sar_input": False,
            "is_binary": True,
            "lut_inputs": 5,
            "c8_beta": 2.0,
        }
        source = BinaryComplexResNet(phase=2.1, **common)
        target = BinaryComplexResNet(phase=3.1, **common)
        for parameter in source.parameters():
            parameter.data.normal_()

        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint_file:
            torch.save(
                {
                    "phase": 2.1,
                    "model": source.state_dict(),
                    "args": {"phase": 2.1, "lut_inputs": 5},
                },
                checkpoint_file.name,
            )
            load_phase_checkpoint(
                target,
                checkpoint_file.name,
                expected_phase=2.1,
                expected_lut_inputs=5,
            )

        source_parameters = dict(source.named_parameters())
        target_parameters = dict(target.named_parameters())
        self.assertEqual(set(source_parameters), set(target_parameters))
        for name, source_parameter in source_parameters.items():
            self.assertTrue(
                torch.equal(source_parameter, target_parameters[name]),
                msg=name,
            )

    def test_phase3p1_checkpoint_rejects_mismatched_c8_configuration(self):
        model = nn.Linear(1, 1)
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint_file:
            torch.save(
                {
                    "phase": 2.1,
                    "model": model.state_dict(),
                    "args": {
                        "phase": 2.1,
                        "lut_inputs": 5,
                        "c8_codebook": "octants",
                        "c8_grad_mode": "bireal_ste",
                    },
                },
                checkpoint_file.name,
            )
            with self.assertRaisesRegex(ValueError, "C8 codebook"):
                load_phase_checkpoint(
                    model,
                    checkpoint_file.name,
                    expected_phase=2.1,
                    expected_lut_inputs=5,
                    expected_c8_codebook="roots",
                    expected_c8_grad_mode="bireal_ste",
                )

    def test_occupancy_tracker_supports_standalone_c8_activation(self):
        activation = C8ComplexActivation(beta=2.0)
        model = nn.Sequential(activation)
        tracker = C8CodeOccupancyTracker(model)
        codebook = activation.c8_codebook
        source = torch.complex(
            codebook[:, 0],
            codebook[:, 1],
        ).reshape(1, 1, 1, 8)

        tracker.start()
        model(source)
        stats = tracker.finish()
        tracker.close()

        self.assertEqual(stats["counts"], [1] * 8)
        self.assertEqual(stats["active_codes"], 8)
        self.assertAlmostEqual(stats["added_axis_ratio"], 0.5)
        self.assertAlmostEqual(stats["normalized_entropy"], 1.0)
        self.assertLess(stats["mean_angle_error_deg"], 0.1)

    def test_occupancy_tracker_labels_octant_indices_without_axis_codes(self):
        activation = C8ComplexActivation(codebook_mode="octants")
        tracker = C8CodeOccupancyTracker(nn.Sequential(activation))
        source = torch.complex(
            activation.c8_codebook[:, 0],
            activation.c8_codebook[:, 1],
        ).reshape(1, 1, 1, 8)

        tracker.start()
        activation(source)
        stats = tracker.finish()
        tracker.close()

        self.assertEqual(stats["codebook_mode"], "octants")
        self.assertAlmostEqual(stats["odd_index_ratio"], 0.5)
        self.assertNotIn("added_axis_ratio", stats)

    def test_fixed_c8_state_reapplies_cli_beta_after_checkpoint_load(self):
        model = BinaryComplexResNet(
            num_blocks=1,
            start_filters=2,
            num_classes=10,
            is_sar_input=False,
            is_binary=True,
            phase=3.1,
            lut_inputs=5,
            c8_beta=7.0,
        )
        stats = set_fixed_c8_qat_state(model, beta=1.5)
        self.assertEqual(stats["activations"], 6)
        self.assertEqual(stats["local_comparators"], 6)
        for module in model.modules():
            if isinstance(module, C8ComplexActivation):
                self.assertEqual(float(module.c8_beta.item()), 1.5)
            if isinstance(module, C8LUTAwareComplexBinaryConv2d):
                self.assertEqual(float(module.c8_beta.item()), 1.5)
                self.assertEqual(float(module.c8_progress.item()), 1.0)

    def test_cli_predecessors_and_checkpoint_names(self):
        args2p1 = parse_args(
            [
                "--phase",
                "2.1",
                "--lut-inputs",
                "5",
                "--c8-beta",
                "3.0",
                "--c8-codebook",
                "octants",
                "--c8-grad-mode",
                "bireal_ste",
            ]
        )
        args3p1 = parse_args(
            [
                "--phase",
                "3p1",
                "--lut-inputs",
                "5",
                "--phase3p1-padding-mode",
                "zero",
                "--phase3p1-transition-epochs",
                "100",
                "--phase3p1-transition-schedule",
                "cosine",
            ]
        )
        args_semantic = parse_args(
            [
                "--phase",
                "2.1",
                "--lut-inputs",
                "5",
                "--c8-codebook",
                "octants",
                "--c8-grad-mode",
                "semantic_ste",
            ]
        )
        args_semantic_phase = parse_args(
            [
                "--phase",
                "2.1",
                "--lut-inputs",
                "5",
                "--c8-codebook",
                "octants",
                "--c8-grad-mode",
                "semantic_phase_ste",
            ]
        )
        self.assertEqual(args2p1.phase, 2.1)
        self.assertEqual(args2p1.c8_beta, 3.0)
        self.assertEqual(args2p1.c8_codebook, "octants")
        self.assertEqual(args2p1.c8_grad_mode, "bireal_ste")
        self.assertEqual(args_semantic.c8_grad_mode, "semantic_ste")
        self.assertEqual(
            args_semantic_phase.c8_grad_mode,
            "semantic_phase_ste",
        )
        self.assertEqual(args3p1.phase, 3.1)
        self.assertEqual(args3p1.phase3p1_padding_mode, "zero")
        self.assertEqual(args3p1.phase3p1_transition_epochs, 100)
        self.assertEqual(args3p1.phase3p1_transition_schedule, "cosine")
        self.assertEqual(phase_initialization_sources(2.1, 5), (1,))
        self.assertEqual(phase_initialization_sources(3.1, 5), (2.1,))
        self.assertEqual(
            phase_checkpoint_filename(2.1),
            "Bestmodel_phase2p1.pt",
        )
        self.assertEqual(
            phase_checkpoint_filename(3.1),
            "Bestmodel_phase3p1.pt",
        )


if __name__ == "__main__":
    unittest.main()
