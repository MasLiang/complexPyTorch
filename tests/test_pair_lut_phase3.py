import argparse
import tempfile
import unittest
from pathlib import Path

import torch

import training
from complexPyTorch.complexBinaryResNet import BinaryComplexResNet
from complexPyTorch.complexLayers import (
    AnalyticPairComparatorConv2d,
    PairLUTNeuronConv2d,
    annealed_binary_table,
)


class PairLUTNeuronTests(unittest.TestCase):
    def test_analytic_pair_forward_matches_hard_lut_with_padding_and_tail(self):
        torch.manual_seed(5)
        analytic = AnalyticPairComparatorConv2d(
            3,
            2,
            kernel_size=3,
            padding=1,
            kernel_mode="floating",
        )
        lut = PairLUTNeuronConv2d(
            3,
            2,
            kernel_size=3,
            padding=1,
            training_mode="real_compatible",
            kernel_mode="floating",
        )
        lut.initialize_from_phase2_weights(
            analytic.conv_r.weight,
            analytic.conv_i.weight,
        )
        real = torch.randn(2, 3, 5, 5)
        imag = torch.randn(2, 3, 5, 5)
        inputs = torch.complex(real, imag)
        torch.testing.assert_close(
            analytic(inputs),
            lut(inputs),
            rtol=0.0,
            atol=0.0,
        )

    def test_analytic_pair_uses_latent_weights_and_input_ste(self):
        torch.manual_seed(7)
        layer = AnalyticPairComparatorConv2d(
            2,
            2,
            kernel_size=1,
            kernel_mode="floating",
        )
        trainable_names = dict(layer.named_parameters())
        self.assertIn("conv_r.weight", trainable_names)
        self.assertIn("conv_i.weight", trainable_names)
        self.assertNotIn("pair_lut.lut_r", trainable_names)
        self.assertNotIn("pair_lut.lut_i", trainable_names)

        real = torch.randn(2, 2, 3, 3, requires_grad=True)
        imag = torch.randn(2, 2, 3, 3, requires_grad=True)
        output = layer(torch.complex(real, imag))
        output.abs().mean().backward()
        self.assertGreater(real.grad.abs().sum().item(), 0.0)
        self.assertGreater(imag.grad.abs().sum().item(), 0.0)
        self.assertGreater(layer.conv_r.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(layer.conv_i.weight.grad.abs().sum().item(), 0.0)

    def test_phase3_analytic_mode_routes_without_lut_schedule(self):
        args = training.parse_args(
            [
                "--phase",
                "3",
                "--phase3-mode",
                "analytic_pair",
                "--train-from-scratch",
            ]
        )
        model = training.build_model(args, num_classes=10)
        self.assertIsInstance(
            model.stage2[0].conv,
            AnalyticPairComparatorConv2d,
        )
        state = training.update_pair_lut_annealing(model, 0, args)
        self.assertTrue(state["fully_hard"])
        self.assertEqual(state["hard_ratio"], 1.0)

    def test_random_initialization_is_trainable_and_records_baseline(self):
        torch.manual_seed(11)
        model = BinaryComplexResNet(
            in_channels=3,
            num_blocks=1,
            start_filters=2,
            num_classes=10,
            is_sar_input=False,
            phase=3,
        )
        report = training.initialize_phase3_random(model, logit_std=0.7)
        self.assertEqual(report["mode"], "random_normal")
        self.assertEqual(report["layers"], 6)
        self.assertEqual(training.pair_lut_sign_diff(model)["total"], 0)
        logits = torch.cat(
            [
                parameter.detach().flatten()
                for name, parameter in model.named_parameters()
                if name.endswith(("lut_r", "lut_i"))
            ]
        )
        self.assertAlmostEqual(logits.mean().item(), 0.0, delta=0.08)
        self.assertAlmostEqual(
            logits.std(unbiased=False).item(),
            0.7,
            delta=0.08,
        )
        model(torch.randn(1, 3, 32, 32)).square().mean().backward()

    def test_phase3_train_from_scratch_needs_no_checkpoint(self):
        args = training.parse_args(
            ["--phase", "3", "--train-from-scratch"]
        )
        self.assertEqual(training._resolve_initial_checkpoint(args), (None, None))

    def test_bimodal_initialization_matches_real_lut_recipe(self):
        torch.manual_seed(17)
        layer = PairLUTNeuronConv2d(8, 8, kernel_size=3)
        report = layer.initialize_bimodal()
        logits = torch.cat([layer.lut_r.flatten(), layer.lut_i.flatten()])
        negative = logits[logits < 0.0]
        positive = logits[logits >= 0.0]
        self.assertEqual(report["mode"], "bimodal")
        self.assertAlmostEqual(negative.mean().item(), -1.0, delta=0.03)
        self.assertAlmostEqual(
            negative.std(unbiased=False).item(),
            0.2,
            delta=0.03,
        )
        self.assertAlmostEqual(positive.mean().item(), 1.0, delta=0.03)
        self.assertAlmostEqual(
            positive.std(unbiased=False).item(),
            0.1,
            delta=0.03,
        )
        self.assertAlmostEqual(
            positive.numel() / float(logits.numel()),
            0.5,
            delta=0.03,
        )

    def test_real_compatible_table_is_hard_with_identity_logit_ste(self):
        logits = torch.tensor([-2.0, -0.1, 0.0, 3.0], requires_grad=True)
        table = annealed_binary_table(
            logits,
            tau=100.0,
            hard_ratio=0.0,
            training_mode="real_compatible",
        )
        torch.testing.assert_close(
            table,
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
        )
        table.sum().backward()
        torch.testing.assert_close(logits.grad, torch.ones_like(logits))

    def test_binary_kernel_rejects_soft_annealing_tables(self):
        with self.assertRaisesRegex(ValueError, "requires real_compatible"):
            PairLUTNeuronConv2d(
                2,
                1,
                kernel_size=1,
                training_mode="anneal",
                kernel_mode="binary",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_binary_and_floating_kernels_match_forward_and_backward(self):
        torch.manual_seed(23)
        floating = PairLUTNeuronConv2d(
            18,
            35,
            kernel_size=1,
            training_mode="real_compatible",
            kernel_mode="floating",
        ).cuda()
        binary = PairLUTNeuronConv2d(
            18,
            35,
            kernel_size=1,
            training_mode="real_compatible",
            kernel_mode="binary",
        ).cuda()
        floating.initialize_bimodal()
        binary.load_state_dict(floating.state_dict())

        real_f = torch.randn(4, 18, 5, 5, device="cuda", requires_grad=True)
        imag_f = torch.randn(4, 18, 5, 5, device="cuda", requires_grad=True)
        real_b = real_f.detach().clone().requires_grad_(True)
        imag_b = imag_f.detach().clone().requires_grad_(True)

        output_f = floating(torch.complex(real_f, imag_f))
        output_b = binary(torch.complex(real_b, imag_b))
        torch.testing.assert_close(output_b, output_f, rtol=0.0, atol=0.0)

        probe_r = torch.randn_like(output_f.real)
        probe_i = torch.randn_like(output_f.imag)
        (output_f.real * probe_r + output_f.imag * probe_i).sum().backward()
        (output_b.real * probe_r + output_b.imag * probe_i).sum().backward()

        torch.testing.assert_close(real_b.grad, real_f.grad)
        torch.testing.assert_close(imag_b.grad, imag_f.grad)
        torch.testing.assert_close(binary.lut_r.grad, floating.lut_r.grad)
        torch.testing.assert_close(binary.lut_i.grad, floating.lut_i.grad)

    def test_initialized_truth_table_matches_two_complex_products(self):
        layer = PairLUTNeuronConv2d(2, 1, kernel_size=1)
        weight_r = torch.tensor([[[[1.0]], [[-1.0]]]])
        weight_i = torch.tensor([[[[1.0]], [[1.0]]]])
        layer.initialize_from_phase2_weights(weight_r, weight_i)
        layer.hard_ratio.fill_(1.0)

        bits = torch.tensor(
            [
                [((state >> shift) & 1) * 2 - 1 for shift in (3, 2, 1, 0)]
                for state in range(16)
            ],
            dtype=torch.float32,
        )
        inputs = torch.complex(
            bits[:, (0, 2)].reshape(16, 2, 1, 1),
            bits[:, (1, 3)].reshape(16, 2, 1, 1),
        )
        output = layer(inputs)
        sum_r = bits[:, 0] - bits[:, 1] - bits[:, 2] - bits[:, 3]
        sum_i = bits[:, 0] + bits[:, 1] + bits[:, 2] - bits[:, 3]
        alpha = torch.tensor(2.0 ** 0.5)
        expected = torch.complex(
            torch.where(sum_r >= 0, 2 * alpha, -2 * alpha),
            torch.where(sum_i >= 0, 2 * alpha, -2 * alpha),
        ).reshape(16, 1, 1, 1)
        torch.testing.assert_close(output, expected)

    def test_odd_tail_uses_constant_low_dummy_input(self):
        layer = PairLUTNeuronConv2d(1, 1, kernel_size=1)
        layer.initialize_from_phase2_weights(
            torch.ones(1, 1, 1, 1),
            torch.ones(1, 1, 1, 1),
        )
        real = layer.initial_lut_r_sign[0]
        imag = layer.initial_lut_i_sign[0]
        for state in range(4):
            values = list(range(state * 4, state * 4 + 4))
            self.assertTrue(
                torch.equal(real[values, 0], real[values[0], 0].expand(4))
            )
            self.assertTrue(
                torch.equal(imag[values, 0], imag[values[0], 0].expand(4))
            )

    def test_soft_and_hard_ste_reach_inputs_and_entries(self):
        for hard_ratio in (0.0, 1.0):
            with self.subTest(hard_ratio=hard_ratio):
                layer = PairLUTNeuronConv2d(2, 1, kernel_size=1)
                layer.initialize_from_phase2_weights(
                    torch.randn(1, 2, 1, 1),
                    torch.randn(1, 2, 1, 1),
                )
                layer.hard_ratio.fill_(hard_ratio)
                real = torch.randn(2, 2, 2, 2, requires_grad=True)
                imag = torch.randn(2, 2, 2, 2, requires_grad=True)
                output = layer(torch.complex(real, imag))
                output.abs().mean().backward()
                self.assertGreater(real.grad.abs().sum().item(), 0.0)
                self.assertGreater(imag.grad.abs().sum().item(), 0.0)
                self.assertGreater(layer.lut_r.grad.abs().sum().item(), 0.0)
                self.assertGreater(layer.lut_i.grad.abs().sum().item(), 0.0)

    def test_phase2_checkpoint_converts_every_main_convolution(self):
        phase2 = BinaryComplexResNet(
            in_channels=3,
            num_blocks=1,
            start_filters=2,
            num_classes=10,
            is_sar_input=False,
            phase=2,
        )
        phase3 = BinaryComplexResNet(
            in_channels=3,
            num_blocks=1,
            start_filters=2,
            num_classes=10,
            is_sar_input=False,
            phase=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase2.pt"
            torch.save({"phase": 2, "model": phase2.state_dict()}, path)
            _, report = training.initialize_phase3_from_phase2(phase3, path)
        self.assertEqual(report["layers"], 6)
        self.assertTrue(
            all(
                bool(module.initialized)
                for module in phase3.modules()
                if isinstance(module, PairLUTNeuronConv2d)
            )
        )
        self.assertEqual(training.pair_lut_sign_diff(phase3)["total"], 0)
        phase3(torch.randn(1, 3, 32, 32)).abs().mean().backward()

    def test_annealing_reaches_fully_hard_on_final_default_epoch(self):
        model = BinaryComplexResNet(
            in_channels=3,
            num_blocks=1,
            start_filters=2,
            num_classes=10,
            is_sar_input=False,
            phase=3,
        )
        args = argparse.Namespace(
            phase=3,
            lut_anneal_epochs=160,
            lut_hard_transition_epochs=40,
            lut_tau_min=0.5,
            lut_tau_max=10.0,
        )
        start = training.update_pair_lut_annealing(model, 0, args)
        transition = training.update_pair_lut_annealing(model, 160, args)
        final = training.update_pair_lut_annealing(model, 199, args)
        self.assertEqual(start["hard_ratio"], 0.0)
        self.assertAlmostEqual(transition["hard_ratio"], 1.0 / 40.0)
        self.assertTrue(final["fully_hard"])
        self.assertAlmostEqual(final["tau"], 10.0)

    def test_real_compatible_mode_is_fully_hard_from_epoch_one(self):
        model = BinaryComplexResNet(
            in_channels=3,
            num_blocks=1,
            start_filters=2,
            num_classes=10,
            is_sar_input=False,
            phase=3,
            lut_training_mode="real_compatible",
        )
        args = argparse.Namespace(
            phase=3,
            lut_training_mode="real_compatible",
            lut_tau_max=10.0,
        )
        state = training.update_pair_lut_annealing(model, 0, args)
        self.assertTrue(state["fully_hard"])
        self.assertEqual(state["hard_ratio"], 1.0)
        self.assertTrue(
            all(
                module.hard_ratio.item() == 1.0
                for module in model.modules()
                if isinstance(module, PairLUTNeuronConv2d)
            )
        )

    def test_linear_learning_rates_decay_without_warmup(self):
        args = argparse.Namespace(
            lr=0.01,
            lut_lr=0.02,
            num_epochs=100,
            schedule="linear",
            lut_schedule="linear",
        )
        self.assertEqual(training.learning_rate_for_epoch(0, args), 0.01)
        self.assertAlmostEqual(
            training.learning_rate_for_epoch(50, args),
            0.005,
        )
        self.assertAlmostEqual(
            training.lut_learning_rate_for_epoch(99, args),
            0.0002,
        )


if __name__ == "__main__":
    unittest.main()
