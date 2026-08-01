import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch

import training
from complexPyTorch.complexBinaryResNet import BinaryComplexResNet
from complexPyTorch.complexFunctions import complex_binary_weight
from complexPyTorch.complexLayers import (
    BinaryComplexConv2d,
    ComplexAvgPool2d,
    ComplexBatchNorm2d,
    ComplexConv2d,
    ComplexLUTConv2d,
    LUTAwareComplexBinaryConv2d,
    NaiveComplexBatchNorm2d,
    PairLUTNeuronConv2d,
)


class ActiveRouteTests(unittest.TestCase):
    def make_model(self, phase):
        return BinaryComplexResNet(
            in_channels=3,
            num_blocks=1,
            start_filters=2,
            num_classes=10,
            is_sar_input=False,
            phase=phase,
        )

    def test_only_phase1_phase2_and_phase3_are_exposed(self):
        self.assertEqual(set(training.PHASE_DESCRIPTIONS), {1, 2, 3})
        for phase in (1, 2, 3):
            self.assertEqual(
                training.parse_args(["--phase", str(phase)]).phase,
                phase,
            )
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                training.parse_args(["--phase", "4"])
        with self.assertRaises(ValueError):
            self.make_model(4)

    def test_cli_has_no_experimental_route_options(self):
        argument_names = vars(training.parse_args([]))
        forbidden = ("c8", "mlp", "phase4", "phase5", "magnitude")
        for name in argument_names:
            self.assertFalse(
                any(token in name.lower() for token in forbidden),
                name,
            )

    def test_phase_models_use_expected_convolution(self):
        phase1 = self.make_model(1)
        phase2 = self.make_model(2)
        phase3 = self.make_model(3)
        self.assertIsInstance(phase1.stage2[0].conv, ComplexConv2d)
        self.assertIsInstance(phase2.stage2[0].conv, BinaryComplexConv2d)
        self.assertIsInstance(phase3.stage2[0].conv, PairLUTNeuronConv2d)
        for model in (phase1, phase2):
            self.assertFalse(
                any(
                    "LUT" in module.__class__.__name__
                    for module in model.modules()
                )
            )
        self.assertFalse(
            any(
                ".conv.conv_" in name
                for name, _ in phase3.named_parameters()
            )
        )

    def test_residual_batch_norm_modes_are_configurable(self):
        defaults = training.parse_args([])
        self.assertEqual(defaults.pre_bn_mode, "covariance")
        self.assertEqual(defaults.post_bn_mode, "covariance")

        expected_types = {
            "covariance": ComplexBatchNorm2d,
            "naive": NaiveComplexBatchNorm2d,
            "none": torch.nn.Identity,
        }
        for pre_mode, pre_type in expected_types.items():
            for post_mode, post_type in expected_types.items():
                with self.subTest(pre=pre_mode, post=post_mode):
                    model = BinaryComplexResNet(
                        in_channels=3,
                        num_blocks=1,
                        start_filters=2,
                        num_classes=10,
                        is_sar_input=False,
                        phase=2,
                        pre_bn_mode=pre_mode,
                        post_bn_mode=post_mode,
                    )
                    block = model.stage3[0]
                    self.assertIsInstance(block.bn_pre, pre_type)
                    self.assertIsInstance(block.bn_post, post_type)
                    self.assertIsInstance(block.proj[1], post_type)

    def test_naive_and_no_bn_routes_forward_and_backward(self):
        for pre_mode, post_mode in (("naive", "naive"), ("none", "none")):
            with self.subTest(pre=pre_mode, post=post_mode):
                model = BinaryComplexResNet(
                    in_channels=3,
                    num_blocks=1,
                    start_filters=2,
                    num_classes=10,
                    is_sar_input=False,
                    phase=2,
                    pre_bn_mode=pre_mode,
                    post_bn_mode=post_mode,
                )
                output = model(torch.randn(2, 3, 32, 32))
                self.assertEqual(tuple(output.shape), (2, 10))
                output.square().mean().backward()

        with self.assertRaisesRegex(ValueError, "BatchNorm mode"):
            BinaryComplexResNet(pre_bn_mode="unsupported")

    def test_phase1_and_phase2_state_keys_match(self):
        phase1 = self.make_model(1)
        phase2 = self.make_model(2)
        self.assertEqual(
            set(phase1.state_dict()),
            set(phase2.state_dict()),
        )

    def test_standard_bireal_topology_keeps_cifar_complex_frontend(self):
        defaults = training.parse_args([])
        self.assertEqual(defaults.bireal_topology, "legacy")
        parsed = training.parse_args(
            ["--phase", "2", "--bireal-topology", "standard"]
        )
        self.assertEqual(parsed.bireal_topology, "standard")

        model = BinaryComplexResNet(
            in_channels=3,
            num_blocks=1,
            start_filters=2,
            num_classes=10,
            is_sar_input=False,
            phase=2,
            bireal_topology="standard",
        )
        self.assertTrue(hasattr(model, "learn_imag"))
        self.assertIsInstance(model.stage2[0].bn_pre, torch.nn.Identity)
        self.assertIsNone(model.stage2[0].proj)
        self.assertEqual(
            model.stage2[0].conv.weight_proxy_mode,
            "bireal",
        )

        downsample = model.stage3[0].proj
        self.assertIsInstance(downsample[0], ComplexAvgPool2d)
        self.assertIsInstance(downsample[1], ComplexConv2d)
        self.assertIsInstance(downsample[2], ComplexBatchNorm2d)
        self.assertEqual(downsample[1].conv_r.stride, (1, 1))
        self.assertEqual(model.stage3[0].conv.stride, 2)

        output = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 10))
        output.square().mean().backward()

    def test_standard_bireal_weight_proxy_matches_clipped_backward(self):
        real = torch.tensor(
            [[[[0.2, -0.4]]]],
            requires_grad=True,
        )
        imag = torch.tensor(
            [[[[0.3, -0.1]]]],
            requires_grad=True,
        )
        weight = torch.complex(real, imag)
        binary = complex_binary_weight(
            weight,
            per_channel=True,
            proxy_mode="bireal",
        )
        alpha = weight.detach().abs().mean(
            dim=(1, 2, 3),
            keepdim=True,
        )
        expected = torch.complex(
            alpha * real.detach().sign(),
            alpha * imag.detach().sign(),
        )
        self.assertTrue(torch.allclose(binary.detach(), expected))

        (binary.real.sum() + binary.imag.sum()).backward()
        self.assertTrue(torch.equal(real.grad, torch.ones_like(real)))
        self.assertTrue(torch.equal(imag.grad, torch.ones_like(imag)))

    def test_standard_phase1_and_phase2_state_keys_match(self):
        models = [
            BinaryComplexResNet(
                in_channels=3,
                num_blocks=1,
                start_filters=2,
                num_classes=10,
                is_sar_input=False,
                phase=phase,
                bireal_topology="standard",
            )
            for phase in (1, 2)
        ]
        self.assertEqual(
            set(models[0].state_dict()),
            set(models[1].state_dict()),
        )

    def test_phase1_checkpoint_fully_initializes_phase2(self):
        phase1 = self.make_model(1)
        phase2 = self.make_model(2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "phase1.pt"
            torch.save(
                {
                    "phase": 1,
                    "model": phase1.state_dict(),
                },
                checkpoint,
            )
            training.load_phase_checkpoint(
                phase2,
                checkpoint,
                expected_phase=1,
            )
        for key, phase1_value in phase1.state_dict().items():
            self.assertTrue(
                torch.equal(phase1_value, phase2.state_dict()[key]),
                key,
            )

    def test_phase_models_forward_and_backward(self):
        for phase in (1, 2):
            model = self.make_model(phase)
            inputs = torch.randn(2, 3, 32, 32)
            output = model(inputs)
            self.assertEqual(tuple(output.shape), (2, 10))
            output.square().mean().backward()
            self.assertTrue(
                any(
                    parameter.grad is not None
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
            )

    def test_learning_rate_schedules_never_exceed_base_lr(self):
        for schedule in (
            "constant",
            "cosine",
            "bireal",
            "bireal_reference",
            "linear",
        ):
            args = argparse.Namespace(
                schedule=schedule,
                lr=0.01,
                min_lr_factor=0.1,
                num_epochs=200,
            )
            rates = [
                training.learning_rate_for_epoch(epoch, args)
                for epoch in range(args.num_epochs)
            ]
            self.assertLessEqual(max(rates), args.lr)
            self.assertGreaterEqual(min(rates), 0.0)

    def test_reference_bireal_schedule_uses_original_milestones(self):
        args = argparse.Namespace(
            schedule="bireal_reference",
            lr=0.001,
            min_lr_factor=0.1,
            num_epochs=256,
        )
        expected = {
            0: 1e-3,
            89: 1e-3,
            90: 1e-4,
            140: 1e-5,
            180: 1e-6,
            220: 1e-7,
        }
        for epoch, learning_rate in expected.items():
            self.assertAlmostEqual(
                training.learning_rate_for_epoch(epoch, args),
                learning_rate,
            )

    def test_retained_lut_layers_remain_available(self):
        self.assertTrue(issubclass(ComplexLUTConv2d, torch.nn.Module))
        self.assertTrue(
            issubclass(LUTAwareComplexBinaryConv2d, torch.nn.Module)
        )


if __name__ == "__main__":
    unittest.main()
