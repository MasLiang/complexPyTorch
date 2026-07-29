import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from complexPyTorch.complexBinaryResNet import BinaryComplexResNet
from complexPyTorch.fixedAnalyticDominanceFlow import (
    ComparatorAwareDominanceConv2d,
    FixedAnalyticDominanceComplexResNet,
    FixedDominanceActivation,
    export_fixed_dominance_lut5,
    fixed_analytic_diagnostics,
)
from complexPyTorch.complexLayers import ComplexLUTConv2d
from scripts.analyze_fixed_analytic_run import analyze_fixed_analytic
from training import load_phase_checkpoint
from training_fixed_analytic import parse_args, validate_args


class FixedAnalyticLayerTest(unittest.TestCase):
    def test_exported_truth_table_uses_dominance(self):
        tables = export_fixed_dominance_lut5()
        self.assertEqual(tuple(tables["real"].shape), (32,))
        self.assertEqual(tuple(tables["imag"].shape), (32,))
        states0 = torch.arange(32, dtype=torch.long)
        states0 = states0[(states0 & 0b00100) == 0]
        states1 = states0 | 0b00100
        differences = sum(
            int(
                (
                    tables[name].index_select(0, states0)
                    != tables[name].index_select(0, states1)
                ).sum().item()
            )
            for name in ("real", "imag")
        )
        self.assertEqual(differences, 16)

    def test_layer_hard_forward_matches_all_32_exported_entries(self):
        layer = ComparatorAwareDominanceConv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=1,
            padding=0,
            per_channel=True,
        )
        weight_states = torch.arange(4, dtype=torch.long)
        with torch.no_grad():
            layer.conv_r.weight.copy_(
                ((((weight_states >> 1) & 1) * 2 - 1).float()).view(4, 1, 1, 1)
            )
            layer.conv_i.weight.copy_(
                (((weight_states & 1) * 2 - 1).float()).view(4, 1, 1, 1)
            )

        activation_states = torch.arange(8, dtype=torch.long)
        bits = torch.stack(
            (
                ((activation_states >> 2) & 1) * 2 - 1,
                ((activation_states >> 1) & 1) * 2 - 1,
                (activation_states & 1) * 2 - 1,
            ),
            dim=1,
        ).to(torch.float32).reshape(8, 3, 1, 1)
        with torch.no_grad():
            output = layer(bits)

        train_bits = bits.clone().requires_grad_(True)
        training_output = layer(train_bits)
        self.assertTrue(torch.equal(training_output.detach(), output))

        tables = export_fixed_dominance_lut5()
        address = activation_states[:, None] * 4 + weight_states[None, :]
        scale = 2.0 * math.sqrt(2.0)
        expected_r = (tables["real"][address].float() * 2.0 - 1.0) * scale
        expected_i = (tables["imag"][address].float() * 2.0 - 1.0) * scale
        self.assertTrue(torch.equal(output.real[:, :, 0, 0], expected_r))
        self.assertTrue(torch.equal(output.imag[:, :, 0, 0], expected_i))

    def test_comparator_surrogate_reaches_bits_source_and_weights(self):
        activation = FixedDominanceActivation(beta=2.0)
        layer = ComparatorAwareDominanceConv2d(
            in_channels=1,
            out_channels=2,
            kernel_size=1,
            padding=0,
            comparator_beta=1.0,
            comparator_scale=1.0,
        )
        with torch.no_grad():
            layer.conv_r.weight.copy_(
                torch.tensor([[[[0.8]]], [[[-0.6]]]])
            )
            layer.conv_i.weight.copy_(
                torch.tensor([[[[-0.4]]], [[[0.9]]]])
            )
        real = torch.tensor(
            [[[[0.7, -0.2], [0.4, -0.8]]]], requires_grad=True
        )
        imag = torch.tensor(
            [[[[0.3, 0.6], [-0.9, -0.1]]]], requires_grad=True
        )
        bits = activation(torch.complex(real, imag))
        bits.retain_grad()
        output = layer(bits)
        loss = output.real.square().mean() + 0.37 * output.imag.square().mean()
        loss.backward()

        self.assertIsNotNone(bits.grad)
        for bit in range(3):
            self.assertGreater(float(bits.grad[:, bit].abs().sum()), 0.0)
        self.assertGreater(float(real.grad.abs().sum()), 0.0)
        self.assertGreater(float(imag.grad.abs().sum()), 0.0)
        self.assertGreater(float(layer.conv_r.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(layer.conv_i.weight.grad.abs().sum()), 0.0)


class FixedAnalyticModelTest(unittest.TestCase):
    def make_model(self):
        return FixedAnalyticDominanceComplexResNet(
            num_blocks=1,
            start_filters=2,
            num_classes=10,
            dominance_beta=2.0,
            comparator_beta=1.0,
        )

    def test_model_contains_no_lut_module_or_parameter(self):
        model = self.make_model()
        self.assertEqual(model.lut_inputs, 5)
        self.assertFalse(
            any(isinstance(module, ComplexLUTConv2d) for module in model.modules())
        )
        self.assertFalse(
            any("lut" in name.lower() for name, _ in model.named_parameters())
        )
        diagnostics = fixed_analytic_diagnostics(model)
        self.assertEqual(diagnostics["analytic_conv_modules"], 6)
        self.assertEqual(diagnostics["lut_modules"], 0)
        self.assertTrue(diagnostics["hardware_deployable"])

    def test_phase1_checkpoint_maps_into_analytic_model(self):
        source = BinaryComplexResNet(
            num_blocks=1,
            start_filters=2,
            num_classes=10,
            is_sar_input=False,
            is_binary=False,
            phase=1,
        )
        target = self.make_model()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "phase1.pt"
            torch.save({"phase": 1, "model": source.state_dict()}, checkpoint)
            load_phase_checkpoint(target, checkpoint, expected_phase=1)
        self.assertTrue(
            torch.equal(
                source.stage2[0].conv.conv_r.weight,
                target.stage2[0].conv.conv_r.weight,
            )
        )
        self.assertTrue(torch.equal(source.fc.weight, target.fc.weight))

    def test_training_cli_has_no_lut_controls(self):
        args = parse_args(
            [
                "--checkpoint",
                "phase1.pt",
                "--workdir",
                "run",
                "--num-epochs",
                "2",
            ]
        )
        validate_args(args)
        self.assertFalse(any("lut" in name.lower() for name in vars(args)))


class FixedAnalyticAnalyzerTest(unittest.TestCase):
    def test_analyzer_reads_structured_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            (workdir / "logs").mkdir()
            (workdir / "logs" / "train.txt").write_text(
                "Epoch     1 train_loss: 1.0, train_acc: 0.5, "
                "val_loss: 0.9, val_acc: 0.6, test_loss: 0.8, "
                "test_acc: 0.7\n",
                encoding="utf-8",
            )
            payload = {
                "args": {"num_epochs": 1},
                "best_checkpoint": "best.pt",
                "best_metrics": {"epoch": 1, "val_acc": 0.6, "test_acc": 0.7},
                "last_metrics": {
                    "epoch": 1,
                    "train_acc": 0.5,
                    "val_acc": 0.6,
                    "test_acc": 0.7,
                },
                "analytic_diagnostics": {
                    "analytic_conv_modules": 6,
                    "lut_modules": 0,
                    "hardware_deployable": True,
                },
                "bit_occupancy": {"active_codes": 8},
                "bit_gradients": {
                    "bits": {
                        "sign_r": {"mean_abs": 1e-3},
                        "sign_i": {"mean_abs": 1e-3},
                        "dominance": {"mean_abs": 1e-3},
                    }
                },
                "training_uses_lut": False,
                "truth_table_trainable": False,
            }
            (workdir / "fixed_analytic_metrics.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            report = analyze_fixed_analytic(workdir)
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["training_uses_lut"])
        self.assertEqual(report["warnings"], [])


if __name__ == "__main__":
    unittest.main()
