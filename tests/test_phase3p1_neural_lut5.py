import tempfile
import unittest
from pathlib import Path

import torch

from complexPyTorch.complexBinaryResNet import BinaryComplexResNet
from complexPyTorch.complexLayers import ComplexLUTConv2d
from scripts.analyze_learned_lut5 import build_report
from training import (
    analyze_lut_activation_bit_sensitivity,
    analyze_lut_change,
    build_optimizer,
    capture_hard_lut_tables,
    capture_lut_reference,
    parse_args,
)


class Phase3p1NeuralLUT5Test(unittest.TestCase):
    def make_layer(self, lut_sets=1, hidden=8):
        return ComplexLUTConv2d(
            1,
            1,
            kernel_size=1,
            padding=0,
            phase=4,
            lut_sets=lut_sets,
            lut_inputs=5,
            lut_init_mode="raw",
            lut5_init_strategy="c8_product",
            c8_codebook_mode="octants",
            lut_init_tau=1.0,
            lut_parameterization="neural",
            neural_lut_hidden=hidden,
            neural_lut_residual_scale=1.0,
        )

    def make_model(self):
        return BinaryComplexResNet(
            in_channels=3,
            num_blocks=1,
            start_filters=2,
            num_classes=10,
            is_sar_input=False,
            is_binary=True,
            phase=3.1,
            lut_inputs=5,
            lut_init_mode="raw",
            c8_codebook="octants",
            c8_grad_mode="semantic_phase_ste",
            phase3p1_mode="neural_lut5",
            lut_init_tau=1.0,
            neural_lut_hidden=8,
            neural_lut_residual_scale=1.0,
        )

    def test_zero_residual_exactly_preserves_analytic_initial_table(self):
        layer = self.make_layer(lut_sets=2)
        effective_r, effective_i = layer.effective_lut_logits()

        self.assertTrue(torch.equal(effective_r, layer.lut_r))
        self.assertTrue(torch.equal(effective_i, layer.lut_i))
        self.assertFalse(layer.lut_r.requires_grad)
        self.assertFalse(layer.lut_i.requires_grad)
        self.assertTrue(layer.lut_generator_output_weight.requires_grad)
        self.assertEqual(
            tuple(layer.lut_generator_hidden_weight.shape),
            (2, 8, 5),
        )
        self.assertEqual(
            tuple(layer.lut_generator_output_weight.shape),
            (2, 2, 8),
        )
        sensitivity = analyze_lut_activation_bit_sensitivity(
            layer,
            ("c8_gray_bit_2", "c8_gray_bit_1", "c8_gray_bit_0"),
        )
        self.assertEqual(
            sensitivity["bits"]["c8_gray_bit_2"]["hard_diff_ratio"],
            0.5,
        )
        self.assertEqual(
            sensitivity["bits"]["c8_gray_bit_1"]["hard_diff_ratio"],
            0.5,
        )
        self.assertEqual(
            sensitivity["bits"]["c8_gray_bit_0"]["hard_diff_ratio"],
            0.0,
        )

    def test_generator_receives_gradients_and_changes_effective_table(self):
        layer = self.make_layer()
        reference = capture_lut_reference(layer)
        with torch.no_grad():
            layer.lut_generator_output_weight.normal_(mean=0.0, std=0.1)

        effective_r, effective_i = layer.effective_lut_logits()
        address_weight = torch.linspace(-1.0, 1.0, 32).unsqueeze(0)
        loss = (
            (effective_r * address_weight).sum()
            + (effective_i * address_weight.flip(1)).sum()
        )
        loss.backward()

        self.assertGreater(
            layer.lut_generator_hidden_weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertGreater(
            layer.lut_generator_output_weight.grad.abs().sum().item(),
            0.0,
        )
        stats = analyze_lut_change(layer, reference)
        self.assertGreater(stats["abs_delta_mean"], 0.0)

    def test_model_routes_only_neural_parameters_to_lut_group(self):
        model = self.make_model()
        block = model.stage2[0]
        self.assertTrue(block.learned_lut5_phase3p1)
        self.assertTrue(block.neural_lut5_phase3p1)
        self.assertEqual(block.conv.lut_parameterization, "neural")
        self.assertFalse(block.conv.lut_r.requires_grad)

        args = parse_args(
            [
                "--phase",
                "3.1",
                "--phase3p1-mode",
                "neural_lut5",
                "--lut-inputs",
                "5",
                "--lut-init-mode",
                "raw",
                "--lut-lr",
                "0.001",
            ]
        )
        optimizer = build_optimizer(args, model)
        lut_group = next(
            group for group in optimizer.param_groups
            if group["name"] == "lut"
        )
        lut_ids = {id(param) for param in lut_group["params"]}
        generator_ids = {
            id(param)
            for name, param in model.named_parameters()
            if "lut_generator_" in name
        }
        self.assertEqual(lut_ids, generator_ids)
        self.assertNotIn(id(block.conv.weight_r), lut_ids)

    def test_hard_export_matches_generated_logit_signs(self):
        layer = self.make_layer()
        with torch.no_grad():
            layer.lut_generator_output_bias.copy_(
                torch.tensor([[0.5, -0.5]])
            )
        exported = capture_hard_lut_tables(layer)
        effective_r, effective_i = layer.effective_lut_logits()

        self.assertEqual(set(exported), {""})
        self.assertTrue(
            torch.equal(
                exported[""]["real"],
                (effective_r >= 0).to(torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(
                exported[""]["imag"],
                (effective_i >= 0).to(torch.uint8),
            )
        )
        self.assertEqual(exported[""]["parameterization"], "neural")

    def test_checkpoint_analyzer_reconstructs_neural_effective_tables(self):
        layer = self.make_layer()
        with torch.no_grad():
            layer.lut_generator_output_bias.copy_(
                torch.tensor([[0.3, -0.2]])
            )
            layer.hard.fill_(1.0)
        checkpoint = {
            "phase": 3.1,
            "epoch": 160,
            "model": layer.state_dict(),
            "args": {
                "phase3p1_mode": "neural_lut5",
                "lut_init_mode": "raw",
                "lut_logit_init": 2.0,
                "lut_tau_min": 1.0,
                "c8_codebook": "octants",
                "neural_lut_hidden": 8,
                "neural_lut_residual_scale": 1.0,
            },
            "metrics": {"hard_mode": True},
            "hard_lut_tables": capture_hard_lut_tables(layer),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "neural_lut5.pt"
            torch.save(checkpoint, path)
            report = build_report(path)

        self.assertEqual(report["parameterization"], "neural_residual")
        self.assertEqual(report["explicit_hard_table_modules"], 1)
        self.assertEqual(report["summary"]["tables"], 2)
        self.assertGreater(report["summary"]["init_sign_diff"], 0)

    def test_parser_keeps_old_default_and_exposes_neural_options(self):
        defaults = parse_args([])
        self.assertEqual(defaults.phase3p1_mode, "fixed")
        self.assertEqual(defaults.neural_lut_hidden, 8)
        self.assertEqual(defaults.neural_lut_hard_eval_interval, 10)

        args = parse_args(
            [
                "--phase3p1-mode",
                "neural_lut5",
                "--neural-lut-hidden",
                "4",
                "--neural-lut-residual-scale",
                "0.5",
                "--neural-lut-hard-eval-interval",
                "5",
            ]
        )
        self.assertEqual(args.phase3p1_mode, "neural_lut5")
        self.assertEqual(args.neural_lut_hidden, 4)
        self.assertEqual(args.neural_lut_residual_scale, 0.5)
        self.assertEqual(args.neural_lut_hard_eval_interval, 5)


if __name__ == "__main__":
    unittest.main()
