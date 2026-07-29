import tempfile
import unittest
from pathlib import Path

import torch

from complexPyTorch.complexBinaryResNet import BinaryComplexResNet
from complexPyTorch.complexLayers import ComplexLUTConv2d
from scripts.analyze_learned_lut5 import build_report
from training import (
    build_optimizer,
    capture_hard_lut_tables,
    parse_args,
)


class Phase3p1PureMLPTest(unittest.TestCase):
    def make_layer(self, lut_sets=1, hidden=32):
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
            lut_parameterization="mlp",
            neural_lut_hidden=hidden,
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
            phase3p1_mode="pure_mlp",
            lut_init_tau=1.0,
            neural_lut_hidden=32,
        )

    def test_teacher_fit_has_no_lut_parameters_and_preserves_c8_start(self):
        layer = self.make_layer(lut_sets=2)
        reference = ComplexLUTConv2d(
            1,
            1,
            kernel_size=1,
            padding=0,
            phase=4,
            lut_sets=2,
            lut_inputs=5,
            lut_init_mode="raw",
            lut5_init_strategy="c8_product",
            c8_codebook_mode="octants",
            lut_init_tau=1.0,
            lut_parameterization="direct",
        )

        self.assertIsNone(layer.lut_r)
        self.assertIsNone(layer.lut_i)
        self.assertNotIn("lut_r", dict(layer.named_parameters()))
        self.assertNotIn("lut_i", dict(layer.named_parameters()))
        self.assertEqual(layer.mlp_init_hard_mismatches, 0)
        self.assertLess(layer.mlp_init_max_abs_error, 1e-5)

        score_r, score_i = layer.effective_lut_logits()
        self.assertTrue(torch.equal(score_r >= 0, reference.lut_r >= 0))
        self.assertTrue(torch.equal(score_i >= 0, reference.lut_i >= 0))

        soft_r, soft_i = layer.mlp_operation_values()
        expected_r = (torch.tanh(reference.lut_r) + 1.0) / 2.0
        expected_i = (torch.tanh(reference.lut_i) + 1.0) / 2.0
        self.assertLess((soft_r - expected_r).abs().max().item(), 1e-5)
        self.assertLess((soft_i - expected_i).abs().max().item(), 1e-5)

    def test_mlp_is_differentiable_and_tau_independent(self):
        layer = self.make_layer()
        soft_r_before, soft_i_before = layer.mlp_operation_values()
        layer.tau.fill_(1000.0)
        soft_r_after, soft_i_after = layer.mlp_operation_values()
        self.assertTrue(torch.equal(soft_r_before, soft_r_after))
        self.assertTrue(torch.equal(soft_i_before, soft_i_after))

        weights = torch.linspace(-1.0, 1.0, 32).unsqueeze(0)
        loss = (soft_r_after * weights).sum() + (
            soft_i_after * weights.flip(1)
        ).sum()
        loss.backward()
        self.assertGreater(
            layer.lut_generator_hidden_weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertGreater(
            layer.lut_generator_output_weight.grad.abs().sum().item(),
            0.0,
        )

        with torch.no_grad():
            layer.hard.fill_(1.0)
        hard_r, hard_i = layer.mlp_operation_values()
        self.assertTrue(torch.all((hard_r == 0) | (hard_r == 1)))
        self.assertTrue(torch.all((hard_i == 0) | (hard_i == 1)))

    def test_model_routes_only_mlp_parameters_to_lut_group(self):
        model = self.make_model()
        block = model.stage2[0]
        self.assertTrue(block.learned_lut5_phase3p1)
        self.assertTrue(block.pure_mlp_phase3p1)
        self.assertFalse(block.neural_lut5_phase3p1)
        self.assertEqual(block.conv.lut_parameterization, "mlp")

        args = parse_args(
            [
                "--phase",
                "3.1",
                "--phase3p1-mode",
                "pure_mlp",
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
        lut_ids = {id(parameter) for parameter in lut_group["params"]}
        mlp_ids = {
            id(parameter)
            for name, parameter in model.named_parameters()
            if "lut_generator_" in name
        }
        self.assertEqual(lut_ids, mlp_ids)
        self.assertFalse(
            any(
                name.endswith(("lut_r", "lut_i"))
                for name, _ in model.named_parameters()
            )
        )

    def test_hard_export_and_checkpoint_analyzer_compile_the_mlp(self):
        layer = self.make_layer()
        with torch.no_grad():
            layer.lut_generator_output_bias.add_(
                torch.tensor([[0.4, -0.3]])
            )
        exported = capture_hard_lut_tables(layer)
        score_r, score_i = layer.effective_lut_logits()
        self.assertEqual(exported[""]["parameterization"], "mlp")
        self.assertTrue(
            torch.equal(
                exported[""]["real"],
                (score_r >= 0).to(torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(
                exported[""]["imag"],
                (score_i >= 0).to(torch.uint8),
            )
        )

        checkpoint = {
            "phase": 3.1,
            "epoch": 10,
            "model": layer.state_dict(),
            "args": {
                "phase3p1_mode": "pure_mlp",
                "lut_init_mode": "raw",
                "lut_logit_init": 2.0,
                "lut_tau_min": 1.0,
                "c8_codebook": "octants",
                "neural_lut_hidden": 32,
            },
            "metrics": {"hard_mode": False},
            "hard_lut_tables": exported,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pure_mlp.pt"
            torch.save(checkpoint, path)
            report = build_report(path)

        self.assertEqual(report["parameterization"], "pure_mlp")
        self.assertEqual(report["explicit_hard_table_modules"], 1)
        self.assertEqual(report["summary"]["tables"], 2)

    def test_parser_exposes_pure_mlp_without_changing_default(self):
        defaults = parse_args([])
        self.assertEqual(defaults.phase3p1_mode, "fixed")

        args = parse_args(
            [
                "--phase3p1-mode",
                "pure_mlp",
                "--neural-lut-hidden",
                "12",
                "--neural-lut-hard-eval-interval",
                "5",
            ]
        )
        self.assertEqual(args.phase3p1_mode, "pure_mlp")
        self.assertEqual(args.neural_lut_hidden, 12)
        self.assertEqual(args.neural_lut_hard_eval_interval, 5)


if __name__ == "__main__":
    unittest.main()
