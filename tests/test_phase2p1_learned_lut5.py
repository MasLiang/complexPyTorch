import unittest
from types import SimpleNamespace

import torch

from complexPyTorch.complexBinaryResNet import BinaryComplexResNet
from complexPyTorch.complexLayers import (
    BinaryComplexActivation,
    ComplexLUTConv2d,
)
from training import (
    analyze_lut5_slice_divergence,
    parse_args,
    update_lut_annealing,
)


class Phase2p1LearnedLUT5Test(unittest.TestCase):
    def make_layer(self):
        return ComplexLUTConv2d(
            1,
            1,
            kernel_size=1,
            padding=0,
            phase=4,
            lut_inputs=5,
            lut_logit_init=2.0,
            lut_init_mode="raw",
            lut5_init_strategy="duplicate_lut4",
        )

    def test_raw_lut5_duplicates_lut4_and_preserves_zero_ties(self):
        layer = self.make_layer()
        state = torch.arange(32)
        slice0 = state[((state >> 2) & 1) == 0]
        slice1 = slice0 | 0b00100

        for table in (layer.lut_r, layer.lut_i):
            self.assertEqual(tuple(table.shape), (1, 32))
            self.assertTrue(
                torch.equal(
                    table.index_select(1, slice0),
                    table.index_select(1, slice1),
                )
            )
            self.assertEqual(int((table == 0).sum().item()), 16)
            self.assertEqual(int((table.abs() == 2).sum().item()), 16)

    def test_phase2p1_mode_builds_trainable_two_output_lut5(self):
        model = BinaryComplexResNet(
            in_channels=3,
            num_blocks=1,
            start_filters=2,
            num_classes=10,
            is_sar_input=False,
            is_binary=True,
            phase=2.1,
            lut_inputs=5,
            lut_init_mode="raw",
            phase2p1_mode="learned_lut5",
        )
        block = model.stage2[0]
        self.assertIsInstance(block.act, BinaryComplexActivation)
        self.assertIsInstance(block.conv, ComplexLUTConv2d)
        self.assertTrue(block.uses_lut5)
        self.assertEqual(block.conv.lut_inputs, 5)
        self.assertEqual(block.conv.lut5_init_strategy, "duplicate_lut4")
        self.assertTrue(block.conv.lut_r.requires_grad)
        self.assertTrue(block.conv.lut_i.requires_grad)

    def test_sign_and_dominance_ste_reach_prequantized_activation(self):
        layer = self.make_layer()
        activation = BinaryComplexActivation(grad_mode="bireal")
        real = torch.tensor([[[[0.25]]]], requires_grad=True)
        imag = torch.tensor([[[[-0.50]]]], requires_grad=True)
        source = torch.complex(real, imag)
        quantized = activation(source)

        x_cat = layer._make_x_cat(quantized, phase_source=source)
        sign_loss = x_cat[:, 0].sum() + 2.0 * x_cat[:, 1].sum()
        sign_loss.backward(retain_graph=True)
        sign_real_grad = real.grad.detach().clone()
        sign_imag_grad = imag.grad.detach().clone()
        real.grad.zero_()
        imag.grad.zero_()

        dominance_loss = 3.0 * x_cat[:, 2].sum()
        dominance_loss.backward()
        self.assertGreater(sign_real_grad.abs().sum().item(), 0.0)
        self.assertGreater(sign_imag_grad.abs().sum().item(), 0.0)
        self.assertGreater(real.grad.abs().sum().item(), 0.0)
        self.assertGreater(imag.grad.abs().sum().item(), 0.0)

    def test_d_slice_diagnostic_detects_hard_dependency(self):
        layer = self.make_layer()
        model = torch.nn.Sequential(layer)
        initial = analyze_lut5_slice_divergence(model)
        self.assertEqual(initial["hard_slice_diff"], 0)

        with torch.no_grad():
            layer.lut_r[0, 4] = -1.0
        changed = analyze_lut5_slice_divergence(model)
        self.assertEqual(changed["hard_slice_diff"], 1)
        self.assertTrue(changed["hard_slice_diff_ratio"] > 0.0)

    def test_phase2p1_learned_lut5_uses_phase4_annealing(self):
        layer = self.make_layer()
        args = SimpleNamespace(
            phase2p1_mode="learned_lut5",
            lut_hard_ste=False,
            lut_anneal_epochs=2,
            lut_hard_epoch_fraction=0.9,
            lut_tau_min=1.0,
            lut_tau_max=10.0,
            lut_hard_transition_epochs=2,
        )
        self.assertFalse(
            update_lut_annealing(
                layer,
                epoch=0,
                num_epochs=4,
                phase=2.1,
                train_logger=None,
                is_main=False,
                args=args,
            )
        )
        self.assertEqual(float(layer.hard.item()), 0.0)
        self.assertFalse(
            update_lut_annealing(
                layer,
                epoch=2,
                num_epochs=4,
                phase=2.1,
                train_logger=None,
                is_main=False,
                args=args,
            )
        )
        self.assertAlmostEqual(float(layer.hard.item()), 0.5)
        self.assertTrue(
            update_lut_annealing(
                layer,
                epoch=3,
                num_epochs=4,
                phase=2.1,
                train_logger=None,
                is_main=False,
                args=args,
            )
        )
        self.assertEqual(float(layer.hard.item()), 1.0)

    def test_parser_exposes_new_mode_without_changing_legacy_default(self):
        self.assertEqual(parse_args([]).phase2p1_mode, "c8")
        args = parse_args(
            [
                "--phase",
                "2.1",
                "--phase2p1-mode",
                "learned_lut5",
                "--lut-inputs",
                "5",
            ]
        )
        self.assertEqual(args.phase2p1_mode, "learned_lut5")


if __name__ == "__main__":
    unittest.main()
