import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

import training
from complexPyTorch.complexBinaryResNet import BinaryComplexResNet
from complexPyTorch.complexFunctions import complex_binary_weight
from complexPyTorch.complexLayers import (
    BinaryComplexActivation,
    BinaryComplexBitActivation,
    BinaryComplexConv2d,
    ComplexAvgPool2d,
    ComplexBatchNorm2d,
    ComplexConv2d,
    LUTBinaryConv2d,
    NaiveComplexBatchNorm2d,
    PairLUT4ComplexConv2d,
    TripleLUT6ComplexConv2d,
    TwoLUTComplexConv2d,
    binary_gumbel_softmax,
    hard_binary_table,
)


class ActiveRouteTests(unittest.TestCase):
    def make_model(
        self,
        phase,
        start_filters=2,
        phase3_operator="pair_lut4",
        pair_lut_parameterization="independent",
        pair_lut_inputs=4,
        pair_lut_encoding="standard",
    ):
        return BinaryComplexResNet(
            in_channels=3,
            num_blocks=1,
            start_filters=start_filters,
            num_classes=10,
            is_sar_input=False,
            phase=phase,
            phase3_operator=phase3_operator,
            pair_lut_parameterization=pair_lut_parameterization,
            pair_lut_inputs=pair_lut_inputs,
            pair_lut_encoding=pair_lut_encoding,
        )

    def test_only_three_phases_are_exposed(self):
        self.assertEqual(set(training.PHASE_DESCRIPTIONS), {1, 2, 3})
        for phase in (1, 2, 3):
            self.assertEqual(training.parse_args(["--phase", str(phase)]).phase, phase)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                training.parse_args(["--phase", "4"])

    def test_batch_progress_logging_is_disabled_by_default(self):
        self.assertEqual(training.parse_args([]).log_interval, 0)

    def test_cli_contains_no_archived_route_options(self):
        names = vars(training.parse_args([]))
        forbidden = (
            "phase3_mode",
            "magnitude",
            "c8",
            "mlp",
            "pre_bn",
            "bireal_topology",
        )
        for name in names:
            self.assertFalse(any(token in name.lower() for token in forbidden), name)

    def test_phase_models_use_expected_main_operator(self):
        phase1 = self.make_model(1)
        phase2 = self.make_model(2)
        phase3 = self.make_model(3)
        self.assertIsInstance(phase1.stage2[0].conv, ComplexConv2d)
        self.assertIsInstance(phase2.stage2[0].conv, BinaryComplexConv2d)
        self.assertIsInstance(phase3.stage2[0].conv, PairLUT4ComplexConv2d)
        self.assertIsInstance(phase2.stage2[0].act, BinaryComplexActivation)
        self.assertNotIsInstance(
            phase2.stage2[0].act,
            BinaryComplexBitActivation,
        )
        self.assertIsInstance(
            phase3.stage2[0].act,
            BinaryComplexBitActivation,
        )
        for model in (phase1, phase2):
            self.assertFalse(any("LUT" in type(module).__name__ for module in model.modules()))

    def test_phase3_pair_lut4_groups_two_complex_inputs(self):
        model = self.make_model(3, start_filters=16)
        operators = [
            block.conv
            for stage in (model.stage2, model.stage3, model.stage4)
            for block in stage
        ]
        self.assertTrue(operators)
        self.assertTrue(
            all(isinstance(op, PairLUT4ComplexConv2d) for op in operators)
        )
        first = operators[0]
        self.assertEqual(first.pair_num, 72)
        self.assertEqual(tuple(first.weight.shape), (32, 72, 16))


    def test_pair_lut_inputs_default_to_four(self):
        self.assertEqual(training.parse_args([]).pair_lut_inputs, 4)
        layer = self.make_model(3).stage2[0].conv
        self.assertEqual(layer.lut_inputs, 4)
        self.assertEqual(layer.complex_inputs_per_lut, 2)

    def test_categorical_lut6_shape_and_three_complex_grouping(self):
        model = self.make_model(
            3,
            start_filters=11,
            pair_lut_parameterization="categorical",
            pair_lut_inputs=6,
        )
        layer = model.stage2[0].conv
        self.assertEqual(layer.lut_inputs, 6)
        self.assertEqual(layer.complex_inputs_per_lut, 3)
        self.assertEqual(layer.group_num, 33)
        self.assertEqual(tuple(layer.weight.shape), (11, 33, 64, 4))
        self.assertEqual(tuple(layer.materialize_table().shape), (22, 33, 64))

    def test_dominance_lut6_groups_two_complex_inputs(self):
        model = self.make_model(
            3,
            start_filters=11,
            pair_lut_parameterization="categorical",
            pair_lut_inputs=6,
            pair_lut_encoding="dominance",
        )
        layer = model.stage2[0].conv
        self.assertEqual(layer.lut_inputs, 6)
        self.assertEqual(layer.complex_inputs_per_lut, 2)
        self.assertEqual(layer.group_num, 50)
        self.assertEqual(tuple(layer.weight.shape), (11, 50, 64, 4))

    def test_dominance_lut6_address_and_gradient_reach_raw_activation(self):
        layer = PairLUT4ComplexConv2d(
            2,
            1,
            kernel_size=1,
            parameterization="categorical",
            lut_inputs=6,
            activation_encoding="dominance",
            dominance_grad_mode="ste",
        )
        with torch.no_grad():
            layer.weight.fill_(-1.0)
            classes = 2 * layer.state_bits[:, 2].long()
            layer.weight[0, 0].scatter_(1, classes.unsqueeze(1), 1.0)

        raw_real = torch.tensor([[[[0.4]], [[0.2]]]], requires_grad=True)
        raw_imag = torch.tensor([[[[0.2]], [[0.4]]]], requires_grad=True)
        raw = torch.complex(raw_real, raw_imag)
        bits = BinaryComplexBitActivation()(raw)
        patches = layer._group_patches(bits, dominance_source=raw)
        self.assertTrue(
            torch.equal(
                patches.detach(),
                torch.tensor([[[[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]]]]),
            )
        )
        layer(bits, dominance_source=raw).real.sum().backward()
        self.assertGreater(float(raw_real.grad.abs().sum()), 0.0)
        self.assertGreater(float(raw_imag.grad.abs().sum()), 0.0)

    def test_dominance_lut6_stops_raw_gradient_by_default(self):
        layer = PairLUT4ComplexConv2d(
            2,
            1,
            kernel_size=1,
            parameterization="categorical",
            lut_inputs=6,
            activation_encoding="dominance",
        )
        raw_real = torch.tensor([[[[0.4]]]], requires_grad=True)
        raw_imag = torch.tensor([[[[0.2]]]], requires_grad=True)
        dominance = layer._dominance_bit(torch.complex(raw_real, raw_imag))
        self.assertFalse(dominance.requires_grad)
        self.assertEqual(float(dominance.item()), 1.0)

    def test_phase2_checkpoint_keeps_dominance_lut6_random(self):
        phase2 = self.make_model(2, start_filters=16)
        phase3 = self.make_model(
            3,
            start_filters=16,
            pair_lut_parameterization="categorical",
            pair_lut_inputs=6,
            pair_lut_encoding="dominance",
        )
        target_lut = phase3.stage2[0].conv
        initial_lut = target_lut.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "phase2.pt"
            torch.save({"phase": 2, "model": phase2.state_dict()}, checkpoint)
            training.load_phase3_shared_checkpoint(phase3, checkpoint)
        self.assertTrue(torch.equal(target_lut.weight, initial_lut))
        self.assertTrue(torch.equal(phase2.fc.weight, phase3.fc.weight))

    def test_lut4_checkpoint_warm_starts_dominance_lut6_exactly(self):
        torch.manual_seed(23)
        source = self.make_model(
            3,
            pair_lut_parameterization="categorical",
            pair_lut_inputs=4,
        )
        target = self.make_model(
            3,
            pair_lut_parameterization="categorical",
            pair_lut_inputs=6,
            pair_lut_encoding="dominance",
        )
        source_luts = [
            module
            for module in source.modules()
            if isinstance(module, PairLUT4ComplexConv2d)
        ]
        target_luts = [
            module
            for module in target.modules()
            if isinstance(module, PairLUT4ComplexConv2d)
        ]

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "lut4.pt"
            torch.save({"phase": 3, "model": source.state_dict()}, checkpoint)
            training.load_phase3_shared_checkpoint(target, checkpoint)

        for old_lut, new_lut in zip(source_luts, target_luts):
            bits = new_lut.state_bits.long()
            old_addresses = (
                bits[:, 0] * 8
                + bits[:, 1] * 4
                + bits[:, 3] * 2
                + bits[:, 4]
            )
            expected = old_lut.weight.index_select(2, old_addresses)
            self.assertTrue(torch.equal(new_lut.weight, expected))
            self.assertEqual(
                torch.nonzero(old_addresses == 0).flatten().tolist(),
                [0, 1, 8, 9],
            )

        source.eval()
        target.eval()
        images = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            old_output = source(images)
            new_output = target(images)
        self.assertTrue(torch.allclose(old_output, new_output, atol=1e-6))


    def test_lut4_warm_starts_zero_mean_dominance_residual_exactly(self):
        torch.manual_seed(29)
        source = self.make_model(
            3,
            pair_lut_parameterization="categorical",
            pair_lut_inputs=4,
        )
        target = self.make_model(
            3,
            pair_lut_parameterization="categorical_residual",
            pair_lut_inputs=6,
            pair_lut_encoding="dominance",
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "lut4.pt"
            torch.save({"phase": 3, "model": source.state_dict()}, checkpoint)
            training.load_phase3_shared_checkpoint(target, checkpoint)

        source_luts = [
            module for module in source.modules()
            if isinstance(module, PairLUT4ComplexConv2d)
        ]
        target_luts = [
            module for module in target.modules()
            if isinstance(module, PairLUT4ComplexConv2d)
        ]
        for old_lut, new_lut in zip(source_luts, target_luts):
            self.assertTrue(torch.equal(new_lut.dominance_base, old_lut.weight))
            self.assertEqual(float(new_lut.weight.detach().abs().max()), 0.0)
            expected = old_lut.weight.index_select(
                2, new_lut.dominance_old_addresses
            )
            self.assertTrue(torch.equal(new_lut._categorical_logits(), expected))

        source.eval()
        target.eval()
        images = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            self.assertTrue(
                torch.allclose(source(images), target(images), atol=1e-6)
            )

    def test_dominance_residual_is_zero_mean_per_lut4_address(self):
        layer = PairLUT4ComplexConv2d(
            2,
            1,
            kernel_size=1,
            parameterization="categorical_residual",
            lut_inputs=6,
            activation_encoding="dominance",
        )
        with torch.no_grad():
            layer.dominance_base.normal_()
            layer.weight.normal_()
            layer.set_dominance_residual_alpha(1.0)
        logits = layer._categorical_logits()
        expanded_base = layer.dominance_base.index_select(
            2, layer.dominance_old_addresses
        )
        residual = logits - expanded_base
        assignment = F.one_hot(
            layer.dominance_old_addresses, num_classes=16
        ).to(residual.dtype)
        residual_sum = torch.einsum("ogsc,sa->ogac", residual, assignment)
        self.assertTrue(
            torch.allclose(
                residual_sum, torch.zeros_like(residual_sum), atol=1e-6
            )
        )

    def test_dominance_residual_alpha_and_optimizer_lr_schedules(self):
        args = training.parse_args([
            "--phase", "3",
            "--pair-lut-parameterization", "categorical_residual",
            "--pair-lut-inputs", "6",
            "--pair-lut-encoding", "dominance",
            "--lr", "0.001",
            "--dominance-residual-lr", "0.002",
            "--dominance-residual-alpha-start", "0.1",
            "--dominance-residual-alpha-end", "1.0",
            "--dominance-residual-ramp-epochs", "3",
        ])
        self.assertAlmostEqual(
            training.dominance_residual_alpha_for_epoch(0, args), 0.1
        )
        self.assertAlmostEqual(
            training.dominance_residual_alpha_for_epoch(1, args), 0.55
        )
        self.assertAlmostEqual(
            training.dominance_residual_alpha_for_epoch(2, args), 1.0
        )

        model = self.make_model(
            3,
            pair_lut_parameterization="categorical_residual",
            pair_lut_inputs=6,
            pair_lut_encoding="dominance",
        )
        optimizer = training.build_optimizer(args, model)
        training.set_optimizer_lr(optimizer, 0.001)
        group_lrs = {
            group["name"]: group["lr"] for group in optimizer.param_groups
        }
        self.assertAlmostEqual(group_lrs["decay"], 0.001)
        self.assertAlmostEqual(group_lrs["no_decay"], 0.001)
        self.assertAlmostEqual(group_lrs["dominance_residual"], 0.002)


    def test_lut6_groups_three_complex_positions_channel_major(self):
        layer = PairLUT4ComplexConv2d(
            2,
            1,
            kernel_size=2,
            parameterization="categorical",
            lut_inputs=6,
        )
        real = torch.stack(
            [
                torch.arange(4, dtype=torch.float32) + 10 * channel
                for channel in range(2)
            ]
        ).view(1, 2, 2, 2)
        imag = real + 100.0
        groups = layer._group_patches(torch.complex(real, imag))
        expected = torch.tensor(
            [[[
                [0.0, 100.0, 1.0, 101.0, 2.0, 102.0],
                [3.0, 103.0, 10.0, 110.0, 11.0, 111.0],
                [12.0, 112.0, 13.0, 113.0, 0.0, 100.0],
            ]]]
        )
        self.assertEqual(layer.group_num, 3)
        self.assertTrue(torch.equal(groups, expected))

    def test_phase2_init_categorical_lut6_is_hard_and_has_gradients(self):
        torch.manual_seed(17)
        layer = PairLUT4ComplexConv2d(
            3,
            2,
            kernel_size=1,
            parameterization="categorical",
            lut_inputs=6,
        )
        layer.initialize_from_binary_complex_weights(
            torch.randn(2, 3, 1, 1),
            torch.randn(2, 3, 1, 1),
        )
        table = layer.materialize_table()
        self.assertEqual(tuple(table.shape), (4, 1, 64))
        self.assertTrue(
            torch.equal(torch.unique(table.detach()), torch.tensor([0.0, 1.0]))
        )

        real = torch.randint(
            0, 2, (1, 3, 1, 1), dtype=torch.float32
        ).requires_grad_()
        imag = torch.randint(
            0, 2, (1, 3, 1, 1), dtype=torch.float32
        ).requires_grad_()
        output = layer(torch.complex(real, imag))
        (output.real + 2.0 * output.imag).sum().backward()
        self.assertGreater(float(layer.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(real.grad.abs().sum()), 0.0)
        self.assertGreater(float(imag.grad.abs().sum()), 0.0)

    def test_pair_lut_parameterization_defaults_to_independent(self):
        args = training.parse_args([])
        self.assertEqual(args.pair_lut_parameterization, "independent")
        model = self.make_model(3)
        layer = model.stage2[0].conv
        self.assertEqual(layer.parameterization, "independent")
        self.assertEqual(
            tuple(layer.weight.shape),
            (2 * layer.out_channels, layer.pair_num, 16),
        )

    def test_categorical_pair_lut_maps_argmax_to_complex_codebook(self):
        layer = PairLUT4ComplexConv2d(
            2,
            1,
            kernel_size=1,
            parameterization="categorical",
        )
        with torch.no_grad():
            layer.weight.zero_()
            for address, category in enumerate((0, 1, 2, 3)):
                layer.weight[0, 0, address, category] = 1.0

        table = layer.materialize_table()
        self.assertEqual(tuple(layer.weight.shape), (1, 1, 16, 4))
        self.assertTrue(
            torch.equal(table[0, 0, :4], torch.tensor([0.0, 0.0, 1.0, 1.0]))
        )
        self.assertTrue(
            torch.equal(table[1, 0, :4], torch.tensor([0.0, 1.0, 0.0, 1.0]))
        )
        self.assertTrue(
            torch.equal(torch.unique(table.detach()), torch.tensor([0.0, 1.0]))
        )

    def test_categorical_pair_lut_backward_reaches_logits_and_inputs(self):
        torch.manual_seed(9)
        layer = PairLUT4ComplexConv2d(
            2,
            1,
            kernel_size=1,
            parameterization="categorical",
        )
        real = torch.tensor([[[[1.0]], [[0.0]]]], requires_grad=True)
        imag = torch.tensor([[[[0.0]], [[1.0]]]], requires_grad=True)
        output = layer(torch.complex(real, imag))
        (output.real + 2.0 * output.imag).sum().backward()
        self.assertGreater(float(layer.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(real.grad.abs().sum()), 0.0)
        self.assertGreater(float(imag.grad.abs().sum()), 0.0)

    def test_phase2_init_materializes_same_independent_and_categorical_tables(self):
        torch.manual_seed(11)
        weight_real = torch.randn(2, 3, 3, 3)
        weight_imag = torch.randn(2, 3, 3, 3)
        independent = PairLUT4ComplexConv2d(3, 2, kernel_size=3)
        categorical = PairLUT4ComplexConv2d(
            3,
            2,
            kernel_size=3,
            parameterization="categorical",
        )
        independent.initialize_from_binary_complex_weights(
            weight_real, weight_imag
        )
        categorical.initialize_from_binary_complex_weights(
            weight_real, weight_imag
        )
        self.assertTrue(
            torch.equal(
                independent.materialize_table().detach(),
                categorical.materialize_table().detach(),
            )
        )
    def test_triple_lut6_operator_remains_available(self):
        model = self.make_model(3, phase3_operator="triple_lut6")
        self.assertIsInstance(model.stage2[0].conv, TripleLUT6ComplexConv2d)

    def test_shared_lut6_operator_remains_available(self):
        model = self.make_model(3, phase3_operator="shared_lut6")
        self.assertIsInstance(model.stage2[0].conv, TwoLUTComplexConv2d)
        self.assertIsInstance(model.stage2[0].conv.conv_r, LUTBinaryConv2d)
        self.assertIsInstance(model.stage2[0].conv.conv_i, LUTBinaryConv2d)

    def test_triple_lut6_forward_uses_ririri_address_and_two_outputs(self):
        layer = TripleLUT6ComplexConv2d(3, 1, kernel_size=1)
        with torch.no_grad():
            layer.weight.fill_(-1.0)
            layer.weight[0, 0, 38] = 1.0
            layer.weight[1, 0, 38] = 1.0
        inp = torch.complex(
            torch.tensor([[[[1.0]], [[0.0]], [[1.0]]]]),
            torch.tensor([[[[0.0]], [[1.0]], [[0.0]]]]),
        )
        output = layer(inp)
        self.assertTrue(torch.equal(output.real, torch.ones_like(output.real)))
        self.assertTrue(torch.equal(output.imag, torch.ones_like(output.imag)))

    def test_triple_lut6_channel_priority_and_odd_tail(self):
        layer = TripleLUT6ComplexConv2d(5, 1, kernel_size=2)
        real = torch.stack(
            [
                torch.arange(4, dtype=torch.float32) + 10 * channel
                for channel in range(5)
            ]
        ).view(1, 5, 2, 2)
        imag = real + 100.0
        patches = layer._triple_patches(torch.complex(real, imag))
        expected = torch.tensor(
            [[[
                [0.0, 100.0, 10.0, 110.0, 20.0, 120.0],
                [30.0, 130.0, 40.0, 140.0, 0.0, 100.0],
                [1.0, 101.0, 11.0, 111.0, 21.0, 121.0],
                [31.0, 131.0, 41.0, 141.0, 1.0, 101.0],
                [2.0, 102.0, 12.0, 112.0, 22.0, 122.0],
                [32.0, 132.0, 42.0, 142.0, 2.0, 102.0],
                [3.0, 103.0, 13.0, 113.0, 23.0, 123.0],
                [33.0, 133.0, 43.0, 143.0, 3.0, 103.0],
            ]]]
        )
        self.assertEqual(layer.lut_num, 8)
        self.assertTrue(torch.equal(patches, expected))

    def test_triple_lut6_backward_reaches_inputs_and_entries(self):
        torch.manual_seed(5)
        layer = TripleLUT6ComplexConv2d(3, 1, kernel_size=1)
        real = torch.tensor(
            [[[[1.0]], [[0.0]], [[1.0]]]], requires_grad=True
        )
        imag = torch.tensor(
            [[[[0.0]], [[1.0]], [[0.0]]]], requires_grad=True
        )
        output = layer(torch.complex(real, imag))
        (output.real + output.imag).sum().backward()
        self.assertGreater(float(real.grad.abs().sum()), 0.0)
        self.assertGreater(float(imag.grad.abs().sum()), 0.0)
        self.assertGreater(float(layer.weight.grad.abs().sum()), 0.0)

    def test_pair_lut4_forward_uses_riri_address_and_two_outputs(self):
        layer = PairLUT4ComplexConv2d(2, 1, kernel_size=1)
        with torch.no_grad():
            layer.weight.fill_(-1.0)
            layer.weight[0, 0, 9] = 1.0
            layer.weight[1, 0, 9] = 1.0
        inp = torch.complex(
            torch.tensor([[[[1.0]], [[0.0]]]]),
            torch.tensor([[[[0.0]], [[1.0]]]]),
        )
        output = layer(inp)
        self.assertTrue(torch.equal(output.real, torch.ones_like(output.real)))
        self.assertTrue(torch.equal(output.imag, torch.ones_like(output.imag)))


    def test_pair_lut4_pairs_spatial_positions_in_channel_major_order(self):
        layer = PairLUT4ComplexConv2d(4, 1, kernel_size=2)
        real = torch.stack(
            [
                torch.arange(4, dtype=torch.float32) + 10 * channel
                for channel in range(4)
            ]
        ).view(1, 4, 2, 2)
        imag = real + 100.0
        patches = layer._pair_patches(torch.complex(real, imag))
        expected = torch.tensor(
            [[[
                [0.0, 100.0, 1.0, 101.0],
                [2.0, 102.0, 3.0, 103.0],
                [10.0, 110.0, 11.0, 111.0],
                [12.0, 112.0, 13.0, 113.0],
                [20.0, 120.0, 21.0, 121.0],
                [22.0, 122.0, 23.0, 123.0],
                [30.0, 130.0, 31.0, 131.0],
                [32.0, 132.0, 33.0, 133.0],
            ]]]
        )
        self.assertTrue(torch.equal(patches, expected))

    def test_pair_lut4_crosses_channel_boundary_for_odd_kernel_area(self):
        layer = PairLUT4ComplexConv2d(2, 1, kernel_size=3)
        real = torch.stack(
            [
                torch.arange(9, dtype=torch.float32) + 10 * channel
                for channel in range(2)
            ]
        ).view(1, 2, 3, 3)
        imag = real + 100.0
        patches = layer._pair_patches(torch.complex(real, imag))
        expected = torch.tensor(
            [[[
                [0.0, 100.0, 1.0, 101.0],
                [2.0, 102.0, 3.0, 103.0],
                [4.0, 104.0, 5.0, 105.0],
                [6.0, 106.0, 7.0, 107.0],
                [8.0, 108.0, 10.0, 110.0],
                [11.0, 111.0, 12.0, 112.0],
                [13.0, 113.0, 14.0, 114.0],
                [15.0, 115.0, 16.0, 116.0],
                [17.0, 117.0, 18.0, 118.0],
            ]]]
        )
        self.assertEqual(layer.pair_num, 9)
        self.assertTrue(torch.equal(patches, expected))

    def test_pair_lut4_global_odd_tail_repeats_first_position(self):
        layer = PairLUT4ComplexConv2d(1, 1, kernel_size=3)
        real = torch.arange(9, dtype=torch.float32).view(1, 1, 3, 3)
        imag = real + 100.0
        patches = layer._pair_patches(torch.complex(real, imag))
        self.assertEqual(layer.pair_num, 5)
        self.assertTrue(
            torch.equal(
                patches[0, 0, -1],
                torch.tensor([8.0, 108.0, 0.0, 100.0]),
            )
        )

    def test_pair_lut4_backward_reaches_inputs_and_entries(self):
        torch.manual_seed(4)
        layer = PairLUT4ComplexConv2d(2, 1, kernel_size=1)
        real = torch.tensor([[[[1.0]], [[0.0]]]], requires_grad=True)
        imag = torch.tensor([[[[0.0]], [[1.0]]]], requires_grad=True)
        output = layer(torch.complex(real, imag))
        (output.real + output.imag).sum().backward()
        self.assertGreater(float(real.grad.abs().sum()), 0.0)
        self.assertGreater(float(imag.grad.abs().sum()), 0.0)
        self.assertGreater(float(layer.weight.grad.abs().sum()), 0.0)

    def test_two_lut_complex_formula_reuses_each_branch(self):
        class Scale(torch.nn.Module):
            def __init__(self, factor):
                super().__init__()
                self.factor = factor

            def forward(self, inp):
                return self.factor * inp

        operator = TwoLUTComplexConv2d(2, 2, kernel_size=3, padding=1)
        operator.conv_r = Scale(2.0)
        operator.conv_i = Scale(3.0)
        real = torch.randn(2, 2, 4, 4)
        imag = torch.randn(2, 2, 4, 4)
        output = operator(torch.complex(real, imag))
        self.assertTrue(torch.equal(output.real, 2.0 * real - 3.0 * imag))
        self.assertTrue(torch.equal(output.imag, 2.0 * imag + 3.0 * real))

    def test_standard_bireal_topology_and_frontend_are_fixed(self):
        model = self.make_model(2)
        self.assertTrue(hasattr(model, "learn_imag"))
        self.assertFalse(hasattr(model.stage2[0], "bn_pre"))
        self.assertIsNone(model.stage2[0].proj)
        self.assertEqual(model.stage2[0].conv.weight_proxy_mode, "bireal")
        projection = model.stage3[0].proj
        self.assertIsInstance(projection[0], ComplexAvgPool2d)
        self.assertIsInstance(projection[1], ComplexConv2d)
        self.assertIsInstance(projection[2], ComplexBatchNorm2d)
        self.assertEqual(projection[1].conv_r.stride, (1, 1))
        self.assertEqual(model.stage3[0].conv.stride, 2)

    def test_post_batch_norm_modes_are_configurable(self):
        expected = {
            "covariance": ComplexBatchNorm2d,
            "naive": NaiveComplexBatchNorm2d,
            "none": torch.nn.Identity,
        }
        for mode, module_type in expected.items():
            model = BinaryComplexResNet(
                in_channels=3,
                num_blocks=1,
                start_filters=2,
                num_classes=10,
                is_sar_input=False,
                phase=2,
                post_bn_mode=mode,
            )
            self.assertIsInstance(model.stage3[0].bn_post, module_type)
            self.assertIsInstance(model.stage3[0].proj[-1], module_type)

    def test_phase1_and_phase2_state_keys_match(self):
        self.assertEqual(
            set(self.make_model(1).state_dict()),
            set(self.make_model(2).state_dict()),
        )

    def test_phase1_checkpoint_fully_initializes_phase2(self):
        phase1 = self.make_model(1)
        phase2 = self.make_model(2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "phase1.pt"
            torch.save({"phase": 1, "model": phase1.state_dict()}, checkpoint)
            training.load_phase_checkpoint(phase2, checkpoint, expected_phase=1)
        for key, value in phase1.state_dict().items():
            self.assertTrue(torch.equal(value, phase2.state_dict()[key]), key)


    def test_phase2_checkpoint_compiles_pair_lut4_tables(self):
        phase2 = self.make_model(2, start_filters=16)
        phase3 = self.make_model(3, start_filters=16)
        source_conv = phase2.stage2[0].conv
        target_lut = phase3.stage2[0].conv
        with torch.no_grad():
            source_conv.conv_r.weight.fill_(1.0)
            source_conv.conv_i.weight.fill_(1.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "phase2.pt"
            torch.save({"phase": 2, "model": phase2.state_dict()}, checkpoint)
            training.load_phase3_shared_checkpoint(phase3, checkpoint)

        states = target_lut.state_bits * 2.0 - 1.0
        expected_real = []
        expected_imag = []
        for xr0, xi0, xr1, xi1 in states.tolist():
            expected_real.append(
                1.0 if xr0 - xi0 + xr1 - xi1 >= 0.0 else -1.0
            )
            expected_imag.append(
                1.0 if xi0 + xr0 + xi1 + xr1 >= 0.0 else -1.0
            )
        self.assertTrue(
            torch.equal(target_lut.weight[0, 0], torch.tensor(expected_real))
        )
        self.assertTrue(
            torch.equal(
                target_lut.weight[target_lut.out_channels, 0],
                torch.tensor(expected_imag),
            )
        )
        self.assertTrue(torch.equal(phase2.fc.weight, phase3.fc.weight))


    def test_phase3_loader_maps_legacy_projection_indices(self):
        phase2 = self.make_model(2)
        phase3 = self.make_model(3)
        legacy_state = {}
        for key, value in phase2.state_dict().items():
            legacy_key = key
            if ".proj.1." in key:
                legacy_key = key.replace(".proj.1.", ".proj.0.", 1)
            elif ".proj.2." in key:
                legacy_key = key.replace(".proj.2.", ".proj.1.", 1)
            legacy_state[legacy_key] = value

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy_phase2.pt"
            torch.save({"phase": 2, "model": legacy_state}, checkpoint)
            training.load_phase3_shared_checkpoint(phase3, checkpoint)

        self.assertTrue(
            torch.equal(
                phase2.stage3[0].proj[1].conv_r.weight,
                phase3.stage3[0].proj[1].conv_r.weight,
            )
        )
        self.assertTrue(
            torch.equal(
                phase2.stage3[0].proj[2].weight,
                phase3.stage3[0].proj[2].weight,
            )
        )
    def test_pair_lut4_checkpoint_init_zero_pads_odd_tail_weight(self):
        layer = PairLUT4ComplexConv2d(3, 1, kernel_size=1)
        weight_real = torch.ones(1, 3, 1, 1)
        weight_imag = torch.ones(1, 3, 1, 1)
        layer.initialize_from_binary_complex_weights(weight_real, weight_imag)

        tail_table_real = layer.weight[0, 1]
        tail_table_imag = layer.weight[1, 1]
        for base_state in (0, 4, 8, 12):
            self.assertTrue(
                torch.equal(
                    tail_table_real[base_state : base_state + 4],
                    tail_table_real[base_state].expand(4),
                )
            )
            self.assertTrue(
                torch.equal(
                    tail_table_imag[base_state : base_state + 4],
                    tail_table_imag[base_state].expand(4),
                )
            )

    def test_hard_table_has_identity_ste(self):
        logits = torch.tensor([-1.0, -0.1, 0.0, 0.2], requires_grad=True)
        table = hard_binary_table(logits)
        self.assertTrue(torch.equal(table.detach(), torch.tensor([0.0, 0.0, 1.0, 1.0])))
        table.sum().backward()
        self.assertTrue(torch.equal(logits.grad, torch.ones_like(logits)))

    def test_lut_temperature_increase_sharpens_soft_entries(self):
        logits = torch.tensor([-1.0, 1.0])
        low = binary_gumbel_softmax(logits, tau=0.5, hard=0)
        high = binary_gumbel_softmax(logits, tau=2.0, hard=0)
        self.assertLess(float(high[0]), float(low[0]))
        self.assertGreater(float(high[1]), float(low[1]))

    def test_nonfinite_gradient_report_identifies_bad_parameter(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.tensor([[float("inf"), float("nan")]])
        with tempfile.TemporaryDirectory() as directory:
            path, report = training._write_nonfinite_gradient_report(
                model=model,
                workdir=directory,
                epoch=18,
                batch_index=123,
                loss=torch.tensor(1.0),
                output=torch.tensor([[0.5]]),
                clipnorm=5.0,
                clip_error=RuntimeError("non-finite norm"),
            )
            saved = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(report["diagnosis"], "nonfinite_gradient_elements")
        self.assertEqual(saved["nonfinite_gradients"][0]["name"], "weight")
        self.assertEqual(saved["nonfinite_gradients"][0]["nan"], 1)
        self.assertEqual(saved["nonfinite_gradients"][0]["pos_inf"], 1)

    def test_phase3_lut_flow_is_hard_from_first_epoch(self):
        model = self.make_model(3)
        args = type("Args", (), {"phase": 3})()
        state = training.configure_lut_training_flow(model, 0, args)
        operators = [
            module
            for module in model.modules()
            if isinstance(
                module, (TwoLUTComplexConv2d, PairLUT4ComplexConv2d, TripleLUT6ComplexConv2d)
            )
        ]
        self.assertEqual(state["stage"], "fully_hard")
        self.assertIsNone(state["temperature"])
        self.assertEqual(state["hard_operators"], len(operators))
        self.assertTrue(state["fully_hard"])

    def test_phase3_bit_activation_is_exact_and_has_scaled_gradient(self):
        real = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
        imag = real.detach().clone().requires_grad_(True)
        activation = BinaryComplexBitActivation(grad_mode="bireal")
        output = activation(torch.complex(real, imag))
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
        self.assertTrue(torch.equal(output.real, expected))
        self.assertTrue(torch.equal(output.imag, expected))
        (output.real.sum() + output.imag.sum()).backward()
        expected_grad = torch.tensor([0.0, 0.5, 1.0, 0.5, 0.0])
        self.assertTrue(torch.allclose(real.grad, expected_grad))
        self.assertTrue(torch.allclose(imag.grad, expected_grad))

    def test_lut_initialization_is_bimodal(self):
        layer = LUTBinaryConv2d(16, 8, kernel_size=3)
        weight = layer.weight.detach()
        self.assertGreater(float((weight > 0).float().mean()), 0.35)
        self.assertLess(float((weight > 0).float().mean()), 0.65)
        self.assertGreater(float(weight.abs().mean()), 0.8)
        self.assertLess(float(weight.abs().mean()), 1.2)

    def test_binary_lut_repeats_inputs_to_complete_tail_group(self):
        layer = LUTBinaryConv2d(11, 8, kernel_size=3)
        self.assertEqual(layer.logical_positions, 99)
        self.assertEqual(layer.lut_num, 17)
        self.assertEqual(tuple(layer.weight.shape), (8, 17, 64))

    def test_phase1_and_phase2_forward_backward(self):
        for phase in (1, 2):
            model = self.make_model(phase)
            output = model(torch.randn(2, 3, 32, 32))
            self.assertEqual(tuple(output.shape), (2, 10))
            output.square().mean().backward()

    def test_bireal_weight_proxy_backward(self):
        real = torch.tensor([[[[0.2, -0.4]]]], requires_grad=True)
        imag = torch.tensor([[[[0.3, -0.1]]]], requires_grad=True)
        binary = complex_binary_weight(
            torch.complex(real, imag), per_channel=True, proxy_mode="bireal"
        )
        (binary.real.sum() + binary.imag.sum()).backward()
        self.assertTrue(torch.equal(real.grad, torch.ones_like(real)))
        self.assertTrue(torch.equal(imag.grad, torch.ones_like(imag)))

    def test_learning_rate_schedules_do_not_exceed_base_lr(self):
        for schedule in (
            "constant",
            "cosine",
            "bireal",
            "bireal_reference",
            "multistep",
            "linear",
        ):
            args = argparse.Namespace(
                schedule=schedule,
                lr=0.01,
                min_lr_factor=0.1,
                num_epochs=256,
            )
            rates = [training.learning_rate_for_epoch(epoch, args) for epoch in range(256)]
            self.assertLessEqual(max(rates), args.lr)
            self.assertGreaterEqual(min(rates), 0.0)

    def test_bireal_reference_schedule_is_linear(self):
        args = argparse.Namespace(
            schedule="bireal_reference",
            lr=0.001,
            min_lr_factor=0.1,
            num_epochs=256,
        )
        expected = (0.001, 0.0005, 0.001 / 256.0)
        actual = tuple(
            training.learning_rate_for_epoch(epoch, args)
            for epoch in (0, 128, 255)
        )
        for observed, target in zip(actual, expected):
            self.assertAlmostEqual(observed, target)

        args.schedule = "linear"
        for epoch in range(args.num_epochs):
            self.assertEqual(
                training.learning_rate_for_epoch(epoch, args),
                training.learning_rate_for_epoch(
                    epoch,
                    argparse.Namespace(
                        schedule="bireal_reference",
                        lr=args.lr,
                        min_lr_factor=args.min_lr_factor,
                        num_epochs=args.num_epochs,
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
