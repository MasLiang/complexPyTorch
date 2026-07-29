import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from complexPyTorch.complexBinaryResNet import BinaryComplexResNet
from complexPyTorch.complexLayers import (
    C8ComplexActivation,
    ComplexLUTConv2d,
    _c8_gray_bits,
    _c8_unit_codebook,
    binary_annealing,
)
from training import (
    C8CodeOccupancyTracker,
    load_phase_checkpoint,
    parse_args,
    update_lut_annealing,
)


class Phase3p1LearnedLUT5Test(unittest.TestCase):
    def make_layer(self, codebook="octants", tau=1.0):
        return ComplexLUTConv2d(
            1,
            1,
            kernel_size=1,
            padding=0,
            phase=4,
            lut_inputs=5,
            lut_init_mode="raw",
            lut5_init_strategy="c8_product",
            c8_codebook_mode=codebook,
            lut_init_tau=tau,
        )

    def expected_local_products(self, codebook, output):
        state = torch.arange(32)
        activation_address = state >> 2
        wr = (((state >> 1) & 1).float() * 2.0) - 1.0
        wi = ((state & 1).float() * 2.0) - 1.0

        phase_index = torch.arange(8, dtype=torch.long)
        gray_address = phase_index ^ (phase_index >> 1)
        address_to_phase = torch.empty(8, dtype=torch.long)
        address_to_phase[gray_address] = phase_index
        decoded = _c8_unit_codebook(codebook)[
            address_to_phase[activation_address]
        ]
        if output == "real":
            return decoded[:, 0] * wr - decoded[:, 1] * wi
        return decoded[:, 0] * wi + decoded[:, 1] * wr

    def test_soft_initial_table_exactly_recovers_c8_local_product(self):
        for codebook in ("roots", "octants"):
            for tau in (0.5, 1.0, 2.0):
                layer = self.make_layer(codebook=codebook, tau=tau)
                for table, output in (
                    (layer.lut_r, "real"),
                    (layer.lut_i, "imag"),
                ):
                    probability = binary_annealing(
                        table,
                        tau=torch.tensor(tau),
                        hard=torch.tensor(0.0),
                    )
                    recovered = 4.0 * (probability[0] - 0.5)
                    expected = self.expected_local_products(
                        codebook,
                        output,
                    )
                    self.assertTrue(
                        torch.allclose(
                            recovered,
                            expected,
                            atol=2e-6,
                            rtol=1e-6,
                        )
                    )

    def test_c8_gray_bits_have_exact_hard_forward_and_input_gradients(self):
        activation = C8ComplexActivation(
            beta=2.0,
            codebook_mode="octants",
            grad_mode="semantic_phase_ste",
        )
        codebook = activation.c8_codebook
        real = codebook[:, 0].reshape(1, 1, 1, 8).clone()
        imag = codebook[:, 1].reshape(1, 1, 1, 8).clone()
        real.requires_grad_()
        imag.requires_grad_()
        bits = activation.phase_code_bits(torch.complex(real, imag))

        expected = (_c8_gray_bits().t() * 2.0 - 1.0).reshape(1, 3, 1, 8)
        self.assertTrue(torch.equal(bits, expected))
        for bit_index in range(3):
            gradient = torch.autograd.grad(
                bits[:, bit_index].sum(),
                (real, imag),
                retain_graph=True,
            )
            self.assertGreater(
                gradient[0].abs().sum().item()
                + gradient[1].abs().sum().item(),
                0.0,
            )

    def test_semantic_bits_are_hard_direct_codes_with_input_gradients(self):
        activation = C8ComplexActivation(
            beta=2.0,
            codebook_mode="octants",
            grad_mode="semantic_ste",
        )
        codebook = activation.c8_codebook
        real = codebook[:, 0].reshape(1, 1, 1, 8).clone()
        imag = codebook[:, 1].reshape(1, 1, 1, 8).clone()
        real.requires_grad_()
        imag.requires_grad_()
        bits = activation.semantic_code_bits(torch.complex(real, imag))

        expected = torch.stack(
            (
                torch.where(codebook[:, 0] >= 0.0, 1.0, -1.0),
                torch.where(codebook[:, 1] >= 0.0, 1.0, -1.0),
                torch.where(
                    codebook[:, 0].abs() >= codebook[:, 1].abs(),
                    1.0,
                    -1.0,
                ),
            ),
            dim=0,
        ).reshape(1, 3, 1, 8)
        self.assertTrue(torch.equal(bits, expected))

        for bit_index in range(3):
            gradient = torch.autograd.grad(
                bits[:, bit_index].sum(),
                (real, imag),
                retain_graph=True,
            )
            self.assertGreater(
                gradient[0].abs().sum().item()
                + gradient[1].abs().sum().item(),
                0.0,
            )

    def test_semantic_soft_table_recovers_c8_product_with_distinct_slices(self):
        layer = ComplexLUTConv2d(
            1,
            1,
            kernel_size=1,
            padding=0,
            phase=4,
            lut_inputs=5,
            lut_init_mode="raw",
            lut5_init_strategy="semantic_c8_product",
            c8_codebook_mode="octants",
            lut_init_tau=1.0,
        )
        state = torch.arange(32)
        activation_address = state >> 2
        sign_r = (((activation_address >> 2) & 1).float() * 2.0) - 1.0
        sign_i = (((activation_address >> 1) & 1).float() * 2.0) - 1.0
        dominance = (activation_address & 1).float()
        high = torch.cos(torch.tensor(torch.pi / 8.0))
        low = torch.sin(torch.tensor(torch.pi / 8.0))
        decoded_r = sign_r * (low + (high - low) * dominance)
        decoded_i = sign_i * (high - (high - low) * dominance)
        wr = (((state >> 1) & 1).float() * 2.0) - 1.0
        wi = ((state & 1).float() * 2.0) - 1.0

        recovered_r = 4.0 * (
            binary_annealing(layer.lut_r, tau=1.0, hard=False)[0] - 0.5
        )
        recovered_i = 4.0 * (
            binary_annealing(layer.lut_i, tau=1.0, hard=False)[0] - 0.5
        )
        self.assertTrue(
            torch.allclose(recovered_r, decoded_r * wr - decoded_i * wi)
        )
        self.assertTrue(
            torch.allclose(recovered_i, decoded_r * wi + decoded_i * wr)
        )

        slice0 = state[((state >> 2) & 1) == 0]
        slice1 = slice0 | 0b00100
        hard_differences = 0
        for table in (layer.lut_r, layer.lut_i):
            hard_differences += int(
                (
                    (table[:, slice0] >= 0)
                    != (table[:, slice1] >= 0)
                ).sum().item()
            )
        self.assertGreater(hard_differences, 0)

    def test_semantic_lut_gradient_reaches_source_through_dominance_only(self):
        activation = C8ComplexActivation(
            beta=2.0,
            codebook_mode="octants",
            grad_mode="semantic_ste",
        )
        layer = ComplexLUTConv2d(
            1, 1, kernel_size=1, padding=0, phase=4, lut_inputs=5,
            lut_init_mode="raw", lut5_init_strategy="semantic_c8_product",
            c8_codebook_mode="octants", lut_init_tau=1.0,
        )
        real = torch.tensor([[[[0.35]]]], requires_grad=True)
        imag = torch.tensor([[[[0.60]]]], requires_grad=True)
        bits = activation.semantic_code_bits(torch.complex(real, imag))
        probabilities = (bits.reshape(3) + 1.0) / 2.0
        probabilities = torch.stack(
            (probabilities[0].detach(), probabilities[1].detach(), probabilities[2])
        )
        state_bits = activation.c8_semantic_state_bits
        state_probabilities = torch.where(
            state_bits.bool(), probabilities, 1.0 - probabilities
        ).prod(dim=1)
        weight_address = 3
        addresses = torch.arange(8) * 4 + weight_address
        local_r = 4.0 * (
            binary_annealing(layer.lut_r[0, addresses], 1.0, False) - 0.5
        )
        local_i = 4.0 * (
            binary_annealing(layer.lut_i[0, addresses], 1.0, False) - 0.5
        )
        loss = (state_probabilities * (local_r + 0.37 * local_i)).sum()
        bit_grad, real_grad, imag_grad = torch.autograd.grad(
            loss, (bits, real, imag)
        )
        self.assertGreater(bit_grad[:, 2].abs().sum().item(), 0.0)
        self.assertGreater(real_grad.abs().sum().item(), 0.0)
        self.assertGreater(imag_grad.abs().sum().item(), 0.0)

    def test_occupancy_tracker_observes_direct_phase_code_bits_path(self):
        class PassthroughLUT(torch.nn.Module):
            def forward(self, phase_source, activation_bits=None):
                self.activation_bits = activation_bits
                return phase_source

        class DirectLUTBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.learned_lut5_phase3p1 = True
                self.act = C8ComplexActivation(
                    codebook_mode="octants",
                    grad_mode="semantic_phase_ste",
                )
                self.conv = PassthroughLUT()

            def forward(self, phase_source):
                activation_bits = self.act.phase_code_bits(phase_source)
                return self.conv(
                    phase_source,
                    activation_bits=activation_bits,
                )

        block = DirectLUTBlock()
        tracker = C8CodeOccupancyTracker(block)
        codebook = block.act.c8_codebook
        source = torch.complex(
            codebook[:, 0],
            codebook[:, 1],
        ).reshape(1, 1, 1, 8)

        tracker.start()
        block(source)
        stats = tracker.finish()
        tracker.close()

        self.assertEqual(stats["counts"], [1] * 8)
        self.assertEqual(stats["active_codes"], 8)
        self.assertEqual(stats["total"], 8)

    def test_phase3p1_mode_uses_c8_activation_and_trainable_lut5(self):
        model = BinaryComplexResNet(
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
            phase3p1_mode="learned_lut5",
            lut_init_tau=1.0,
        )
        block = model.stage2[0]
        self.assertIsInstance(block.act, C8ComplexActivation)
        self.assertIsInstance(block.conv, ComplexLUTConv2d)
        self.assertTrue(block.uses_lut5)
        self.assertTrue(block.learned_lut5_phase3p1)
        self.assertEqual(block.conv.lut5_init_strategy, "c8_product")
        self.assertEqual(block.conv.c8_codebook_mode, "octants")
        self.assertTrue(block.conv.lut_r.requires_grad)
        self.assertTrue(block.conv.weight_r.requires_grad)

    def test_semantic_phase3p1_mode_uses_direct_address_and_product_init(self):
        model = BinaryComplexResNet(
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
            c8_grad_mode="semantic_ste",
            phase3p1_mode="semantic_lut5",
            lut_init_tau=1.0,
        )
        block = model.stage2[0]
        self.assertIsInstance(block.act, C8ComplexActivation)
        self.assertIsInstance(block.conv, ComplexLUTConv2d)
        self.assertTrue(block.learned_lut5_phase3p1)
        self.assertTrue(block.semantic_lut5_phase3p1)
        self.assertEqual(
            block.conv.lut5_init_strategy,
            "semantic_c8_product",
        )
        self.assertTrue(block.conv.lut_r.requires_grad)
        self.assertTrue(block.conv.weight_r.requires_grad)

    def test_explicit_activation_bits_validate_shape(self):
        layer = self.make_layer()
        source = torch.complex(
            torch.ones(2, 1, 3, 3),
            -torch.ones(2, 1, 3, 3),
        )
        good = torch.ones(2, 3, 3, 3)
        self.assertIs(layer._make_x_cat(source, activation_bits=good), good)
        with self.assertRaisesRegex(ValueError, "activation_bits must have shape"):
            layer._make_x_cat(
                source,
                activation_bits=torch.ones(2, 2, 3, 3),
            )

    def test_phase2p1_checkpoint_maps_spatial_weights_to_lut_model(self):
        source = BinaryComplexResNet(
            in_channels=3,
            num_blocks=1,
            start_filters=2,
            num_classes=10,
            is_sar_input=False,
            is_binary=True,
            phase=2.1,
            lut_inputs=5,
            c8_codebook="octants",
            c8_grad_mode="semantic_phase_ste",
        )
        target = BinaryComplexResNet(
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
            phase3p1_mode="learned_lut5",
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "phase2p1.pt"
            torch.save(
                {
                    "phase": 2.1,
                    "model": source.state_dict(),
                    "args": {
                        "phase": 2.1,
                        "lut_inputs": 5,
                        "phase2p1_mode": "c8",
                        "c8_codebook": "octants",
                        "c8_grad_mode": "semantic_phase_ste",
                    },
                },
                checkpoint,
            )
            load_phase_checkpoint(
                target,
                checkpoint,
                expected_phase=2.1,
                expected_lut_inputs=5,
                expected_c8_codebook="octants",
                expected_c8_grad_mode="semantic_phase_ste",
                expected_phase2p1_mode="c8",
            )

        for source_block, target_block in zip(
            source.stage2,
            target.stage2,
        ):
            self.assertTrue(
                torch.equal(
                    source_block.conv.conv_r.weight,
                    target_block.conv.weight_r,
                )
            )
            self.assertTrue(
                torch.equal(
                    source_block.conv.conv_i.weight,
                    target_block.conv.weight_i,
                )
            )

    def test_phase3p1_annealing_holds_tau_then_becomes_hard(self):
        layer = self.make_layer(tau=1.0)
        args = SimpleNamespace(
            phase2p1_mode="c8",
            phase3p1_mode="learned_lut5",
            lut_hard_ste=False,
            lut_anneal_epochs=14,
            lut_soft_warmup_epochs=4,
            lut_hard_epoch_fraction=0.9,
            lut_tau_min=1.0,
            lut_tau_max=10.0,
            lut_hard_transition_epochs=2,
        )
        for epoch in (0, 3, 4):
            self.assertFalse(
                update_lut_annealing(
                    layer,
                    epoch=epoch,
                    num_epochs=20,
                    phase=3.1,
                    train_logger=None,
                    is_main=False,
                    args=args,
                )
            )
            self.assertAlmostEqual(float(layer.tau.item()), 1.0)

        update_lut_annealing(
            layer,
            epoch=9,
            num_epochs=20,
            phase=3.1,
            train_logger=None,
            is_main=False,
            args=args,
        )
        self.assertGreater(float(layer.tau.item()), 1.0)
        self.assertFalse(
            update_lut_annealing(
                layer,
                epoch=14,
                num_epochs=20,
                phase=3.1,
                train_logger=None,
                is_main=False,
                args=args,
            )
        )
        self.assertAlmostEqual(float(layer.hard.item()), 0.5)
        self.assertTrue(
            update_lut_annealing(
                layer,
                epoch=15,
                num_epochs=20,
                phase=3.1,
                train_logger=None,
                is_main=False,
                args=args,
            )
        )
        self.assertEqual(float(layer.hard.item()), 1.0)

    def test_parser_keeps_fixed_default_and_exposes_learned_mode(self):
        defaults = parse_args([])
        self.assertEqual(defaults.phase3p1_mode, "fixed")
        self.assertEqual(defaults.lut_soft_warmup_epochs, 0)
        args = parse_args(
            [
                "--phase",
                "3.1",
                "--phase3p1-mode",
                "learned_lut5",
                "--lut-inputs",
                "5",
                "--lut-soft-warmup-epochs",
                "10",
            ]
        )
        self.assertEqual(args.phase3p1_mode, "learned_lut5")
        self.assertEqual(args.lut_soft_warmup_epochs, 10)

        semantic_args = parse_args(
            [
                "--phase",
                "3.1",
                "--phase3p1-mode",
                "semantic_lut5",
                "--lut-inputs",
                "5",
            ]
        )
        self.assertEqual(semantic_args.phase3p1_mode, "semantic_lut5")


if __name__ == "__main__":
    unittest.main()
