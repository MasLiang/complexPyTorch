import itertools
import math
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from complexPyTorch.complexLayers import (
    C8LUTAwareComplexQATConv2d,
    LUTFloatingConvFunction,
)
from training import (
    parse_args,
    phase3p6_qat_state_for_epoch,
    phase_checkpoint_filename,
)


class Phase3p6C8QATTest(unittest.TestCase):
    def make_layer(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        padding=0,
    ):
        return C8LUTAwareComplexQATConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def test_layer_has_phase3_weights_and_no_lut_parameters(self):
        layer = self.make_layer()
        self.assertEqual(
            set(dict(layer.named_parameters())),
            {"conv_r.weight", "conv_i.weight"},
        )
        self.assertFalse(hasattr(layer, "lut_r"))
        self.assertFalse(hasattr(layer, "lut_i"))
        self.assertFalse(hasattr(layer, "hard"))
        self.assertFalse(hasattr(layer, "tau"))

    def test_all_32_local_states_match_decode_multiply_compare(self):
        layer = self.make_layer()
        layer.set_qat_state(progress=1.0, beta=12.0)
        source = torch.complex(
            layer.c8_codebook[:, 0],
            layer.c8_codebook[:, 1],
        ).reshape(1, 1, 1, 8)
        binary_input = torch.zeros_like(source)

        for weight_r, weight_i in itertools.product((-1.0, 1.0), repeat=2):
            layer.conv_r.weight.data.fill_(weight_r)
            layer.conv_i.weight.data.fill_(weight_i)

            hard_r, hard_i = layer.hard_local_sums(source)
            expected_r = (
                layer.c8_codebook[:, 0] * weight_r
                - layer.c8_codebook[:, 1] * weight_i
                >= 0
            ).to(torch.float32)
            expected_i = (
                layer.c8_codebook[:, 0] * weight_i
                + layer.c8_codebook[:, 1] * weight_r
                >= 0
            ).to(torch.float32)
            self.assertTrue(torch.equal(hard_r.flatten(), expected_r))
            self.assertTrue(torch.equal(hard_i.flatten(), expected_i))

            output = layer(binary_input, phase_source=source)
            alpha = math.sqrt(2.0)
            recovered_r = output.real / (4.0 * alpha) + 0.5
            recovered_i = output.imag / (4.0 * alpha) + 0.5
            self.assertTrue(
                torch.allclose(recovered_r.flatten(), expected_r, atol=1e-6)
            )
            self.assertTrue(
                torch.allclose(recovered_i.flatten(), expected_i, atol=1e-6)
            )

    def test_kernel_accumulation_matches_explicit_local_comparators(self):
        torch.manual_seed(17)
        layer = self.make_layer(
            in_channels=2,
            out_channels=3,
            kernel_size=3,
            padding=1,
        )
        layer.set_qat_state(progress=1.0, beta=12.0)
        source = torch.complex(
            torch.randn(1, 2, 4, 4),
            torch.randn(1, 2, 4, 4),
        )

        actual_r, actual_i = layer.hard_local_sums(source)
        phase_index = F.pad(
            layer.hard_phase_indices(source),
            (1, 1, 1, 1),
            "constant",
            0,
        )
        decoded = layer.c8_codebook[phase_index]
        patch_r = F.unfold(decoded[..., 0], kernel_size=3)
        patch_i = F.unfold(decoded[..., 1], kernel_size=3)
        weight_r = torch.where(
            layer.conv_r.weight >= 0,
            1.0,
            -1.0,
        ).flatten(1)
        weight_i = torch.where(
            layer.conv_i.weight >= 0,
            1.0,
            -1.0,
        ).flatten(1)

        raw_r = (
            patch_r.unsqueeze(1) * weight_r[None, :, :, None]
            - patch_i.unsqueeze(1) * weight_i[None, :, :, None]
        )
        raw_i = (
            patch_r.unsqueeze(1) * weight_i[None, :, :, None]
            + patch_i.unsqueeze(1) * weight_r[None, :, :, None]
        )
        expected_r = (raw_r >= 0).to(torch.float32).sum(dim=2)
        expected_i = (raw_i >= 0).to(torch.float32).sum(dim=2)
        expected_r = expected_r.reshape_as(actual_r)
        expected_i = expected_i.reshape_as(actual_i)

        self.assertTrue(torch.equal(actual_r, expected_r))
        self.assertTrue(torch.equal(actual_i, expected_i))

    def test_progress_enables_only_the_four_axis_codes(self):
        layer = self.make_layer()
        source = torch.complex(
            torch.tensor([[[[-1.0, 0.0, 1.0, 0.0, 1.0]]]]),
            torch.tensor([[[[-1.0, 1.0, 1.0, -1.0, 0.0]]]]),
        )

        layer.set_qat_state(progress=0.0, beta=2.0)
        c4_indices = layer.hard_phase_indices(source)
        self.assertTrue(bool((c4_indices.remainder(2) == 0).all()))

        layer.set_qat_state(progress=1.0, beta=12.0)
        c8_indices = layer.hard_phase_indices(source).flatten().tolist()
        self.assertEqual(c8_indices, [0, 3, 4, 7, 5])
        bits = layer.phase_code_bits(source)
        self.assertEqual(tuple(bits.shape), (1, 3, 1, 5))
        self.assertEqual(set(bits.unique().tolist()), {-1.0, 1.0})

    def test_analytic_forward_avoids_lut_backend_and_backpropagates(self):
        torch.manual_seed(11)
        layer = self.make_layer(
            in_channels=2,
            out_channels=3,
            kernel_size=3,
            padding=1,
        )
        layer.set_qat_state(progress=0.5, beta=4.0)
        real = torch.randn(2, 2, 5, 5, requires_grad=True)
        imag = torch.randn(2, 2, 5, 5, requires_grad=True)
        source = torch.complex(real, imag)
        binary_input = torch.complex(
            torch.where(real >= 0, 1.0, -1.0),
            torch.where(imag >= 0, 1.0, -1.0),
        )

        with mock.patch.object(
            LUTFloatingConvFunction,
            "apply",
            side_effect=AssertionError("LUT backend must not be called"),
        ):
            output = layer(binary_input, phase_source=source)
            output.abs().mean().backward()

        self.assertTrue(torch.isfinite(output).all())
        self.assertGreater(real.grad.abs().sum().item(), 0.0)
        self.assertGreater(imag.grad.abs().sum().item(), 0.0)
        self.assertGreater(layer.conv_r.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(layer.conv_i.weight.grad.abs().sum().item(), 0.0)

    def test_schedule_cli_and_checkpoint_tag(self):
        start = phase3p6_qat_state_for_epoch(
            0,
            10,
            100,
            2.0,
            12.0,
            "cosine",
        )
        end = phase3p6_qat_state_for_epoch(
            109,
            10,
            100,
            2.0,
            12.0,
            "cosine",
        )
        linear_mid = phase3p6_qat_state_for_epoch(
            59,
            10,
            100,
            2.0,
            12.0,
            "linear",
        )
        self.assertEqual(start["code_progress"], 0.0)
        self.assertFalse(start["hard_c8_ready"])
        self.assertAlmostEqual(end["code_progress"], 1.0)
        self.assertTrue(end["hard_c8_ready"])
        self.assertAlmostEqual(linear_mid["code_progress"], 49.0 / 99.0)

        args = parse_args(["--phase", "3.6", "--lut-inputs", "5"])
        self.assertEqual(args.phase, 3.6)
        self.assertEqual(args.phase3p6_schedule, "cosine")
        self.assertEqual(
            phase_checkpoint_filename(3.6),
            "Bestmodel_phase3p6.pt",
        )


if __name__ == "__main__":
    unittest.main()
