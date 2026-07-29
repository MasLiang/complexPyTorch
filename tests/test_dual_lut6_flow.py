import itertools
import math
import unittest
from types import SimpleNamespace

import torch

from complexPyTorch.complexBinaryResNet import BinaryComplexResNet
from complexPyTorch.dualLut6Flow import (
    DUAL_LUT6_PHASE,
    DualLUT6ComplexConv2d,
    DualLUT6ComplexResNet,
    TwoBitComplexActivation,
    dual_lut6_hardware_spec,
    dual_lut6_init,
    dual_lut6_local_output_bits,
    evaluate_dual_lut6_init,
    set_two_bit_rho,
)
from training_dual_lut6 import rho_for_epoch


class TwoBitComplexActivationTest(unittest.TestCase):
    def test_rho_zero_matches_phase2_binary_activation(self):
        activation = TwoBitComplexActivation(
            channels=2,
            threshold_init=0.675,
            high_ratio=3.0,
        )
        value = torch.complex(
            torch.tensor([[[[-2.0, -0.2]], [[0.3, 1.7]]]]),
            torch.tensor([[[[0.4, -1.4]], [[-0.6, 2.2]]]]),
        )
        output = activation(value)
        expected = torch.complex(value.real.sign(), value.imag.sign())
        torch.testing.assert_close(output, expected)

    def test_final_codes_are_ordered_and_decode_exactly(self):
        activation = TwoBitComplexActivation(
            channels=1,
            threshold_init=0.675,
            high_ratio=3.0,
        )
        activation.set_rho(1.0)
        value = torch.complex(
            torch.tensor([[[[-2.0, -0.2, 0.2, 2.0]]]]),
            torch.tensor([[[[2.0, 0.2, -0.2, -2.0]]]]),
        )
        real_bits, imag_bits = activation.hard_code_bits(value)
        self.assertEqual(
            real_bits.reshape(-1, 2).tolist(),
            [[0, 0], [0, 1], [1, 0], [1, 1]],
        )
        self.assertEqual(
            imag_bits.reshape(-1, 2).tolist(),
            [[1, 1], [1, 0], [0, 1], [0, 0]],
        )
        output = activation(value)
        torch.testing.assert_close(
            activation.decode_code_bits(real_bits),
            output.real,
        )
        torch.testing.assert_close(
            activation.decode_code_bits(imag_bits),
            output.imag,
        )
        low, high = activation.levels()
        self.assertAlmostEqual(
            float((low.square() + high.square()) / 2.0),
            1.0,
            places=6,
        )

    def test_input_and_threshold_receive_gradients(self):
        activation = TwoBitComplexActivation(
            channels=1,
            threshold_init=0.675,
            high_ratio=3.0,
            beta=4.0,
        )
        activation.set_rho(1.0)
        real = torch.tensor([[[[0.55, 0.70, 0.90]]]], requires_grad=True)
        imag = torch.tensor([[[[0.60, 0.75, 1.10]]]], requires_grad=True)
        output = activation(torch.complex(real, imag))
        (output.real.sum() + 0.3 * output.imag.sum()).backward()
        self.assertGreater(real.grad.abs().sum().item(), 0.0)
        self.assertGreater(imag.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            activation.threshold_unconstrained.grad.abs().sum().item(),
            0.0,
        )


class DualLUT6ComplexProductTest(unittest.TestCase):
    def test_standard_and_hardware_convolution_match_forward_and_backward(self):
        torch.manual_seed(17)
        layer = DualLUT6ComplexConv2d(
            in_channels=2,
            out_channels=3,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        choices = torch.tensor([-3.0, -1.0, 1.0, 3.0]) / math.sqrt(5.0)
        real_base = choices[torch.randint(0, 4, (2, 2, 5, 5))]
        imag_base = choices[torch.randint(0, 4, (2, 2, 5, 5))]

        real = real_base.clone().requires_grad_(True)
        imag = imag_base.clone().requires_grad_(True)
        layer.set_implementation("standard")
        standard = layer(torch.complex(real, imag))
        standard_loss = standard.real.square().mean() + standard.imag.square().mean()
        standard_loss.backward()
        standard_grads = {
            "real": real.grad.clone(),
            "imag": imag.grad.clone(),
            "wr": layer.conv_r.weight.grad.clone(),
            "wi": layer.conv_i.weight.grad.clone(),
        }

        layer.zero_grad(set_to_none=True)
        real = real_base.clone().requires_grad_(True)
        imag = imag_base.clone().requires_grad_(True)
        layer.set_implementation("hardware")
        hardware = layer(torch.complex(real, imag))
        hardware_loss = hardware.real.square().mean() + hardware.imag.square().mean()
        hardware_loss.backward()
        hardware_grads = {
            "real": real.grad,
            "imag": imag.grad,
            "wr": layer.conv_r.weight.grad,
            "wi": layer.conv_i.weight.grad,
        }

        torch.testing.assert_close(hardware, standard, rtol=2e-5, atol=2e-5)
        for name in standard_grads:
            torch.testing.assert_close(
                hardware_grads[name],
                standard_grads[name],
                rtol=3e-5,
                atol=3e-5,
            )

    def test_all_codes_and_weight_signs_match_complex_multiplication(self):
        activation = TwoBitComplexActivation(
            channels=1,
            threshold_init=0.675,
            high_ratio=3.0,
        )
        activation.set_rho(1.0)

        for xr_code, xi_code, wr_bit, wi_bit in itertools.product(
            range(4), range(4), (0, 1), (0, 1)
        ):
            xr_bits = torch.tensor(
                [(xr_code >> 1) & 1, xr_code & 1]
            )
            xi_bits = torch.tensor(
                [(xi_code >> 1) & 1, xi_code & 1]
            )
            ur_bits = []
            ui_bits = []
            for plane in range(2):
                ur_bit, ui_bit = dual_lut6_local_output_bits(
                    xr_bits[plane],
                    xi_bits[plane],
                    wr_bit,
                    wi_bit,
                )
                ur_bits.append(ur_bit)
                ui_bits.append(ui_bit)

            xr = activation.decode_code_bits(xr_bits)
            xi = activation.decode_code_bits(xi_bits)
            ur = activation.decode_code_bits(torch.tensor(ur_bits))
            ui = activation.decode_code_bits(torch.tensor(ui_bits))
            routed_product = torch.complex(ur - ui, ur + ui)
            wr = 1.0 if wr_bit else -1.0
            wi = 1.0 if wi_bit else -1.0
            expected = torch.complex(xr, xi) * complex(wr, wi)
            torch.testing.assert_close(routed_product, expected)

    def test_lut6_2_init_matches_truth_table(self):
        init_value = dual_lut6_init()
        for inputs in itertools.product((0, 1), repeat=4):
            self.assertEqual(
                evaluate_dual_lut6_init(init_value, *inputs),
                dual_lut6_local_output_bits(*inputs),
            )
        spec = dual_lut6_hardware_spec(high_ratio=3.0)
        self.assertEqual(spec["lut6_2_units_per_complex_product"], 2)
        self.assertEqual(spec["local_outputs"], 4)
        self.assertFalse(spec["local_sign_truncation"])
        self.assertTrue(spec["same_init_for_high_and_low_planes"])


class DualLUT6ModelCompatibilityTest(unittest.TestCase):
    def test_phase2_state_is_shape_compatible(self):
        kwargs = {
            "num_blocks": 1,
            "start_filters": 2,
            "num_classes": 10,
            "is_sar_input": False,
            "spectral_pool_scheme": "none",
        }
        phase2 = BinaryComplexResNet(phase=2, **kwargs)
        phase2p7 = DualLUT6ComplexResNet(**kwargs)
        source = phase2.state_dict()
        target = phase2p7.state_dict()
        self.assertTrue(set(source).issubset(target))
        for key, value in source.items():
            self.assertEqual(value.shape, target[key].shape, key)
        missing, unexpected = phase2p7.load_state_dict(source, strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(missing)
        self.assertTrue(
            all(
                key.endswith(
                    (
                        "threshold_unconstrained",
                        "rho",
                        "beta",
                        "high_ratio",
                    )
                )
                for key in missing
            )
        )
        self.assertEqual(phase2p7.phase, DUAL_LUT6_PHASE)
        set_two_bit_rho(phase2p7, 1.0)

    def test_default_rho_schedule_reaches_one_at_epoch_90(self):
        args = SimpleNamespace(
            rho_warmup_epochs=10,
            rho_transition_epochs=80,
            rho_schedule="cosine",
        )
        self.assertEqual(rho_for_epoch(0, args), 0.0)
        self.assertEqual(rho_for_epoch(9, args), 0.0)
        self.assertGreater(rho_for_epoch(10, args), 0.0)
        self.assertLess(rho_for_epoch(88, args), 1.0)
        self.assertEqual(rho_for_epoch(89, args), 1.0)


if __name__ == "__main__":
    unittest.main()
