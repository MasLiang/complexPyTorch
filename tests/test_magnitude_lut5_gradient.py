import unittest

import torch

from complexPyTorch.complexLayers import (
    ComplexLUTConv2d,
    binary_annealing,
)
from complexPyTorch.lut_backend import (
    LUTFloatingConvFunction,
    LUTFloatingShadowInputGradFunction,
)


class MagnitudeLUT5GradientTest(unittest.TestCase):
    def make_layer(self, in_channels=2):
        return ComplexLUTConv2d(
            in_channels=in_channels,
            out_channels=2,
            kernel_size=1,
            stride=1,
            padding=0,
            phase=4,
            lut_inputs=5,
            lut_logit_init=2.0,
            lut_init_mode="binary",
            lut5_init_strategy="duplicate_lut4",
            lut_extra_bit="magnitude_ste",
            magnitude_threshold_init=1.0,
            magnitude_bit_beta=2.0,
            magnitude_shadow_epsilon=0.05,
        )

    def test_physical_slices_are_duplicated_but_shadow_is_asymmetric(self):
        layer = self.make_layer()
        state = torch.arange(32)
        slice0 = state[((state >> 2) & 1) == 0]
        slice1 = slice0 | 0b00100

        for logits in (layer.lut_r, layer.lut_i):
            self.assertTrue(
                torch.equal(logits[:, slice0], logits[:, slice1])
            )
            operation = binary_annealing(logits, tau=1.0, hard=True)
            shadow = layer._magnitude_shadow_operation(operation)
            self.assertTrue(
                torch.equal(operation >= 0.5, shadow >= 0.5)
            )
            self.assertTrue(
                torch.allclose(
                    (shadow[:, slice1] - shadow[:, slice0]).abs(),
                    torch.full_like(shadow[:, slice0], 0.05),
                )
            )

    def test_magnitude_bit_backward_reaches_threshold_and_source(self):
        layer = self.make_layer()
        real = torch.full((1, 2, 2, 2), 0.8, requires_grad=True)
        imag = torch.full((1, 2, 2, 2), 0.6, requires_grad=True)
        phase_source = torch.complex(real, imag)
        binary_input = torch.complex(
            torch.ones_like(real),
            -torch.ones_like(imag),
        )

        x_cat = layer._make_x_cat(
            binary_input,
            phase_source=phase_source,
        )
        magnitude_sign = x_cat[:, 4:]
        self.assertTrue(
            torch.all((magnitude_sign == -1.0) | (magnitude_sign == 1.0))
        )

        magnitude_sign.sum().backward()
        self.assertIsNotNone(layer.mag_threshold_raw.grad)
        self.assertGreater(
            float(layer.mag_threshold_raw.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(float(real.grad.abs().sum()), 0.0)
        self.assertGreater(float(imag.grad.abs().sum()), 0.0)

    def test_threshold_parameter_uses_requested_physical_initial_value(self):
        layer = self.make_layer()
        threshold = layer.physical_mag_threshold()
        self.assertTrue(
            torch.allclose(threshold, torch.ones_like(threshold), atol=1e-6)
        )
        self.assertTrue(layer.mag_threshold_raw.requires_grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_shadow_backend_keeps_forward_and_unlocks_fifth_bit_gradient(self):
        device = torch.device("cuda")
        x_base = torch.tensor(
            [[[[0.25]], [[0.75]], [[0.0]]]],
            device=device,
            requires_grad=True,
        )
        x_shadow = x_base.detach().clone().requires_grad_(True)

        states = torch.arange(8, device=device)
        physical = (((states >> 2) & 1).float()).view(1, 8, 1)
        shadow = physical.clone()
        d_is_one = (states & 1) == 1
        direction = torch.where(
            physical[0, :, 0] >= 0.5,
            -torch.ones(8, device=device),
            torch.ones(8, device=device),
        )
        shadow[0, d_is_one, 0] += 0.05 * direction[d_is_one]

        w_base = physical.clone().requires_grad_(True)
        w_shadow = physical.clone().requires_grad_(True)
        offsets = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

        out_base = LUTFloatingConvFunction.apply(
            x_base, w_base, offsets, 1, 3, 1, 1, 0
        )
        out_shadow = LUTFloatingShadowInputGradFunction.apply(
            x_shadow, w_shadow, shadow, offsets, 1, 3, 1, 1, 0
        )
        self.assertTrue(torch.equal(out_base, out_shadow))

        out_base.sum().backward()
        out_shadow.sum().backward()

        self.assertEqual(float(x_base.grad[0, 2, 0, 0]), 0.0)
        self.assertGreater(float(x_shadow.grad[0, 2, 0, 0].abs()), 0.0)
        self.assertTrue(torch.allclose(w_base.grad, w_shadow.grad))


if __name__ == "__main__":
    unittest.main()
