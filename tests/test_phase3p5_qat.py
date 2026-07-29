import unittest

import torch

from complexPyTorch.complexFunctions import complex_binary_activation
from complexPyTorch.complexLayers import (
    BinaryComplexConv2d,
    LUT5AwareComplexQATConv2d,
    LUTAwareComplexBinaryConv2d,
)
from training import (
    phase3p5_qat_state_for_epoch,
    phase_checkpoint_filename,
)


class Phase3p5QATTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.source = torch.complex(
            torch.randn(2, 2, 7, 7),
            torch.randn(2, 2, 7, 7),
        )
        self.binary_input = complex_binary_activation(self.source)
        self.base = BinaryComplexConv2d(2, 3, 3, padding=1, bias=False)
        self.qat = LUT5AwareComplexQATConv2d(
            2, 3, 3, padding=1, bias=False
        )
        self.qat.conv_r.weight.data.copy_(self.base.conv_r.weight)
        self.qat.conv_i.weight.data.copy_(self.base.conv_i.weight)

    def test_phase2_endpoint_is_exact(self):
        self.qat.set_qat_state(strength=0.0, beta=1.0)
        expected = self.base(self.binary_input)
        actual = self.qat(self.binary_input, phase_source=self.source)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_hard_lut5_endpoint_matches_analytic_layer(self):
        expected_layer = LUTAwareComplexBinaryConv2d(
            2, 3, 3, padding=1, bias=False, lut_inputs=5
        )
        expected_layer.conv_r.weight.data.copy_(self.base.conv_r.weight)
        expected_layer.conv_i.weight.data.copy_(self.base.conv_i.weight)
        expected_layer.set_phase_mix(1.0)

        self.qat.set_qat_state(strength=1.0, beta=8.0)
        expected = expected_layer(
            self.binary_input,
            phase_source=self.source,
        )
        actual = self.qat(
            self.binary_input,
            phase_source=self.source,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_comparator_backward_is_nonzero(self):
        source = self.source.detach().requires_grad_()
        binary_input = complex_binary_activation(source)
        self.qat.set_qat_state(strength=0.5, beta=2.0)
        output = self.qat(binary_input, phase_source=source)
        (output.real.square() + output.imag.square()).mean().backward()
        self.assertIsNotNone(source.grad)
        self.assertTrue(torch.isfinite(source.grad).all())
        self.assertGreater(source.grad.abs().sum().item(), 0.0)

    def test_schedule_and_checkpoint_tag(self):
        start = phase3p5_qat_state_for_epoch(0, 5, 100, 1.0, 8.0)
        end = phase3p5_qat_state_for_epoch(104, 5, 100, 1.0, 8.0)
        self.assertEqual(start["strength"], 0.0)
        self.assertFalse(start["hard_lut5_ready"])
        self.assertAlmostEqual(end["strength"], 1.0)
        self.assertTrue(end["hard_lut5_ready"])
        self.assertEqual(
            phase_checkpoint_filename(3.5),
            "Bestmodel_phase3p5.pt",
        )


if __name__ == "__main__":
    unittest.main()
