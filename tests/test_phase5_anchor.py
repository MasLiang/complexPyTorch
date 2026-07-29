import os
import tempfile
import unittest

import torch

from complexPyTorch.complexLayers import ComplexLUTConv2d
from training import (
    duplicate_lut4_table_to_lut5,
    load_phase_checkpoint,
    parse_args,
    phase_checkpoint_filename,
    verify_lut5_slices_duplicated,
)


class Phase5AnchorTest(unittest.TestCase):
    def make_layer(self, phase, lut_inputs, lut_sets=1):
        return ComplexLUTConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=1,
            padding=0,
            phase=phase,
            lut_inputs=lut_inputs,
            lut_sets=lut_sets,
            lut_logit_init=2.0,
        )

    def duplicated_values(self, rows=1):
        source = torch.linspace(-2.0, 2.0, 16).unsqueeze(0)
        return duplicate_lut4_table_to_lut5(source, (rows, 32))

    def test_address_mapping_duplicates_both_phase_slices(self):
        source = torch.arange(16, dtype=torch.float32).unsqueeze(0)
        duplicated = duplicate_lut4_table_to_lut5(source)

        for state5 in range(32):
            state4 = ((state5 & 0b11000) >> 1) | (state5 & 0b00011)
            self.assertEqual(duplicated[0, state5].item(), source[0, state4].item())

        state = torch.arange(32)
        slice0 = state[((state >> 2) & 1) == 0]
        torch.testing.assert_close(
            duplicated[:, slice0],
            duplicated[:, slice0 | 0b00100],
            rtol=0.0,
            atol=0.0,
        )

    def test_phase4_checkpoint_is_expanded_to_exact_lut5(self):
        source = self.make_layer(phase=4, lut_inputs=4)
        target = self.make_layer(phase=5, lut_inputs=5)
        source_values_r = torch.linspace(-2.0, 2.0, 16).unsqueeze(0)
        source_values_i = torch.linspace(2.0, -2.0, 16).unsqueeze(0)
        source.lut_r.data.copy_(source_values_r)
        source.lut_i.data.copy_(source_values_i)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "phase4.pt")
            torch.save(
                {
                    "phase": 4,
                    "args": {"phase": 4, "lut_inputs": 4},
                    "model": source.state_dict(),
                },
                path,
            )
            load_phase_checkpoint(
                target,
                path,
                expected_phase=4,
                expected_lut_inputs=4,
                duplicate_lut4_to_lut5=True,
            )

        torch.testing.assert_close(
            target.lut_r,
            duplicate_lut4_table_to_lut5(source_values_r),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            target.lut_i,
            duplicate_lut4_table_to_lut5(source_values_i),
            rtol=0.0,
            atol=0.0,
        )
        stats = verify_lut5_slices_duplicated(target)
        self.assertEqual(stats["value_mismatches"], 0)
        self.assertEqual(stats["sign_mismatches"], 0)

    def test_anchor_slice_cannot_be_committed(self):
        layer = self.make_layer(phase=5, lut_inputs=5)
        values = self.duplicated_values()
        layer.lut_r.data.copy_(values)
        layer.lut_i.data.copy_(values)
        layer.begin_lut_flip_search(
            init_flip_prob=0.05,
            score_temperature=0.25,
            anchor_slice=0,
            ties_only=False,
        )

        state = torch.arange(32)
        phase_bit = (state >> 2) & 1
        self.assertTrue(torch.equal(layer.lut_search_eligible_r[0], phase_bit == 1))
        self.assertEqual(int(layer.lut_search_eligible_r.sum()), 16)
        self.assertEqual(int(layer.lut_search_eligible_i.sum()), 16)

        with self.assertRaisesRegex(ValueError, "frozen LUT5 anchor"):
            layer.microcommit_lut_flip_entries([0], [])

        original_anchor = layer.lut_search_base_r[:, phase_bit == 0].clone()
        layer.microcommit_lut_flip_entries([4], [], cooldown_commits=2)
        torch.testing.assert_close(
            layer.lut_search_base_r[:, phase_bit == 0],
            original_anchor,
            rtol=0.0,
            atol=0.0,
        )
        stats = layer.finalize_lut_flip_search(logit_abs_value=2.0)
        self.assertEqual(stats["net_real_flips"], 1)
        self.assertEqual(stats["frozen_real_diff"], 0)
        self.assertEqual(stats["frozen_imag_diff"], 0)

    def test_ties_only_halves_the_trainable_slice(self):
        layer = self.make_layer(phase=5, lut_inputs=5, lut_sets=2)
        layer.begin_lut_flip_search(
            init_flip_prob=0.05,
            score_temperature=0.25,
            anchor_slice=1,
            ties_only=True,
        )
        self.assertEqual(int(layer.lut_search_eligible_r.sum()), 16)
        self.assertEqual(int(layer.lut_search_eligible_i.sum()), 16)

        state = torch.arange(32).repeat(2, 1)
        phase_bit = (state >> 2) & 1
        self.assertTrue(
            bool((phase_bit[layer.lut_search_eligible_r] == 0).all())
        )
        self.assertTrue(
            bool((phase_bit[layer.lut_search_eligible_i] == 0).all())
        )

    def test_phase5_cli_and_checkpoint_name(self):
        args = parse_args(["--phase", "5", "--lut-inputs", "5"])
        self.assertEqual(args.phase, 5)
        self.assertEqual(args.phase5_anchor_slice, 0)
        self.assertEqual(phase_checkpoint_filename(5), "Bestmodel_phase5.pt")


if __name__ == "__main__":
    unittest.main()
