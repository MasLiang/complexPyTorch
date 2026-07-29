import tempfile
import unittest
from pathlib import Path

from scripts.analyze_training_run import parse_training_log


def epoch_line(epoch, accuracy):
    return (
        "Epoch {epoch} train_loss: 1.0, train_acc: {acc:.4f}, "
        "val_loss: 1.0, val_acc: {acc:.4f}, "
        "test_loss: 1.0, test_acc: {acc:.4f}\n"
    ).format(epoch=epoch, acc=accuracy)


class AnalyzeTrainingRunTest(unittest.TestCase):
    def test_restart_continues_most_recent_matching_segment(self):
        lines = [
            "[Phase3.1 Learned LUT5 Annealing Scheduler] "
            "Epoch 0: tau=1.0000, hard_ratio=0.0000, hard_mode=False\n",
            epoch_line(1, 0.10),
            epoch_line(2, 0.20),
            epoch_line(1, 0.30),
            epoch_line(2, 0.40),
            "[Phase3.1 Neural LUT5 Annealing Scheduler] "
            "Epoch 2: tau=1.5000, hard_ratio=0.0000, hard_mode=False\n",
            "[Phase3.1 Neural LUT5 Hard Projection] Epoch 3: "
            "val_loss=0.500000, val_acc=0.8100, "
            "test_loss=0.600000, test_acc=0.8000. "
            "Soft training state restored.\n",
            epoch_line(3, 0.50),
            "[Phase3.1 Neural LUT5] Epoch 3: "
            "init_sign_diff=1 / 1152 (0.086806%), "
            "c8_gray_bit_0_slice_hard_diff=1 / 576 (0.173611%), "
            "c8_gray_bit_0_slice_soft_abs_diff_mean=0.579100, "
            "max=0.880519, hard_checkpoint_eligible=False.\n",
            "[Phase3.1 Neural LUT5 Bit Sensitivity] Epoch 3: "
            "c8_gray_bit_2=288/576 (50.000000%), "
            "c8_gray_bit_1=288/576 (50.000000%), "
            "c8_gray_bit_0=1/576 (0.173611%).\n",
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.txt"
            path.write_text("".join(lines), encoding="utf-8")
            parsed = parse_training_log(path)

        self.assertEqual(len(parsed["segments"]), 2)
        first, second = parsed["segments"]
        self.assertEqual(
            [record["epoch"] for record in first["epochs"]],
            [1, 2],
        )
        self.assertEqual(
            [record["epoch"] for record in second["epochs"]],
            [1, 2, 3],
        )
        self.assertEqual(second["lut_schedule"][3]["tau"], 1.5)
        diagnostic = second["lut_diagnostics"][3]
        self.assertEqual(diagnostic["init_sign_diff"], 1)
        self.assertEqual(diagnostic["slice_label"], "c8_gray_bit_0")
        self.assertAlmostEqual(
            diagnostic["slice_hard_diff_ratio"],
            0.00173611,
        )
        projection = second["hard_projections"][3]
        self.assertEqual(projection["val_acc"], 0.81)
        self.assertEqual(projection["test_acc"], 0.80)
        sensitivity = second["lut_bit_sensitivity"][3]["bits"]
        self.assertEqual(sensitivity["c8_gray_bit_2"]["hard_diff"], 288)
        self.assertAlmostEqual(
            sensitivity["c8_gray_bit_0"]["hard_diff_ratio"],
            0.00173611,
        )

    def test_pure_mlp_diagnostics_and_hard_projection_are_parsed(self):
        lines = [
            epoch_line(1, 0.82),
            "[Phase3.1 Pure MLP5] Epoch 1: "
            "init_sign_diff=2 / 1152 (0.173611%), "
            "c8_gray_bit_0_slice_hard_diff=2 / 576 (0.347222%), "
            "c8_gray_bit_0_slice_soft_abs_diff_mean=0.600000, "
            "max=1.100000, hard_checkpoint_eligible=True.\n",
            "[Phase3.1 Pure MLP5 Bit Sensitivity] Epoch 1: "
            "c8_gray_bit_2=288/576 (50.000000%), "
            "c8_gray_bit_1=288/576 (50.000000%), "
            "c8_gray_bit_0=2/576 (0.347222%).\n",
            "[Phase3.1 Pure MLP5 Hard Projection] Epoch 1: "
            "val_loss=0.500000, val_acc=0.8300, "
            "test_loss=0.510000, test_acc=0.8200. "
            "Soft training state restored.\n",
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.txt"
            path.write_text("".join(lines), encoding="utf-8")
            parsed = parse_training_log(path)

        segment = parsed["segments"][0]
        self.assertEqual(segment["lut_diagnostics"][1]["init_sign_diff"], 2)
        self.assertEqual(
            segment["lut_bit_sensitivity"][1]["bits"][
                "c8_gray_bit_0"
            ]["hard_diff"],
            2,
        )
        self.assertEqual(segment["hard_projections"][1]["val_acc"], 0.83)

    def test_semantic_lut5_diagnostics_and_bit_sensitivity_are_parsed(self):
        lines = [
            epoch_line(1, 0.82),
            "[Phase3.1 Semantic LUT5] Epoch 1: "
            "init_sign_diff=3 / 1152 (0.260417%), "
            "dominance_slice_hard_diff=285 / 576 (49.479167%), "
            "dominance_slice_soft_abs_diff_mean=0.275000, "
            "max=0.620000, hard_checkpoint_eligible=False.\n",
            "[Phase3.1 Semantic LUT5 Bit Sensitivity] Epoch 1: "
            "sign_r=288/576 (50.000000%), "
            "sign_i=288/576 (50.000000%), "
            "dominance=285/576 (49.479167%).\n",
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.txt"
            path.write_text("".join(lines), encoding="utf-8")
            parsed = parse_training_log(path)

        segment = parsed["segments"][0]
        diagnostic = segment["lut_diagnostics"][1]
        self.assertEqual(diagnostic["stage"], "Phase3.1 Semantic LUT5")
        self.assertEqual(diagnostic["slice_label"], "dominance")
        self.assertEqual(diagnostic["slice_hard_diff"], 285)
        sensitivity = segment["lut_bit_sensitivity"][1]["bits"]
        self.assertEqual(sensitivity["dominance"]["hard_diff"], 285)


if __name__ == "__main__":
    unittest.main()
