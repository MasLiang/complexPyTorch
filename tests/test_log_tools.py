import tempfile
import unittest
from pathlib import Path

from scripts.analyze_training_run import summarize
from scripts.append_experiment_log import append_entry


EPOCH_TEMPLATE = (
    "[2026-07-29 12:00:00 ~~ INFO    ] "
    "Epoch {epoch:5d} train_loss: 1.000000, train_acc: {train:.4f}, "
    "val_loss: 0.900000, val_acc: {val:.4f}, "
    "test_loss: 0.800000, test_acc: {test:.4f} (1.00s)\n"
)


class LogToolTests(unittest.TestCase):
    def test_analyzer_uses_latest_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            logdir = workdir / "logs"
            logdir.mkdir()
            old_run = (
                "[x] INVOCATION: training.py --phase 1 --num-epochs 1\n"
                + EPOCH_TEMPLATE.format(
                    epoch=1,
                    train=0.9,
                    val=0.9,
                    test=0.9,
                )
            )
            new_run = (
                "[x] INVOCATION: training.py --phase 2 --num-epochs 2 "
                "--batch-size 128 --lr 0.1 --schedule bireal\n"
                + EPOCH_TEMPLATE.format(
                    epoch=1,
                    train=0.5,
                    val=0.6,
                    test=0.55,
                )
                + EPOCH_TEMPLATE.format(
                    epoch=2,
                    train=0.7,
                    val=0.65,
                    test=0.62,
                )
            )
            (logdir / "train.txt").write_text(
                old_run + new_run,
                encoding="utf-8",
            )
            result = summarize(workdir)
        self.assertEqual(result["configuration"]["phase"], 2)
        self.assertEqual(result["epochs_logged"], 2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["best_validation"]["epoch"], 2)

    def test_log_append_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiments.md"
            first = append_entry(
                path,
                "route reset",
                "body",
                "2026-07-29",
                dedupe_key="route-reset",
            )
            second = append_entry(
                path,
                "route reset",
                "body",
                "2026-07-29",
                dedupe_key="route-reset",
            )
            text = path.read_text(encoding="utf-8")
        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(text.count("## 2026-07-29 - route reset"), 1)


if __name__ == "__main__":
    unittest.main()
