from pathlib import Path
import tempfile
import unittest

from scripts.analyze_mlp_flow import parse_training_log


def test_parse_training_log_tracks_phase_and_operation_metrics(tmp_path):
    log_path = Path(tmp_path) / "train.txt"
    log_path.write_text(
        "[x] MLP flow Phase 1.2 (floating MLP operation), allocation=layer\n"
        "[x] Initial val_acc=0.9000 test_acc=0.8900 hard_changed=0.0000 "
        "d_sensitive=0.0000 d_coeff=0.000000\n"
        "[x] Epoch 1 train_loss=0.1 train_acc=0.9100 val_loss=0.2 "
        "val_acc=0.9050 test_loss=0.3 test_acc=0.8950 lrs={} "
        "hard_changed=0.0100 d_sensitive=0.0200 d_coeff=0.030000 "
        "best_val=0.9050 time=1.0s\n",
        encoding="utf-8",
    )

    result = parse_training_log(log_path)

    assert result["1.2"]["initial"]["test_acc"] == 0.89
    assert result["1.2"]["last"]["epoch"] == 1
    assert result["1.2"]["last"]["d_sensitive"] == 0.02


class AnalyzeMLPFlowTest(unittest.TestCase):
    def test_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            test_parse_training_log_tracks_phase_and_operation_metrics(
                Path(directory)
            )
