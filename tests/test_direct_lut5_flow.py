import json
import tempfile
import unittest
from pathlib import Path

import torch

from complexPyTorch.complexBinaryResNet import BinaryComplexResNet
from complexPyTorch.complexLayers import binary_annealing
from complexPyTorch.directLut5Flow import (
    DIRECT_LUT5_PHASE,
    DirectBitGradientTracker,
    DirectBitOccupancyTracker,
    DirectLUT5ComplexResNet,
    capture_hard_lut_tables,
    direct_lut5_bit_names,
    direct_lut5_diagnostics,
    direct_lut5_hardware_spec,
    direct_lut5_margin_loss,
    iter_direct_lut5_convs,
    set_direct_lut5_trainable,
)
from scripts.analyze_direct_lut5_run import analyze_direct_lut5
from training import load_phase_checkpoint


def make_model(encoder):
    return DirectLUT5ComplexResNet(
        in_channels=3,
        num_blocks=1,
        start_filters=2,
        num_classes=10,
        is_sar_input=False,
        encoder_mode=encoder,
        initial_logit_margin=0.25,
    )


class DirectLUT5ModelTest(unittest.TestCase):
    def test_encoder_modes_are_hard_and_use_expected_gradient_paths(self):
        for encoder, codebook, gradient, strategy in (
            ("phase", "roots", "softmax", "c8_product"),
            (
                "dominance",
                "octants",
                "semantic_ste",
                "semantic_c8_product",
            ),
        ):
            model = make_model(encoder)
            self.assertEqual(model.phase, DIRECT_LUT5_PHASE)
            self.assertEqual(model.direct_lut5_encoder, encoder)
            layers = list(iter_direct_lut5_convs(model))
            self.assertGreater(len(layers), 0)
            for layer in layers:
                self.assertEqual(layer.lut_inputs, 5)
                self.assertEqual(layer.c8_codebook_mode, codebook)
                self.assertEqual(layer.lut5_init_strategy, strategy)
                self.assertEqual(float(layer.hard.item()), 1.0)
                self.assertEqual(float(layer.tau.item()), 1.0)
                self.assertGreaterEqual(layer.lut_r.abs().min().item(), 0.25)
                self.assertEqual(
                    model.stage2[0].act.c8_grad_mode,
                    gradient,
                )
                hard = binary_annealing(
                    layer.lut_r,
                    tau=layer.tau,
                    hard=layer.hard,
                )
                self.assertTrue(
                    torch.logical_or(hard == 0.0, hard == 1.0).all()
                )

    def test_all_three_activation_bits_have_gradients_to_source(self):
        source_values = torch.tensor(
            [
                [-1.2, -0.4],
                [-0.3, 1.4],
                [0.2, 0.9],
                [1.1, -0.6],
                [0.6, 0.5],
                [-0.7, -0.8],
            ],
            dtype=torch.float32,
        )
        for encoder in ("phase", "dominance"):
            activation = make_model(encoder).stage2[0].act
            real = source_values[:, 0].reshape(1, 1, 1, -1).clone()
            imag = source_values[:, 1].reshape(1, 1, 1, -1).clone()
            real.requires_grad_()
            imag.requires_grad_()
            source = torch.complex(real, imag)
            bits = (
                activation.phase_code_bits(source)
                if encoder == "phase"
                else activation.semantic_code_bits(source)
            )
            self.assertTrue(
                torch.logical_or(bits == -1.0, bits == 1.0).all()
            )
            for bit in range(3):
                gradients = torch.autograd.grad(
                    bits[:, bit].sum(),
                    (real, imag),
                    retain_graph=True,
                )
                self.assertGreater(
                    gradients[0].abs().sum().item()
                    + gradients[1].abs().sum().item(),
                    0.0,
                    msg="{} bit {} has no source gradient".format(
                        encoder, bit
                    ),
                )

    def test_lut_state_diagnostics_export_and_freezing(self):
        model = make_model("dominance")
        diagnostics = direct_lut5_diagnostics(model)
        self.assertEqual(diagnostics["sign_diff"], 0)
        self.assertTrue(diagnostics["fully_hard"])
        self.assertGreater(
            diagnostics["extra_bit_slice_hard_diff"],
            0,
        )
        self.assertGreater(direct_lut5_margin_loss(model, 0.3).item(), 0.0)

        set_direct_lut5_trainable(model, False)
        self.assertTrue(
            all(
                not layer.lut_r.requires_grad
                for layer in iter_direct_lut5_convs(model)
            )
        )
        set_direct_lut5_trainable(model, True)
        self.assertTrue(
            all(
                layer.lut_i.requires_grad
                for layer in iter_direct_lut5_convs(model)
            )
        )

        tables = capture_hard_lut_tables(model)
        self.assertGreater(len(tables), 0)
        for table in tables.values():
            self.assertEqual(table["real"].shape[-1], 32)
            self.assertTrue(
                torch.logical_or(
                    table["imag"] == 0,
                    table["imag"] == 1,
                ).all()
            )
        spec = direct_lut5_hardware_spec("dominance")
        self.assertEqual(spec["physical_inputs_per_lut"], 5)
        self.assertEqual(spec["physical_outputs_per_complex_product"], 2)
        self.assertEqual(spec["lut_units_per_complex_product"], 2)

    def test_phase1_checkpoint_maps_spatial_weights(self):
        source = BinaryComplexResNet(
            in_channels=3,
            num_blocks=1,
            start_filters=2,
            num_classes=10,
            is_sar_input=False,
            is_binary=False,
            phase=1,
        )
        target = make_model("phase")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase1.pt"
            torch.save(
                {
                    "phase": 1,
                    "model": source.state_dict(),
                    "args": {"phase": 1},
                },
                path,
            )
            load_phase_checkpoint(
                target,
                path,
                expected_phase=1,
            )
        for source_block, target_block in zip(
            source.stage2,
            target.stage2,
        ):
            torch.testing.assert_close(
                source_block.conv.conv_r.weight,
                target_block.conv.weight_r,
            )
            torch.testing.assert_close(
                source_block.conv.conv_i.weight,
                target_block.conv.weight_i,
            )


class DirectLUT5TrackerTest(unittest.TestCase):
    def test_trackers_measure_codes_and_per_bit_gradients(self):
        tracker = DirectBitOccupancyTracker(torch.nn.Identity())
        signed_bits = torch.tensor(
            [
                [1.0, 1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0],
            ]
        ).reshape(1, 3, 1, 1, 4)
        activation_bits = signed_bits.reshape(1, 3, 1, 4)
        tracker._hook(None, (), {"activation_bits": activation_bits})
        occupancy = tracker.finish()
        self.assertEqual(occupancy["total"], 4)
        self.assertEqual(occupancy["active_codes"], 4)

        gradient_tracker = DirectBitGradientTracker(torch.nn.Identity())
        gradient_tracker._gradient_hook(
            torch.tensor(
                [
                    [[[1.0, 0.0]]],
                    [[[0.5, 0.5]]],
                    [[[0.0, 2.0]]],
                ]
            ).reshape(1, 3, 1, 2)
        )
        result = gradient_tracker.finish(
            direct_lut5_bit_names("dominance")
        )
        self.assertEqual(result["captured_tensors"], 1)
        self.assertGreater(result["bits"]["dominance"]["mean_abs"], 0.0)
        gradient_tracker.close()


class DirectLUT5AnalyzerTest(unittest.TestCase):
    def test_analyzer_reads_reusable_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            (workdir / "logs").mkdir()
            (workdir / "logs" / "train.txt").write_text(
                "INVOCATION: python training_direct_lut5.py "
                "--encoder-mode dominance --num-epochs 200\n"
                "Epoch     1 train_loss: 1.000000, train_acc: 0.5000, "
                "val_loss: 0.900000, val_acc: 0.6000, "
                "test_loss: 0.800000, test_acc: 0.6100\n",
                encoding="utf-8",
            )
            (workdir / "phase_metrics.json").write_text(
                json.dumps(
                    {
                        "phase2p5": {
                            "epoch": 1,
                            "val_acc": 0.6,
                            "test_acc": 0.61,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (workdir / "direct_lut5_metrics.json").write_text(
                json.dumps(
                    {
                        "encoder_mode": "dominance",
                        "best_metrics": {"epoch": 1, "val_acc": 0.6},
                        "last_metrics": {"epoch": 1, "val_acc": 0.6},
                        "lut_diagnostics": {
                            "sign_diff": 0,
                            "entries": 64,
                        },
                        "bit_gradients": {
                            "captured_tensors": 1,
                            "bits": {
                                "dominance": {"mean_abs": 0.1}
                            },
                        },
                        "bit_occupancy": {"active_codes": 8},
                    }
                ),
                encoding="utf-8",
            )
            report = analyze_direct_lut5(workdir)
        self.assertEqual(report["encoder_mode"], "dominance")
        self.assertEqual(report["configured_epochs"], 200)
        self.assertEqual(report["status"], "running_or_stopped")


if __name__ == "__main__":
    unittest.main()
