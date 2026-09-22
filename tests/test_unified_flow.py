"""Regression coverage for the dataset-agnostic staged training flow."""

from types import SimpleNamespace

import torch

from experiments.cifar10.protocol import Cifar10Protocol
from experiments.flow.checkpoints import (
    compile_lut4_from_bireal,
    expand_lut6_from_lut4,
    load_matching_stage,
)
from experiments.san_francisco.protocol import SanFranciscoProtocol
from experiments.pol_insar_island.protocol import PolInSARIslandT3Protocol
from experiments.opensarship_slc.protocol import OpenSARShipSLCProtocol


def _args():
    return SimpleNamespace(
        start_filters=4,
        num_blocks=1,
        lut4_parameterization="categorical",
        fp_epochs=1,
        shared_fp_epochs=1,
        bireal_epochs=1,
        lut4_epochs=1,
        lut6_epochs=1,
    )


def _check_transitions(protocol, num_classes, tmp_path):
    args = _args()
    stages = {stage.name: stage for stage in protocol.stages(args)}

    fp = protocol.build_model(stages["shared_fp"], args, num_classes)
    fp_checkpoint = tmp_path / "shared_fp.pt"
    torch.save({"stage": "shared_fp", "model": fp.state_dict()}, fp_checkpoint)

    bireal = protocol.build_model(stages["bireal"], args, num_classes)
    load_matching_stage(bireal, fp_checkpoint, "shared_fp")
    bireal_checkpoint = tmp_path / "bireal.pt"
    torch.save(
        {"stage": "bireal", "model": bireal.state_dict()}, bireal_checkpoint
    )

    lut4 = protocol.build_model(stages["lut4"], args, num_classes).eval()
    _, compiled = compile_lut4_from_bireal(lut4, bireal_checkpoint)
    assert compiled > 0
    lut4_checkpoint = tmp_path / "lut4.pt"
    torch.save({"stage": "lut4", "model": lut4.state_dict()}, lut4_checkpoint)

    lut6 = protocol.build_model(stages["lut6_residual"], args, num_classes).eval()
    _, expanded = expand_lut6_from_lut4(lut6, lut4_checkpoint)
    assert expanded == compiled

    channels = 3 if protocol.name == "cifar10" else 6
    spatial = 32 if protocol.name == "cifar10" else 11
    inputs = torch.randn(2, channels, spatial, spatial, dtype=torch.complex64)
    with torch.no_grad():
        assert torch.equal(lut4(inputs), lut6(inputs))


def test_cifar10_protocol_stage_transitions(tmp_path):
    _check_transitions(Cifar10Protocol(), 10, tmp_path)


def test_san_francisco_protocol_stage_transitions(tmp_path):
    _check_transitions(SanFranciscoProtocol(), 5, tmp_path)


def _check_paper_protocol_transitions(protocol, num_classes, channels, spatial, tmp_path):
    args = _args()
    stages = {stage.name: stage for stage in protocol.stages(args)}
    fp = protocol.build_model(stages["bireal_fp"], args, num_classes)
    fp_checkpoint = tmp_path / "bireal_fp.pt"
    torch.save({"stage": "bireal_fp", "model": fp.state_dict()}, fp_checkpoint)
    bireal = protocol.build_model(stages["bireal"], args, num_classes)
    load_matching_stage(bireal, fp_checkpoint, "bireal_fp")
    bireal_checkpoint = tmp_path / "bireal.pt"
    torch.save({"stage": "bireal", "model": bireal.state_dict()}, bireal_checkpoint)
    lut4 = protocol.build_model(stages["lut4"], args, num_classes).eval()
    _, compiled = compile_lut4_from_bireal(lut4, bireal_checkpoint)
    lut4_checkpoint = tmp_path / "lut4.pt"
    torch.save({"stage": "lut4", "model": lut4.state_dict()}, lut4_checkpoint)
    lut6 = protocol.build_model(stages["lut6_residual"], args, num_classes).eval()
    _, expanded = expand_lut6_from_lut4(lut6, lut4_checkpoint)
    assert compiled == expanded > 0
    inputs = torch.randn(2, channels, spatial, spatial, dtype=torch.complex64)
    with torch.no_grad():
        assert torch.equal(lut4(inputs), lut6(inputs))


def test_pol_insar_protocol_stage_transitions(tmp_path):
    _check_paper_protocol_transitions(PolInSARIslandT3Protocol(), 12, 6, 12, tmp_path)


def test_opensarship_protocol_stage_transitions(tmp_path):
    _check_paper_protocol_transitions(OpenSARShipSLCProtocol(), 3, 2, 128, tmp_path)
