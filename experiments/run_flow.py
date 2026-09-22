#!/usr/bin/env python3
"""Run the unified FP -> BiReal -> LUT4 -> LUT6 residual workflow."""

import argparse
import logging

from experiments.cifar10.protocol import Cifar10Protocol
from experiments.flow.engine import run_flow
from experiments.opensarship_slc.protocol import OpenSARShipSLCProtocol
from experiments.pol_insar_island.protocol import PolInSARIslandT3Protocol
from experiments.san_francisco.protocol import SanFranciscoProtocol


PROTOCOLS = {
    "cifar10": Cifar10Protocol,
    "san_francisco": SanFranciscoProtocol,
    "pol_insar_island_t3": PolInSARIslandT3Protocol,
    "opensarship_slc": OpenSARShipSLCProtocol,
}


def parse_args(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--dataset", choices=tuple(PROTOCOLS), required=True)
    known, _ = bootstrap.parse_known_args(argv)
    protocol = PROTOCOLS[known.dataset]()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(PROTOCOLS), required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--stages", help="Comma-separated stage subset; default is the complete protocol flow")
    parser.add_argument(
        "--skip-parent-stages",
        action="store_true",
        help="Run only --stages and initialize their parents from --parent-workdir",
    )
    parser.add_argument(
        "--parent-workdir",
        help="Existing flow root that provides parent best.pt checkpoints when parent stages are skipped",
    )
    parser.add_argument(
        "--select-any-residual-alpha",
        action="store_true",
        help="Allow residual-ramp epochs to compete for the validation-selected best checkpoint",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--fp-epochs", type=int, default=100)
    parser.add_argument("--shared-fp-epochs", type=int, default=200)
    parser.add_argument("--bireal-epochs", type=int, default=200)
    parser.add_argument("--lut4-epochs", type=int, default=200)
    parser.add_argument("--lut6-epochs", type=int, default=200)
    protocol.add_arguments(parser)
    return parser.parse_args(argv), protocol


def main(argv=None):
    args, protocol = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s")
    selected = None if not args.stages else [item for item in args.stages.split(",") if item]
    run_flow(protocol, args, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
