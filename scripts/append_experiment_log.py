#!/usr/bin/env python3
"""Append a dated, optionally deduplicated section to the experiment log."""

import argparse
import datetime as dt
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = REPO_ROOT / "PHASE4_LUT_EXPERIMENT_LOG.md"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--body-file", type=Path)
    parser.add_argument(
        "--dedupe-key",
        help="Skip the append when this exact marker already exists",
    )
    parser.add_argument(
        "--date",
        default=dt.date.today().isoformat(),
        help="Section date in YYYY-MM-DD form",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    body = (
        args.body_file.read_text(encoding="utf-8")
        if args.body_file is not None
        else sys.stdin.read()
    ).strip()
    if not body:
        raise SystemExit("Refusing to append an empty log section")

    existing = (
        args.log.read_text(encoding="utf-8")
        if args.log.exists()
        else ""
    )
    marker = args.dedupe_key or ""
    if marker and marker in existing:
        print("Log section already present; skipped: {}".format(marker))
        return

    section = "\n\n## {} {}\n\n{}\n".format(
        args.date,
        args.title,
        body,
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as stream:
        stream.write(section)
    print("Appended experiment log section to {}".format(args.log))


if __name__ == "__main__":
    main()
