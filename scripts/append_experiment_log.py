#!/usr/bin/env python3
"""Append a structured, optionally deduplicated experiment-log entry."""

import argparse
import datetime as dt
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="EXPERIMENT_LOG.md")
    parser.add_argument("--title", required=True)
    parser.add_argument("--body", default=None)
    parser.add_argument("--body-file", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--dedupe-key", default=None)
    return parser.parse_args(argv)


def load_body(args):
    sources = sum(
        value is not None for value in (args.body, args.body_file)
    )
    if sources > 1:
        raise ValueError("Use only one of --body or --body-file")
    if args.body_file is not None:
        return Path(args.body_file).read_text(encoding="utf-8").strip()
    if args.body is not None:
        return args.body.strip()
    return sys.stdin.read().strip()


def append_entry(path, title, body, date, dedupe_key=None):
    path = Path(path)
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    marker = None
    if dedupe_key:
        marker = "<!-- experiment-entry:{} -->".format(dedupe_key)
        if marker in existing:
            return False

    parts = []
    if existing and not existing.endswith("\n"):
        parts.append("\n")
    if existing:
        parts.append("\n")
    if marker:
        parts.append(marker + "\n")
    parts.append("## {} - {}\n\n".format(date, title))
    parts.append(body + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("".join(parts))
    return True


def main(argv=None):
    args = parse_args(argv)
    date = args.date or dt.date.today().isoformat()
    body = load_body(args)
    if not body:
        raise ValueError("Experiment log body cannot be empty")
    appended = append_entry(
        args.log,
        args.title,
        body,
        date,
        dedupe_key=args.dedupe_key,
    )
    print("appended" if appended else "already present")


if __name__ == "__main__":
    main()
