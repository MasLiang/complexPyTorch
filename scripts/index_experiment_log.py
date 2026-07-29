#!/usr/bin/env python3
"""Build a reusable section index for the unified experiment log."""

import argparse
import json
import re
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = REPO_ROOT / "PHASE4_LUT_EXPERIMENT_LOG.md"
HEADING_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$")
DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")

CATEGORY_PATTERNS = {
    "lut4_baseline_and_scaling": (
        "baseline",
        "lut4",
        "multi lut",
        "channel-level",
        "channel wise",
    ),
    "lut_optimization": (
        "anneal",
        "hard ste",
        "hard flip",
        "truth-table",
        "micro-commit",
        "lut-soft",
        "lut lr",
        "init margin",
        "schedule",
    ),
    "lut5_extra_bit": (
        "lut5",
        "fifth bit",
        "第五 bit",
        "magnitude",
        "phase3p5",
        "phase 5",
        "phase-dominance",
    ),
    "c8_phase_quantization": (
        "c8",
        "8-psk",
        "phase2.1",
        "phase3.1",
        "phase 3.6",
        "phase3.6",
        "semantic-ste",
        "octants",
    ),
    "learned_operation": (
        "learned lut5",
        "neural lut5",
        "5-to-2",
        "mlp operation",
        "mlp flow",
        "binary mlp",
        "network in network",
    ),
    "maintenance_and_tooling": (
        "目录",
        "清理",
        "故障",
        "修复",
        "report",
        "日志",
    ),
}


def valid_date(value):
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected an ISO date in YYYY-MM-DD form"
        ) from exc
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument(
        "--since",
        type=valid_date,
        help="Only include dated sections on/after YYYY-MM-DD",
    )
    parser.add_argument(
        "--category",
        choices=sorted(CATEGORY_PATTERNS),
        help="Only include sections assigned to this category",
    )
    return parser.parse_args(argv)


def categorize(title, body):
    searchable = "{}\n{}".format(title, body[:1000]).lower()
    return [
        category
        for category, patterns in CATEGORY_PATTERNS.items()
        if any(pattern.lower() in searchable for pattern in patterns)
    ]


def parse_sections(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    headings = []
    for index, line in enumerate(lines):
        match = HEADING_RE.match(line)
        if match:
            headings.append((index, match.group("title")))

    sections = []
    for heading_index, (start, title) in enumerate(headings):
        end = (
            headings[heading_index + 1][0]
            if heading_index + 1 < len(headings)
            else len(lines)
        )
        body = "\n".join(lines[start + 1 : end]).strip()
        date_match = DATE_RE.search(title)
        sections.append(
            {
                "index": heading_index,
                "title": title,
                "date": date_match.group(1) if date_match else None,
                "line_start": start + 1,
                "line_end": end,
                "body_lines": max(end - start - 1, 0),
                "body_chars": len(body),
                "categories": categorize(title, body),
            }
        )
    return sections


def filter_sections(sections, since=None, category=None):
    result = []
    for section in sections:
        if since and (not section["date"] or section["date"] < since):
            continue
        if category and category not in section["categories"]:
            continue
        result.append(section)
    return result


def markdown(path, sections):
    lines = [
        "# Experiment Log Index",
        "",
        "- Source: `{}`".format(path.resolve()),
        "- Indexed sections: `{}`".format(len(sections)),
        "",
        "| # | Date | Lines | Categories | Title |",
        "| ---: | --- | ---: | --- | --- |",
    ]
    for section in sections:
        row = dict(section)
        row.update(
            {
                "date": section["date"] or "-",
                "categories": ", ".join(section["categories"]) or "-",
                "title": section["title"].replace("|", "\\|"),
            }
        )
        lines.append(
            "| {index} | {date} | {line_start}-{line_end} | {categories} | {title} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    sections = filter_sections(
        parse_sections(args.log),
        since=args.since,
        category=args.category,
    )
    payload = {
        "source": str(args.log.resolve()),
        "section_count": len(sections),
        "sections": sections,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("Wrote {}".format(args.json_output.resolve()))
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            markdown(args.log, sections),
            encoding="utf-8",
        )
        print("Wrote {}".format(args.markdown_output.resolve()))
    if not args.json_output and not args.markdown_output:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
