#!/usr/bin/env python3

# What this script does:
#   Filters the OpenNeuro participant table to rows with MRI = 1.
# How to run it:
#   Run from the repo root with:
#   python code/primary/01_filter_mri_participants.py
# Main output:
#   data/processed/screening/ds005752_mri_participants.tsv

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter ds005752 participants.tsv to MRI-eligible rows only."
    )
    parser.add_argument(
        "--input",
        default="data/raw/ds005752/participants.tsv",
        help="Path to the raw OpenNeuro participants.tsv file.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/screening/ds005752_mri_participants.tsv",
        help="Output TSV for MRI-eligible participants.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_path = resolve_project_path(args.output)

    with input_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    filtered = []
    for row in rows:
        if row.get("MRI", "").strip() == "1":
            filtered.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(filtered)

    print(f"Wrote {len(filtered)} MRI-eligible participants to {output_path}")


if __name__ == "__main__":
    main()
