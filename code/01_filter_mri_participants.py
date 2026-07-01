#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter participants.tsv to rows with MRI=1."
    )
    parser.add_argument(
        "--participants",
        default="data/raw/ds005752/participants.tsv",
        help="Input participants.tsv file.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/screening/ds005752_mri_participants.tsv",
        help="Output TSV path for MRI participants.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return reader.fieldnames, list(reader)


def main() -> None:
    args = parse_args()
    participants_path = resolve_project_path(args.participants)
    output_path = resolve_project_path(args.output)

    fieldnames, rows = load_rows(participants_path)
    filtered_rows = [row for row in rows if row.get("MRI") == "1"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"Wrote {len(filtered_rows)} MRI participants to {output_path}")


if __name__ == "__main__":
    main()
