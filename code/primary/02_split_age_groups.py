#!/usr/bin/env python3

# What this script does:
#   Splits the MRI-eligible table into the younger and older age bands.
# How to run it:
#   Run from the repo root with:
#   python code/primary/02_split_age_groups.py
# Main outputs:
#   data/processed/screening/ds005752_mri_participants_age_20_25.tsv
#   data/processed/screening/ds005752_mri_participants_age_50_75.tsv

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split the MRI-eligible ds005752 participant table into the younger "
            "and older dissertation age bands."
        )
    )
    parser.add_argument(
        "--input",
        default="data/processed/screening/ds005752_mri_participants.tsv",
        help="Input TSV of MRI-eligible participants.",
    )
    parser.add_argument(
        "--young-min-age",
        type=int,
        default=20,
        help="Minimum age for the younger age band.",
    )
    parser.add_argument(
        "--young-max-age",
        type=int,
        default=25,
        help="Maximum age for the younger age band.",
    )
    parser.add_argument(
        "--older-min-age",
        type=int,
        default=50,
        help="Minimum age for the older age band.",
    )
    parser.add_argument(
        "--older-max-age",
        type=int,
        default=75,
        help="Maximum age for the older age band.",
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
        return reader.fieldnames or [], list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def filter_age_band(rows: list[dict[str, str]], min_age: int, max_age: int) -> list[dict[str, str]]:
    filtered = []
    for row in rows:
        age_text = row.get("age", "").strip()
        if not age_text:
            continue
        age = int(float(age_text))
        if min_age <= age <= max_age:
            filtered.append(row)
    return filtered


def default_output_path(min_age: int, max_age: int) -> Path:
    return Path(
        f"data/processed/screening/ds005752_mri_participants_age_{min_age}_{max_age}.tsv"
    )


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    fieldnames, rows = load_rows(input_path)

    young_rows = filter_age_band(rows, args.young_min_age, args.young_max_age)
    older_rows = filter_age_band(rows, args.older_min_age, args.older_max_age)

    young_output = resolve_project_path(str(default_output_path(args.young_min_age, args.young_max_age)))
    older_output = resolve_project_path(str(default_output_path(args.older_min_age, args.older_max_age)))

    write_rows(young_output, fieldnames, young_rows)
    write_rows(older_output, fieldnames, older_rows)

    print(f"Wrote {len(young_rows)} younger participants to {young_output}")
    print(f"Wrote {len(older_rows)} older participants to {older_output}")


if __name__ == "__main__":
    main()
