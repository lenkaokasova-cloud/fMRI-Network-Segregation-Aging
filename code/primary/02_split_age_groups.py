#!/usr/bin/env python3

# What this script does:
#   Splits the MRI-eligible participant table into the younger and older age
#   groups used later in the project.
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
        description="Split the MRI participant table into younger and older age bands."
    )
    parser.add_argument(
        "--participants",
        default="data/processed/screening/ds005752_mri_participants.tsv",
        help="Input TSV of MRI participants.",
    )
    parser.add_argument("--young-min", type=int, default=20, help="Younger-group minimum age.")
    parser.add_argument("--young-max", type=int, default=25, help="Younger-group maximum age.")
    parser.add_argument("--older-min", type=int, default=50, help="Older-group minimum age.")
    parser.add_argument("--older-max", type=int, default=75, help="Older-group maximum age.")
    parser.add_argument(
        "--young-output",
        default=None,
        help="Optional output TSV path for the younger group.",
    )
    parser.add_argument(
        "--older-output",
        default=None,
        help="Optional output TSV path for the older group.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def default_output(stem: str, min_age: int, max_age: int) -> Path:
    return resolve_project_path(
        f"data/processed/screening/ds005752_mri_participants_age_{min_age}_{max_age}.tsv"
    )


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return reader.fieldnames, list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_age(row: dict[str, str]) -> int | None:
    age = row.get("age", "")
    return int(age) if age.isdigit() else None


def main() -> None:
    args = parse_args()
    participants_path = resolve_project_path(args.participants)
    young_output = (
        resolve_project_path(args.young_output)
        if args.young_output
        else default_output("young", args.young_min, args.young_max)
    )
    older_output = (
        resolve_project_path(args.older_output)
        if args.older_output
        else default_output("older", args.older_min, args.older_max)
    )

    fieldnames, rows = load_rows(participants_path)
    young_rows: list[dict[str, str]] = []
    older_rows: list[dict[str, str]] = []

    # This is where one MRI-only table gets turned into the two age bands I actually analyze.
    for row in rows:
        age = parse_age(row)
        if age is None:
            continue
        if args.young_min <= age <= args.young_max:
            young_rows.append(row)
        if args.older_min <= age <= args.older_max:
            older_rows.append(row)

    write_rows(young_output, fieldnames, young_rows)
    write_rows(older_output, fieldnames, older_rows)

    print(
        f"Wrote {len(young_rows)} participants ages {args.young_min}-{args.young_max} "
        f"to {young_output}"
    )
    print(
        f"Wrote {len(older_rows)} participants ages {args.older_min}-{args.older_max} "
        f"to {older_output}"
    )


if __name__ == "__main__":
    main()
