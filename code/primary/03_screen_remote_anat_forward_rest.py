#!/usr/bin/env python3

# What this script does:
#   Checks remote OpenNeuro metadata and keeps only the subjects who have both
#   structural anatomy and a forward resting-state run.
# How to run it:
#   Run from the repo root with:
#   python code/primary/03_screen_remote_anat_forward_rest.py
# Main outputs:
#   Annotated younger and older remote-file tables plus the filtered
#   remote_anat_forward TSV files in data/processed/screening/

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENNEURO_PYTHON = os.environ.get("OPENNEURO_PYTHON", "python3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate age-band participant TSV files with remote OpenNeuro "
            "availability and filter to subjects with anat plus forward rest."
        )
    )
    parser.add_argument("--dataset", default="ds005752")
    parser.add_argument(
        "--young-input",
        default="data/processed/screening/ds005752_mri_participants_age_20_25.tsv",
        help="Input TSV for the younger age group.",
    )
    parser.add_argument(
        "--older-input",
        default="data/processed/screening/ds005752_mri_participants_age_50_75.tsv",
        help="Input TSV for the older age group.",
    )
    parser.add_argument(
        "--young-annotated-output",
        default="data/processed/screening/ds005752_mri_participants_age_20_25_remote_files.tsv",
        help="Annotated output TSV for the younger group.",
    )
    parser.add_argument(
        "--older-annotated-output",
        default="data/processed/screening/ds005752_mri_participants_age_50_75_remote_files.tsv",
        help="Annotated output TSV for the older group.",
    )
    parser.add_argument(
        "--young-filtered-output",
        default="data/processed/screening/ds005752_mri_participants_age_20_25_remote_anat_forward.tsv",
        help="Filtered younger-group output TSV with remote anat and forward rest only.",
    )
    parser.add_argument(
        "--older-filtered-output",
        default="data/processed/screening/ds005752_mri_participants_age_50_75_remote_anat_forward.tsv",
        help="Filtered older-group output TSV with remote anat and forward rest only.",
    )
    parser.add_argument(
        "--allow-partial-results",
        action="store_true",
        help="Write outputs even if some remote metadata queries fail.",
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


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def query_single_subject(dataset: str, subject: str) -> dict[str, object]:
    code = f"""
from tqdm.std import tqdm
import openneuro._download as d

d.tqdm = tqdm
dataset = "{dataset}"
root = d._get_download_metadata(dataset_id=dataset, max_retries=5)
tag = root["id"].replace(f"{{dataset}}:", "")
subject_ids = {{
    item["filename"]: item["id"]
    for item in root["files"]
    if item.get("directory")
}}

subject_id = subject_ids["{subject}"]
meta = d._get_download_metadata(
    dataset_id=dataset,
    tag=tag,
    tree=f'"{{subject_id}}"',
    max_retries=5,
    check_snapshot=False,
)

files = set()
stack = list(meta.get("files", []))
while stack:
    item = stack.pop()
    if item.get("directory"):
        stack.extend(item.get("files", []))
    else:
        files.add(item.get("filename", ""))

has_anat = any("/anat/" in file for file in files)
forward_rest_files = [file for file in files if "/func/" in file and "task-rest" in file and "dir-forward" in file]
print(int(has_anat))
print(int(bool(forward_rest_files)))
print(len(forward_rest_files))
"""
    completed = subprocess.run(
        [str(OPENNEURO_PYTHON), "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {
        "has_remote_anat": int(lines[0]),
        "has_remote_forward_rest": int(lines[1]),
        "n_remote_forward_rest_files": int(lines[2]),
    }


def annotate_rows(dataset: str, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    annotated = []
    for row in rows:
        subject = row["participant_id"]
        query = query_single_subject(dataset, subject)
        annotated.append(
            {
                **row,
                "subject_id": subject,
                **query,
            }
        )
    return annotated


def filter_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if int(row["has_remote_anat"]) == 1 and int(row["has_remote_forward_rest"]) == 1
    ]


def main() -> None:
    args = parse_args()
    young_input = resolve_project_path(args.young_input)
    older_input = resolve_project_path(args.older_input)
    young_annotated_output = resolve_project_path(args.young_annotated_output)
    older_annotated_output = resolve_project_path(args.older_annotated_output)
    young_filtered_output = resolve_project_path(args.young_filtered_output)
    older_filtered_output = resolve_project_path(args.older_filtered_output)

    young_fieldnames, young_rows = load_rows(young_input)
    older_fieldnames, older_rows = load_rows(older_input)

    young_annotated = annotate_rows(args.dataset, young_rows)
    older_annotated = annotate_rows(args.dataset, older_rows)

    annotated_fieldnames = young_fieldnames + [
        "subject_id",
        "has_remote_anat",
        "has_remote_forward_rest",
        "n_remote_forward_rest_files",
    ]

    write_rows(young_annotated_output, annotated_fieldnames, young_annotated)
    write_rows(older_annotated_output, annotated_fieldnames, older_annotated)

    young_filtered = filter_rows(young_annotated)
    older_filtered = filter_rows(older_annotated)

    write_rows(young_filtered_output, annotated_fieldnames, young_filtered)
    write_rows(older_filtered_output, annotated_fieldnames, older_filtered)

    print(f"Wrote annotated younger table to {young_annotated_output}")
    print(f"Wrote annotated older table to {older_annotated_output}")
    print(f"Wrote filtered younger table to {young_filtered_output}")
    print(f"Wrote filtered older table to {older_filtered_output}")


if __name__ == "__main__":
    main()
