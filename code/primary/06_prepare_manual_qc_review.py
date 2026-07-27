#!/usr/bin/env python3

# What this script does:
#   Creates or refreshes the TSV I use to manually review each fMRIPrep HTML
#   report.
# How to run it:
#   Run from the repo root with:
#   python code/primary/06_prepare_manual_qc_review.py
# Main output:
#   data/processed/qc/manual_fmriprep_report_review.tsv

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = "data/processed/qc/manual_fmriprep_report_review.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create or refresh a manual fMRIPrep report-review table. Fill this TSV "
            "after visually inspecting each subject's HTML report."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "Optional subject IDs or TSV files containing subject IDs. If omitted, "
            "all subjects with a local fMRIPrep report are included."
        ),
    )
    parser.add_argument(
        "--derivatives-dir",
        default="data/derivatives/fmriprep",
        help="Path to the fMRIPrep derivatives directory.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output TSV path. Defaults to {DEFAULT_OUTPUT}.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_subjects_from_tsv(path: Path) -> list[str]:
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        key = "subject_id" if "subject_id" in fieldnames else "participant_id"
        return [row[key].strip() for row in reader if row.get(key, "").strip()]


def collect_subjects(inputs: list[str], derivatives_dir: Path) -> list[str]:
    subjects: list[str] = []
    seen: set[str] = set()

    def add(subject: str) -> None:
        if subject and subject not in seen:
            seen.add(subject)
            subjects.append(subject)

    for item in inputs:
        path = Path(item)
        if path.exists() and path.suffix == ".tsv":
            for subject in read_subjects_from_tsv(path):
                add(subject)
        elif item.startswith("sub-"):
            add(item)
        else:
            raise ValueError(f"Unrecognized input: {item}")

    if subjects:
        return sorted(subjects)

    # If I do not give inputs, I just build the sheet from every local fMRIPrep report I have.
    for report_path in sorted(derivatives_dir.glob("sub-*.html")):
        add(report_path.stem)
    return subjects


def load_existing_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {row["subject_id"]: row for row in reader}


def main() -> None:
    args = parse_args()
    derivatives_dir = resolve_project_path(args.derivatives_dir)
    output_path = resolve_project_path(args.output)
    existing_rows = load_existing_rows(output_path)
    subjects = collect_subjects(args.inputs, derivatives_dir)
    for subject in sorted(existing_rows):
        if subject not in subjects:
            subjects.append(subject)
    subjects = sorted(subjects)

    fieldnames = [
        "subject_id",
        "report_exists",
        "manual_qc_status",
        "t1w_brainmask_status",
        "bold_to_t1w_status",
        "t1w_to_mni_status",
        "forward_bold_mask_status",
        "susceptibility_artifact_status",
        "review_date",
        "manual_qc_notes",
    ]

    rows: list[dict[str, str]] = []
    for subject in subjects:
        report_exists = (derivatives_dir / f"{subject}.html").exists()
        row = {
            "subject_id": subject,
            "report_exists": "1" if report_exists else "0",
            "manual_qc_status": "pending",
            "t1w_brainmask_status": "",
            "bold_to_t1w_status": "",
            "t1w_to_mni_status": "",
            "forward_bold_mask_status": "",
            "susceptibility_artifact_status": "",
            "review_date": "",
            "manual_qc_notes": "",
        }
        existing = existing_rows.get(subject, {})
        # I keep any older manual notes here so rerunning the template does not wipe my review work.
        for fieldname in fieldnames:
            existing_value = existing.get(fieldname, "").strip()
            if existing_value:
                row[fieldname] = existing_value
        row["report_exists"] = "1" if report_exists else "0"
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote manual QC review template to {output_path}")
    print("Fill manual_qc_status with pass or fail after checking each fMRIPrep HTML report.")


if __name__ == "__main__":
    main()
