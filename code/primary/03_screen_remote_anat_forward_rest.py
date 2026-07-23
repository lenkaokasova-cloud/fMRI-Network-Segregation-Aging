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
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENNEURO_PYTHON = Path(
    "/Library/Frameworks/Python.framework/Versions/3.9/bin/python3"
)


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
    # I do the remote check subject-by-subject here so I can tell exactly what each person has
    # before I download anything locally.
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

info = {{
    "subject": "{subject}",
    "has_anat": 0,
    "has_rest_bold": 0,
    "has_rest_forward": 0,
    "has_rest_reverse": 0,
    "has_fmap": 0,
    "rest_bold_files": 0,
}}
subject_id = subject_ids.get("{subject}")
if subject_id:
    meta = d._get_download_metadata(
        dataset_id=dataset,
        tag=tag,
        tree=f'"{{subject_id}}"',
        max_retries=5,
        check_snapshot=False,
    )
    for file in d._iterate_filenames(
        meta["files"],
        dataset_id=dataset,
        tag=tag,
        max_retries=5,
        include=[],
    ):
        filename = file["filename"]
        if (
            filename.endswith(".nii.gz")
            and (filename.startswith("anat/") or "/anat/" in filename)
        ):
            info["has_anat"] = 1
        if filename.startswith("fmap/") or "/fmap/" in filename:
            info["has_fmap"] = 1
        if (
            (filename.startswith("func/") or "/func/" in filename)
            and "task-rest" in filename
            and filename.endswith("_bold.nii.gz")
        ):
            info["has_rest_bold"] = 1
            info["rest_bold_files"] += 1
            if "dir-forward" in filename:
                info["has_rest_forward"] = 1
            if "dir-reverse" in filename:
                info["has_rest_reverse"] = 1
print(
    "\\t".join(
        str(info[key]) for key in (
            "subject",
            "has_anat",
            "has_rest_bold",
            "has_rest_forward",
            "has_rest_reverse",
            "has_fmap",
            "rest_bold_files",
        )
    )
)
"""
    proc = subprocess.run(
        [str(OPENNEURO_PYTHON), "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "Metadata query failed")
    for line in proc.stdout.splitlines():
        if not line or line.startswith(("👋", "👉", "🌍", "📁", "🍪")):
            continue
        (
            _subject,
            has_anat,
            has_rest_bold,
            has_rest_forward,
            has_rest_reverse,
            has_fmap,
            rest_bold_files,
        ) = line.split("\t")
        return {
            "has_anat": int(has_anat),
            "has_rest_bold": int(has_rest_bold),
            "has_rest_forward": int(has_rest_forward),
            "has_rest_reverse": int(has_rest_reverse),
            "has_fmap": int(has_fmap),
            "rest_bold_files": int(rest_bold_files),
            "query_failed": 0,
            "query_error": "",
        }
    raise RuntimeError(f"No metadata row parsed for {subject}")


def fetch_subject_availability(
    dataset: str, subjects: list[str]
) -> dict[str, dict[str, object]]:
    availability: dict[str, dict[str, object]] = {}
    for subject in subjects:
        last_error: RuntimeError | None = None
        for _ in range(2):
            try:
                availability[subject] = query_single_subject(dataset, subject)
                last_error = None
                break
            except RuntimeError as exc:
                last_error = exc
        if last_error is not None:
            availability[subject] = {
                "has_anat": 0,
                "has_rest_bold": 0,
                "has_rest_forward": 0,
                "has_rest_reverse": 0,
                "has_fmap": 0,
                "rest_bold_files": 0,
                "query_failed": 1,
                "query_error": str(last_error),
            }
    return availability


def annotate_rows(
    rows: list[dict[str, str]], availability: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for row in rows:
        subject = row["participant_id"]
        out = dict(row)
        out.update(availability[subject])
        annotated.append(out)
    return annotated


def filter_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    # This is the actual gate for the project: the subject needs anat plus the forward rest run.
    return [
        row
        for row in rows
        if int(row["has_anat"]) == 1 and int(row["has_rest_forward"]) == 1
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
    all_subjects = [row["participant_id"] for row in young_rows + older_rows]
    # I query availability once up front so the younger and older tables are filtered consistently.
    availability = fetch_subject_availability(args.dataset, all_subjects)

    failed_subjects = [
        subject for subject, remote in availability.items() if int(remote["query_failed"]) == 1
    ]
    if failed_subjects and not args.allow_partial_results:
        raise RuntimeError(
            "Remote metadata queries failed for "
            f"{len(failed_subjects)} participants. "
            "Rerun when the connection is stable, or use --allow-partial-results."
        )

    extra_fields = [
        "has_anat",
        "has_rest_bold",
        "has_rest_forward",
        "has_rest_reverse",
        "has_fmap",
        "rest_bold_files",
        "query_failed",
        "query_error",
    ]
    young_annotated = annotate_rows(young_rows, availability)
    older_annotated = annotate_rows(older_rows, availability)
    young_filtered = filter_rows(young_annotated)
    older_filtered = filter_rows(older_annotated)

    write_rows(young_annotated_output, young_fieldnames + extra_fields, young_annotated)
    write_rows(older_annotated_output, older_fieldnames + extra_fields, older_annotated)
    write_rows(young_filtered_output, young_fieldnames + extra_fields, young_filtered)
    write_rows(older_filtered_output, older_fieldnames + extra_fields, older_filtered)

    print(f"Wrote annotated younger table to {young_annotated_output}")
    print(f"Wrote annotated older table to {older_annotated_output}")
    print(
        f"Wrote {len(young_filtered)} younger participants with remote anat and forward rest "
        f"to {young_filtered_output}"
    )
    print(
        f"Wrote {len(older_filtered)} older participants with remote anat and forward rest "
        f"to {older_filtered_output}"
    )
    if failed_subjects:
        print(
            f"Warning: remote queries failed for {len(failed_subjects)} participants. "
            "Those rows were marked unavailable."
        )


if __name__ == "__main__":
    main()
