#!/usr/bin/env python3

# What this script does:
#   Queries the remote OpenNeuro JSON metadata and adds repetition time (TR)
#   information to a participant TSV before full download.
# How to run it:
#   Run from the repo root with:
#   python code/utilities/13_annotate_remote_tr.py
# Main output:
#   An annotated *_with_tr.tsv file next to the input screening TSV

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENNEURO_PYTHON = os.environ.get("OPENNEURO_PYTHON", "python3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query remote OpenNeuro forward-run JSON metadata and annotate a "
            "participant TSV with repetition time (TR) before full download."
        )
    )
    parser.add_argument("--dataset", default="ds005752")
    parser.add_argument(
        "--input",
        default="data/processed/screening/ds005752_mri_participants_age_20_25_remote_anat_forward.tsv",
        help="Input TSV containing participant_id or subject_id.",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Output TSV path. Defaults to the input path with _with_tr appended "
            "before the .tsv suffix."
        ),
    )
    parser.add_argument(
        "--allow-partial-results",
        action="store_true",
        help="Write the output TSV even if some subject-level TR queries fail.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_with_tr.tsv")


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        if "participant_id" in reader.fieldnames:
            key = "participant_id"
        elif "subject_id" in reader.fieldnames:
            key = "subject_id"
        else:
            raise ValueError(
                f"Input TSV must contain participant_id or subject_id: {path}"
            )
        return reader.fieldnames, list(reader), key


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def query_remote_tr(dataset: str, subjects: list[str]) -> dict[str, dict[str, object]]:
    code = f"""
import json
import requests
from tqdm.std import tqdm
import openneuro._download as d

d.tqdm = tqdm
dataset = {json.dumps(dataset)}
subjects = {json.dumps(subjects)}

root = d._get_download_metadata(dataset_id=dataset, max_retries=5)
tag = root["id"].replace(f"{{dataset}}:", "")
subject_ids = {{
    item["filename"]: item["id"]
    for item in root["files"]
    if item.get("directory")
}}

def detect_protocol_hint(n_forward_jsons):
    if n_forward_jsons > 1:
        return "multi_echo"
    if n_forward_jsons == 1:
        return "single_echo"
    return "unknown"

session = requests.Session()

for subject in subjects:
    result = {{
        "subject_id": subject,
        "tr_seconds": "",
        "unique_tr_values": "",
        "n_forward_jsons": 0,
        "protocol_hint": "unknown",
        "query_failed": 0,
        "query_error": "",
    }}
    try:
        subject_id = subject_ids.get(subject)
        if not subject_id:
            raise RuntimeError("Subject not found in remote dataset metadata.")

        meta = d._get_download_metadata(
            dataset_id=dataset,
            tag=tag,
            tree=f'"{{subject_id}}"',
            max_retries=5,
            check_snapshot=False,
        )

        stack = list(meta.get("files", []))
        forward_json_urls = []
        while stack:
            item = stack.pop()
            if item.get("directory"):
                stack.extend(item.get("files", []))
            else:
                filename = item.get("filename", "")
                if "/func/" in filename and "task-rest" in filename and "dir-forward" in filename and filename.endswith(".json"):
                    forward_json_urls.append(item.get("urls", [""])[0])

        result["n_forward_jsons"] = len(forward_json_urls)
        result["protocol_hint"] = detect_protocol_hint(len(forward_json_urls))

        tr_values = []
        for url in forward_json_urls:
            if not url:
                continue
            response = session.get(url, timeout=60)
            response.raise_for_status()
            metadata = response.json()
            tr = metadata.get("RepetitionTime")
            if tr is not None:
                tr_values.append(float(tr))

        unique = sorted(set(tr_values))
        result["unique_tr_values"] = ",".join(str(value) for value in unique)
        result["tr_seconds"] = unique[0] if len(unique) == 1 else ""
    except Exception as exc:
        result["query_failed"] = 1
        result["query_error"] = str(exc)

    print(json.dumps(result))
"""
    completed = subprocess.run(
        [str(OPENNEURO_PYTHON), "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    results = {}
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        results[row["subject_id"]] = row
    return results


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_path = resolve_project_path(args.output) if args.output else default_output_path(input_path)
    fieldnames, rows, key = load_rows(input_path)
    subjects = [row[key] for row in rows]
    query_results = query_remote_tr(args.dataset, subjects)

    missing = [subject for subject in subjects if query_results.get(subject, {}).get("query_failed") == 1]
    if missing and not args.allow_partial_results:
        raise RuntimeError(
            "TR annotation failed for one or more subjects. Re-run with "
            "--allow-partial-results to write the partial table."
        )

    output_fieldnames = fieldnames + [
        "tr_seconds",
        "unique_tr_values",
        "n_forward_jsons",
        "protocol_hint",
        "query_failed",
        "query_error",
    ]
    output_rows = []
    for row in rows:
        subject = row[key]
        query = query_results.get(subject, {})
        output_rows.append(
            {
                **row,
                "tr_seconds": query.get("tr_seconds", ""),
                "unique_tr_values": query.get("unique_tr_values", ""),
                "n_forward_jsons": query.get("n_forward_jsons", ""),
                "protocol_hint": query.get("protocol_hint", ""),
                "query_failed": query.get("query_failed", 0),
                "query_error": query.get("query_error", ""),
            }
        )

    write_rows(output_path, output_fieldnames, output_rows)
    print(f"Wrote TR-annotated table to {output_path}")


if __name__ == "__main__":
    main()
