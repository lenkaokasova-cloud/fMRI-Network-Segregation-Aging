#!/usr/bin/env python3

# What this script does:
#   Queries the remote OpenNeuro JSON metadata and adds repetition time (TR)
#   information to a participant TSV before full download.
# How to run it:
#   Run from the repo root with:
#   python code/utilities/14_annotate_remote_tr.py
# Main output:
#   An annotated *_with_tr.tsv file next to the input screening TSV

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENNEURO_PYTHON = Path(
    "/Library/Frameworks/Python.framework/Versions/3.9/bin/python3"
)


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
    # I ask the remote JSONs for TR here so I can screen protocol differences before full download.
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

        forward_jsons = []
        for file in d._iterate_filenames(
            meta["files"],
            dataset_id=dataset,
            tag=tag,
            max_retries=5,
            include=[],
        ):
            filename = file["filename"]
            if (
                (filename.startswith("func/") or "/func/" in filename)
                and "task-rest" in filename
                and "dir-forward" in filename
                and filename.endswith("_bold.json")
            ):
                forward_jsons.append(file)

        result["n_forward_jsons"] = len(forward_jsons)
        result["protocol_hint"] = detect_protocol_hint(len(forward_jsons))

        tr_values = []
        for file in forward_jsons:
            response = session.get(file["urls"][0], timeout=60)
            response.raise_for_status()
            payload = response.json()
            value = payload.get("RepetitionTime")
            if value is None:
                continue
            tr_values.append(float(value))

        # I keep all unique TR values so I can spot inconsistent protocol metadata early.
        unique_trs = sorted(set(tr_values))
        result["unique_tr_values"] = ",".join(f"{{value:g}}" for value in unique_trs)
        if len(unique_trs) == 1:
            result["tr_seconds"] = f"{{unique_trs[0]:g}}"
        elif len(unique_trs) > 1:
            result["tr_seconds"] = "inconsistent"
    except Exception as exc:
        result["query_failed"] = 1
        result["query_error"] = str(exc).replace("\\t", " ").replace("\\n", " ")

    print(
        "\\t".join(
            str(result[key]) for key in (
                "subject_id",
                "tr_seconds",
                "unique_tr_values",
                "n_forward_jsons",
                "protocol_hint",
                "query_failed",
                "query_error",
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
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "TR query failed")

    results: dict[str, dict[str, object]] = {}
    for line in proc.stdout.splitlines():
        if not line or line.startswith(("👋", "👉", "🌍", "📁", "🍪")):
            continue
        parts = line.split("\t")
        if len(parts) != 7:
            continue
        (
            subject_id,
            tr_seconds,
            unique_tr_values,
            n_forward_jsons,
            protocol_hint,
            query_failed,
            query_error,
        ) = parts
        results[subject_id] = {
            "tr_seconds": tr_seconds,
            "unique_tr_values": unique_tr_values,
            "n_forward_jsons": int(n_forward_jsons),
            "protocol_hint": protocol_hint,
            "tr_query_failed": int(query_failed),
            "tr_query_error": query_error,
        }

    missing = sorted(set(subjects).difference(results))
    if missing:
        raise RuntimeError(
            "No TR metadata row parsed for: " + ", ".join(missing)
        )
    return results


def annotate_rows(
    rows: list[dict[str, str]],
    key: str,
    tr_results: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for row in rows:
        subject = row[key]
        out = dict(row)
        out.update(tr_results[subject])
        annotated.append(out)
    return annotated


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_path = (
        resolve_project_path(args.output)
        if args.output
        else default_output_path(input_path)
    )

    fieldnames, rows, key = load_rows(input_path)
    subjects = [row[key] for row in rows]
    # This keeps the TR annotation tied to the exact subject list that came from the TSV.
    tr_results = query_remote_tr(args.dataset, subjects)

    failures = sorted(
        subject for subject, result in tr_results.items() if int(result["tr_query_failed"]) == 1
    )
    if failures and not args.allow_partial_results:
        raise RuntimeError(
            "TR query failed for: "
            + ", ".join(failures)
            + ". Re-run with --allow-partial-results if you still want an output TSV."
        )

    annotated_rows = annotate_rows(rows, key, tr_results)
    extra_fields = [
        "tr_seconds",
        "unique_tr_values",
        "n_forward_jsons",
        "protocol_hint",
        "tr_query_failed",
        "tr_query_error",
    ]
    output_fields = fieldnames + [field for field in extra_fields if field not in fieldnames]
    write_rows(output_path, output_fields, annotated_rows)

    print(f"Wrote TR-annotated TSV to {output_path}")
    for tr_label in sorted(
        {row["tr_seconds"] for row in annotated_rows if str(row["tr_seconds"]).strip()}
    ):
        count = sum(1 for row in annotated_rows if row["tr_seconds"] == tr_label)
        print(f"TR {tr_label}: {count} subjects")


if __name__ == "__main__":
    main()
