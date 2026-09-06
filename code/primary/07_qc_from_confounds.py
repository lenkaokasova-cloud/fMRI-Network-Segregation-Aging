#!/usr/bin/env python3

# What this script does:
#   Reads the fMRIPrep confounds file for a subject and turns it into a simple
#   QC summary with motion and retained-data measures.
# How to run it:
#   Run from the repo root with:
#   python code/primary/07_qc_from_confounds.py --subject sub-ON01016
# Main output:
#   data/processed/qc/sub-*_qc_summary.tsv

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRUB_FD_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize fMRIPrep confounds files into simple QC metrics."
    )
    parser.add_argument(
        "--derivatives-dir",
        default="data/derivatives/fmriprep",
        help="Path to the fMRIPrep derivatives directory.",
    )
    parser.add_argument(
        "--subject",
        required=True,
        help="BIDS subject ID to summarize, for example sub-ON01016.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output TSV path. Defaults to data/processed/qc/<subject>_qc_summary.tsv.",
    )
    parser.add_argument(
        "--scrub-fd-threshold",
        type=float,
        default=DEFAULT_SCRUB_FD_THRESHOLD,
        help=(
            "Framewise-displacement threshold used to estimate how many volumes would "
            "remain after frame censoring."
        ),
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_float(value: str) -> float | None:
    if value in ("", "n/a", "NaN", None):
        return None
    number = float(value)
    if math.isnan(number):
        return None
    return number


def summarize_confounds(
    confounds_file: Path, scrub_fd_threshold: float
) -> dict[str, object]:
    with confounds_file.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {confounds_file}")

    fd = [parse_float(row.get("framewise_displacement", "")) for row in rows]
    dvars = [parse_float(row.get("dvars", "")) for row in rows]
    std_dvars = [parse_float(row.get("std_dvars", "")) for row in rows]

    fd_valid = [x for x in fd if x is not None]
    dvars_valid = [x for x in dvars if x is not None]
    std_dvars_valid = [x for x in std_dvars if x is not None]

    nonsteady_columns = [
        name for name in rows[0].keys() if name.startswith("non_steady_state_outlier")
    ]
    nonsteady_count = 0
    scrubbed_count = 0
    for row in rows:
        flagged = any(row.get(col, "") == "1" for col in nonsteady_columns)
        if flagged:
            nonsteady_count += 1
        fd_value = parse_float(row.get("framewise_displacement", ""))
        if flagged or (fd_value is not None and fd_value > scrub_fd_threshold):
            scrubbed_count += 1

    retained_count = len(rows) - scrubbed_count

    return {
        "subject_id": confounds_file.name.split("_ses-")[0],
        "session": "ses-" + confounds_file.name.split("_ses-")[1].split("_")[0],
        "run_label": confounds_file.name.replace("_desc-confounds_timeseries.tsv", ""),
        "n_volumes": len(rows),
        "n_nonsteady_volumes": nonsteady_count,
        "mean_fd": round(mean(fd_valid), 6) if fd_valid else "",
        "median_fd": round(median(fd_valid), 6) if fd_valid else "",
        "max_fd": round(max(fd_valid), 6) if fd_valid else "",
        "n_fd_gt_0p2": sum(x > 0.2 for x in fd_valid),
        "pct_fd_gt_0p2": (
            round(100 * sum(x > 0.2 for x in fd_valid) / len(fd_valid), 2)
            if fd_valid
            else ""
        ),
        "n_fd_gt_0p5": sum(x > 0.5 for x in fd_valid),
        "pct_fd_gt_0p5": (
            round(100 * sum(x > 0.5 for x in fd_valid) / len(fd_valid), 2)
            if fd_valid
            else ""
        ),
        "scrub_fd_threshold": round(scrub_fd_threshold, 3),
        "n_scrubbed": scrubbed_count,
        "pct_scrubbed": round(100 * scrubbed_count / len(rows), 2),
        "n_retained_after_scrub": retained_count,
        "pct_retained_after_scrub": round(100 * retained_count / len(rows), 2),
        "mean_dvars": round(mean(dvars_valid), 6) if dvars_valid else "",
        "mean_std_dvars": round(mean(std_dvars_valid), 6) if std_dvars_valid else "",
    }


def main() -> None:
    args = parse_args()
    derivatives_dir = resolve_project_path(args.derivatives_dir)
    func_dir = derivatives_dir / args.subject / "ses-01" / "func"
    confounds_files = sorted(func_dir.glob("*desc-confounds_timeseries.tsv"))

    if not confounds_files:
        raise FileNotFoundError(f"No confounds files found in {func_dir}")

    summaries = [
        summarize_confounds(path, args.scrub_fd_threshold) for path in confounds_files
    ]

    output_path = (
        Path(args.output)
        if args.output
        else resolve_project_path(f"data/processed/qc/{args.subject}_qc_summary.tsv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(summaries[0].keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote QC summary to {output_path}")
    for summary in summaries:
        print(
            f"{summary['run_label']}: mean FD={summary['mean_fd']}, "
            f"FD>0.2mm={summary['n_fd_gt_0p2']}, FD>0.5mm={summary['n_fd_gt_0p5']}, "
            f"retained={summary['n_retained_after_scrub']}"
        )


if __name__ == "__main__":
    main()
