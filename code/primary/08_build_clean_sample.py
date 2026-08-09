#!/usr/bin/env python3

# What this script does:
#   Combines automated QC, manual HTML review, and output existence checks to
#   decide which subjects pass into the QC-pass sample.
# How to run it:
#   Run from the repo root with:
#   python code/primary/08_build_clean_sample.py
# Main outputs:
#   data/processed/screening/ds005752_qc_pass_sample.tsv
#   data/processed/screening/ds005752_qc_pass_sample_decisions.tsv

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANUAL_QC = "data/processed/qc/manual_fmriprep_report_review.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the QC-pass younger/older sample after download, "
            "fMRIPrep, automated QC, and manual report review."
        )
    )
    parser.add_argument(
        "--young-input",
        default="data/processed/screening/ds005752_mri_participants_age_20_25_remote_anat_forward.tsv",
        help="Remote-screened younger candidate table.",
    )
    parser.add_argument(
        "--older-input",
        default="data/processed/screening/ds005752_mri_participants_age_50_75_remote_anat_forward.tsv",
        help="Remote-screened older candidate table.",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/ds005752",
        help="Raw BIDS dataset directory.",
    )
    parser.add_argument(
        "--derivatives-dir",
        default="data/derivatives/fmriprep",
        help="fMRIPrep derivatives directory.",
    )
    parser.add_argument(
        "--qc-dir",
        default="data/processed/qc",
        help="Directory containing subject-level QC summary TSV files.",
    )
    parser.add_argument(
        "--mean-fd-max",
        type=float,
        default=0.2,
        help="Maximum allowed mean FD for the forward resting-state run.",
    )
    parser.add_argument(
        "--pct-fd-0p2-max",
        type=float,
        default=25.0,
        help="Maximum allowed percent of volumes with FD > 0.2 mm.",
    )
    parser.add_argument(
        "--min-volumes",
        type=int,
        default=200,
        help="Minimum required number of volumes in the forward resting-state run.",
    )
    parser.add_argument(
        "--min-retained-volumes",
        type=int,
        default=180,
        help=(
            "Legacy retained-volume threshold. This is ignored whenever "
            "--min-retained-minutes is set."
        ),
    )
    parser.add_argument(
        "--min-retained-minutes",
        type=float,
        default=9.0,
        help=(
            "Minimum retained time in minutes after censoring. This is the "
            "primary current QC rule and overrides --min-retained-volumes."
        ),
    )
    parser.add_argument(
        "--manual-qc",
        default=DEFAULT_MANUAL_QC,
        help=(
            "Manual fMRIPrep report-review TSV produced by "
            "code/primary/06_prepare_manual_qc_review.py."
        ),
    )
    parser.add_argument(
        "--require-manual-qc",
        type=int,
        choices=(0, 1),
        default=1,
        help="Require a manual fMRIPrep report review pass before inclusion.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/screening/ds005752_qc_pass_sample.tsv",
        help="Output TSV of included QC-pass subjects.",
    )
    parser.add_argument(
        "--decisions-output",
        default="data/processed/screening/ds005752_qc_pass_sample_decisions.tsv",
        help="Output TSV with inclusion and exclusion decisions for all candidates.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_forward_qc(qc_dir: Path, subject: str) -> dict[str, str] | None:
    qc_path = qc_dir / f"{subject}_qc_summary.tsv"
    if not qc_path.exists():
        return None
    with qc_path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows:
        if row["run_label"].endswith("task-rest_dir-forward"):
            return row
    return None


def load_manual_qc_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {row["subject_id"]: row for row in reader}


def load_repetition_time(path: Path) -> float | None:
    with path.open() as f:
        metadata = json.load(f)
    value = metadata.get("RepetitionTime")
    return float(value) if value is not None else None


def normalize_manual_qc_status(value: str | None) -> str:
    text = (value or "").strip().lower()
    if text in {"1", "pass", "passed", "true", "yes"}:
        return "pass"
    if text in {"0", "fail", "failed", "false", "no"}:
        return "fail"
    return "pending"


def write_rows(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    young_input = resolve_project_path(args.young_input)
    older_input = resolve_project_path(args.older_input)
    raw_dir = resolve_project_path(args.raw_dir)
    derivatives_dir = resolve_project_path(args.derivatives_dir)
    qc_dir = resolve_project_path(args.qc_dir)
    manual_qc_path = resolve_project_path(args.manual_qc)
    output_path = resolve_project_path(args.output)
    decisions_output = resolve_project_path(args.decisions_output)

    if args.require_manual_qc and not manual_qc_path.exists():
        raise FileNotFoundError(
            "Manual QC review file is missing. Run code/primary/06_prepare_manual_qc_review.py "
            "and complete the TSV after checking each fMRIPrep HTML report."
        )

    young_rows = load_rows(young_input)
    older_rows = load_rows(older_input)
    candidates = [("young", row) for row in young_rows] + [
        ("older", row) for row in older_rows
    ]
    manual_qc_rows = (
        load_manual_qc_rows(manual_qc_path) if manual_qc_path.exists() else {}
    )

    decisions: list[dict[str, object]] = []
    included: list[dict[str, object]] = []

    for age_group, row in candidates:
        subject = row["participant_id"]
        func_dir = derivatives_dir / subject / "ses-01" / "func"
        raw_subject_dir = raw_dir / subject
        bold_path = next(
            func_dir.glob(
                f"{subject}_ses-01_task-rest_dir-forward_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
            ),
            None,
        )
        confounds_path = next(
            func_dir.glob(
                f"{subject}_ses-01_task-rest_dir-forward_desc-confounds_timeseries.tsv"
            ),
            None,
        )
        json_path = next(
            func_dir.glob(
                f"{subject}_ses-01_task-rest_dir-forward_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.json"
            ),
            None,
        )
        html_path = derivatives_dir / f"{subject}.html"
        qc_row = load_forward_qc(qc_dir, subject)
        manual_qc_row = manual_qc_rows.get(subject, {})
        manual_qc_status = normalize_manual_qc_status(manual_qc_row.get("review_status"))

        has_required_outputs = all(
            path is not None and path.exists()
            for path in (bold_path, confounds_path, json_path)
        ) and html_path.exists()
        has_raw_subject = raw_subject_dir.exists()
        mean_fd = float(qc_row["mean_fd"]) if qc_row and qc_row.get("mean_fd") else None
        pct_fd_0p2 = (
            float(qc_row["pct_fd_gt_0p2"]) if qc_row and qc_row.get("pct_fd_gt_0p2") else None
        )
        n_volumes = int(float(qc_row["n_volumes"])) if qc_row and qc_row.get("n_volumes") else None
        retained_volumes = (
            int(float(qc_row["n_volumes_retained_after_scrub"]))
            if qc_row and qc_row.get("n_volumes_retained_after_scrub")
            else None
        )
        retained_minutes = (
            float(qc_row["retained_minutes_after_scrub"])
            if qc_row and qc_row.get("retained_minutes_after_scrub")
            else None
        )
        repetition_time = load_repetition_time(json_path) if json_path and json_path.exists() else None

        include = True
        exclusion_reasons: list[str] = []

        if not has_raw_subject:
            include = False
            exclusion_reasons.append("missing_raw_subject")
        if not has_required_outputs:
            include = False
            exclusion_reasons.append("missing_required_outputs")
        if qc_row is None:
            include = False
            exclusion_reasons.append("missing_forward_qc_row")
        if manual_qc_status != "pass" and args.require_manual_qc:
            include = False
            exclusion_reasons.append(f"manual_qc_{manual_qc_status}")
        if n_volumes is None or n_volumes < args.min_volumes:
            include = False
            exclusion_reasons.append("too_few_volumes")
        if mean_fd is None or mean_fd >= args.mean_fd_max:
            include = False
            exclusion_reasons.append("mean_fd_too_high")
        if pct_fd_0p2 is None or pct_fd_0p2 >= args.pct_fd_0p2_max:
            include = False
            exclusion_reasons.append("pct_fd_gt_0p2_too_high")

        if args.min_retained_minutes is not None:
            if retained_minutes is None or retained_minutes < args.min_retained_minutes:
                include = False
                exclusion_reasons.append("retained_minutes_too_low")
        elif retained_volumes is None or retained_volumes < args.min_retained_volumes:
            include = False
            exclusion_reasons.append("retained_volumes_too_low")

        decision = {
            "subject_id": subject,
            "age_group": age_group,
            "age": row.get("age", ""),
            "sex": row.get("sex", ""),
            "include": int(include),
            "exclusion_reasons": ";".join(exclusion_reasons),
            "manual_qc_status": manual_qc_status,
            "has_raw_subject": int(has_raw_subject),
            "has_required_outputs": int(has_required_outputs),
            "n_volumes": n_volumes if n_volumes is not None else "",
            "retained_volumes_after_scrub": retained_volumes if retained_volumes is not None else "",
            "retained_minutes_after_scrub": retained_minutes if retained_minutes is not None else "",
            "mean_fd": mean_fd if mean_fd is not None else "",
            "pct_fd_gt_0p2": pct_fd_0p2 if pct_fd_0p2 is not None else "",
            "repetition_time": repetition_time if repetition_time is not None else "",
        }
        decisions.append(decision)

        if include:
            included.append(
                {
                    "subject_id": subject,
                    "age_group": age_group,
                    "age": row.get("age", ""),
                    "sex": row.get("sex", ""),
                    "mean_fd": mean_fd,
                    "pct_fd_gt_0p2": pct_fd_0p2,
                    "n_volumes": n_volumes,
                    "retained_volumes_after_scrub": retained_volumes,
                    "retained_minutes_after_scrub": retained_minutes,
                    "repetition_time": repetition_time,
                }
            )

    decision_fieldnames = list(decisions[0].keys()) if decisions else []
    included_fieldnames = list(included[0].keys()) if included else []

    write_rows(output_path, included_fieldnames, included)
    write_rows(decisions_output, decision_fieldnames, decisions)

    print(f"Wrote QC-pass sample with {len(included)} subjects to {output_path}")
    print(f"Wrote decision log for {len(decisions)} candidates to {decisions_output}")


if __name__ == "__main__":
    main()
