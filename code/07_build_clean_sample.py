#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUAL_QC = "data/processed/qc/manual_fmriprep_report_review.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the final clean younger/older sample after download, "
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
        default=0.25,
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
        default=180,
        help="Minimum required number of volumes in the forward resting-state run.",
    )
    parser.add_argument(
        "--min-retained-volumes",
        type=int,
        default=150,
        help=(
            "Minimum required number of volumes remaining after censoring "
            "nonsteady volumes and FD spikes."
        ),
    )
    parser.add_argument(
        "--manual-qc",
        default=DEFAULT_MANUAL_QC,
        help=(
            "Manual fMRIPrep report-review TSV produced by "
            "code/06_prepare_manual_qc_review.py."
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
        default="data/processed/screening/ds005752_clean_age_sample.tsv",
        help="Output TSV of included clean subjects.",
    )
    parser.add_argument(
        "--decisions-output",
        default="data/processed/screening/ds005752_clean_age_sample_decisions.tsv",
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


def normalize_manual_qc_status(value: str | None) -> str:
    text = (value or "").strip().lower()
    if text in {"1", "pass", "passed", "true", "yes"}:
        return "pass"
    if text in {"0", "fail", "failed", "false", "no"}:
        return "fail"
    return "pending"


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
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
            "Manual QC review file is missing. Run code/06_prepare_manual_qc_review.py "
            "and complete the TSV after checking each fMRIPrep HTML report."
        )

    young_rows = load_rows(young_input)
    older_rows = load_rows(older_input)
    candidates = [("young", row) for row in young_rows] + [("older", row) for row in older_rows]
    manual_qc_rows = load_manual_qc_rows(manual_qc_path) if manual_qc_path.exists() else {}

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
            func_dir.glob(f"{subject}_ses-01_task-rest_dir-forward_desc-confounds_timeseries.tsv"),
            None,
        )
        json_path = next(
            func_dir.glob(
                f"{subject}_ses-01_task-rest_dir-forward_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.json"
            ),
            None,
        )
        report_path = derivatives_dir / f"{subject}.html"
        qc_row = load_forward_qc(qc_dir, subject)
        manual_qc_row = manual_qc_rows.get(subject, {})
        manual_qc_status = normalize_manual_qc_status(manual_qc_row.get("manual_qc_status"))

        decision = "include"
        reason = "Accepted for connectivity analysis."
        n_volumes = ""
        mean_fd = ""
        pct_fd = ""
        n_retained_after_scrub = ""
        pct_retained_after_scrub = ""

        if not raw_subject_dir.exists():
            decision = "exclude"
            reason = "Raw data were not downloaded."
        elif not report_path.exists():
            decision = "exclude"
            reason = "fMRIPrep HTML report is missing."
        elif bold_path is None or confounds_path is None or json_path is None:
            decision = "exclude"
            reason = "Required forward-run fMRIPrep outputs are missing."
        elif qc_row is None:
            decision = "exclude"
            reason = "Forward-run QC summary is missing."
        elif args.require_manual_qc and manual_qc_status != "pass":
            decision = "exclude"
            reason = (
                "Manual fMRIPrep report review is not marked pass."
                if manual_qc_status == "fail"
                else "Manual fMRIPrep report review is still pending."
            )
        else:
            if "n_retained_after_scrub" not in qc_row:
                decision = "exclude"
                reason = (
                    "QC summary predates the current pipeline. Rerun "
                    "code/06_qc_from_confounds.py for this subject."
                )
            else:
                n_volumes = int(qc_row["n_volumes"])
                mean_fd = float(qc_row["mean_fd"])
                pct_fd = float(qc_row["pct_fd_gt_0p2"])
                n_retained_after_scrub = int(qc_row["n_retained_after_scrub"])
                pct_retained_after_scrub = float(qc_row["pct_retained_after_scrub"])
                if n_volumes < args.min_volumes:
                    decision = "exclude"
                    reason = f"Forward run has only {n_volumes} volumes (< {args.min_volumes})."
                elif n_retained_after_scrub < args.min_retained_volumes:
                    decision = "exclude"
                    reason = (
                        f"Only {n_retained_after_scrub} volumes remain after censoring "
                        f"(< {args.min_retained_volumes})."
                    )
                elif mean_fd >= args.mean_fd_max:
                    decision = "exclude"
                    reason = (
                        f"Mean FD {mean_fd:.3f} mm exceeded the threshold of "
                        f"{args.mean_fd_max:.3f} mm."
                    )
                elif pct_fd >= args.pct_fd_0p2_max:
                    decision = "exclude"
                    reason = (
                        f"{pct_fd:.2f}% of volumes exceeded FD > 0.2 mm, above the "
                        f"{args.pct_fd_0p2_max:.2f}% threshold."
                    )

        decision_row = {
            "subject_id": subject,
            "age": int(row["age"]),
            "sex": row["sex"],
            "age_group": age_group,
            "handedness": row["handedness"],
            "release_1": row["release_1"],
            "release_2": row["release_2"],
            "has_remote_anat": int(row.get("has_anat", 0)),
            "has_remote_rest_forward": int(row.get("has_rest_forward", 0)),
            "raw_downloaded": int(raw_subject_dir.exists()),
            "report_exists": int(report_path.exists()),
            "forward_bold_exists": int(bold_path is not None),
            "forward_confounds_exists": int(confounds_path is not None),
            "forward_json_exists": int(json_path is not None),
            "n_volumes": n_volumes,
            "n_retained_after_scrub": n_retained_after_scrub,
            "pct_retained_after_scrub": pct_retained_after_scrub,
            "mean_fd": mean_fd,
            "pct_fd_gt_0p2": pct_fd,
            "manual_qc_status": manual_qc_status,
            "decision": decision,
            "decision_reason": reason,
        }
        decisions.append(decision_row)
        if decision == "include":
            included.append(decision_row)

    fieldnames = list(decisions[0].keys()) if decisions else []
    write_rows(decisions_output, fieldnames, decisions)
    write_rows(output_path, fieldnames, included)

    print(f"Wrote clean-sample decision table to {decisions_output}")
    print(f"Wrote {len(included)} included subjects to {output_path}")


if __name__ == "__main__":
    main()
