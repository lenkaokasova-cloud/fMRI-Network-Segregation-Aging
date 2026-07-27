#!/usr/bin/env python3

# What this script does:
#   Runs the study-specific parcel-level denoising step for the final analysis
#   sample.
#   For each subject, it loads the preprocessed forward resting-state BOLD run,
#   the matching confounds TSV, and the JSON metadata from fMRIPrep. It then
#   applies the dissertation denoising pipeline: censoring non-steady-state and
#   high-motion volumes, nuisance regression, temporal filtering, and Schaefer
#   parcel time-series extraction in MNI space.
# How to run it:
#   Run from the repo root with:
#   python code/primary/10_run_denoising.py
# Main outputs:
#   data/processed/denoising/timeseries/
#   data/processed/denoising/metrics/
#   In practice, this writes one denoised parcel time-series file per subject
#   in timeseries/, plus atlas_labels.tsv, subject_denoising_summary.tsv, and
#   denoising_settings.tsv in metrics/.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import nilearn
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRUB_FD_THRESHOLD = 0.5
DEFAULT_CONFOUND_COLUMNS = [
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "trans_x_derivative1",
    "trans_y_derivative1",
    "trans_z_derivative1",
    "rot_x_derivative1",
    "rot_y_derivative1",
    "rot_z_derivative1",
    "white_matter",
    "csf",
]
GLOBAL_SIGNAL_COLUMNS = ["global_signal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the dissertation denoising pipeline to fMRIPrep forward resting-state "
            "runs and save denoised Schaefer parcel time series."
        )
    )
    parser.add_argument(
        "subjects",
        nargs="*",
        help="Optional subject IDs to denoise. Defaults to subjects in --sample.",
    )
    parser.add_argument(
        "--sample",
        default="data/processed/screening/ds005752_final_analysis_sample_tr_3s.tsv",
        help="Analysis sample TSV used to choose subjects and carry subject metadata.",
    )
    parser.add_argument(
        "--derivatives-dir",
        default="data/derivatives/fmriprep",
        help="Path to fMRIPrep derivatives.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/denoising",
        help="Directory for denoised time series and denoising summaries.",
    )
    parser.add_argument(
        "--atlas-data-dir",
        default="data/external/nilearn_data",
        help="Writable directory where Nilearn atlas files should be cached.",
    )
    parser.add_argument("--n-rois", type=int, default=200, help="Number of Schaefer parcels.")
    parser.add_argument(
        "--yeo-networks", type=int, default=7, help="Number of Yeo networks for the atlas."
    )
    parser.add_argument("--high-pass", type=float, default=0.008, help="High-pass filter in Hz.")
    parser.add_argument("--low-pass", type=float, default=0.09, help="Low-pass filter in Hz.")
    parser.add_argument(
        "--scrub-fd-threshold",
        type=float,
        default=DEFAULT_SCRUB_FD_THRESHOLD,
        help="Censor volumes with framewise displacement above this threshold.",
    )
    parser.add_argument(
        "--min-retained-minutes",
        type=float,
        default=7.5,
        help="Fail if fewer than this many minutes remain after censoring.",
    )
    parser.add_argument(
        "--dvars-threshold",
        type=float,
        default=None,
        help=(
            "Optional standardized DVARS threshold for an additional censoring "
            "sensitivity branch. If omitted, DVARS is not used for censoring."
        ),
    )
    parser.add_argument(
        "--fd-backward-neighbors",
        type=int,
        default=0,
        help="Also censor this many volumes before each flagged motion spike.",
    )
    parser.add_argument(
        "--fd-forward-neighbors",
        type=int,
        default=0,
        help="Also censor this many volumes after each flagged motion spike.",
    )
    parser.add_argument(
        "--include-global-signal",
        action="store_true",
        help=(
            "Include the fMRIPrep global_signal confound as a sensitivity-analysis "
            "version of denoising."
        ),
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_sample(sample_path: Path) -> pd.DataFrame:
    sample = pd.read_csv(sample_path, sep="\t")
    required = {"subject_id", "age", "sex", "age_group", "mean_fd", "pct_fd_gt_0p2", "n_volumes"}
    missing = required.difference(sample.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Sample file is missing required columns: {missing_str}")
    return sample


def get_forward_run_paths(derivatives_dir: Path, subject: str) -> tuple[Path, Path, Path, Path]:
    func_dir = derivatives_dir / subject / "ses-01" / "func"
    bold = next(
        func_dir.glob(
            f"{subject}_ses-01_task-rest_dir-forward_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
        )
    )
    confounds = next(
        func_dir.glob(f"{subject}_ses-01_task-rest_dir-forward_desc-confounds_timeseries.tsv")
    )
    bold_json = next(
        func_dir.glob(
            f"{subject}_ses-01_task-rest_dir-forward_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.json"
        )
    )
    confounds_json = next(
        func_dir.glob(f"{subject}_ses-01_task-rest_dir-forward_desc-confounds_timeseries.json")
    )
    return bold, confounds, bold_json, confounds_json


def get_tr(bold_json: Path) -> float:
    with bold_json.open() as f:
        metadata = json.load(f)
    if "RepetitionTime" not in metadata:
        raise KeyError(f"RepetitionTime missing from {bold_json}")
    return float(metadata["RepetitionTime"])


def load_confounds_table(confounds_path: Path) -> pd.DataFrame:
    return pd.read_csv(confounds_path, sep="\t")


def load_confounds_metadata(confounds_json: Path) -> dict[str, object]:
    with confounds_json.open() as f:
        return json.load(f)


def acompcor_summary(confounds_metadata: dict[str, object]) -> dict[str, object]:
    keys = sorted(
        [key for key in confounds_metadata if key.startswith("a_comp_cor_")]
    )[:6]
    if not keys:
        return {
            "acompcor_components_used": 0,
            "acompcor_first6_cumulative_variance_explained": float("nan"),
            "acompcor_first6_mean_variance_explained": float("nan"),
            "acompcor_mask": "",
        }

    entries = [confounds_metadata[key] for key in keys]
    cumulative = entries[-1].get("CumulativeVarianceExplained", float("nan"))
    mean_var = float(
        np.nanmean([entry.get("VarianceExplained", float("nan")) for entry in entries])
    )
    masks = sorted({str(entry.get("Mask", "")) for entry in entries if entry.get("Mask") is not None})
    return {
        "acompcor_components_used": len(keys),
        "acompcor_first6_cumulative_variance_explained": float(cumulative),
        "acompcor_first6_mean_variance_explained": mean_var,
        "acompcor_mask": ",".join(masks),
    }


def expand_spike_mask(spike_mask: np.ndarray, backward: int, forward: int) -> np.ndarray:
    if backward <= 0 and forward <= 0:
        return spike_mask.copy()
    expanded = spike_mask.copy()
    n_volumes = len(spike_mask)
    for idx in np.flatnonzero(spike_mask):
        start = max(0, idx - backward)
        stop = min(n_volumes, idx + forward + 1)
        expanded[start:stop] = True
    return expanded


def build_sample_mask(
    confounds: pd.DataFrame,
    scrub_fd_threshold: float,
    dvars_threshold: float | None,
    fd_backward_neighbors: int,
    fd_forward_neighbors: int,
) -> tuple[np.ndarray, dict[str, object]]:
    # This turns the confounds table into the keep-mask that decides which volumes survive cleaning.
    keep_mask = np.ones(len(confounds), dtype=bool)
    dvars_column = "std_dvars" if "std_dvars" in confounds.columns else "dvars" if "dvars" in confounds.columns else ""
    fd_spike_mask = np.zeros(len(confounds), dtype=bool)
    dvars_spike_mask = np.zeros(len(confounds), dtype=bool)

    nonsteady_columns = [
        column for column in confounds.columns if column.startswith("non_steady_state_outlier")
    ]
    if nonsteady_columns:
        nonsteady_mask = confounds[nonsteady_columns].fillna(0.0).eq(0.0).all(axis=1).to_numpy()
        keep_mask &= nonsteady_mask
        censored_nonsteady = int((~nonsteady_mask).sum())
    else:
        censored_nonsteady = 0

    if "framewise_displacement" in confounds.columns:
        fd = pd.to_numeric(confounds["framewise_displacement"], errors="coerce")
        fd_spike_mask = fd.gt(scrub_fd_threshold).fillna(False).to_numpy()

    if dvars_threshold is not None and dvars_column:
        dvars = pd.to_numeric(confounds[dvars_column], errors="coerce")
        dvars_spike_mask = dvars.gt(dvars_threshold).fillna(False).to_numpy()

    spike_mask = fd_spike_mask | dvars_spike_mask
    expanded_spike_mask = expand_spike_mask(
        spike_mask,
        backward=fd_backward_neighbors,
        forward=fd_forward_neighbors,
    )
    keep_mask &= ~expanded_spike_mask

    summary = {
        "dvars_column_used": dvars_column,
        "censored_nonsteady_state_volumes": censored_nonsteady,
        "censored_fd_spike_volumes": int(fd_spike_mask.sum()),
        "censored_dvars_spike_volumes": int(dvars_spike_mask.sum()),
        "censored_neighbor_expansion_volumes": int((expanded_spike_mask & ~spike_mask).sum()),
        "censored_total_volumes": int((~keep_mask).sum()),
    }

    return np.flatnonzero(keep_mask), summary


def pick_confounds(confounds: pd.DataFrame, *, include_global_signal: bool) -> pd.DataFrame:
    # These are the nuisance regressors that actually go into the cleaning step.
    columns = [column for column in DEFAULT_CONFOUND_COLUMNS if column in confounds.columns]
    if include_global_signal:
        columns.extend([column for column in GLOBAL_SIGNAL_COLUMNS if column in confounds.columns])
    acompcor = [column for column in confounds.columns if column.startswith("a_comp_cor_")][:6]
    columns.extend(acompcor)
    selected = confounds[columns].copy()
    return selected.fillna(0.0)


def decode_label(label: object) -> str:
    if isinstance(label, bytes):
        return label.decode("utf-8")
    return str(label)


def parse_network_name(label: str) -> str:
    parts = label.split("_")
    if parts and parts[0].endswith("Networks") and len(parts) > 2:
        return parts[2]
    if len(parts) > 1:
        return parts[1]
    return label


def relative_project_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    sample_path = resolve_project_path(args.sample)
    derivatives_dir = resolve_project_path(args.derivatives_dir)
    output_dir = resolve_project_path(args.output_dir)
    atlas_data_dir = resolve_project_path(args.atlas_data_dir)
    timeseries_dir = output_dir / "timeseries"
    metrics_dir = output_dir / "metrics"
    for directory in (output_dir, timeseries_dir, metrics_dir, atlas_data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    sample = load_sample(sample_path)
    if args.subjects:
        sample = sample[sample["subject_id"].isin(args.subjects)].copy()
    if sample.empty:
        raise ValueError("No subjects remain to denoise after applying the requested filter.")

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=args.n_rois,
        yeo_networks=args.yeo_networks,
        resolution_mm=2,
        data_dir=str(atlas_data_dir),
    )
    # I save the atlas labels here as well so every downstream step knows exactly which parcel is which.
    atlas_labels = [decode_label(label) for label in atlas.labels]
    if len(atlas_labels) == args.n_rois + 1 and atlas_labels[0].lower() == "background":
        atlas_labels = atlas_labels[1:]

    pd.DataFrame(
        {
            "parcel_index": list(range(len(atlas_labels))),
            "parcel_label": atlas_labels,
            "network": [parse_network_name(label) for label in atlas_labels],
        }
    ).to_csv(metrics_dir / "atlas_labels.tsv", sep="\t", index=False)

    subject_rows: list[dict[str, object]] = []
    for row in sample.sort_values(["age_group", "age", "subject_id"]).itertuples(index=False):
        subject = row.subject_id
        bold_path, confounds_path, bold_json, confounds_json = get_forward_run_paths(
            derivatives_dir, subject
        )
        tr = get_tr(bold_json)
        confounds_table = load_confounds_table(confounds_path)
        confounds_metadata = load_confounds_metadata(confounds_json)
        n_volumes = int(len(confounds_table))
        sample_mask, censor_summary = build_sample_mask(
            confounds_table,
            args.scrub_fd_threshold,
            args.dvars_threshold,
            args.fd_backward_neighbors,
            args.fd_forward_neighbors,
        )
        acompcor = acompcor_summary(confounds_metadata)
        retained_volumes = int(len(sample_mask))
        retained_minutes = float(retained_volumes * tr / 60.0)
        pct_retained = float(100 * retained_volumes / n_volumes)
        if retained_minutes < args.min_retained_minutes:
            raise ValueError(
                f"{subject} retained only {retained_minutes:.2f} minutes after censoring, "
                f"below the required minimum of {args.min_retained_minutes:.2f} minutes."
            )

        confounds = pick_confounds(
            confounds_table,
            include_global_signal=args.include_global_signal,
        )
        masker = NiftiLabelsMasker(
            labels_img=atlas.maps,
            standardize="zscore_sample",
            detrend=True,
            low_pass=args.low_pass,
            high_pass=args.high_pass,
            t_r=tr,
            standardize_confounds=True,
        )
        # Nilearn does the parcel extraction and the cleaning together here, using the subject's own TR.
        time_series = masker.fit_transform(
            str(bold_path),
            confounds=confounds,
            sample_mask=sample_mask,
        )

        timeseries_path = timeseries_dir / f"{subject}_forward_timeseries.npy"
        np.save(timeseries_path, time_series)

        subject_rows.append(
            {
                "subject_id": subject,
                "age": int(row.age),
                "sex": row.sex,
                "age_group": row.age_group,
                "n_volumes": n_volumes,
                "retained_volumes": retained_volumes,
                "retained_minutes_after_scrub": round(retained_minutes, 2),
                "pct_retained_after_scrub": round(pct_retained, 2),
                "mean_fd": float(row.mean_fd),
                "pct_fd_gt_0p2": float(row.pct_fd_gt_0p2),
                "repetition_time_seconds": round(tr, 3),
                **acompcor,
                **censor_summary,
                "timeseries_file": relative_project_path(timeseries_path),
            }
        )

    pd.DataFrame(subject_rows).to_csv(
        metrics_dir / "subject_denoising_summary.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        [
            {
                "sample": relative_project_path(sample_path),
                "n_rois": args.n_rois,
                "yeo_networks": args.yeo_networks,
                "atlas_data_dir": relative_project_path(atlas_data_dir),
                "nilearn_version": nilearn.__version__,
                "cleaning_library": "nilearn.maskers.NiftiLabelsMasker",
                "cleaning_backend": "nilearn.signal.clean",
                "sample_mask_application": (
                    "sample_mask passed to nilearn.signal.clean; with Butterworth "
                    "filtering, flagged volumes are spline-interpolated for filtering "
                    "and censored before final confound regression/standardization"
                ),
                "cleaning_order_after_mask": (
                    "sample_mask,spline_interpolation_for_flagged_volumes,detrend,"
                    "butterworth_filter,censor,remove_confounds,standardize"
                ),
                "filter_confound_orthogonalization": (
                    "nilearn.signal.clean performs confound removal orthogonally "
                    "to temporal filters when both are specified"
                ),
                "high_pass_hz": args.high_pass,
                "low_pass_hz": args.low_pass,
                "scrub_fd_threshold": args.scrub_fd_threshold,
                "dvars_threshold": args.dvars_threshold,
                "fd_backward_neighbors": args.fd_backward_neighbors,
                "fd_forward_neighbors": args.fd_forward_neighbors,
                "min_retained_minutes": args.min_retained_minutes,
                "include_global_signal": int(args.include_global_signal),
                "confounds": ",".join(
                    DEFAULT_CONFOUND_COLUMNS
                    + (GLOBAL_SIGNAL_COLUMNS if args.include_global_signal else [])
                    + ["first_6_a_comp_cor"]
                ),
                "censoring": "non_steady_state_volumes,fd_spikes,optional_dvars,optional_adjacent_volume_expansion",
            }
        ]
    ).to_csv(metrics_dir / "denoising_settings.tsv", sep="\t", index=False)
    # I write the settings file so I can always show exactly what this denoising branch did.

    print(f"Denoised {len(subject_rows)} subjects from {sample_path}")
    print(f"Wrote denoised time series to {timeseries_dir}")
    print(f"Wrote denoising summary to {metrics_dir / 'subject_denoising_summary.tsv'}")


if __name__ == "__main__":
    main()
