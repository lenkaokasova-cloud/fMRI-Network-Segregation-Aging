#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Schaefer atlas time series from fMRIPrep outputs, compute "
            "connectivity matrices, and calculate segregation metrics for the "
            "clean age-analysis sample after nuisance regression and frame censoring."
        )
    )
    parser.add_argument(
        "subjects",
        nargs="*",
        help="Optional subject IDs to analyze. Defaults to subjects in --sample.",
    )
    parser.add_argument(
        "--sample",
        default="data/processed/screening/ds005752_clean_age_sample.tsv",
        help="Clean sample TSV produced by code/07_build_clean_sample.py.",
    )
    parser.add_argument(
        "--derivatives-dir",
        default="data/derivatives/fmriprep",
        help="Path to fMRIPrep derivatives.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/connectivity",
        help="Directory for connectivity and segregation outputs.",
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
        "--min-retained-volumes",
        type=int,
        default=150,
        help="Fail if fewer than this many volumes remain after censoring.",
    )
    parser.add_argument(
        "--include-global-signal",
        action="store_true",
        help="Include global signal in the nuisance-regression design as a sensitivity analysis.",
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


def get_forward_run_paths(derivatives_dir: Path, subject: str) -> tuple[Path, Path, Path]:
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
    return bold, confounds, bold_json


def get_tr(bold_json: Path) -> float:
    with bold_json.open() as f:
        metadata = json.load(f)
    if "RepetitionTime" not in metadata:
        raise KeyError(f"RepetitionTime missing from {bold_json}")
    return float(metadata["RepetitionTime"])


def load_confounds_table(confounds_path: Path) -> pd.DataFrame:
    return pd.read_csv(confounds_path, sep="\t")


def build_sample_mask(confounds: pd.DataFrame, scrub_fd_threshold: float) -> np.ndarray:
    keep_mask = np.ones(len(confounds), dtype=bool)

    nonsteady_columns = [
        column for column in confounds.columns if column.startswith("non_steady_state_outlier")
    ]
    if nonsteady_columns:
        nonsteady_mask = confounds[nonsteady_columns].fillna(0.0).eq(0.0).all(axis=1).to_numpy()
        keep_mask &= nonsteady_mask

    if "framewise_displacement" in confounds.columns:
        fd = pd.to_numeric(confounds["framewise_displacement"], errors="coerce")
        fd_ok = ~fd.gt(scrub_fd_threshold).fillna(False).to_numpy()
        keep_mask &= fd_ok

    return np.flatnonzero(keep_mask)


def pick_confounds(confounds: pd.DataFrame, include_global_signal: bool) -> pd.DataFrame:
    columns = [col for col in DEFAULT_CONFOUND_COLUMNS if col in confounds.columns]
    acompcor = [col for col in confounds.columns if col.startswith("a_comp_cor_")][:6]
    columns.extend(acompcor)
    if include_global_signal and "global_signal" in confounds.columns:
        columns.append("global_signal")
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


def fisher_z(matrix: np.ndarray) -> np.ndarray:
    clipped = np.clip(matrix, -0.999999, 0.999999)
    return np.arctanh(clipped)


def network_groups(labels: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[parse_network_name(label)].append(idx)
    return dict(groups)


def mean_off_diagonal(matrix: np.ndarray) -> float:
    if matrix.shape[0] <= 1:
        return float("nan")
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(matrix[mask].mean())


def compute_network_segregation(
    z_matrix: np.ndarray, groups: dict[str, list[int]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for network, indices in groups.items():
        within = z_matrix[np.ix_(indices, indices)]
        between_indices = [idx for idx in range(z_matrix.shape[0]) if idx not in indices]
        between = z_matrix[np.ix_(indices, between_indices)]
        within_mean = mean_off_diagonal(within)
        between_mean = float(between.mean()) if between.size else float("nan")
        rows.append(
            {
                "network": network,
                "n_parcels": len(indices),
                "within_mean_z": within_mean,
                "between_mean_z": between_mean,
                "segregation": within_mean - between_mean,
            }
        )
    return rows


def compute_global_segregation(z_matrix: np.ndarray, groups: dict[str, list[int]]) -> float:
    within_vals: list[float] = []
    between_vals: list[float] = []
    for indices in groups.values():
        within = z_matrix[np.ix_(indices, indices)]
        within_vals.append(mean_off_diagonal(within))
        between_indices = [idx for idx in range(z_matrix.shape[0]) if idx not in indices]
        between = z_matrix[np.ix_(indices, between_indices)]
        if between.size:
            between_vals.append(float(between.mean()))
    return float(np.nanmean(within_vals) - np.nanmean(between_vals))


def save_matrix_csv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(path)


def main() -> None:
    args = parse_args()
    sample_path = resolve_project_path(args.sample)
    derivatives_dir = resolve_project_path(args.derivatives_dir)
    output_dir = resolve_project_path(args.output_dir)
    matrices_dir = output_dir / "matrices"
    timeseries_dir = output_dir / "timeseries"
    metrics_dir = output_dir / "metrics"
    for directory in (output_dir, matrices_dir, timeseries_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    sample = load_sample(sample_path)
    if args.subjects:
        sample = sample[sample["subject_id"].isin(args.subjects)].copy()
    if sample.empty:
        raise ValueError("No subjects remain to analyze after applying the requested filter.")

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=args.n_rois,
        yeo_networks=args.yeo_networks,
        resolution_mm=2,
    )
    atlas_labels = [decode_label(label) for label in atlas.labels]
    if len(atlas_labels) == args.n_rois + 1 and atlas_labels[0].lower() == "background":
        atlas_labels = atlas_labels[1:]
    groups = network_groups(atlas_labels)

    connectivity = ConnectivityMeasure(kind="correlation", standardize="zscore_sample")
    subject_rows: list[dict[str, object]] = []
    network_rows: list[dict[str, object]] = []
    z_matrices: list[np.ndarray] = []

    for row in sample.sort_values(["age_group", "age", "subject_id"]).itertuples(index=False):
        subject = row.subject_id
        bold_path, confounds_path, bold_json = get_forward_run_paths(derivatives_dir, subject)
        tr = get_tr(bold_json)
        confounds_table = load_confounds_table(confounds_path)
        n_volumes = int(len(confounds_table))
        sample_mask = build_sample_mask(confounds_table, args.scrub_fd_threshold)
        if len(sample_mask) < args.min_retained_volumes:
            raise ValueError(
                f"{subject} retained only {len(sample_mask)} volumes after censoring, "
                f"below the required minimum of {args.min_retained_volumes}."
            )
        confounds = pick_confounds(confounds_table, args.include_global_signal)

        masker = NiftiLabelsMasker(
            labels_img=atlas.maps,
            standardize="zscore_sample",
            detrend=True,
            low_pass=args.low_pass,
            high_pass=args.high_pass,
            t_r=tr,
            standardize_confounds=True,
        )
        time_series = masker.fit_transform(
            str(bold_path),
            confounds=confounds,
            sample_mask=sample_mask,
        )
        corr = connectivity.fit_transform([time_series])[0]
        z_matrix = fisher_z(corr)
        np.fill_diagonal(z_matrix, 0.0)
        z_matrices.append(z_matrix)

        np.save(matrices_dir / f"{subject}_forward_z_matrix.npy", z_matrix)
        save_matrix_csv(matrices_dir / f"{subject}_forward_z_matrix.csv", z_matrix, atlas_labels)
        np.save(timeseries_dir / f"{subject}_forward_timeseries.npy", time_series)

        subject_rows.append(
            {
                "subject_id": subject,
                "age": int(row.age),
                "sex": row.sex,
                "age_group": row.age_group,
                "n_volumes": n_volumes,
                "retained_volumes": int(len(sample_mask)),
                "pct_retained_after_scrub": round(100 * len(sample_mask) / n_volumes, 2),
                "mean_fd": float(row.mean_fd),
                "pct_fd_gt_0p2": float(row.pct_fd_gt_0p2),
                "global_segregation": compute_global_segregation(z_matrix, groups),
            }
        )

        for network_row in compute_network_segregation(z_matrix, groups):
            network_row.update(
                {
                    "subject_id": subject,
                    "age": int(row.age),
                    "sex": row.sex,
                    "age_group": row.age_group,
                    "retained_volumes": int(len(sample_mask)),
                    "mean_fd": float(row.mean_fd),
                }
            )
            network_rows.append(network_row)

    if z_matrices:
        mean_matrix = np.mean(z_matrices, axis=0)
        np.save(matrices_dir / "group_mean_z_matrix.npy", mean_matrix)
        save_matrix_csv(matrices_dir / "group_mean_z_matrix.csv", mean_matrix, atlas_labels)

    pd.DataFrame(subject_rows).to_csv(
        metrics_dir / "subject_global_segregation.tsv", sep="\t", index=False
    )
    pd.DataFrame(network_rows).to_csv(
        metrics_dir / "subject_network_segregation.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        {
            "parcel_index": list(range(len(atlas_labels))),
            "parcel_label": atlas_labels,
            "network": [parse_network_name(label) for label in atlas_labels],
        }
    ).to_csv(metrics_dir / "atlas_labels.tsv", sep="\t", index=False)
    confound_description = DEFAULT_CONFOUND_COLUMNS + ["first_6_a_comp_cor"]
    if args.include_global_signal:
        confound_description.append("global_signal")

    pd.DataFrame(
        [
            {
                "n_rois": args.n_rois,
                "yeo_networks": args.yeo_networks,
                "high_pass_hz": args.high_pass,
                "low_pass_hz": args.low_pass,
                "scrub_fd_threshold": args.scrub_fd_threshold,
                "min_retained_volumes": args.min_retained_volumes,
                "include_global_signal": int(args.include_global_signal),
                "confounds": ",".join(confound_description),
            }
        ]
    ).to_csv(metrics_dir / "denoising_settings.tsv", sep="\t", index=False)

    print(f"Processed {len(subject_rows)} subjects from {sample_path}")
    print(f"Wrote subject metrics to {metrics_dir / 'subject_global_segregation.tsv'}")
    print(f"Wrote network metrics to {metrics_dir / 'subject_network_segregation.tsv'}")


if __name__ == "__main__":
    main()
