#!/usr/bin/env python3

# What this script does:
#   Turns the denoised parcel time series into subject-level connectivity
#   matrices and network/global segregation measures.
# How to run it:
#   Run from the repo root with:
#   python code/primary/12_run_connectivity_analysis.py
# Main outputs:
#   data/processed/connectivity/matrices/
#   data/processed/connectivity/metrics/

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Schaefer atlas connectivity matrices and segregation metrics "
            "from denoised parcel time series."
        )
    )
    parser.add_argument(
        "subjects",
        nargs="*",
        help="Optional subject IDs to analyze. Defaults to subjects in the denoising summary.",
    )
    parser.add_argument(
        "--denoising-dir",
        default="data/processed/denoising",
        help="Directory produced by code/primary/11_run_denoising.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/connectivity",
        help="Directory for connectivity matrices and segregation outputs.",
    )
    parser.add_argument(
        "--connectivity-kind",
        choices=["correlation", "partial_correlation"],
        default="correlation",
        help="Connectivity estimator to apply to the denoised parcel time series.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_denoising_summary(denoising_dir: Path) -> pd.DataFrame:
    summary_path = denoising_dir / "metrics" / "subject_denoising_summary.tsv"
    summary = pd.read_csv(summary_path, sep="\t")
    required = {
        "subject_id",
        "age",
        "sex",
        "age_group",
        "retained_volumes",
        "retained_minutes_after_scrub",
        "pct_retained_after_scrub",
        "mean_fd",
        "pct_fd_gt_0p2",
        "timeseries_file",
    }
    missing = required.difference(summary.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Denoising summary is missing required columns: {missing_str}")
    return summary


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


def positive_only(matrix: np.ndarray) -> np.ndarray:
    out = matrix.copy()
    out[out < 0] = 0.0
    return out


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


def proportional_segregation(within_mean: float, between_mean: float) -> float:
    if not np.isfinite(within_mean) or not np.isfinite(between_mean) or np.isclose(within_mean, 0.0):
        return float("nan")
    return float((within_mean - between_mean) / within_mean)


def compute_network_segregation(
    z_matrix: np.ndarray, posonly_z_matrix: np.ndarray, groups: dict[str, list[int]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for network, indices in groups.items():
        within = z_matrix[np.ix_(indices, indices)]
        between_indices = [idx for idx in range(z_matrix.shape[0]) if idx not in indices]
        between = z_matrix[np.ix_(indices, between_indices)]
        within_mean = mean_off_diagonal(within)
        between_mean = float(between.mean()) if between.size else float("nan")

        within_posonly = posonly_z_matrix[np.ix_(indices, indices)]
        between_posonly = posonly_z_matrix[np.ix_(indices, between_indices)]
        within_mean_posonly = mean_off_diagonal(within_posonly)
        between_mean_posonly = float(between_posonly.mean()) if between_posonly.size else float("nan")
        raw_diff = within_mean - between_mean
        rows.append(
            {
                "network": network,
                "n_parcels": len(indices),
                "within_mean_z": within_mean,
                "between_mean_z": between_mean,
                "segregation": proportional_segregation(within_mean, between_mean),
                "segregation_raw_diff": raw_diff,
                "segregation_prop": proportional_segregation(within_mean, between_mean),
                "within_mean_z_posonly": within_mean_posonly,
                "between_mean_z_posonly": between_mean_posonly,
                "segregation_prop_posonly": proportional_segregation(
                    within_mean_posonly, between_mean_posonly
                ),
            }
        )
    return rows


def compute_global_metrics(z_matrix: np.ndarray, groups: dict[str, list[int]]) -> dict[str, float]:
    within_vals: list[float] = []
    between_vals: list[float] = []
    for indices in groups.values():
        within = z_matrix[np.ix_(indices, indices)]
        within_vals.append(mean_off_diagonal(within))
        between_indices = [idx for idx in range(z_matrix.shape[0]) if idx not in indices]
        between = z_matrix[np.ix_(indices, between_indices)]
        if between.size:
            between_vals.append(float(between.mean()))
    global_within = float(np.nanmean(within_vals))
    global_between = float(np.nanmean(between_vals))
    raw_diff = global_within - global_between
    return {
        "global_within_mean_z": global_within,
        "global_between_mean_z": global_between,
        "global_segregation": proportional_segregation(global_within, global_between),
        "global_segregation_raw_diff": raw_diff,
        "global_segregation_prop": proportional_segregation(global_within, global_between),
    }


def save_matrix_csv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(path)


def main() -> None:
    args = parse_args()
    denoising_dir = resolve_project_path(args.denoising_dir)
    output_dir = resolve_project_path(args.output_dir)
    matrices_dir = output_dir / "matrices"
    metrics_dir = output_dir / "metrics"
    for directory in (output_dir, matrices_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    summary = load_denoising_summary(denoising_dir)
    if args.subjects:
        summary = summary[summary["subject_id"].isin(args.subjects)].copy()
    if summary.empty:
        raise ValueError("No subjects remain to analyze after applying the requested filter.")

    atlas_labels_df = pd.read_csv(denoising_dir / "metrics" / "atlas_labels.tsv", sep="\t")
    atlas_labels = [decode_label(label) for label in atlas_labels_df["parcel_label"].tolist()]
    groups = network_groups(atlas_labels)

    kind_map = {
        "correlation": "correlation",
        "partial_correlation": "partial correlation",
    }
    # I keep the estimator switch here so Pearson can stay primary and partial correlation can stay sensitivity-only.
    connectivity = ConnectivityMeasure(
        kind=kind_map[args.connectivity_kind],
        standardize="zscore_sample",
    )
    subject_rows: list[dict[str, object]] = []
    network_rows: list[dict[str, object]] = []
    z_matrices: list[np.ndarray] = []

    for row in summary.sort_values(["age_group", "age", "subject_id"]).itertuples(index=False):
        subject = row.subject_id
        timeseries_path = resolve_project_path(row.timeseries_file)
        time_series = np.load(timeseries_path)
        # This is where the denoised parcel series finally become subject-level connectivity matrices.
        corr = connectivity.fit_transform([time_series])[0]
        z_matrix = fisher_z(corr)
        np.fill_diagonal(z_matrix, 0.0)
        posonly_z_matrix = positive_only(z_matrix)
        z_matrices.append(z_matrix)

        np.save(matrices_dir / f"{subject}_forward_z_matrix.npy", z_matrix)
        save_matrix_csv(matrices_dir / f"{subject}_forward_z_matrix.csv", z_matrix, atlas_labels)

        global_metrics = compute_global_metrics(z_matrix, groups)
        global_metrics_posonly = compute_global_metrics(posonly_z_matrix, groups)

        subject_rows.append(
            {
                "subject_id": subject,
                "age": int(row.age),
                "sex": row.sex,
                "age_group": row.age_group,
                "n_volumes": int(row.n_volumes),
                "retained_volumes": int(row.retained_volumes),
                "retained_minutes_after_scrub": float(row.retained_minutes_after_scrub),
                "pct_retained_after_scrub": float(row.pct_retained_after_scrub),
                "mean_fd": float(row.mean_fd),
                "pct_fd_gt_0p2": float(row.pct_fd_gt_0p2),
                **global_metrics,
                "global_within_mean_z_posonly": global_metrics_posonly["global_within_mean_z"],
                "global_between_mean_z_posonly": global_metrics_posonly["global_between_mean_z"],
                "global_segregation_prop_posonly": global_metrics_posonly["global_segregation_prop"],
            }
        )

        for network_row in compute_network_segregation(z_matrix, posonly_z_matrix, groups):
            network_row.update(
                {
                    "subject_id": subject,
                    "age": int(row.age),
                    "sex": row.sex,
                    "age_group": row.age_group,
                    "retained_volumes": int(row.retained_volumes),
                    "retained_minutes_after_scrub": float(row.retained_minutes_after_scrub),
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
    atlas_labels_df.to_csv(metrics_dir / "atlas_labels.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "denoising_dir": str(denoising_dir),
                "connectivity_measure": args.connectivity_kind,
                "fisher_z_transform": 1,
                "time_series_standardization": "zscore_sample",
                "primary_global_metric": "global_segregation_prop",
                "primary_network_metric": "segregation_prop",
                "legacy_global_sensitivity_metric": "global_segregation_raw_diff",
                "legacy_network_sensitivity_metric": "segregation_raw_diff",
                "positive_only_sensitivity_metric": "segregation_prop_posonly",
                "segregation_formula_primary": "(within_mean_z - between_mean_z) / within_mean_z",
                "segregation_formula_legacy_raw_difference": "within_mean_z - between_mean_z",
            }
        ]
    ).to_csv(metrics_dir / "connectivity_settings.tsv", sep="\t", index=False)
    # I save the formulas too so the exact segregation definition is never ambiguous later on.

    print(f"Processed {len(subject_rows)} subjects from {denoising_dir}")
    print(f"Wrote subject metrics to {metrics_dir / 'subject_global_segregation.tsv'}")
    print(f"Wrote network metrics to {metrics_dir / 'subject_network_segregation.tsv'}")


if __name__ == "__main__":
    main()
