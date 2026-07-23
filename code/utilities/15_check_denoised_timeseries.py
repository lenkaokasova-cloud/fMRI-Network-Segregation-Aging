#!/usr/bin/env python3

# What this script does:
#   Makes quick QC figures for the denoised parcel time series so I can check
#   whether the saved time series look sensible before connectivity analysis.
# How to run it:
#   Run from the repo root with:
#   python code/utilities/15_check_denoised_timeseries.py
# Main outputs:
#   data/processed/denoising/qc/
#   including subject figures and a timeseries QC summary TSV

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("data/processed/.matplotlib")).resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DPI = 300
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
GRID_COLOR = "#B8BDC7"

plt.rcParams.update(
    {
        "font.size": TICK_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "figure.titlesize": TITLE_SIZE,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect denoised parcel time series and save simple QC figures and a "
            "summary table for one or more subjects."
        )
    )
    parser.add_argument(
        "subjects",
        nargs="*",
        help="Optional subject IDs to inspect. Defaults to all subjects in the summary TSV.",
    )
    parser.add_argument(
        "--summary",
        default="data/processed/denoising/metrics/subject_denoising_summary.tsv",
        help="Denoising summary TSV produced by code/primary/11_run_denoising.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/denoising/qc",
        help="Directory for QC figures and the inspection summary TSV.",
    )
    parser.add_argument(
        "--n-display-parcels",
        type=int,
        default=5,
        help="Number of parcel traces to draw in the line-plot panel.",
    )
    parser.add_argument(
        "--heatmap-vmin",
        type=float,
        default=-3.0,
        help="Lower color limit for the heatmap panel.",
    )
    parser.add_argument(
        "--heatmap-vmax",
        type=float,
        default=3.0,
        help="Upper color limit for the heatmap panel.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display plots interactively instead of only saving them.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_summary(path: Path) -> pd.DataFrame:
    summary = pd.read_csv(path, sep="\t")
    required = {"subject_id", "retained_volumes", "timeseries_file"}
    missing = required.difference(summary.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Summary TSV is missing required columns: {missing_str}")
    return summary


def relative_project_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def make_subject_figure(
    subject: str,
    time_series: np.ndarray,
    output_path: Path,
    n_display_parcels: int,
    heatmap_vmin: float,
    heatmap_vmax: float,
    show: bool,
) -> None:
    n_display = min(max(n_display_parcels, 1), time_series.shape[1])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.5),
        gridspec_kw={"height_ratios": [1, 2]},
        constrained_layout=True,
    )

    axes[0].plot(time_series[:, :n_display], linewidth=1.1)
    axes[0].set_title(f"{subject}: first {n_display} parcel time series")
    axes[0].set_xlabel("Retained volume")
    axes[0].set_ylabel("Signal")
    axes[0].grid(color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    heatmap = axes[1].imshow(
        time_series.T,
        aspect="auto",
        cmap="coolwarm",
        vmin=heatmap_vmin,
        vmax=heatmap_vmax,
    )
    axes[1].set_title(f"{subject}: all parcel time series")
    axes[1].set_xlabel("Retained volume")
    axes[1].set_ylabel("Parcel")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    colorbar = fig.colorbar(heatmap, ax=axes[1], shrink=0.9)
    colorbar.outline.set_linewidth(0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)


def inspect_time_series(
    subject: str,
    time_series: np.ndarray,
    expected_retained_volumes: int,
    figure_path: Path,
) -> dict[str, object]:
    # This is just a sanity check that the saved parcel series still look numerically usable.
    has_nan = bool(np.isnan(time_series).any())
    has_inf = bool(np.isinf(time_series).any())
    n_timepoints, n_parcels = time_series.shape

    parcel_means = time_series.mean(axis=0)
    parcel_stds = time_series.std(axis=0)
    near_flat_parcels = int(np.sum(parcel_stds < 1e-6))
    mean_abs_parcel_mean = float(np.mean(np.abs(parcel_means)))
    median_parcel_std = float(np.median(parcel_stds))
    min_parcel_std = float(np.min(parcel_stds))
    max_parcel_std = float(np.max(parcel_stds))

    status = "ok"
    notes: list[str] = []
    if n_timepoints != expected_retained_volumes:
        status = "check"
        notes.append(
            f"Observed {n_timepoints} timepoints, expected {expected_retained_volumes}."
        )
    if has_nan:
        status = "check"
        notes.append("Contains NaN values.")
    if has_inf:
        status = "check"
        notes.append("Contains infinite values.")
    if near_flat_parcels:
        status = "check"
        notes.append(f"{near_flat_parcels} parcel(s) have near-zero variance.")

    return {
        "subject_id": subject,
        "status": status,
        "expected_retained_volumes": expected_retained_volumes,
        "observed_timepoints": n_timepoints,
        "n_parcels": n_parcels,
        "has_nan": int(has_nan),
        "has_inf": int(has_inf),
        "mean_abs_parcel_mean": round(mean_abs_parcel_mean, 6),
        "median_parcel_std": round(median_parcel_std, 6),
        "min_parcel_std": round(min_parcel_std, 6),
        "max_parcel_std": round(max_parcel_std, 6),
        "near_flat_parcels": near_flat_parcels,
        "figure_file": relative_project_path(figure_path),
        "notes": " ".join(notes),
    }


def main() -> None:
    args = parse_args()
    summary_path = resolve_project_path(args.summary)
    output_dir = resolve_project_path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)
    if args.subjects:
        requested = set(args.subjects)
        summary = summary[summary["subject_id"].isin(requested)].copy()
        missing = requested.difference(set(summary["subject_id"]))
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Requested subjects not found in summary TSV: {missing_str}")

    if summary.empty:
        raise ValueError("No subjects remain to inspect.")

    summary_rows: list[dict[str, object]] = []

    for row in summary.sort_values(["subject_id"]).itertuples(index=False):
        # I save one figure per subject so I can quickly spot anything that looks obviously wrong.
        subject = row.subject_id
        timeseries_path = resolve_project_path(row.timeseries_file)
        if not timeseries_path.exists():
            raise FileNotFoundError(f"Time series file is missing for {subject}: {timeseries_path}")

        time_series = np.load(timeseries_path)
        if time_series.ndim != 2:
            raise ValueError(
                f"{subject} time series should be 2D, got shape {time_series.shape}."
            )

        figure_path = figures_dir / f"{subject}_timeseries_qc.png"
        make_subject_figure(
            subject=subject,
            time_series=time_series,
            output_path=figure_path,
            n_display_parcels=args.n_display_parcels,
            heatmap_vmin=args.heatmap_vmin,
            heatmap_vmax=args.heatmap_vmax,
            show=args.show,
        )

        summary_rows.append(
            inspect_time_series(
                subject=subject,
                time_series=time_series,
                expected_retained_volumes=int(row.retained_volumes),
                figure_path=figure_path,
            )
        )

    output_path = output_dir / "timeseries_qc_summary.tsv"
    pd.DataFrame(summary_rows).to_csv(output_path, sep="\t", index=False)

    print(f"Inspected {len(summary_rows)} subject(s) from {summary_path}")
    print(f"Wrote QC figures to {figures_dir}")
    print(f"Wrote QC summary to {output_path}")


if __name__ == "__main__":
    main()
