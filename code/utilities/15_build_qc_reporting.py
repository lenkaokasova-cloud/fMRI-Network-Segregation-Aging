#!/usr/bin/env python3

# What this script does:
#   Builds the extra QC reporting figures and tables that summarize sample flow,
#   motion, censoring, and QC-FC patterns.
# How to run it:
#   Run from the repo root with:
#   python code/utilities/15_build_qc_reporting.py
# Main output:
#   data/processed/qc/reporting/

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str((Path("data/processed/.matplotlib")).resolve())
)

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from nilearn import datasets
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DPI = 300
TITLE_SIZE = 14
LABEL_SIZE = 11
TICK_SIZE = 10
TEXT_COLOR = "#253547"
GRID_COLOR = "#D7DCE4"
YOUNG_COLOR = "#4F739C"
OLDER_COLOR = "#B87053"
REFERENCE_LINE_COLOR = "#253547"
FONT_FAMILY = "serif"
FONT_SERIF = [
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "TeX Gyre Termes",
    "STIX Two Text",
    "Liberation Serif",
    "DejaVu Serif",
]
MIN_RETAINED_MINUTES_REFERENCE = 9.0
NETWORK_ORDER = [
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
]

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.serif": FONT_SERIF,
        "font.size": TICK_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "figure.titlesize": TITLE_SIZE,
        "axes.titleweight": "semibold",
        "axes.labelcolor": TEXT_COLOR,
        "axes.edgecolor": TEXT_COLOR,
        "axes.linewidth": 0.9,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "mathtext.fontset": "stix",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build dissertation-style QC reporting figures and summary tables for the "
            "final analysis sample."
        )
    )
    parser.add_argument(
        "--sample",
        default="data/processed/screening/ds005752_final_analysis_sample_tr_3s.tsv",
        help="Primary final analysis sample TSV.",
    )
    parser.add_argument(
        "--denoising-summary",
        default="data/processed/denoising/metrics/subject_denoising_summary.tsv",
        help="Denoising summary TSV produced by code/primary/10_run_denoising.py.",
    )
    parser.add_argument(
        "--denoising-settings",
        default="data/processed/denoising/metrics/denoising_settings.tsv",
        help="Denoising settings TSV produced by code/primary/10_run_denoising.py.",
    )
    parser.add_argument(
        "--derivatives-dir",
        default="data/derivatives/fmriprep",
        help="fMRIPrep derivatives directory.",
    )
    parser.add_argument(
        "--connectivity-dir",
        default="data/processed/connectivity/metrics",
        help="Connectivity metrics directory.",
    )
    parser.add_argument(
        "--matrices-dir",
        default="data/processed/connectivity/matrices",
        help="Connectivity matrices directory.",
    )
    parser.add_argument(
        "--screening-dir",
        default="data/processed/screening",
        help="Screening directory used for the sample-flow figure.",
    )
    parser.add_argument(
        "--atlas-data-dir",
        default="data/external/nilearn_data",
        help="Directory where Nilearn atlas files are cached.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/qc/reporting",
        help="Directory for QC reporting outputs.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def style_axis(
    ax: plt.Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    if title:
        ax.set_title(title, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT_COLOR)
    ax.spines["bottom"].set_color(TEXT_COLOR)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(length=4.2, width=0.8, color=TEXT_COLOR)


def save_figure(fig: plt.Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ordered_networks(networks: list[str]) -> list[str]:
    order_index = {network: idx for idx, network in enumerate(NETWORK_ORDER)}
    unique = list(dict.fromkeys(networks))
    return sorted(
        unique, key=lambda name: (order_index.get(name, len(order_index)), name)
    )


def expand_spike_mask(
    spike_mask: np.ndarray, backward: int, forward: int
) -> np.ndarray:
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
) -> tuple[np.ndarray, np.ndarray, str]:
    keep_mask = np.ones(len(confounds), dtype=bool)
    dvars_column = (
        "std_dvars"
        if "std_dvars" in confounds.columns
        else "dvars" if "dvars" in confounds.columns else ""
    )
    fd_spike_mask = np.zeros(len(confounds), dtype=bool)
    dvars_spike_mask = np.zeros(len(confounds), dtype=bool)

    nonsteady_columns = [
        column
        for column in confounds.columns
        if column.startswith("non_steady_state_outlier")
    ]
    if nonsteady_columns:
        nonsteady_mask = (
            confounds[nonsteady_columns].fillna(0.0).eq(0.0).all(axis=1).to_numpy()
        )
        keep_mask &= nonsteady_mask

    if "framewise_displacement" in confounds.columns:
        fd = pd.to_numeric(confounds["framewise_displacement"], errors="coerce")
        fd_spike_mask = fd.gt(scrub_fd_threshold).fillna(False).to_numpy()

    if dvars_threshold is not None and dvars_column:
        dvars = pd.to_numeric(confounds[dvars_column], errors="coerce")
        dvars_spike_mask = dvars.gt(dvars_threshold).fillna(False).to_numpy()

    spike_mask = fd_spike_mask | dvars_spike_mask
    expanded_spike_mask = expand_spike_mask(
        spike_mask, fd_backward_neighbors, fd_forward_neighbors
    )
    keep_mask &= ~expanded_spike_mask
    return keep_mask, expanded_spike_mask, dvars_column


def draw_flow_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    text: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.6,
        edgecolor=color,
        facecolor=color,
        alpha=0.18,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=16.5,
        weight="semibold",
        color="#081A2D",
    )
    ax.text(
        x + width / 2,
        y + height * 0.34,
        text,
        ha="center",
        va="center",
        fontsize=15.5,
        color="#081A2D",
    )


def plot_sample_flow(screening_dir: Path, outpath: Path) -> None:
    mri = pd.read_csv(screening_dir / "ds005752_mri_participants.tsv", sep="\t")
    young_remote = pd.read_csv(
        screening_dir / "ds005752_mri_participants_age_20_25_remote_anat_forward.tsv",
        sep="\t",
    )
    older_remote = pd.read_csv(
        screening_dir / "ds005752_mri_participants_age_50_75_remote_anat_forward.tsv",
        sep="\t",
    )
    qc_pass = pd.read_csv(screening_dir / "ds005752_qc_pass_sample.tsv", sep="\t")
    final_sample = pd.read_csv(
        screening_dir / "ds005752_final_analysis_sample_tr_3s.tsv", sep="\t"
    )

    counts = {
        "young_age_band": int(((mri["age"] >= 20) & (mri["age"] <= 25)).sum()),
        "older_age_band": int(((mri["age"] >= 50) & (mri["age"] <= 75)).sum()),
        "young_remote": int(len(young_remote)),
        "older_remote": int(len(older_remote)),
        "young_qc_pass": int((qc_pass["age_group"] == "young").sum()),
        "older_qc_pass": int((qc_pass["age_group"] == "older").sum()),
        "young_final": int((final_sample["age_group"] == "young").sum()),
        "older_final": int((final_sample["age_group"] == "older").sum()),
    }

    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    ax.axis("off")
    flow_young_text = "#123B63"
    flow_older_text = "#7B2E1F"
    ax.set_title("Sample selection flow", pad=16, fontsize=21, color="#081A2D")

    y_positions = [0.78, 0.54, 0.30, 0.06]
    left_x = 0.05
    right_x = 0.65
    width = 0.30
    height = 0.14

    steps = [
        (
            "MRI-eligible in age band",
            f"{counts['young_age_band']} young\n{counts['older_age_band']} older",
        ),
        (
            "Remote anat + forward-rest candidates",
            f"{counts['young_remote']} young\n{counts['older_remote']} older",
        ),
        (
            "QC-pass preprocessed sample",
            f"{counts['young_qc_pass']} young\n{counts['older_qc_pass']} older",
        ),
        (
            "Primary final TR = 3 s sample",
            f"{counts['young_final']} young\n{counts['older_final']} older",
        ),
    ]
    exclusions = [
        (
            "Missing\nscans",
            counts["young_age_band"] - counts["young_remote"],
            counts["older_age_band"] - counts["older_remote"],
        ),
        (
            "Preprocessing +\nQC exclusions",
            counts["young_remote"] - counts["young_qc_pass"],
            counts["older_remote"] - counts["older_qc_pass"],
        ),
        (
            "TR ≠ 3 s\nexclusions",
            counts["young_qc_pass"] - counts["young_final"],
            counts["older_qc_pass"] - counts["older_final"],
        ),
    ]

    for idx, (title, text) in enumerate(steps):
        draw_flow_box(
            ax,
            left_x,
            y_positions[idx],
            width,
            height,
            title,
            text.split("\n")[0],
            YOUNG_COLOR,
        )
        draw_flow_box(
            ax,
            right_x,
            y_positions[idx],
            width,
            height,
            title,
            text.split("\n")[1],
            OLDER_COLOR,
        )
        if idx < len(steps) - 1:
            exclusion_title, young_excluded, older_excluded = exclusions[idx]
            arrow_top = y_positions[idx] - 0.01
            arrow_bottom = y_positions[idx + 1] + height + 0.01
            arrow_midpoint = (arrow_top + arrow_bottom) / 2
            for x_center in (left_x + width / 2, right_x + width / 2):
                ax.annotate(
                    "",
                    xy=(x_center, arrow_bottom),
                    xytext=(x_center, arrow_top),
                    arrowprops={
                        "arrowstyle": "->",
                        "linewidth": 1.4,
                        "color": REFERENCE_LINE_COLOR,
                    },
                )
            ax.text(
                0.50,
                arrow_midpoint,
                exclusion_title,
                ha="center",
                va="center",
                fontsize=13.5,
                color=REFERENCE_LINE_COLOR,
            )
            ax.text(
                0.43,
                arrow_midpoint,
                f"n = {young_excluded}",
                ha="right",
                va="center",
                fontsize=15.0,
                weight="semibold",
                color=flow_young_text,
            )
            ax.text(
                0.57,
                arrow_midpoint,
                f"n = {older_excluded}",
                ha="left",
                va="center",
                fontsize=15.0,
                weight="semibold",
                color=flow_older_text,
            )

    ax.text(
        left_x + width / 2,
        0.97,
        "Younger branch",
        ha="center",
        va="center",
        fontsize=18.5,
        weight="semibold",
        color=flow_young_text,
    )
    ax.text(
        right_x + width / 2,
        0.97,
        "Older branch",
        ha="center",
        va="center",
        fontsize=18.5,
        weight="semibold",
        color=flow_older_text,
    )
    save_figure(fig, outpath)


def plot_retained_minutes(denoising_summary: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for age_group, color in [("young", YOUNG_COLOR), ("older", OLDER_COLOR)]:
        subset = denoising_summary[denoising_summary["age_group"] == age_group]
        ax.hist(
            subset["retained_minutes_after_scrub"],
            bins=10,
            alpha=0.45,
            color=color,
            label=age_group,
            edgecolor="white",
        )
    ax.axvline(
        MIN_RETAINED_MINUTES_REFERENCE,
        color=REFERENCE_LINE_COLOR,
        linestyle="--",
        linewidth=1.6,
    )
    style_axis(
        ax,
        title="Retained Data Duration After Censoring",
        xlabel="Retained minutes",
        ylabel="Number of participants",
    )
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    ax.legend(frameon=False)
    save_figure(fig, outpath)


def plot_within_mean_stability(network_df: pd.DataFrame, outpath: Path) -> None:
    networks = ordered_networks(network_df["network"].tolist())
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    data = [
        network_df.loc[network_df["network"] == network, "within_mean_z"].to_numpy()
        for network in networks
    ]
    box = ax.boxplot(
        data,
        labels=networks,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.8},
        whiskerprops={"linewidth": 1.4},
        capprops={"linewidth": 1.4},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(YOUNG_COLOR)
        patch.set_alpha(0.22)
        patch.set_edgecolor(YOUNG_COLOR)
        patch.set_linewidth(1.6)
    ax.axhline(0.0, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1.5)
    style_axis(
        ax,
        title="Within-Network Mean Connectivity by Network",
        xlabel="Network",
        ylabel="within_mean_z",
    )
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    save_figure(fig, outpath)


def get_confounds_path(derivatives_dir: Path, subject_id: str) -> Path:
    func_dir = derivatives_dir / subject_id / "ses-01" / "func"
    return next(
        func_dir.glob(
            f"{subject_id}_ses-01_task-rest_dir-forward_desc-confounds_timeseries.tsv"
        )
    )


def plot_motion_trace(
    subject_id: str,
    confounds: pd.DataFrame,
    keep_mask: np.ndarray,
    expanded_spike_mask: np.ndarray,
    fd_threshold: float,
    dvars_threshold: float | None,
    dvars_column: str,
    output_path: Path,
) -> None:
    fd = (
        pd.to_numeric(confounds["framewise_displacement"], errors="coerce")
        if "framewise_displacement" in confounds.columns
        else pd.Series(np.nan, index=confounds.index)
    )
    dvars = (
        pd.to_numeric(confounds[dvars_column], errors="coerce")
        if dvars_column
        else pd.Series(np.nan, index=confounds.index)
    )
    x = np.arange(len(confounds))

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.8, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 0.45]},
    )

    axes[0].plot(x, fd, color=YOUNG_COLOR, linewidth=1.25)
    axes[0].axhline(
        fd_threshold, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1.4
    )
    style_axis(
        axes[0], title=f"{subject_id}: FD, DVARS, and censor mask", ylabel="FD (mm)"
    )
    axes[0].grid(color=GRID_COLOR, alpha=0.22, linewidth=0.8)

    axes[1].plot(x, dvars, color=OLDER_COLOR, linewidth=1.25)
    if dvars_threshold is not None and dvars_column:
        axes[1].axhline(
            dvars_threshold, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1.4
        )
    style_axis(axes[1], ylabel=dvars_column or "DVARS")
    axes[1].grid(color=GRID_COLOR, alpha=0.22, linewidth=0.8)

    mask_array = np.where(keep_mask, 1.0, 0.0)[None, :]
    axes[2].imshow(mask_array, aspect="auto", cmap="Greys", vmin=0.0, vmax=1.0)
    axes[2].set_yticks([])
    axes[2].set_xlabel("Volume")
    axes[2].set_ylabel("keep")
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    censored_idx = np.flatnonzero(expanded_spike_mask)
    if len(censored_idx):
        axes[2].scatter(
            censored_idx,
            np.zeros_like(censored_idx),
            color=REFERENCE_LINE_COLOR,
            s=6,
            alpha=0.8,
        )

    save_figure(fig, output_path)


def build_motion_qc_figures(
    sample: pd.DataFrame,
    denoising_summary: pd.DataFrame,
    denoising_settings: pd.Series,
    derivatives_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    fd_threshold = float(denoising_settings.get("scrub_fd_threshold", 0.5))
    dvars_threshold = (
        None
        if pd.isna(denoising_settings.get("dvars_threshold", np.nan))
        else float(denoising_settings.get("dvars_threshold"))
    )
    backward = int(denoising_settings.get("fd_backward_neighbors", 0))
    forward = int(denoising_settings.get("fd_forward_neighbors", 0))

    rows: list[dict[str, object]] = []
    summary_lookup = denoising_summary.set_index("subject_id")
    for subject_id in sorted(sample["subject_id"].tolist()):
        confounds = pd.read_csv(
            get_confounds_path(derivatives_dir, subject_id), sep="\t"
        )
        keep_mask, expanded_spike_mask, dvars_column = build_sample_mask(
            confounds,
            scrub_fd_threshold=fd_threshold,
            dvars_threshold=dvars_threshold,
            fd_backward_neighbors=backward,
            fd_forward_neighbors=forward,
        )
        output_path = output_dir / f"{subject_id}_fd_dvars_censor.png"
        plot_motion_trace(
            subject_id,
            confounds,
            keep_mask,
            expanded_spike_mask,
            fd_threshold,
            dvars_threshold,
            dvars_column,
            output_path,
        )
        row = summary_lookup.loc[subject_id].to_dict()
        row["subject_id"] = subject_id
        row["motion_figure_file"] = str(output_path.relative_to(PROJECT_ROOT))
        row["dvars_column_used"] = dvars_column
        rows.append(row)
    return pd.DataFrame(rows)


def atlas_centroids(n_rois: int, atlas_data_dir: Path) -> np.ndarray:
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois,
        yeo_networks=7,
        resolution_mm=2,
        data_dir=str(atlas_data_dir),
    )
    img = nib.load(atlas.maps)
    data = np.asanyarray(img.dataobj)
    centroids = []
    for label_idx in range(1, n_rois + 1):
        coords = np.argwhere(data == label_idx)
        if len(coords) == 0:
            centroids.append([np.nan, np.nan, np.nan])
            continue
        world = nib.affines.apply_affine(img.affine, coords)
        centroids.append(world.mean(axis=0))
    return np.asarray(centroids)


def compute_qcfc(
    matrices_dir: Path,
    denoising_summary: pd.DataFrame,
    atlas_data_dir: Path,
    n_rois: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    summary = denoising_summary.sort_values("subject_id").reset_index(drop=True)
    triu = np.triu_indices(n_rois, k=1)
    edge_values = []
    for subject_id in summary["subject_id"]:
        matrix = np.load(matrices_dir / f"{subject_id}_forward_z_matrix.npy")
        edge_values.append(matrix[triu])
    edge_matrix = np.vstack(edge_values)
    fd = summary["mean_fd"].to_numpy(dtype=float)

    fd_centered = fd - fd.mean()
    edges_centered = edge_matrix - edge_matrix.mean(axis=0, keepdims=True)
    denom = np.sqrt((fd_centered**2).sum()) * np.sqrt((edges_centered**2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r_vals = np.divide(
            fd_centered @ edges_centered,
            denom,
            out=np.zeros_like(denom),
            where=denom > 0,
        )
    r_vals = np.clip(r_vals, -0.999999, 0.999999)
    dfree = len(fd) - 2
    t_vals = r_vals * np.sqrt(dfree / np.maximum(1.0 - r_vals**2, 1e-12))
    p_vals = 2 * stats.t.sf(np.abs(t_vals), df=dfree)

    centroids = atlas_centroids(n_rois=n_rois, atlas_data_dir=atlas_data_dir)
    distances = np.linalg.norm(centroids[triu[0]] - centroids[triu[1]], axis=1)

    qcfc_df = pd.DataFrame(
        {
            "edge_index": np.arange(len(r_vals)),
            "parcel_i": triu[0] + 1,
            "parcel_j": triu[1] + 1,
            "distance_mm": distances,
            "qcfc_r": r_vals,
            "qcfc_p": p_vals,
        }
    )
    return qcfc_df, r_vals, distances


def plot_qcfc_distribution(qcfc_df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(qcfc_df["qcfc_r"], bins=40, color=YOUNG_COLOR, alpha=0.7, edgecolor="white")
    ax.axvline(0.0, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1.5)
    style_axis(
        ax,
        title="QC-FC Edge Distribution",
        xlabel="Edgewise correlation with mean FD",
        ylabel="Number of edges",
    )
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    save_figure(fig, outpath)


def plot_qcfc_distance(qcfc_df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    hb = ax.hexbin(
        qcfc_df["distance_mm"],
        qcfc_df["qcfc_r"],
        gridsize=40,
        cmap="Blues",
        mincnt=1,
    )
    bins = np.linspace(qcfc_df["distance_mm"].min(), qcfc_df["distance_mm"].max(), 13)
    qcfc_df = qcfc_df.copy()
    qcfc_df["distance_bin"] = pd.cut(
        qcfc_df["distance_mm"], bins=bins, include_lowest=True
    )
    binned = qcfc_df.groupby("distance_bin", observed=True).agg(
        distance_mid=("distance_mm", "mean"),
        qcfc_mean=("qcfc_r", "mean"),
    )
    ax.plot(
        binned["distance_mid"], binned["qcfc_mean"], color=OLDER_COLOR, linewidth=2.0
    )
    colorbar = fig.colorbar(hb, ax=ax, shrink=0.88)
    colorbar.set_label("Number of edges")
    ax.axhline(0.0, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1.4)
    style_axis(
        ax,
        title="QC-FC by Inter-Parcel Distance",
        xlabel="Distance between parcel centroids (mm)",
        ylabel="QC-FC correlation (r)",
    )
    ax.grid(color=GRID_COLOR, alpha=0.18, linewidth=0.8)
    save_figure(fig, outpath)


def main() -> None:
    args = parse_args()
    sample_path = resolve_project_path(args.sample)
    denoising_summary_path = resolve_project_path(args.denoising_summary)
    denoising_settings_path = resolve_project_path(args.denoising_settings)
    derivatives_dir = resolve_project_path(args.derivatives_dir)
    connectivity_dir = resolve_project_path(args.connectivity_dir)
    matrices_dir = resolve_project_path(args.matrices_dir)
    screening_dir = resolve_project_path(args.screening_dir)
    atlas_data_dir = resolve_project_path(args.atlas_data_dir)
    output_dir = resolve_project_path(args.output_dir)
    figures_dir = output_dir / "figures"
    motion_figures_dir = output_dir / "motion_traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    motion_figures_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(sample_path, sep="\t")
    denoising_summary = pd.read_csv(denoising_summary_path, sep="\t")
    denoising_settings = pd.read_csv(denoising_settings_path, sep="\t").iloc[0]
    network_df = pd.read_csv(
        connectivity_dir / "subject_network_segregation.tsv", sep="\t"
    )
    atlas_labels = pd.read_csv(connectivity_dir / "atlas_labels.tsv", sep="\t")

    subject_ids = set(sample["subject_id"])
    denoising_summary = denoising_summary[
        denoising_summary["subject_id"].isin(subject_ids)
    ].copy()
    network_df = network_df[network_df["subject_id"].isin(subject_ids)].copy()

    plot_sample_flow(screening_dir, figures_dir / "sample_flow_primary.png")
    plot_retained_minutes(
        denoising_summary, figures_dir / "retained_minutes_distribution.png"
    )
    plot_within_mean_stability(network_df, figures_dir / "within_mean_z_by_network.png")

    motion_summary = build_motion_qc_figures(
        sample=sample,
        denoising_summary=denoising_summary,
        denoising_settings=denoising_settings,
        derivatives_dir=derivatives_dir,
        output_dir=motion_figures_dir,
    )
    motion_summary.to_csv(
        output_dir / "motion_qc_trace_summary.tsv", sep="\t", index=False
    )

    n_rois = int(len(atlas_labels))
    qcfc_df, r_vals, distances = compute_qcfc(
        matrices_dir=matrices_dir,
        denoising_summary=denoising_summary,
        atlas_data_dir=atlas_data_dir,
        n_rois=n_rois,
    )
    qcfc_df.to_csv(output_dir / "qcfc_edge_summary.tsv", sep="\t", index=False)
    plot_qcfc_distribution(qcfc_df, figures_dir / "qcfc_distribution.png")
    plot_qcfc_distance(qcfc_df, figures_dir / "qcfc_distance_relationship.png")

    qcfc_summary = pd.DataFrame(
        [
            {
                "n_subjects": int(denoising_summary["subject_id"].nunique()),
                "n_edges": int(len(qcfc_df)),
                "mean_qcfc_r": float(np.mean(r_vals)),
                "mean_abs_qcfc_r": float(np.mean(np.abs(r_vals))),
                "median_abs_qcfc_r": float(np.median(np.abs(r_vals))),
                "pct_edges_p_lt_0p05": float(100 * np.mean(qcfc_df["qcfc_p"] < 0.05)),
                "qcfc_distance_r": float(stats.pearsonr(distances, r_vals).statistic),
                "qcfc_distance_p": float(stats.pearsonr(distances, r_vals).pvalue),
                "within_mean_z_min": float(network_df["within_mean_z"].min()),
                "within_mean_z_n_le_zero": int(
                    (network_df["within_mean_z"] <= 0).sum()
                ),
                "within_mean_z_n_abs_lt_0p05": int(
                    (network_df["within_mean_z"].abs() < 0.05).sum()
                ),
            }
        ]
    )
    qcfc_summary.to_csv(output_dir / "qcfc_summary.tsv", sep="\t", index=False)

    print(f"Wrote QC reporting figures to {figures_dir}")
    print(f"Wrote motion-trace figures to {motion_figures_dir}")
    print(f"Wrote QC summaries to {output_dir}")


if __name__ == "__main__":
    main()
