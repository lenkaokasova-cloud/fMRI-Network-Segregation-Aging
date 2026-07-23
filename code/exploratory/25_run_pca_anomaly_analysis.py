#!/usr/bin/env python3

# What this script does:
#   Runs the PCA-based anomaly extension by learning the younger reference
#   pattern and scoring subjects by how unusual they are relative to it.
# How to run it:
#   Run from the repo root with:
#   python code/exploratory/25_run_pca_anomaly_analysis.py
# Main output:
#   data/processed/analysis/pca_anomaly/

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("data/processed/.matplotlib")).resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
REFERENCE_GROUP = "young"
COMPARISON_GROUP = "older"
GLOBAL_SEGREGATION_COL = "global_segregation_prop"
NETWORK_SEGREGATION_COL = "segregation_prop"
YOUNG_COLOR = "#4C78A8"
OLDER_COLOR = "#D16A3A"
REFERENCE_LINE_COLOR = "#1F3A5F"
REFERENCE_RANGE_COLOR = "#3E73B3"
GRID_COLOR = "#B8BDC7"
HEATMAP_CMAP = "coolwarm"
FIG_DPI = 300
TITLE_SIZE = 15
LABEL_SIZE = 12
TICK_SIZE = 11
LEGEND_SIZE = 11
NEUTRAL_COLOR = "#C9CED6"
TEXT_DARK = "#2B2B2B"

plt.rcParams.update(
    {
        "font.size": TICK_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "figure.titlesize": TITLE_SIZE,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a PCA-based anomaly analysis by learning the younger TR=3 s reference "
            "pattern of network segregation and scoring all subjects relative to it."
        )
    )
    parser.add_argument(
        "subjects",
        nargs="*",
        help="Optional subject IDs to analyze after loading the sample and connectivity tables.",
    )
    parser.add_argument(
        "--sample",
        default="data/processed/screening/ds005752_final_analysis_sample_tr_3s.tsv",
        help="Final TR=3 s analysis sample TSV.",
    )
    parser.add_argument(
        "--connectivity-dir",
        default="data/processed/connectivity/metrics",
        help="Directory containing connectivity outputs from code/primary/12_run_connectivity_analysis.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/analysis/pca_anomaly",
        help="Directory for PCA anomaly tables, figures, and summary files.",
    )
    parser.add_argument(
        "--reference-group",
        default=REFERENCE_GROUP,
        help="Age-group label to use as the PCA reference group.",
    )
    parser.add_argument(
        "--comparison-group",
        default=COMPARISON_GROUP,
        help="Age-group label to compare against the PCA reference group.",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.90,
        help="Minimum cumulative explained variance to retain in the PCA model.",
    )
    parser.add_argument(
        "--anomaly-percentile",
        type=float,
        default=0.95,
        help="Reference percentile used to flag unusually large anomaly scores.",
    )
    parser.add_argument(
        "--include-global",
        action="store_true",
        help="Include global segregation alongside the seven network segregation features.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def style_axis(
    ax: plt.Axes,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xrotation: float = 0.0,
) -> None:
    ax.set_title(title, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    for label in ax.get_xticklabels():
        label.set_rotation(xrotation)
        if xrotation:
            label.set_ha("right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_sample(sample_path: Path) -> pd.DataFrame:
    sample = pd.read_csv(sample_path, sep="\t")
    required = {"subject_id", "age", "sex", "age_group"}
    missing = required.difference(sample.columns)
    if missing:
        raise ValueError(f"Sample TSV is missing required columns: {', '.join(sorted(missing))}")
    return sample


def load_connectivity_tables(connectivity_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_df = pd.read_csv(connectivity_dir / "subject_global_segregation.tsv", sep="\t")
    network_df = pd.read_csv(connectivity_dir / "subject_network_segregation.tsv", sep="\t")
    return global_df, network_df


def keep_sample_subjects(
    sample: pd.DataFrame,
    global_df: pd.DataFrame,
    network_df: pd.DataFrame,
    requested_subjects: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if requested_subjects:
        requested = set(requested_subjects)
        sample = sample[sample["subject_id"].isin(requested)].copy()
        missing = requested.difference(set(sample["subject_id"]))
        if missing:
            raise ValueError(f"Requested subjects not found in sample TSV: {', '.join(sorted(missing))}")

    sample_subjects = set(sample["subject_id"])
    global_df = global_df[global_df["subject_id"].isin(sample_subjects)].copy()
    network_df = network_df[network_df["subject_id"].isin(sample_subjects)].copy()

    missing_global = sample_subjects.difference(set(global_df["subject_id"]))
    missing_network = sample_subjects.difference(set(network_df["subject_id"]))
    if missing_global:
        raise ValueError(
            f"Connectivity global table is missing subjects from the sample: {', '.join(sorted(missing_global))}"
        )
    if missing_network:
        raise ValueError(
            f"Connectivity network table is missing subjects from the sample: {', '.join(sorted(missing_network))}"
        )

    return sample, global_df, network_df


def ordered_networks(networks: list[str]) -> list[str]:
    order_index = {network: idx for idx, network in enumerate(NETWORK_ORDER)}
    unique_networks = list(dict.fromkeys(networks))
    return sorted(unique_networks, key=lambda value: (order_index.get(value, len(order_index)), value))


def build_feature_matrix(
    sample: pd.DataFrame,
    global_df: pd.DataFrame,
    network_df: pd.DataFrame,
    include_global: bool,
) -> tuple[pd.DataFrame, list[str]]:
    # This turns each subject into one feature row built from the segregation measures I care about.
    network_order = ordered_networks(network_df["network"].tolist())
    pivot = (
        network_df.pivot(index="subject_id", columns="network", values=NETWORK_SEGREGATION_COL)
        .reindex(columns=network_order)
        .reset_index()
    )

    meta_cols = ["subject_id", "age", "sex", "age_group"]
    subject_df = sample[meta_cols].drop_duplicates("subject_id").merge(pivot, on="subject_id", how="inner")

    global_keep = global_df[
        ["subject_id", GLOBAL_SEGREGATION_COL, "mean_fd", "retained_minutes_after_scrub", "pct_retained_after_scrub"]
    ].drop_duplicates("subject_id")
    subject_df = subject_df.merge(global_keep, on="subject_id", how="left")

    feature_columns = network_order.copy()
    if include_global:
        feature_columns.append(GLOBAL_SEGREGATION_COL)

    missing_feature_values = subject_df[feature_columns].isna().any(axis=1)
    if missing_feature_values.any():
        missing_subjects = subject_df.loc[missing_feature_values, "subject_id"].tolist()
        raise ValueError(
            "Feature matrix contains missing values for subjects: "
            + ", ".join(sorted(missing_subjects))
        )

    return subject_df, feature_columns


def reference_standardize(
    subject_df: pd.DataFrame,
    feature_columns: list[str],
    reference_group: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    # I standardize everything using only the younger group because that is the reference population here.
    reference = subject_df[subject_df["age_group"] == reference_group].copy()
    if reference.empty:
        raise ValueError(f"No subjects found in reference group: {reference_group}")

    means = reference[feature_columns].mean()
    sds = reference[feature_columns].std(ddof=1)
    bad_sds = sds[(~np.isfinite(sds)) | (sds <= 0)]
    if not bad_sds.empty:
        raise ValueError(
            "Cannot standardize PCA features because these reference SDs are invalid: "
            + ", ".join(bad_sds.index.tolist())
        )

    standardized = subject_df.copy()
    standardized[feature_columns] = (subject_df[feature_columns] - means) / sds
    return standardized, means, sds


def fit_pca(reference_std: pd.DataFrame, feature_columns: list[str], variance_threshold: float) -> dict[str, object]:
    # PCA is fit on the younger reference only, so "normal" structure is learned without the older group shaping it.
    if not 0 < variance_threshold <= 1:
        raise ValueError("--variance-threshold must be in the interval (0, 1].")

    x_ref = reference_std[feature_columns].to_numpy(dtype=float)
    n_reference = x_ref.shape[0]
    if n_reference < 3:
        raise ValueError("At least 3 reference subjects are needed for PCA anomaly analysis.")

    _, singular_values, vt = np.linalg.svd(x_ref, full_matrices=False)
    explained_variance = (singular_values**2) / (n_reference - 1)
    explained_ratio = explained_variance / explained_variance.sum()
    cumulative_ratio = np.cumsum(explained_ratio)
    n_components = int(np.searchsorted(cumulative_ratio, variance_threshold) + 1)

    components = vt[:n_components]
    retained_variance = explained_variance[:n_components]

    return {
        "components": components,
        "explained_variance": explained_variance,
        "explained_ratio": explained_ratio,
        "cumulative_ratio": cumulative_ratio,
        "n_components": n_components,
        "retained_variance": retained_variance,
        "feature_columns": feature_columns,
    }


def score_subjects(
    standardized_df: pd.DataFrame,
    feature_columns: list[str],
    pca_model: dict[str, object],
    reference_group: str,
    anomaly_percentile: float,
) -> pd.DataFrame:
    # Subjects look more unusual here if they sit far from the younger PCA space or reconstruct badly from it.
    if not 0 < anomaly_percentile < 1:
        raise ValueError("--anomaly-percentile must be in the interval (0, 1).")

    x_all = standardized_df[feature_columns].to_numpy(dtype=float)
    components = np.asarray(pca_model["components"], dtype=float)
    retained_variance = np.asarray(pca_model["retained_variance"], dtype=float)

    scores = x_all @ components.T
    reconstructed = scores @ components
    residual = x_all - reconstructed
    reconstruction_error = np.sqrt(np.mean(residual**2, axis=1))
    pca_distance = np.sqrt(np.sum((scores**2) / retained_variance, axis=1))

    scored = standardized_df.copy()
    for idx in range(scores.shape[1]):
        scored[f"PC{idx + 1}"] = scores[:, idx]
    scored["reconstruction_error"] = reconstruction_error
    scored["pca_distance"] = pca_distance

    ref_mask = scored["age_group"] == reference_group
    ref_error = scored.loc[ref_mask, "reconstruction_error"]
    ref_distance = scored.loc[ref_mask, "pca_distance"]

    error_mean = float(ref_error.mean())
    error_sd = float(ref_error.std(ddof=1))
    distance_mean = float(ref_distance.mean())
    distance_sd = float(ref_distance.std(ddof=1))
    error_p95 = float(ref_error.quantile(anomaly_percentile))
    distance_p95 = float(ref_distance.quantile(anomaly_percentile))

    scored["reconstruction_error_z"] = (
        (scored["reconstruction_error"] - error_mean) / error_sd if error_sd > 0 else np.nan
    )
    scored["pca_distance_z"] = (
        (scored["pca_distance"] - distance_mean) / distance_sd if distance_sd > 0 else np.nan
    )
    scored["error_outside_reference_p95"] = scored["reconstruction_error"] > error_p95
    scored["distance_outside_reference_p95"] = scored["pca_distance"] > distance_p95
    scored["pca_anomaly_flag"] = (
        scored["error_outside_reference_p95"] | scored["distance_outside_reference_p95"]
    )

    scored["reference_error_mean"] = error_mean
    scored["reference_error_sd"] = error_sd
    scored["reference_error_p95"] = error_p95
    scored["reference_distance_mean"] = distance_mean
    scored["reference_distance_sd"] = distance_sd
    scored["reference_distance_p95"] = distance_p95

    return scored


def build_component_summary(pca_model: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    feature_columns = list(pca_model["feature_columns"])
    components = np.asarray(pca_model["components"])
    explained_variance = np.asarray(pca_model["explained_variance"])
    explained_ratio = np.asarray(pca_model["explained_ratio"])
    cumulative_ratio = np.asarray(pca_model["cumulative_ratio"])

    for idx in range(components.shape[0]):
        row: dict[str, object] = {
            "component": f"PC{idx + 1}",
            "explained_variance": float(explained_variance[idx]),
            "explained_variance_ratio": float(explained_ratio[idx]),
            "cumulative_explained_variance_ratio": float(cumulative_ratio[idx]),
        }
        for feature_name, loading in zip(feature_columns, components[idx]):
            row[feature_name] = float(loading)
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_reference_table(means: pd.Series, sds: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": means.index,
            "reference_mean": means.to_numpy(dtype=float),
            "reference_sd": sds.to_numpy(dtype=float),
        }
    )


def build_group_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    return (
        scored_df.groupby("age_group", sort=False)
        .agg(
            n_subjects=("subject_id", "count"),
            mean_reconstruction_error=("reconstruction_error", "mean"),
            sd_reconstruction_error=("reconstruction_error", "std"),
            mean_pca_distance=("pca_distance", "mean"),
            sd_pca_distance=("pca_distance", "std"),
            n_flagged=("pca_anomaly_flag", "sum"),
            pct_flagged=("pca_anomaly_flag", lambda values: 100 * float(np.mean(values))),
        )
        .reset_index()
    )


def build_older_summary(scored_df: pd.DataFrame, comparison_group: str) -> pd.DataFrame:
    older = scored_df[scored_df["age_group"] == comparison_group].copy()
    return older.sort_values(
        ["pca_anomaly_flag", "reconstruction_error_z", "pca_distance_z", "subject_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def plot_pca_space(
    scored_df: pd.DataFrame,
    component_df: pd.DataFrame,
    reference_group: str,
    comparison_group: str,
    outpath: Path,
) -> None:
    if "PC1" not in scored_df.columns:
        return

    plot_df = scored_df.copy()
    if "PC2" not in plot_df.columns:
        plot_df["PC2"] = 0.0

    reference = plot_df[plot_df["age_group"] == reference_group].copy()
    comparison = plot_df[plot_df["age_group"] == comparison_group].copy()

    pc1_ratio = float(component_df.loc[component_df["component"] == "PC1", "explained_variance_ratio"].iloc[0])
    pc2_ratio = (
        float(component_df.loc[component_df["component"] == "PC2", "explained_variance_ratio"].iloc[0])
        if (component_df["component"] == "PC2").any()
        else 0.0
    )

    fig, ax = plt.subplots(figsize=(7.8, 5.1))
    ax.scatter(
        reference["PC1"],
        reference["PC2"],
        color=YOUNG_COLOR,
        s=60,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    ax.scatter(
        comparison["PC1"],
        comparison["PC2"],
        color=OLDER_COLOR,
        s=70,
        alpha=0.94,
        edgecolors="white",
        linewidths=0.5,
        zorder=4,
    )

    if len(reference) >= 3 and reference["PC2"].std(ddof=1) > 0:
        cov = np.cov(reference[["PC1", "PC2"]].to_numpy().T)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
        scale = np.sqrt(5.991)  # chi-square 95% for 2 df
        width, height = 2 * scale * np.sqrt(np.maximum(evals, 0))
        ellipse = Ellipse(
            xy=(reference["PC1"].mean(), reference["PC2"].mean()),
            width=width,
            height=height,
            angle=angle,
            facecolor="none",
            edgecolor=REFERENCE_RANGE_COLOR,
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
        )
        ax.add_patch(ellipse)
        ax.scatter(
            reference["PC1"].mean(),
            reference["PC2"].mean(),
            color=REFERENCE_LINE_COLOR,
            s=42,
            zorder=5,
        )

    ax.axhline(0.0, color=GRID_COLOR, linewidth=0.9, alpha=0.5)
    ax.axvline(0.0, color=GRID_COLOR, linewidth=0.9, alpha=0.5)
    style_axis(
        ax,
        title="PCA Space Relative to Younger Reference",
        xlabel=f"PC1 score ({pc1_ratio * 100:.1f}% variance)",
        ylabel=f"PC2 score ({pc2_ratio * 100:.1f}% variance)",
    )
    ax.grid(color=GRID_COLOR, alpha=0.22, linewidth=0.8)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=YOUNG_COLOR, markeredgecolor="white", markeredgewidth=0.5, markersize=8.5, label=reference_group),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=OLDER_COLOR, markeredgecolor="white", markeredgewidth=0.5, markersize=9, label=comparison_group),
        Line2D([0], [0], color=REFERENCE_RANGE_COLOR, linestyle="--", linewidth=1.8, label="younger 95% ellipse"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    save_figure(fig, outpath)


def plot_reconstruction_error(
    scored_df: pd.DataFrame,
    reference_group: str,
    comparison_group: str,
    outpath: Path,
) -> None:
    reference = scored_df[scored_df["age_group"] == reference_group].copy()
    comparison = scored_df[scored_df["age_group"] == comparison_group].copy()
    ref_threshold = float(reference["reference_error_p95"].iloc[0])
    ref_mean = float(reference["reference_error_mean"].iloc[0])

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    data = [reference["reconstruction_error"].to_numpy(), comparison["reconstruction_error"].to_numpy()]
    box = ax.boxplot(
        data,
        positions=[0, 1],
        widths=0.44,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 1.1, "color": "#444444"},
        capprops={"linewidth": 1.1, "color": "#444444"},
        boxprops={"linewidth": 1.2, "edgecolor": "#444444"},
    )
    for patch, color in zip(box["boxes"], [YOUNG_COLOR, OLDER_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)

    for x_pos, subset, color in [(0, reference, YOUNG_COLOR), (1, comparison, OLDER_COLOR)]:
        rng = np.random.default_rng(202 + x_pos)
        offsets = rng.uniform(-0.08, 0.08, len(subset))
        ax.scatter(
            np.full(len(subset), x_pos) + offsets,
            subset["reconstruction_error"],
            color=color,
            s=52,
            alpha=0.92,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    ax.axhline(ref_mean, color=REFERENCE_LINE_COLOR, linewidth=2.2, label="younger mean error")
    ax.axhline(
        ref_threshold,
        color=REFERENCE_RANGE_COLOR,
        linewidth=1.8,
        linestyle="--",
        label="younger 95th percentile",
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([reference_group, comparison_group])
    style_axis(
        ax,
        title="Reconstruction Error by Group",
        ylabel="Reconstruction error",
    )
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    save_figure(fig, outpath)


def plot_older_anomaly_scores(older_df: pd.DataFrame, outpath: Path) -> None:
    if older_df.empty:
        return

    ordered = older_df.sort_values(
        ["pca_anomaly_flag", "reconstruction_error", "pca_distance"],
        ascending=[False, False, False],
    ).copy()
    threshold = float(ordered["reference_error_p95"].iloc[0])
    colors = [OLDER_COLOR if flagged else NEUTRAL_COLOR for flagged in ordered["pca_anomaly_flag"]]

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    y_positions = np.arange(len(ordered))
    ax.barh(y_positions, ordered["reconstruction_error"], color=colors)
    ax.axvline(
        threshold,
        color=REFERENCE_RANGE_COLOR,
        linewidth=1.8,
        linestyle="--",
        label="younger 95th percentile",
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered["subject_id"])
    ax.invert_yaxis()
    style_axis(
        ax,
        title="Older Participant Anomaly Scores",
        xlabel="Reconstruction error",
        ylabel="Reconstruction error",
    )
    ax.set_ylabel("Older participant")
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    save_figure(fig, outpath)


def plot_loading_heatmap(component_df: pd.DataFrame, feature_columns: list[str], outpath: Path) -> None:
    if component_df.empty:
        return

    display_df = component_df.copy()
    n_rows = min(3, len(display_df))
    display_df = display_df.iloc[:n_rows]
    matrix = display_df[feature_columns].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(6.2, 0.82 * len(feature_columns)), 2.7 + 0.55 * n_rows))
    im = ax.imshow(matrix, aspect="auto", cmap=HEATMAP_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(feature_columns)))
    ax.set_xticklabels(feature_columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(display_df["component"])
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text_color = "white" if abs(value) >= 0.45 else TEXT_DARK
            ax.text(
                col_idx,
                row_idx,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )
    style_axis(
        ax,
        title="PCA Loadings",
        xlabel="Segregation feature",
        ylabel="Principal component",
        xrotation=35,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Loading", fontsize=LABEL_SIZE)
    save_figure(fig, outpath)


def write_summary_markdown(
    outpath: Path,
    scored_df: pd.DataFrame,
    group_summary: pd.DataFrame,
    older_summary: pd.DataFrame,
    component_df: pd.DataFrame,
    feature_columns: list[str],
    reference_group: str,
    comparison_group: str,
) -> None:
    reference_n = int(group_summary.loc[group_summary["age_group"] == reference_group, "n_subjects"].iloc[0])
    comparison_n = int(group_summary.loc[group_summary["age_group"] == comparison_group, "n_subjects"].iloc[0])
    flagged_older = int(older_summary["pca_anomaly_flag"].sum()) if not older_summary.empty else 0

    top_rows = older_summary.head(5)
    top_lines = []
    for _, row in top_rows.iterrows():
        top_lines.append(
            f"- {row['subject_id']}: reconstruction error={row['reconstruction_error']:.4f}, "
            f"error z={row['reconstruction_error_z']:.4f}, PCA distance={row['pca_distance']:.4f}, "
            f"flagged={bool(row['pca_anomaly_flag'])}"
        )

    component_lines = []
    for _, row in component_df.iterrows():
        top_features = sorted(
            feature_columns,
            key=lambda feature: abs(float(row[feature])),
            reverse=True,
        )[:3]
        feature_text = ", ".join(f"{feature} ({row[feature]:+.3f})" for feature in top_features)
        component_lines.append(
            f"- {row['component']}: explained variance ratio={row['explained_variance_ratio']:.4f}; strongest loadings: {feature_text}"
        )

    lines = [
        "# PCA Anomaly Analysis Summary",
        "",
        f"- Reference group: `{reference_group}` (n={reference_n})",
        f"- Comparison group: `{comparison_group}` (n={comparison_n})",
        f"- Features used: {', '.join(feature_columns)}",
        f"- Older participants flagged as anomalous by PCA criteria: {flagged_older}/{comparison_n}",
        "",
        "## Interpretation",
        "",
        "- PCA was fit only on the younger group so that the principal components describe the younger normative pattern.",
        "- Each subject was then projected into that PCA space and scored using reconstruction error and PCA distance.",
        "- Subjects were flagged when either score exceeded the younger 95th-percentile reference threshold.",
        "",
        "## Most Deviant Older Participants",
        "",
        *top_lines,
        "",
        "## PCA Components",
        "",
        *component_lines,
    ]

    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    sample_path = resolve_project_path(args.sample)
    connectivity_dir = resolve_project_path(args.connectivity_dir)
    output_dir = resolve_project_path(args.output_dir)
    figures_dir = output_dir / "figures"

    sample = load_sample(sample_path)
    global_df, network_df = load_connectivity_tables(connectivity_dir)
    sample, global_df, network_df = keep_sample_subjects(
        sample,
        global_df,
        network_df,
        args.subjects,
    )

    subject_df, feature_columns = build_feature_matrix(
        sample=sample,
        global_df=global_df,
        network_df=network_df,
        include_global=args.include_global,
    )
    standardized_df, means, sds = reference_standardize(
        subject_df=subject_df,
        feature_columns=feature_columns,
        reference_group=args.reference_group,
    )
    reference_std = standardized_df[standardized_df["age_group"] == args.reference_group].copy()
    pca_model = fit_pca(
        reference_std=reference_std,
        feature_columns=feature_columns,
        variance_threshold=args.variance_threshold,
    )
    scored_df = score_subjects(
        standardized_df=standardized_df,
        feature_columns=feature_columns,
        pca_model=pca_model,
        reference_group=args.reference_group,
        anomaly_percentile=args.anomaly_percentile,
    )

    component_df = build_component_summary(pca_model)
    feature_reference_df = build_feature_reference_table(means, sds)
    group_summary_df = build_group_summary(scored_df)
    older_summary_df = build_older_summary(scored_df, args.comparison_group)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    subject_df.to_csv(output_dir / "subject_feature_matrix.tsv", sep="\t", index=False)
    feature_reference_df.to_csv(output_dir / "younger_feature_reference.tsv", sep="\t", index=False)
    component_df.to_csv(output_dir / "pca_component_summary.tsv", sep="\t", index=False)
    scored_df.to_csv(output_dir / "subject_pca_scores.tsv", sep="\t", index=False)
    group_summary_df.to_csv(output_dir / "group_pca_anomaly_summary.tsv", sep="\t", index=False)
    older_summary_df.to_csv(output_dir / "older_pca_anomaly_summary.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "reference_group": args.reference_group,
                "comparison_group": args.comparison_group,
                "variance_threshold": args.variance_threshold,
                "anomaly_percentile": args.anomaly_percentile,
                "include_global": args.include_global,
                "n_features": len(feature_columns),
                "n_components_retained": int(pca_model["n_components"]),
                "cumulative_explained_variance_ratio": float(
                    pca_model["cumulative_ratio"][int(pca_model["n_components"]) - 1]
                ),
            }
        ]
    ).to_csv(output_dir / "pca_settings.tsv", sep="\t", index=False)

    plot_pca_space(
        scored_df=scored_df,
        component_df=component_df,
        reference_group=args.reference_group,
        comparison_group=args.comparison_group,
        outpath=figures_dir / "pca_reference_space.png",
    )
    plot_reconstruction_error(
        scored_df=scored_df,
        reference_group=args.reference_group,
        comparison_group=args.comparison_group,
        outpath=figures_dir / "reconstruction_error_by_group.png",
    )
    plot_older_anomaly_scores(
        older_df=older_summary_df,
        outpath=figures_dir / "older_pca_anomaly_scores.png",
    )
    plot_loading_heatmap(
        component_df=component_df,
        feature_columns=feature_columns,
        outpath=figures_dir / "pca_loading_heatmap.png",
    )

    write_summary_markdown(
        outpath=output_dir / "pca_anomaly_summary.md",
        scored_df=scored_df,
        group_summary=group_summary_df,
        older_summary=older_summary_df,
        component_df=component_df,
        feature_columns=feature_columns,
        reference_group=args.reference_group,
        comparison_group=args.comparison_group,
    )

    print(f"Reference group: {args.reference_group}")
    print(f"Comparison group: {args.comparison_group}")
    print(f"Features used: {', '.join(feature_columns)}")
    print(f"PCA retained {pca_model['n_components']} components")
    print(f"Wrote PCA anomaly tables to {output_dir}")
    print(f"Wrote PCA anomaly figures to {figures_dir}")


if __name__ == "__main__":
    main()
