#!/usr/bin/env python3

# What this script does:
#   Uses the younger group as the reference distribution and scores the older
#   group relative to that younger normative range.
# How to run it:
#   Run from the repo root with:
#   python code/followup/17_run_normative_analysis.py
# Main output:
#   data/processed/analysis/normative/

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("data/processed/.matplotlib")).resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
REFERENCE_GROUP = "young"
COMPARISON_GROUP = "older"
NORMATIVE_Z_THRESHOLD = 1.96
GLOBAL_SEGREGATION_COL = "global_segregation_prop"
NETWORK_SEGREGATION_COL = "segregation_prop"
YOUNG_COLOR = "#4C78A8"
OLDER_COLOR = "#D16A3A"
REFERENCE_LINE_COLOR = "#1F3A5F"
REFERENCE_RANGE_COLOR = "#3E73B3"
POSITIVE_COLOR = "#4C78A8"
NEGATIVE_COLOR = "#D16A3A"
BURDEN_COLOR = "#B55223"
BURDEN_NEUTRAL_COLOR = "#B8BDC7"
GRID_COLOR = "#B8BDC7"
HEATMAP_CMAP = "coolwarm"
FIG_DPI = 300
TITLE_SIZE = 15
LABEL_SIZE = 12
TICK_SIZE = 11
LEGEND_SIZE = 11

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
            "Run a normative resting-state analysis by using the younger TR=3 s sample "
            "as the reference distribution and scoring each older participant relative "
            "to that reference."
        )
    )
    parser.add_argument(
        "subjects",
        nargs="*",
        help=(
            "Optional subject IDs to analyze. If supplied, only these subjects are kept "
            "after loading the sample and connectivity files."
        ),
    )
    parser.add_argument(
        "--sample",
        default="data/processed/screening/ds005752_final_analysis_sample_tr_3s.tsv",
        help="Final TR=3 s analysis sample TSV.",
    )
    parser.add_argument(
        "--connectivity-dir",
        default="data/processed/connectivity/metrics",
        help="Directory containing connectivity outputs from code/primary/11_run_connectivity_analysis.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/analysis/normative",
        help="Directory for normative-analysis tables, figures, and summary files.",
    )
    parser.add_argument(
        "--reference-group",
        default=REFERENCE_GROUP,
        help="Age-group label to use as the normative reference.",
    )
    parser.add_argument(
        "--comparison-group",
        default=COMPARISON_GROUP,
        help="Age-group label to compare against the reference group.",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=NORMATIVE_Z_THRESHOLD,
        help="Absolute z-score threshold used to flag deviation from the reference range.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_sample(sample_path: Path) -> pd.DataFrame:
    sample = pd.read_csv(sample_path, sep="\t")
    required = {"subject_id", "age", "sex", "age_group"}
    missing = required.difference(sample.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Sample TSV is missing required columns: {missing_str}")
    return sample


def load_connectivity_tables(connectivity_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_path = connectivity_dir / "subject_global_segregation.tsv"
    network_path = connectivity_dir / "subject_network_segregation.tsv"
    global_df = pd.read_csv(global_path, sep="\t")
    network_df = pd.read_csv(network_path, sep="\t")

    global_required = {
        "subject_id",
        "age",
        "sex",
        "age_group",
        "retained_volumes",
        "retained_minutes_after_scrub",
        "pct_retained_after_scrub",
        "mean_fd",
        "pct_fd_gt_0p2",
        GLOBAL_SEGREGATION_COL,
    }
    network_required = {
        "subject_id",
        "age",
        "sex",
        "age_group",
        "network",
        "within_mean_z",
        "between_mean_z",
        NETWORK_SEGREGATION_COL,
        "retained_volumes",
        "retained_minutes_after_scrub",
        "mean_fd",
    }

    missing_global = global_required.difference(global_df.columns)
    missing_network = network_required.difference(network_df.columns)
    if missing_global:
        missing_str = ", ".join(sorted(missing_global))
        raise ValueError(f"Global connectivity table is missing required columns: {missing_str}")
    if missing_network:
        missing_str = ", ".join(sorted(missing_network))
        raise ValueError(
            f"Network connectivity table is missing required columns: {missing_str}"
        )

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
        missing_requested = requested.difference(set(sample["subject_id"]))
        if missing_requested:
            missing_str = ", ".join(sorted(missing_requested))
            raise ValueError(f"Requested subjects not found in sample TSV: {missing_str}")

    sample_subjects = set(sample["subject_id"])
    global_subjects = set(global_df["subject_id"])
    network_subjects = set(network_df["subject_id"])

    missing_global = sample_subjects.difference(global_subjects)
    missing_network = sample_subjects.difference(network_subjects)
    if missing_global:
        missing_str = ", ".join(sorted(missing_global))
        raise ValueError(f"Connectivity global table is missing subjects from the sample: {missing_str}")
    if missing_network:
        missing_str = ", ".join(sorted(missing_network))
        raise ValueError(
            f"Connectivity network table is missing subjects from the sample: {missing_str}"
        )

    global_df = global_df[global_df["subject_id"].isin(sample_subjects)].copy()
    network_df = network_df[network_df["subject_id"].isin(sample_subjects)].copy()
    return sample, global_df, network_df


def sample_sd(values: pd.Series) -> float:
    if len(values) < 2:
        return float("nan")
    return float(values.std(ddof=1))


def reference_distribution_table(
    df: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
    reference_group: str,
    z_threshold: float,
) -> pd.DataFrame:
    # This table is the younger reference itself: mean, spread, and practical cutoffs for each metric.
    reference = df[df["age_group"] == reference_group].copy()
    if reference.empty:
        raise ValueError(f"No subjects found in reference group: {reference_group}")

    rows: list[dict[str, object]] = []
    grouped = reference.groupby(group_columns, dropna=False) if group_columns else [((), reference)]
    for key, subset in grouped:
        values = subset[value_column].astype(float)
        mean_value = float(values.mean())
        sd_value = sample_sd(values)
        row: dict[str, object] = {
            "reference_group": reference_group,
            "metric": value_column,
            "n_reference": int(len(subset)),
            "reference_mean": mean_value,
            "reference_sd": sd_value,
            "reference_median": float(values.median()),
            "reference_p025": float(values.quantile(0.025)),
            "reference_p05": float(values.quantile(0.05)),
            "reference_p95": float(values.quantile(0.95)),
            "reference_p975": float(values.quantile(0.975)),
        }
        if np.isfinite(sd_value):
            row["reference_z_lower"] = mean_value - z_threshold * sd_value
            row["reference_z_upper"] = mean_value + z_threshold * sd_value
        else:
            row["reference_z_lower"] = np.nan
            row["reference_z_upper"] = np.nan

        if group_columns:
            if not isinstance(key, tuple):
                key = (key,)
            for column, value in zip(group_columns, key):
                row[column] = value
        rows.append(row)

    columns = group_columns + [
        "reference_group",
        "metric",
        "n_reference",
        "reference_mean",
        "reference_sd",
        "reference_median",
        "reference_p025",
        "reference_p05",
        "reference_p95",
        "reference_p975",
        "reference_z_lower",
        "reference_z_upper",
    ]
    return pd.DataFrame(rows)[columns]


def add_normative_scores(
    df: pd.DataFrame,
    reference_table: pd.DataFrame,
    merge_columns: list[str],
    value_column: str,
    z_threshold: float,
) -> pd.DataFrame:
    # This is the step where raw values turn into deviation-from-younger-reference scores.
    out = df.merge(reference_table, on=merge_columns, how="left", validate="many_to_one")
    deviation = out[value_column] - out["reference_mean"]
    out["deviation_from_reference_mean"] = deviation

    sd = out["reference_sd"].replace(0.0, np.nan)
    out["z_score"] = deviation / sd
    out["abs_z_score"] = out["z_score"].abs()
    out["outside_reference_z95"] = out["abs_z_score"] > z_threshold
    out["below_reference_p05"] = out[value_column] < out["reference_p05"]
    out["above_reference_p95"] = out[value_column] > out["reference_p95"]
    out["outside_reference_empirical_95"] = (
        (out[value_column] < out["reference_p025"]) | (out[value_column] > out["reference_p975"])
    )
    return out


def ordered_networks(networks: list[str]) -> list[str]:
    order_index = {network: idx for idx, network in enumerate(NETWORK_ORDER)}
    unique_networks = list(dict.fromkeys(networks))
    return sorted(
        unique_networks,
        key=lambda name: (order_index.get(name, len(order_index)), name),
    )


def style_axis(
    ax,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xrotation: float = 0.0,
) -> None:
    if title:
        ax.set_title(title, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelrotation=xrotation)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right" if xrotation else "center")


def save_figure(fig: plt.Figure, outpath: Path) -> None:
    fig.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_sample_summary(global_df: pd.DataFrame) -> pd.DataFrame:
    return (
        global_df.groupby("age_group", as_index=False)
        .agg(
            n_subjects=("subject_id", "count"),
            age_mean=("age", "mean"),
            age_sd=("age", "std"),
            mean_fd_mean=("mean_fd", "mean"),
            mean_fd_sd=("mean_fd", "std"),
            retained_minutes_mean=("retained_minutes_after_scrub", "mean"),
            retained_minutes_sd=("retained_minutes_after_scrub", "std"),
            global_segregation_mean=(GLOBAL_SEGREGATION_COL, "mean"),
            global_segregation_sd=(GLOBAL_SEGREGATION_COL, "std"),
        )
        .sort_values("age_group")
    )


def build_network_summary(normative_network_df: pd.DataFrame, comparison_group: str) -> pd.DataFrame:
    older = normative_network_df[normative_network_df["age_group"] == comparison_group].copy()
    rows: list[dict[str, object]] = []
    for network, subset in older.groupby("network", sort=False):
        rows.append(
            {
                "network": network,
                "n_older": int(len(subset)),
                "reference_mean": float(subset["reference_mean"].iloc[0]),
                "reference_sd": float(subset["reference_sd"].iloc[0]),
                "reference_p025": float(subset["reference_p025"].iloc[0]),
                "reference_p975": float(subset["reference_p975"].iloc[0]),
                "older_mean": float(subset[NETWORK_SEGREGATION_COL].mean()),
                "older_sd": sample_sd(subset[NETWORK_SEGREGATION_COL]),
                "mean_older_minus_reference": float(subset["deviation_from_reference_mean"].mean()),
                "mean_older_z_score": float(subset["z_score"].mean()),
                "median_older_z_score": float(subset["z_score"].median()),
                "pct_older_outside_reference_z95": float(100 * subset["outside_reference_z95"].mean()),
                "pct_older_below_reference_p05": float(100 * subset["below_reference_p05"].mean()),
                "pct_older_above_reference_p95": float(100 * subset["above_reference_p95"].mean()),
                "pct_older_outside_reference_empirical_95": float(
                    100 * subset["outside_reference_empirical_95"].mean()
                ),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["network"] = pd.Categorical(out["network"], categories=ordered_networks(out["network"].tolist()), ordered=True)
        out = out.sort_values("network").reset_index(drop=True)
    return out


def build_global_summary(normative_global_df: pd.DataFrame, comparison_group: str) -> pd.DataFrame:
    older = normative_global_df[normative_global_df["age_group"] == comparison_group].copy()
    if older.empty:
        raise ValueError(f"No subjects found in comparison group: {comparison_group}")
    return pd.DataFrame(
        [
            {
                "metric": GLOBAL_SEGREGATION_COL,
                "n_older": int(len(older)),
                "reference_mean": float(older["reference_mean"].iloc[0]),
                "reference_sd": float(older["reference_sd"].iloc[0]),
                "reference_p025": float(older["reference_p025"].iloc[0]),
                "reference_p975": float(older["reference_p975"].iloc[0]),
                "older_mean": float(older[GLOBAL_SEGREGATION_COL].mean()),
                "older_sd": sample_sd(older[GLOBAL_SEGREGATION_COL]),
                "mean_older_minus_reference": float(older["deviation_from_reference_mean"].mean()),
                "mean_older_z_score": float(older["z_score"].mean()),
                "median_older_z_score": float(older["z_score"].median()),
                "pct_older_outside_reference_z95": float(100 * older["outside_reference_z95"].mean()),
                "pct_older_below_reference_p05": float(100 * older["below_reference_p05"].mean()),
                "pct_older_above_reference_p95": float(100 * older["above_reference_p95"].mean()),
                "pct_older_outside_reference_empirical_95": float(
                    100 * older["outside_reference_empirical_95"].mean()
                ),
            }
        ]
    )


def build_subject_summary(
    normative_global_df: pd.DataFrame,
    normative_network_df: pd.DataFrame,
    comparison_group: str,
) -> pd.DataFrame:
    older_global = normative_global_df[normative_global_df["age_group"] == comparison_group].copy()
    older_network = normative_network_df[normative_network_df["age_group"] == comparison_group].copy()

    rows: list[dict[str, object]] = []
    for subject, network_subset in older_network.groupby("subject_id", sort=False):
        global_row = older_global[older_global["subject_id"] == subject].iloc[0]
        top_row = network_subset.loc[network_subset["abs_z_score"].idxmax()]
        rows.append(
            {
                "subject_id": subject,
                "age": int(global_row["age"]),
                "sex": global_row["sex"],
                "mean_fd": float(global_row["mean_fd"]),
                "retained_minutes_after_scrub": float(global_row["retained_minutes_after_scrub"]),
                "global_segregation": float(global_row[GLOBAL_SEGREGATION_COL]),
                "global_z_score": float(global_row["z_score"]),
                "global_outside_reference_z95": int(bool(global_row["outside_reference_z95"])),
                "n_networks": int(len(network_subset)),
                "n_networks_outside_reference_z95": int(network_subset["outside_reference_z95"].sum()),
                "n_networks_below_reference_p05": int(network_subset["below_reference_p05"].sum()),
                "mean_abs_network_z_score": float(network_subset["abs_z_score"].mean()),
                "max_abs_network_z_score": float(network_subset["abs_z_score"].max()),
                "most_deviant_network": top_row["network"],
                "most_deviant_network_z_score": float(top_row["z_score"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["max_abs_network_z_score", "subject_id"], ascending=[False, True])


def plot_global_reference(
    normative_global_df: pd.DataFrame,
    reference_group: str,
    comparison_group: str,
    outpath: Path,
) -> None:
    reference = normative_global_df[normative_global_df["age_group"] == reference_group].copy()
    older = normative_global_df[normative_global_df["age_group"] == comparison_group].copy()
    ref_mean = float(reference["reference_mean"].iloc[0])
    ref_lower = float(reference["reference_z_lower"].iloc[0])
    ref_upper = float(reference["reference_z_upper"].iloc[0])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ref_x = np.zeros(len(reference))
    old_x = np.ones(len(older))
    if len(reference):
        ref_x += np.linspace(-0.12, 0.12, len(reference))
    if len(older):
        old_x += np.linspace(-0.12, 0.12, len(older))

    ax.scatter(ref_x, reference[GLOBAL_SEGREGATION_COL], color=YOUNG_COLOR, s=60, label=reference_group)
    ax.scatter(old_x, older[GLOBAL_SEGREGATION_COL], color=OLDER_COLOR, s=60, label=comparison_group)
    ax.axhline(ref_mean, color=REFERENCE_LINE_COLOR, linestyle="-", linewidth=2.2, alpha=0.95)
    ax.axhline(ref_lower, color=REFERENCE_RANGE_COLOR, linestyle="--", linewidth=1.6, alpha=0.9)
    ax.axhline(ref_upper, color=REFERENCE_RANGE_COLOR, linestyle="--", linewidth=1.6, alpha=0.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([reference_group, comparison_group])
    style_axis(
        ax,
        title="Global Segregation Relative to Younger Norms",
        ylabel="Global segregation",
    )
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    save_figure(fig, outpath)


def plot_global_distribution(
    normative_global_df: pd.DataFrame,
    reference_group: str,
    comparison_group: str,
    outpath: Path,
) -> None:
    reference = normative_global_df[normative_global_df["age_group"] == reference_group].copy()
    older = normative_global_df[normative_global_df["age_group"] == comparison_group].copy()
    ref_mean = float(reference["reference_mean"].iloc[0])
    ref_lower = float(reference["reference_p025"].iloc[0])
    ref_upper = float(reference["reference_p975"].iloc[0])

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    data = [reference[GLOBAL_SEGREGATION_COL].to_numpy(), older[GLOBAL_SEGREGATION_COL].to_numpy()]
    box = ax.boxplot(
        data,
        positions=[0, 1],
        widths=0.45,
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

    for x_pos, subset, color in [(0, reference, YOUNG_COLOR), (1, older, OLDER_COLOR)]:
        rng = np.random.default_rng(100 + x_pos)
        offsets = rng.uniform(-0.08, 0.08, len(subset)) if len(subset) else np.array([])
        ax.scatter(
            np.full(len(subset), x_pos) + offsets,
            subset[GLOBAL_SEGREGATION_COL],
            color=color,
            s=52,
            alpha=0.92,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    ax.axhline(
        ref_mean,
        color=REFERENCE_LINE_COLOR,
        linewidth=2.2,
        linestyle="-",
        alpha=0.95,
        label="younger mean",
    )
    ax.axhline(
        ref_lower,
        color=REFERENCE_RANGE_COLOR,
        linewidth=1.6,
        linestyle="--",
        alpha=0.9,
        label="younger 95% range",
    )
    ax.axhline(ref_upper, color=REFERENCE_RANGE_COLOR, linewidth=1.6, linestyle="--", alpha=0.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([reference_group, comparison_group])
    style_axis(
        ax,
        title="Global Segregation by Group",
        ylabel="Global segregation",
    )
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    save_figure(fig, outpath)


def plot_network_mean_z(network_summary: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [NEGATIVE_COLOR if value < 0 else POSITIVE_COLOR for value in network_summary["mean_older_z_score"]]
    ax.bar(network_summary["network"], network_summary["mean_older_z_score"], color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axhline(-NORMATIVE_Z_THRESHOLD, color="gray", linestyle="--", linewidth=1)
    ax.axhline(NORMATIVE_Z_THRESHOLD, color="gray", linestyle="--", linewidth=1)
    style_axis(
        ax,
        title="Mean Older Deviation by Network",
        ylabel="Mean z-score vs younger reference",
        xrotation=35,
    )
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    save_figure(fig, outpath)


def plot_network_reference_ranges(
    normative_network_df: pd.DataFrame,
    reference_group: str,
    comparison_group: str,
    outpath: Path,
) -> None:
    reference = normative_network_df[normative_network_df["age_group"] == reference_group].copy()
    older = normative_network_df[normative_network_df["age_group"] == comparison_group].copy()
    network_order = ordered_networks(normative_network_df["network"].tolist())

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x_positions = np.arange(len(network_order))

    for idx, network in enumerate(network_order):
        ref_subset = reference[reference["network"] == network]
        old_subset = older[older["network"] == network]
        ref_mean = float(ref_subset["reference_mean"].iloc[0])
        ref_lower = float(ref_subset["reference_p025"].iloc[0])
        ref_upper = float(ref_subset["reference_p975"].iloc[0])

        ax.vlines(idx, ref_lower, ref_upper, color=REFERENCE_RANGE_COLOR, linewidth=3.2, alpha=0.95)
        ax.hlines(ref_mean, idx - 0.18, idx + 0.18, color=REFERENCE_LINE_COLOR, linewidth=2.6)

        if len(ref_subset):
            ref_offsets = np.linspace(-0.12, 0.12, len(ref_subset))
            ax.scatter(
                np.full(len(ref_subset), idx) + ref_offsets,
                ref_subset[NETWORK_SEGREGATION_COL],
                color=YOUNG_COLOR,
                alpha=0.30,
                s=25,
                zorder=2,
            )
        if len(old_subset):
            old_offsets = np.linspace(-0.08, 0.08, len(old_subset))
            ax.scatter(
                np.full(len(old_subset), idx) + old_offsets,
                old_subset[NETWORK_SEGREGATION_COL],
                color=OLDER_COLOR,
                s=42,
                zorder=3,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(network_order, rotation=35, ha="right")
    style_axis(
        ax,
        title="Network Segregation Relative to Younger Reference Ranges",
        ylabel="Network segregation",
        xrotation=35,
    )
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    ax.scatter([], [], color=YOUNG_COLOR, alpha=0.30, s=25, label=reference_group)
    ax.scatter([], [], color=OLDER_COLOR, s=42, label=comparison_group)
    ax.plot([], [], color=REFERENCE_RANGE_COLOR, linewidth=3.2, label="younger 95% range")
    ax.plot([], [], color=REFERENCE_LINE_COLOR, linewidth=2.6, label="younger mean")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    save_figure(fig, outpath)


def plot_network_outside_rate(network_summary: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        network_summary["network"],
        network_summary["pct_older_outside_reference_z95"],
        color="#7B6FD0",
    )
    style_axis(
        ax,
        title="Older Participants Outside Younger Network Range",
        ylabel="% older outside |z| > 1.96",
        xrotation=35,
    )
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    save_figure(fig, outpath)


def plot_out_of_range_burden(subject_summary: pd.DataFrame, outpath: Path) -> None:
    ordered = subject_summary.sort_values(
        ["n_networks_outside_reference_z95", "max_abs_network_z_score", "subject_id"],
        ascending=[False, False, True],
    ).copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = [
        BURDEN_COLOR if count > 0 else BURDEN_NEUTRAL_COLOR
        for count in ordered["n_networks_outside_reference_z95"]
    ]
    ax.bar(ordered["subject_id"], ordered["n_networks_outside_reference_z95"], color=colors)
    style_axis(
        ax,
        title="Older Participant Deviation Burden",
        xlabel="Older participant",
        ylabel="Networks outside younger |z| > 1.96",
        xrotation=45,
    )
    ax.set_ylim(0, max(ordered["n_networks_outside_reference_z95"].max() + 0.8, 1.2))
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    save_figure(fig, outpath)


def plot_older_network_heatmap(normative_network_df: pd.DataFrame, comparison_group: str, outpath: Path) -> None:
    older = normative_network_df[normative_network_df["age_group"] == comparison_group].copy()
    subject_order = older.groupby("subject_id")["abs_z_score"].mean().sort_values(ascending=False).index.tolist()
    network_order = ordered_networks(older["network"].tolist())
    heatmap_df = (
        older.pivot(index="subject_id", columns="network", values="z_score")
        .reindex(index=subject_order, columns=network_order)
    )

    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.35 * len(heatmap_df))))
    im = ax.imshow(heatmap_df.to_numpy(), aspect="auto", cmap=HEATMAP_CMAP, vmin=-3, vmax=3)
    ax.set_xticks(np.arange(len(network_order)))
    ax.set_xticklabels(network_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(subject_order)))
    ax.set_yticklabels(subject_order)
    style_axis(
        ax,
        title="Older Participant Network Deviation Profiles",
        xlabel="Network",
        ylabel="Older participant",
        xrotation=45,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("z-score vs younger reference", fontsize=LABEL_SIZE)
    save_figure(fig, outpath)


def write_summary_markdown(
    outpath: Path,
    sample_summary: pd.DataFrame,
    global_summary: pd.DataFrame,
    network_summary: pd.DataFrame,
    subject_summary: pd.DataFrame,
    reference_group: str,
    comparison_group: str,
) -> None:
    reference_n = int(sample_summary.loc[sample_summary["age_group"] == reference_group, "n_subjects"].iloc[0])
    comparison_n = int(sample_summary.loc[sample_summary["age_group"] == comparison_group, "n_subjects"].iloc[0])
    global_row = global_summary.iloc[0]
    most_deviant = network_summary.reindex(
        network_summary["mean_older_z_score"].abs().sort_values(ascending=False).index
    ).head(3)
    highest_outside = network_summary.sort_values(
        "pct_older_outside_reference_z95", ascending=False
    ).head(3)

    lines = [
        "# Normative Analysis Summary",
        "",
        "## Design",
        "",
        (
            f"- Reference group: {reference_group} (n={reference_n})"
        ),
        (
            f"- Comparison group: {comparison_group} (n={comparison_n})"
        ),
        "- Primary outcomes: one global segregation metric plus seven network-specific segregation metrics",
        "- Older participants were standardized relative to the younger reference mean and SD",
        f"- Primary flag for deviation: |z| > {NORMATIVE_Z_THRESHOLD:.2f}",
        "",
        "## Global Segregation",
        "",
        (
            f"- Younger reference mean={global_row['reference_mean']:.4f}, "
            f"SD={global_row['reference_sd']:.4f}"
        ),
        (
            f"- Older mean={global_row['older_mean']:.4f}, "
            f"mean older z-score={global_row['mean_older_z_score']:.4f}"
        ),
        (
            f"- {global_row['pct_older_outside_reference_z95']:.1f}% of older participants "
            "fell outside the younger |z| > 1.96 range"
        ),
        "",
        "## Networks With Largest Mean Older Deviation",
        "",
    ]

    for _, row in most_deviant.iterrows():
        lines.append(
            f"- {row['network']}: mean older z-score={row['mean_older_z_score']:.4f}, "
            f"older mean - younger mean={row['mean_older_minus_reference']:.4f}"
        )

    lines.extend(["", "## Networks With Highest Older Out-of-Range Rate", ""])
    for _, row in highest_outside.iterrows():
        lines.append(
            f"- {row['network']}: {row['pct_older_outside_reference_z95']:.1f}% outside |z| > 1.96"
        )

    lines.extend(
        [
            "",
            "## Older-Participant Summary",
            "",
            (
                f"- Mean number of out-of-range networks per older participant: "
                f"{subject_summary['n_networks_outside_reference_z95'].mean():.2f}"
            ),
            (
                f"- Maximum number of out-of-range networks in one older participant: "
                f"{int(subject_summary['n_networks_outside_reference_z95'].max())}"
            ),
        ]
    )

    outpath.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    sample_path = resolve_project_path(args.sample)
    connectivity_dir = resolve_project_path(args.connectivity_dir)
    output_dir = resolve_project_path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    sample = load_sample(sample_path)
    global_df, network_df = load_connectivity_tables(connectivity_dir)
    sample, global_df, network_df = keep_sample_subjects(
        sample,
        global_df,
        network_df,
        args.subjects,
    )

    if args.reference_group not in set(sample["age_group"]):
        raise ValueError(f"Reference group not present in sample: {args.reference_group}")
    if args.comparison_group not in set(sample["age_group"]):
        raise ValueError(f"Comparison group not present in sample: {args.comparison_group}")

    sample_summary = build_sample_summary(global_df)

    reference_global = reference_distribution_table(
        global_df,
        group_columns=[],
        value_column=GLOBAL_SEGREGATION_COL,
        reference_group=args.reference_group,
        z_threshold=args.z_threshold,
    )
    reference_network = reference_distribution_table(
        network_df,
        group_columns=["network"],
        value_column=NETWORK_SEGREGATION_COL,
        reference_group=args.reference_group,
        z_threshold=args.z_threshold,
    )

    global_df = global_df.copy()
    global_df["metric"] = GLOBAL_SEGREGATION_COL
    network_df = network_df.copy()
    network_df["metric"] = NETWORK_SEGREGATION_COL

    normative_global = add_normative_scores(
        global_df,
        reference_global,
        merge_columns=["metric"],
        value_column=GLOBAL_SEGREGATION_COL,
        z_threshold=args.z_threshold,
    )
    normative_network = add_normative_scores(
        network_df,
        reference_network,
        merge_columns=["network", "metric"],
        value_column=NETWORK_SEGREGATION_COL,
        z_threshold=args.z_threshold,
    )

    older_global = normative_global[normative_global["age_group"] == args.comparison_group].copy()
    older_network = normative_network[normative_network["age_group"] == args.comparison_group].copy()

    global_summary = build_global_summary(normative_global, args.comparison_group)
    network_summary = build_network_summary(normative_network, args.comparison_group)
    subject_summary = build_subject_summary(
        normative_global,
        normative_network,
        args.comparison_group,
    )

    reference_global.to_csv(output_dir / "reference_global_distribution.tsv", sep="\t", index=False)
    reference_network.to_csv(output_dir / "reference_network_distribution.tsv", sep="\t", index=False)
    normative_global.to_csv(output_dir / "all_subject_global_normative_scores.tsv", sep="\t", index=False)
    normative_network.to_csv(output_dir / "all_subject_network_normative_scores.tsv", sep="\t", index=False)
    older_global.to_csv(output_dir / "older_global_normative_scores.tsv", sep="\t", index=False)
    older_network.to_csv(output_dir / "older_network_normative_scores.tsv", sep="\t", index=False)
    global_summary.to_csv(output_dir / "global_normative_summary.tsv", sep="\t", index=False)
    network_summary.to_csv(output_dir / "network_normative_summary.tsv", sep="\t", index=False)
    subject_summary.to_csv(output_dir / "older_subject_normative_summary.tsv", sep="\t", index=False)
    sample_summary.to_csv(output_dir / "sample_summary.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "sample": str(sample_path),
                "connectivity_dir": str(connectivity_dir),
                "reference_group": args.reference_group,
                "comparison_group": args.comparison_group,
                "z_threshold": args.z_threshold,
                "primary_global_metric": GLOBAL_SEGREGATION_COL,
                "primary_network_metric": NETWORK_SEGREGATION_COL,
            }
        ]
    ).to_csv(output_dir / "normative_settings.tsv", sep="\t", index=False)

    plot_global_reference(
        normative_global,
        args.reference_group,
        args.comparison_group,
        figures_dir / "global_normative_reference.png",
    )
    plot_global_distribution(
        normative_global,
        args.reference_group,
        args.comparison_group,
        figures_dir / "global_segregation_distribution.png",
    )
    plot_network_mean_z(network_summary, figures_dir / "network_mean_older_z_score.png")
    plot_network_reference_ranges(
        normative_network,
        args.reference_group,
        args.comparison_group,
        figures_dir / "network_reference_ranges.png",
    )
    plot_network_outside_rate(
        network_summary, figures_dir / "network_pct_older_outside_reference_z95.png"
    )
    plot_out_of_range_burden(
        subject_summary,
        figures_dir / "older_network_deviation_burden.png",
    )
    plot_older_network_heatmap(
        normative_network,
        args.comparison_group,
        figures_dir / "older_network_z_score_heatmap.png",
    )
    write_summary_markdown(
        output_dir / "normative_analysis_summary.md",
        sample_summary,
        global_summary,
        network_summary,
        subject_summary,
        args.reference_group,
        args.comparison_group,
    )

    print(f"Reference group: {args.reference_group}")
    print(f"Comparison group: {args.comparison_group}")
    print(f"Wrote normative-analysis tables to {output_dir}")
    print(f"Wrote normative-analysis figures to {figures_dir}")


if __name__ == "__main__":
    main()
