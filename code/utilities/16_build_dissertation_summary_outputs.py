#!/usr/bin/env python3

# What this script does:
#   Collects the main analysis outputs and rewrites them into cleaner summary
#   tables and figures that are easier to use in the dissertation.
# How to run it:
#   Run from the repo root with:
#   python code/utilities/16_build_dissertation_summary_outputs.py
# Main output:
#   data/processed/analysis/dissertation_summary/
#   This now includes:
#   - the main age-effects summary table and figure
#   - a global-effect sensitivity forest-style figure
#   - a 7 x 7 older-minus-younger network connectivity heatmap

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
FIG_DPI = 300
TITLE_SIZE = 14
LABEL_SIZE = 11
TICK_SIZE = 10
TEXT_COLOR = "#223245"
SUBTLE_TEXT_COLOR = "#33485D"
AXIS_TEXT_COLOR = "#18222D"
GRID_COLOR = "#D4DAE3"
REFERENCE_LINE_COLOR = "#6A7D92"
OVERALL_COLOR = "#355D4C"
NETWORK_COLOR = "#7A9C8B"
HIGHLIGHT_COLOR = "#5E7B5F"
HEADLINE_AXIS_LABEL_SIZE = 12.5
HEADLINE_TICK_LABEL_SIZE = 11.5
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
            "Build dissertation-ready summary outputs from the main permutation "
            "results: one compact headline table and one coefficient-style figure."
        )
    )
    parser.add_argument(
        "--sample",
        default="data/processed/screening/ds005752_final_analysis_sample_tr_3s.tsv",
        help="Final TR = 3 s analysis sample TSV used to build Table 1.",
    )
    parser.add_argument(
        "--permutation-dir",
        default="data/processed/analysis/permutation_inference",
        help="Directory containing overall_permutation_summary.tsv and network_permutation_summary.tsv.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/analysis/dissertation_summary",
        help="Directory for the summary table and figure.",
    )
    parser.add_argument(
        "--sensitivity-branch-table",
        default="data/processed/analysis/literature_sensitivity/sensitivity_branch_comparison.tsv",
        help="TSV summarizing the main literature-style sensitivity branches.",
    )
    parser.add_argument(
        "--stricter-motion-table",
        default="data/processed/analysis/robustness/stricter_motion_global.tsv",
        help="TSV summarizing the stricter motion subset global model.",
    )
    parser.add_argument(
        "--retained-minutes-table",
        default="data/processed/analysis/robustness/retained_minutes_covariate_global.tsv",
        help="TSV summarizing the retained-minutes covariate sensitivity model.",
    )
    parser.add_argument(
        "--balanced-overall-table",
        default="data/processed/analysis/balanced_tr3_11x11_sensitivity/permutation_inference/overall_permutation_summary.tsv",
        help="Overall permutation summary TSV for the balanced 11 x 11 sensitivity branch.",
    )
    parser.add_argument(
        "--matrices-dir",
        default="data/processed/connectivity/matrices",
        help="Directory containing subject Fisher z connectivity matrices.",
    )
    parser.add_argument(
        "--atlas-labels",
        default="data/processed/connectivity/metrics/atlas_labels.tsv",
        help="Atlas labels TSV with parcel-to-network assignments.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def save_figure(fig: plt.Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def matrix_subject_id_from_path(path: Path) -> str:
    return path.name.replace("_forward_z_matrix.npy", "")


def clean_overall_table(overall_df: pd.DataFrame) -> pd.DataFrame:
    keep = overall_df[overall_df["outcome"].isin(
        [
            "global_segregation_prop",
            "overall_within_mean",
            "overall_between_mean",
        ]
    )].copy()

    label_map = {
        "global_segregation_prop": "Global segregation",
        "overall_within_mean": "Overall within-network connectivity",
        "overall_between_mean": "Overall between-network connectivity",
    }
    order_map = {
        "global_segregation_prop": 0,
        "overall_within_mean": 1,
        "overall_between_mean": 2,
    }
    keep["result_group"] = "overall"
    keep["network"] = ""
    keep["result_label"] = keep["outcome"].map(label_map)
    keep["display_order"] = keep["outcome"].map(order_map)
    return keep


def clean_network_table(network_df: pd.DataFrame) -> pd.DataFrame:
    keep = network_df[network_df["component"].astype(str) == "segregation_prop"].copy()
    network_order_map = {name: idx for idx, name in enumerate(NETWORK_ORDER)}
    keep["result_group"] = "network_specific"
    keep["result_label"] = keep["network"].astype(str) + " segregation"
    keep["display_order"] = keep["network"].astype(str).map(network_order_map)
    return keep


def build_headline_table(overall_df: pd.DataFrame, network_df: pd.DataFrame) -> pd.DataFrame:
    overall = clean_overall_table(overall_df)
    network = clean_network_table(network_df)
    combined = pd.concat([overall, network], ignore_index=True, sort=False)
    combined["group_order"] = combined["result_group"].map({"overall": 0, "network_specific": 1})
    combined["significant_fdr_lt_0p05"] = combined["fdr_q_value"].astype(float) < 0.05
    combined["ci_text"] = combined.apply(
        lambda row: f"[{row['conf_low']:.4f}, {row['conf_high']:.4f}]",
        axis=1,
    )
    combined = combined[
        [
            "result_group",
            "group_order",
            "display_order",
            "result_label",
            "network",
            "estimate_older_vs_young",
            "conf_low",
            "conf_high",
            "ci_text",
            "permutation_p_value",
            "fdr_q_value",
            "young_mean",
            "older_mean",
            "older_minus_young_mean",
            "n_subjects",
            "significant_fdr_lt_0p05",
        ]
    ].sort_values(["group_order", "display_order", "result_label"]).reset_index(drop=True)
    combined = combined.drop(columns=["group_order"])
    return combined


def build_word_friendly_table(headline_table: pd.DataFrame) -> pd.DataFrame:
    # This is the cleaner dissertation table version with friendlier labels and rounded values for Word.
    out = headline_table.copy()
    out["section"] = out["result_group"].map(
        {
            "overall": "Overall outcomes",
            "network_specific": "Network-specific segregation",
        }
    )
    out["outcome"] = out["result_label"]
    out["young_mean"] = out["young_mean"].astype(float).round(3)
    out["older_mean"] = out["older_mean"].astype(float).round(3)
    out["older_minus_young_mean"] = out["older_minus_young_mean"].astype(float).round(3)
    out["estimate_older_vs_young"] = out["estimate_older_vs_young"].astype(float).round(3)
    out["permutation_p_value"] = out["permutation_p_value"].astype(float).round(3)
    out["fdr_q_value"] = out["fdr_q_value"].astype(float).round(3)
    out["ci_95"] = out.apply(
        lambda row: f"[{row['conf_low']:.3f}, {row['conf_high']:.3f}]",
        axis=1,
    )
    out["fdr_sig"] = out["significant_fdr_lt_0p05"].map({True: "yes", False: "no"})
    out = out[
        [
            "section",
            "outcome",
            "young_mean",
            "older_mean",
            "older_minus_young_mean",
            "estimate_older_vs_young",
            "ci_95",
            "permutation_p_value",
            "fdr_q_value",
            "fdr_sig",
        ]
    ]
    return out.rename(
        columns={
            "section": "Section",
            "outcome": "Outcome",
            "young_mean": "Young mean",
            "older_mean": "Older mean",
            "older_minus_young_mean": "Older minus younger difference",
            "estimate_older_vs_young": "Age-group estimate (older vs younger)",
            "ci_95": "95% CI (model-based)",
            "permutation_p_value": "Permutation p",
            "fdr_q_value": "FDR q",
            "fdr_sig": "FDR < .05",
        }
    )


def format_mean_sd(series: pd.Series, decimals: int = 2) -> str:
    series = pd.to_numeric(series, errors="coerce")
    return f"{series.mean():.{decimals}f} ({series.std(ddof=1):.{decimals}f})"


def format_range(series: pd.Series, decimals: int = 0) -> str:
    series = pd.to_numeric(series, errors="coerce")
    return f"{series.min():.{decimals}f}-{series.max():.{decimals}f}"


def format_count_pct(mask: pd.Series, total_n: int, decimals: int = 1) -> str:
    count = int(mask.sum())
    pct = 100.0 * count / total_n if total_n else float("nan")
    return f"{count} ({pct:.{decimals}f}%)"


def build_participant_characteristics_table(sample_df: pd.DataFrame) -> pd.DataFrame:
    # This builds a dissertation-style Table 1 with Total / Young / Older columns from the locked final sample.
    groups = {
        "Total": sample_df.copy(),
        "Young (20-25)": sample_df[sample_df["age_group"] == "young"].copy(),
        "Older (50-75)": sample_df[sample_df["age_group"] == "older"].copy(),
    }

    rows: list[dict[str, object]] = []

    def add_row(label: str, value_fn) -> None:
        row = {"Characteristic": label}
        for group_name, group_df in groups.items():
            row[group_name] = value_fn(group_df)
        rows.append(row)

    add_row("N", lambda df: str(len(df)))
    add_row("Age, years, mean (SD)", lambda df: format_mean_sd(df["age"], decimals=2))
    add_row("Age range, years", lambda df: format_range(df["age"], decimals=0))
    add_row("Female, n (%)", lambda df: format_count_pct(df["sex"].astype(str).eq("female"), len(df)))
    add_row("Male, n (%)", lambda df: format_count_pct(df["sex"].astype(str).eq("male"), len(df)))
    add_row("Right-handed, n (%)", lambda df: format_count_pct(df["handedness"].astype(str).eq("right"), len(df)))
    add_row(
        "Mean framewise displacement, mm, mean (SD)",
        lambda df: format_mean_sd(df["mean_fd"], decimals=3),
    )
    add_row(
        "Retained time after censoring, min, mean (SD)",
        lambda df: format_mean_sd(df["retained_minutes_after_scrub"], decimals=2),
    )
    add_row(
        "Retained volumes after censoring, mean (SD)",
        lambda df: format_mean_sd(df["n_retained_after_scrub"], decimals=1),
    )

    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT_COLOR)
    ax.spines["bottom"].set_color(TEXT_COLOR)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(length=4.2, width=0.8, color=TEXT_COLOR)
    ax.grid(axis="x", color=GRID_COLOR, alpha=0.62, linewidth=0.85)
    ax.axvline(0.0, color=REFERENCE_LINE_COLOR, linewidth=1.2, alpha=0.92)


def style_y_grid_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT_COLOR)
    ax.spines["bottom"].set_color(TEXT_COLOR)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(length=4.2, width=0.8, color=TEXT_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.62, linewidth=0.85)
    ax.axvline(0.0, color=REFERENCE_LINE_COLOR, linewidth=1.2, alpha=0.92)


def add_q_labels(ax: plt.Axes, values: np.ndarray, y_positions: np.ndarray, q_values: np.ndarray) -> None:
    finite_values = values[np.isfinite(values)]
    finite_lows = []
    finite_highs = []
    for val, q in zip(values, q_values):
        if np.isfinite(val) and np.isfinite(q):
            finite_lows.append(val)
            finite_highs.append(val)
    max_x = float(np.nanmax(finite_values)) if len(finite_values) else 0.0
    x_text = max_x + 0.02
    for y, q in zip(y_positions, q_values):
        if not np.isfinite(q):
            continue
        ax.text(
            x_text,
            y,
            f"q = {q:.3f}",
            va="center",
            ha="left",
            fontsize=9.0,
            color=TEXT_COLOR,
        )


def darken_axis_text(ax: plt.Axes) -> None:
    ax.xaxis.label.set_color(AXIS_TEXT_COLOR)
    ax.yaxis.label.set_color(AXIS_TEXT_COLOR)
    ax.tick_params(axis="both", labelcolor=AXIS_TEXT_COLOR)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(AXIS_TEXT_COLOR)


def enlarge_headline_axis_text(ax: plt.Axes) -> None:
    ax.xaxis.label.set_fontsize(HEADLINE_AXIS_LABEL_SIZE)
    ax.yaxis.label.set_fontsize(HEADLINE_AXIS_LABEL_SIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(HEADLINE_TICK_LABEL_SIZE)


def emphasize_headline_title(ax: plt.Axes) -> None:
    ax.title.set_fontsize(15)
    ax.title.set_fontweight("bold")


def plot_headline_effects(overall_df: pd.DataFrame, network_df: pd.DataFrame, outpath: Path) -> None:
    overall = clean_overall_table(overall_df).sort_values("display_order").reset_index(drop=True)
    network = clean_network_table(network_df).sort_values("display_order").reset_index(drop=True)
    overall_tick_labels = [
        "Global\nsegregation",
        "Within-network\nconnectivity",
        "Between-network\nconnectivity",
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.6, 5.9),
        gridspec_kw={"width_ratios": [1.0, 1.22]},
    )

    # This left panel keeps the three big-picture outcomes together so the reader sees the headline pattern first.
    ax = axes[0]
    y_overall = np.arange(len(overall))[::-1]
    ax.errorbar(
        overall["estimate_older_vs_young"],
        y_overall,
        xerr=[
            overall["estimate_older_vs_young"] - overall["conf_low"],
            overall["conf_high"] - overall["estimate_older_vs_young"],
        ],
        fmt="o",
        color=OVERALL_COLOR,
        ecolor=OVERALL_COLOR,
        elinewidth=1.9,
        capsize=3.8,
        markersize=6.9,
        markerfacecolor="white",
        markeredgewidth=1.8,
        zorder=3,
    )
    ax.set_yticks(y_overall)
    ax.set_yticklabels(overall_tick_labels)
    ax.set_xlabel("Older minus younger estimate")
    ax.set_title("Overall outcomes")
    style_axis(ax)
    darken_axis_text(ax)
    enlarge_headline_axis_text(ax)
    ax.tick_params(axis="y", pad=8)
    add_q_labels(
        ax,
        overall["conf_high"].to_numpy(dtype=float),
        y_overall,
        overall["fdr_q_value"].to_numpy(dtype=float),
    )

    # This right panel focuses only on network-specific segregation, which is the cleaner network-level story.
    ax = axes[1]
    y_network = np.arange(len(network))[::-1]
    colors = np.where(
        network["fdr_q_value"].astype(float).to_numpy() < 0.05,
        HIGHLIGHT_COLOR,
        NETWORK_COLOR,
    )
    for idx, row in network.iterrows():
        y = y_network[idx]
        color = colors[idx]
        ax.errorbar(
            row["estimate_older_vs_young"],
            y,
            xerr=[[row["estimate_older_vs_young"] - row["conf_low"]], [row["conf_high"] - row["estimate_older_vs_young"]]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.8,
            capsize=3.6,
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.7,
            zorder=3,
        )
    ax.set_yticks(y_network)
    ax.set_yticklabels(network["network"].astype(str))
    ax.set_xlabel("Older minus younger estimate")
    ax.set_title("Network segregation")
    style_axis(ax)
    darken_axis_text(ax)
    enlarge_headline_axis_text(ax)
    add_q_labels(
        ax,
        network["conf_high"].to_numpy(dtype=float),
        y_network,
        network["fdr_q_value"].to_numpy(dtype=float),
    )

    fig.suptitle("Primary age effects", y=1.01, fontweight="bold")
    save_figure(fig, outpath)


def build_sensitivity_forest_table(
    branch_df: pd.DataFrame,
    stricter_motion_df: pd.DataFrame,
    retained_minutes_df: pd.DataFrame,
    balanced_overall_df: pd.DataFrame,
) -> pd.DataFrame:
    # This combines the main branch, the named methodological branches, and the two most useful sample/covariate checks.
    branch_keep = branch_df.copy()
    branch_keep["branch_key"] = branch_keep["branch"].astype(str)
    branch_keep["estimate"] = branch_keep["global_estimate"].astype(float)
    branch_keep["conf_low"] = branch_keep["global_conf_low"].astype(float)
    branch_keep["conf_high"] = branch_keep["global_conf_high"].astype(float)
    branch_keep["n_total"] = branch_keep["n_young"].astype(int) + branch_keep["n_older"].astype(int)
    branch_keep["n_label"] = (
        "n="
        + branch_keep["n_total"].astype(int).astype(str)
        + " ["
        + branch_keep["n_young"].astype(int).astype(str)
        + "y, "
        + branch_keep["n_older"].astype(int).astype(str)
        + "o]"
    )
    branch_keep["family"] = "method_branch"
    branch_keep["plot_label"] = branch_keep["branch_key"].map(
        {
            "main": "Main branch",
            "with_gsr": "With GSR",
            "partial_correlation": "Partial correlation",
            "schaefer_100": "Schaefer-100 atlas",
            "adjacent_scrub": "Adjacent-frame scrub",
        }
    )
    branch_keep["display_order"] = branch_keep["branch_key"].map(
        {
            "main": 0,
            "with_gsr": 1,
            "partial_correlation": 2,
            "schaefer_100": 3,
            "adjacent_scrub": 4,
        }
    )
    branch_keep["permutation_q"] = branch_keep["perm_global_q"].astype(float)
    branch_keep = branch_keep[
        [
            "branch_key",
            "plot_label",
            "family",
            "display_order",
            "estimate",
            "conf_low",
            "conf_high",
            "n_total",
            "n_label",
            "permutation_q",
        ]
    ]

    stricter_row = stricter_motion_df[
        stricter_motion_df["label"].astype(str) == "stricter_motion_subset"
    ].copy()
    stricter_row["branch_key"] = "stricter_motion_subset"
    stricter_row["plot_label"] = "Stricter motion subset"
    stricter_row["family"] = "sample_check"
    stricter_row["display_order"] = 5
    stricter_row["estimate"] = stricter_row["estimate_older_vs_young"].astype(float)
    stricter_row["n_total"] = stricter_row["n_subjects"].astype(int)
    stricter_row["n_label"] = (
        "n="
        + stricter_row["n_subjects"].astype(int).astype(str)
        + " ["
        + stricter_row["n_younger"].astype(int).astype(str)
        + "y, "
        + stricter_row["n_older"].astype(int).astype(str)
        + "o]"
    )
    stricter_row["permutation_q"] = np.nan
    stricter_row = stricter_row[
        [
            "branch_key",
            "plot_label",
            "family",
            "display_order",
            "estimate",
            "conf_low",
            "conf_high",
            "n_total",
            "n_label",
            "permutation_q",
        ]
    ]

    retained_row = retained_minutes_df[
        retained_minutes_df["label"].astype(str) == "with_retained_minutes_covariate"
    ].copy()
    retained_row["branch_key"] = "with_retained_minutes_covariate"
    retained_row["plot_label"] = "Retained-minutes covariate"
    retained_row["family"] = "covariate_check"
    retained_row["display_order"] = 6
    retained_row["estimate"] = retained_row["estimate_older_vs_young"].astype(float)
    retained_row["n_total"] = retained_row["n_subjects"].astype(int)
    retained_row["n_label"] = (
        "n="
        + retained_row["n_subjects"].astype(int).astype(str)
        + " ["
        + retained_row["n_younger"].astype(int).astype(str)
        + "y, "
        + retained_row["n_older"].astype(int).astype(str)
        + "o]"
    )
    retained_row["permutation_q"] = np.nan
    retained_row = retained_row[
        [
            "branch_key",
            "plot_label",
            "family",
            "display_order",
            "estimate",
            "conf_low",
            "conf_high",
            "n_total",
            "n_label",
            "permutation_q",
        ]
    ]

    balanced_row = balanced_overall_df[
        balanced_overall_df["outcome"].astype(str) == "global_segregation_prop"
    ].copy()
    balanced_row["branch_key"] = "balanced_tr3_11x11"
    balanced_row["plot_label"] = "Balanced 11 x 11 branch"
    balanced_row["family"] = "sample_check"
    balanced_row["display_order"] = 7
    balanced_row["estimate"] = balanced_row["estimate_older_vs_young"].astype(float)
    balanced_row["n_total"] = balanced_row["n_subjects"].astype(int)
    balanced_row["n_label"] = "n=22 [11y, 11o]"
    balanced_row["permutation_q"] = balanced_row["fdr_q_value"].astype(float)
    balanced_row = balanced_row[
        [
            "branch_key",
            "plot_label",
            "family",
            "display_order",
            "estimate",
            "conf_low",
            "conf_high",
            "n_total",
            "n_label",
            "permutation_q",
        ]
    ]

    combined = pd.concat(
        [branch_keep, stricter_row, retained_row, balanced_row],
        ignore_index=True,
        sort=False,
    )
    return combined.sort_values("display_order").reset_index(drop=True)


def plot_sensitivity_forest(sensitivity_df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.8))

    plot_df = sensitivity_df.copy().sort_values("display_order", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))[::-1]
    colors = []
    for branch_key in plot_df["branch_key"]:
        if branch_key == "main":
            colors.append(OVERALL_COLOR)
        else:
            colors.append(NETWORK_COLOR)

    for idx, row in plot_df.iterrows():
        ax.errorbar(
            row["estimate"],
            y[idx],
            xerr=[[row["estimate"] - row["conf_low"]], [row["conf_high"] - row["estimate"]]],
            fmt="o",
            color=colors[idx],
            ecolor=colors[idx],
            elinewidth=1.9,
            capsize=3.8,
            markersize=6.7,
            markerfacecolor="white",
            markeredgewidth=1.7,
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["plot_label"])
    ax.set_xlabel("Older minus younger estimate")
    ax.set_title("Global effect across sensitivity branches")
    style_y_grid_axis(ax)
    darken_axis_text(ax)
    enlarge_headline_axis_text(ax)
    emphasize_headline_title(ax)

    finite_high = plot_df["conf_high"].astype(float).to_numpy()
    max_high = float(np.nanmax(finite_high)) if len(finite_high) else 0.0
    x_text = max_high + 0.012
    for idx, row in plot_df.iterrows():
        extra = row["n_label"]
        q = row["permutation_q"]
        if pd.notna(q):
            extra = f"{extra}\nq={float(q):.3f}"
        ax.text(
            x_text,
            y[idx],
            extra,
            va="center",
            ha="left",
            fontsize=8.8,
            color=TEXT_COLOR,
            linespacing=1.1,
        )

    save_figure(fig, outpath)


def compute_subject_network_mean_matrix(matrix: np.ndarray, atlas_df: pd.DataFrame) -> np.ndarray:
    network_matrix = np.full((len(NETWORK_ORDER), len(NETWORK_ORDER)), np.nan, dtype=float)
    for i, network_i in enumerate(NETWORK_ORDER):
        idx_i = atlas_df.loc[atlas_df["network"].astype(str) == network_i, "parcel_index"].to_numpy(dtype=int)
        for j, network_j in enumerate(NETWORK_ORDER):
            idx_j = atlas_df.loc[atlas_df["network"].astype(str) == network_j, "parcel_index"].to_numpy(dtype=int)
            block = matrix[np.ix_(idx_i, idx_j)]
            if i == j:
                mask = ~np.eye(block.shape[0], dtype=bool)
                values = block[mask]
            else:
                values = block.reshape(-1)
            network_matrix[i, j] = float(np.nanmean(values))
    return network_matrix


def build_network_connectivity_group_matrices(
    sample_df: pd.DataFrame,
    atlas_df: pd.DataFrame,
    matrices_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subject_matrices: dict[str, np.ndarray] = {}
    for matrix_path in sorted(matrices_dir.glob("sub-*_forward_z_matrix.npy")):
        subject_id = matrix_subject_id_from_path(matrix_path)
        subject_matrices[subject_id] = np.load(matrix_path)

    young_blocks = []
    older_blocks = []
    for row in sample_df.itertuples(index=False):
        matrix = subject_matrices.get(row.subject_id)
        if matrix is None:
            continue
        network_matrix = compute_subject_network_mean_matrix(matrix, atlas_df)
        if str(row.age_group) == "young":
            young_blocks.append(network_matrix)
        elif str(row.age_group) == "older":
            older_blocks.append(network_matrix)

    if not young_blocks or not older_blocks:
        raise RuntimeError("Could not build network connectivity group matrices from the final sample.")

    young_mean = np.nanmean(np.stack(young_blocks, axis=0), axis=0)
    older_mean = np.nanmean(np.stack(older_blocks, axis=0), axis=0)
    diff = older_mean - young_mean
    return young_mean, older_mean, diff


def matrix_to_long_table(matrix: np.ndarray, value_name: str) -> pd.DataFrame:
    rows = []
    for i, row_network in enumerate(NETWORK_ORDER):
        for j, col_network in enumerate(NETWORK_ORDER):
            rows.append(
                {
                    "row_network": row_network,
                    "column_network": col_network,
                    value_name: float(matrix[i, j]),
                }
            )
    return pd.DataFrame(rows)


def plot_network_difference_heatmap(diff_matrix: np.ndarray, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    vmax = float(np.nanmax(np.abs(diff_matrix)))
    vmax = max(vmax, 0.01)
    im = ax.imshow(diff_matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(NETWORK_ORDER)))
    ax.set_yticks(np.arange(len(NETWORK_ORDER)))
    ax.set_xticklabels(NETWORK_ORDER, rotation=35, ha="right")
    ax.set_yticklabels(NETWORK_ORDER)
    ax.set_xlabel("Target network")
    ax.set_ylabel("Seed network")
    ax.set_title("Older minus younger network connectivity", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT_COLOR)
    ax.spines["bottom"].set_color(TEXT_COLOR)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    darken_axis_text(ax)
    enlarge_headline_axis_text(ax)
    emphasize_headline_title(ax)

    # Diagonal cells are within-network means; the off-diagonal cells summarize between-network coupling.
    for i in range(len(NETWORK_ORDER)):
        ax.add_patch(
            plt.Rectangle(
                (i - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor=SUBTLE_TEXT_COLOR,
                linewidth=0.75,
                alpha=0.55,
            )
        )

    for i in range(len(NETWORK_ORDER)):
        for j in range(len(NETWORK_ORDER)):
            value = diff_matrix[i, j]
            text_color = "white" if abs(value) > (0.55 * vmax) else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.9, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        "Older minus younger mean Fisher z",
        fontsize=HEADLINE_AXIS_LABEL_SIZE,
        color=AXIS_TEXT_COLOR,
    )
    cbar.ax.tick_params(labelsize=HEADLINE_TICK_LABEL_SIZE, colors=AXIS_TEXT_COLOR)
    cbar.outline.set_edgecolor(TEXT_COLOR)
    cbar.outline.set_linewidth(0.9)
    save_figure(fig, outpath)


def write_summary_markdown(table: pd.DataFrame, outpath: Path) -> None:
    overall = table[table["result_group"] == "overall"].copy()
    network = table[table["result_group"] == "network_specific"].copy()

    lines = [
        "# Dissertation Summary Outputs",
        "",
        "This folder contains one compact headline table and one coefficient-style figure for dissertation write-up.",
        "",
        "## Overall Outcomes",
        "",
    ]
    for _, row in overall.iterrows():
        lines.append(
            f"- {row['result_label']}: estimate={row['estimate_older_vs_young']:.4f}, "
            f"95% CI {row['ci_text']}, permutation p={row['permutation_p_value']:.4g}, "
            f"q={row['fdr_q_value']:.4g}"
        )

    lines.extend(["", "## Network-Specific Segregation", ""])
    for _, row in network.iterrows():
        lines.append(
            f"- {row['network']}: estimate={row['estimate_older_vs_young']:.4f}, "
            f"95% CI {row['ci_text']}, permutation p={row['permutation_p_value']:.4g}, "
            f"q={row['fdr_q_value']:.4g}"
        )

    lines.extend(
        [
            "",
            "## Important note",
            "",
            "- The confidence intervals in these summary outputs are model-based robust intervals carried through from the regression summaries.",
            "- The p values and q values are permutation-based from the Freedman-Lane analysis.",
            "- These should therefore not be described as permutation-based confidence intervals in the dissertation text.",
            "",
            "## Extra dissertation-strengthening figures",
            "",
            "- `figures/global_effect_sensitivity_forest.png` summarizes how the older-versus-younger global segregation estimate behaves across the main branch, methodological sensitivity branches, and key sample/covariate checks.",
            "- `figures/older_minus_younger_network_connectivity_heatmap.png` summarizes the older-minus-younger 7 x 7 network-level Fisher z connectivity pattern using the final primary sample.",
        ]
    )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    sample_path = resolve_project_path(args.sample)
    permutation_dir = resolve_project_path(args.permutation_dir)
    output_dir = resolve_project_path(args.output_dir)
    sensitivity_branch_table_path = resolve_project_path(args.sensitivity_branch_table)
    stricter_motion_table_path = resolve_project_path(args.stricter_motion_table)
    retained_minutes_table_path = resolve_project_path(args.retained_minutes_table)
    balanced_overall_table_path = resolve_project_path(args.balanced_overall_table)
    matrices_dir = resolve_project_path(args.matrices_dir)
    atlas_labels_path = resolve_project_path(args.atlas_labels)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    sample_df = pd.read_csv(sample_path, sep="\t")
    overall_df = pd.read_csv(permutation_dir / "overall_permutation_summary.tsv", sep="\t")
    network_df = pd.read_csv(permutation_dir / "network_permutation_summary.tsv", sep="\t")
    sensitivity_branch_df = pd.read_csv(sensitivity_branch_table_path, sep="\t")
    stricter_motion_df = pd.read_csv(stricter_motion_table_path, sep="\t")
    retained_minutes_df = pd.read_csv(retained_minutes_table_path, sep="\t")
    balanced_overall_df = pd.read_csv(balanced_overall_table_path, sep="\t")
    atlas_df = pd.read_csv(atlas_labels_path, sep="\t")

    headline_table = build_headline_table(overall_df, network_df)
    word_table = build_word_friendly_table(headline_table)
    characteristics_table = build_participant_characteristics_table(sample_df)
    sensitivity_forest_df = build_sensitivity_forest_table(
        sensitivity_branch_df,
        stricter_motion_df,
        retained_minutes_df,
        balanced_overall_df,
    )
    young_network_mean, older_network_mean, older_minus_young_network_mean = build_network_connectivity_group_matrices(
        sample_df,
        atlas_df,
        matrices_dir,
    )
    network_connectivity_long = (
        matrix_to_long_table(young_network_mean, "young_mean_z")
        .merge(matrix_to_long_table(older_network_mean, "older_mean_z"), on=["row_network", "column_network"])
        .merge(
            matrix_to_long_table(older_minus_young_network_mean, "older_minus_young_mean_z"),
            on=["row_network", "column_network"],
        )
    )

    headline_table.to_csv(output_dir / "main_age_effects_summary.tsv", sep="\t", index=False)
    word_table.to_csv(output_dir / "table2_main_age_effects_word_friendly.tsv", sep="\t", index=False)
    word_table.to_csv(output_dir / "table2_main_age_effects_word_friendly.csv", index=False)
    characteristics_table.to_csv(output_dir / "table1_participant_characteristics.tsv", sep="\t", index=False)
    characteristics_table.to_csv(output_dir / "table1_participant_characteristics.csv", index=False)
    sensitivity_forest_df.to_csv(output_dir / "global_effect_sensitivity_forest.tsv", sep="\t", index=False)
    network_connectivity_long.to_csv(output_dir / "network_connectivity_group_means.tsv", sep="\t", index=False)
    pd.DataFrame(young_network_mean, index=NETWORK_ORDER, columns=NETWORK_ORDER).to_csv(
        output_dir / "young_mean_network_connectivity_matrix.csv"
    )
    pd.DataFrame(older_network_mean, index=NETWORK_ORDER, columns=NETWORK_ORDER).to_csv(
        output_dir / "older_mean_network_connectivity_matrix.csv"
    )
    pd.DataFrame(older_minus_young_network_mean, index=NETWORK_ORDER, columns=NETWORK_ORDER).to_csv(
        output_dir / "older_minus_young_network_connectivity_matrix.csv"
    )

    pd.DataFrame(
        [
            {
                "source_sample_table": str(sample_path),
                "source_overall_table": str(permutation_dir / "overall_permutation_summary.tsv"),
                "source_network_table": str(permutation_dir / "network_permutation_summary.tsv"),
                "included_overall_outcomes": "global_segregation_prop,overall_within_mean,overall_between_mean",
                "included_network_component": "segregation_prop",
                "figure": "figures/main_age_effects_summary.png",
                "participant_characteristics_table_tsv": "table1_participant_characteristics.tsv",
                "participant_characteristics_table_csv": "table1_participant_characteristics.csv",
                "table": "main_age_effects_summary.tsv",
                "word_friendly_table_tsv": "table2_main_age_effects_word_friendly.tsv",
                "word_friendly_table_csv": "table2_main_age_effects_word_friendly.csv",
                "global_effect_sensitivity_table_tsv": "global_effect_sensitivity_forest.tsv",
                "network_connectivity_table_tsv": "network_connectivity_group_means.tsv",
                "sensitivity_figure": "figures/global_effect_sensitivity_forest.png",
                "network_connectivity_heatmap": "figures/older_minus_younger_network_connectivity_heatmap.png",
            }
        ]
    ).to_csv(output_dir / "dissertation_summary_settings.tsv", sep="\t", index=False)

    plot_headline_effects(
        overall_df,
        network_df,
        figures_dir / "main_age_effects_summary.png",
    )
    plot_sensitivity_forest(
        sensitivity_forest_df,
        figures_dir / "global_effect_sensitivity_forest.png",
    )
    plot_network_difference_heatmap(
        older_minus_young_network_mean,
        figures_dir / "older_minus_younger_network_connectivity_heatmap.png",
    )
    write_summary_markdown(headline_table, output_dir / "dissertation_summary.md")

    print(f"Wrote dissertation summary table to {output_dir / 'main_age_effects_summary.tsv'}")
    print(f"Wrote participant characteristics table to {output_dir / 'table1_participant_characteristics.tsv'}")
    print(f"Wrote Word-friendly table to {output_dir / 'table2_main_age_effects_word_friendly.tsv'}")
    print(f"Wrote dissertation summary figure to {figures_dir / 'main_age_effects_summary.png'}")
    print(f"Wrote sensitivity forest figure to {figures_dir / 'global_effect_sensitivity_forest.png'}")
    print(
        "Wrote network connectivity heatmap to "
        f"{figures_dir / 'older_minus_younger_network_connectivity_heatmap.png'}"
    )


if __name__ == "__main__":
    main()
