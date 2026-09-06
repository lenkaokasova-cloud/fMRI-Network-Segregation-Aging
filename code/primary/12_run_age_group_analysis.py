#!/usr/bin/env python3

# What this script does:
#   Runs the main age-group analysis for the final TR = 3 s sample.
#   It combines the final sample TSV with the connectivity outputs
#   subject_global_segregation.tsv and subject_network_segregation.tsv, then
#   fits three main models:
#   1. a global younger-vs-older model using global_segregation_prop
#   2. a network-type model using segregation_prop for sensory/motor vs
#      higher-order networks
#   3. a network-specific model across the individual Yeo 7 networks
#   In these models, sex and mean FD are included as covariates.
#   The script also writes summary tables and figures.
# How to run it:
#   Run from the repo root with:
#   python code/primary/12_run_age_group_analysis.py
# Main output:
#   data/processed/analysis/
#   This includes model_coefficients.tsv, sample_summary.tsv,
#   subject_global_with_metadata.tsv, subject_network_with_metadata.tsv,
#   subject_network_type_summary.tsv, analysis_summary.md, and the main
#   age-group figures.

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str((Path("data/processed/.matplotlib")).resolve())
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dissertation_figure_style import (
    FIG_DPI,
    GRID_COLOR,
    MUTED_REFERENCE_LINE_COLOR,
    OLDER_COLOR,
    SUMMARY_AXIS_LABEL_SIZE,
    SUMMARY_TICK_LABEL_SIZE,
    SUMMARY_TITLE_SIZE,
    YOUNG_COLOR,
    apply_dissertation_rcparams,
    darken_axis_text,
)
from dissertation_figure_style import save_figure as save_dissertation_figure
from dissertation_figure_style import (
    style_spines,
)

HIGHER_ORDER = {"Default", "Cont", "DorsAttn", "SalVentAttn"}
SENSORY_MOTOR = {"Vis", "SomMot"}
TITLE_SIZE = 15
LABEL_SIZE = 12
TICK_SIZE = 11
LEGEND_SIZE = 11

apply_dissertation_rcparams(
    title_size=TITLE_SIZE,
    label_size=LABEL_SIZE,
    tick_size=TICK_SIZE,
    legend_size=LEGEND_SIZE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the main age-group analyses on global and network segregation "
            "while controlling for sex and mean FD."
        )
    )
    parser.add_argument(
        "--sample",
        default="data/processed/screening/ds005752_final_analysis_sample_tr_3s.tsv",
        help="Final TR=3 s analysis sample TSV.",
    )
    parser.add_argument(
        "--connectivity-dir",
        default="data/processed/connectivity/metrics",
        help="Directory containing subject_global_segregation.tsv and subject_network_segregation.tsv.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/analysis",
        help="Directory for age-analysis outputs.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def add_network_type(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["network_type"] = "other"
    out.loc[out["network"].isin(HIGHER_ORDER), "network_type"] = "higher_order"
    out.loc[out["network"].isin(SENSORY_MOTOR), "network_type"] = "sensory_motor"
    return out


def merge_missing_columns(
    df: pd.DataFrame, sample: pd.DataFrame, required_columns: list[str]
) -> pd.DataFrame:
    missing = [column for column in required_columns if column not in df.columns]
    if not missing:
        return df.copy()
    keep = ["subject_id"] + missing
    return df.merge(sample[keep], on="subject_id", how="inner")


def model_to_table(result, model_name: str) -> pd.DataFrame:
    conf_int = result.conf_int()
    conf_int.columns = ["conf_low", "conf_high"]
    conf_int = conf_int.reset_index().rename(columns={"index": "term"})
    table = pd.DataFrame(
        {
            "term": result.params.index,
            "estimate": result.params.values,
            "std_error": result.bse.values,
            "statistic": result.tvalues.values,
            "p_value": result.pvalues.values,
        }
    )
    table = table.merge(conf_int, on="term", how="left", validate="one_to_one")
    table["model"] = model_name
    table["nobs"] = result.nobs
    table["r_squared"] = getattr(result, "rsquared", np.nan)
    return table[
        [
            "model",
            "term",
            "estimate",
            "std_error",
            "statistic",
            "p_value",
            "conf_low",
            "conf_high",
            "nobs",
            "r_squared",
        ]
    ]


def style_axis(
    ax: plt.Axes,
    *,
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
    style_spines(ax)
    darken_axis_text(ax)
    ax.xaxis.label.set_fontsize(SUMMARY_AXIS_LABEL_SIZE)
    ax.yaxis.label.set_fontsize(SUMMARY_AXIS_LABEL_SIZE)
    ax.title.set_fontsize(SUMMARY_TITLE_SIZE)
    ax.title.set_fontweight("bold")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(SUMMARY_TICK_LABEL_SIZE)
    ax.tick_params(axis="x", labelrotation=xrotation)
    for label in ax.get_xticklabels():
        label.set_ha("right" if xrotation else "center")


def save_figure(fig: plt.Figure, outpath: Path) -> None:
    save_dissertation_figure(fig, outpath, dpi=FIG_DPI)


def group_offsets(n_points: int, width: float = 0.12) -> np.ndarray:
    if n_points == 0:
        return np.array([])
    if n_points <= 1:
        return np.array([0.0])
    return np.linspace(-width, width, n_points)


def plot_global_by_group(df: pd.DataFrame, outpath: Path) -> None:
    order = ["young", "older"]
    colors = [YOUNG_COLOR, OLDER_COLOR]
    data = [
        df.loc[df["age_group"] == group, "global_segregation_prop"].to_numpy()
        for group in order
    ]

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    box = ax.boxplot(
        data,
        positions=[0, 1],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.8},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.8)

    for x_pos, group, color in zip([0, 1], order, colors):
        subset = df[df["age_group"] == group].reset_index(drop=True)
        offsets = group_offsets(len(subset))
        ax.scatter(
            np.full(len(subset), x_pos) + offsets,
            subset["global_segregation_prop"],
            color=color,
            s=78,
            alpha=0.92,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )

    style_axis(
        ax,
        title="Global Segregation by Age Group",
        ylabel="Global segregation",
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(order)
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    save_figure(fig, outpath)


def plot_motion_vs_global(df: pd.DataFrame, outpath: Path) -> None:
    colors = {"young": YOUNG_COLOR, "older": OLDER_COLOR}
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for age_group, color in colors.items():
        subset = df[df["age_group"] == age_group].copy()
        if subset.empty:
            continue
        ax.scatter(
            subset["mean_fd"],
            subset["global_segregation_prop"],
            color=color,
            s=80,
            alpha=0.92,
            edgecolors="white",
            linewidths=0.6,
            label=age_group,
            zorder=3,
        )
        if len(subset) >= 2:
            slope, intercept = np.polyfit(
                subset["mean_fd"], subset["global_segregation_prop"], 1
            )
            x_values = np.linspace(
                subset["mean_fd"].min(), subset["mean_fd"].max(), 100
            )
            ax.plot(
                x_values,
                slope * x_values + intercept,
                color=color,
                linewidth=1.8,
                alpha=0.9,
            )

    style_axis(
        ax,
        title="Motion and Global Segregation",
        xlabel="Mean FD (mm)",
        ylabel="Global segregation",
    )
    ax.grid(color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, outpath)


def plot_network_type_summary(df: pd.DataFrame, outpath: Path) -> None:
    order = [
        ("young", "sensory_motor"),
        ("young", "higher_order"),
        ("older", "sensory_motor"),
        ("older", "higher_order"),
    ]
    positions = {
        ("young", "sensory_motor"): 0.0,
        ("young", "higher_order"): 1.0,
        ("older", "sensory_motor"): 3.0,
        ("older", "higher_order"): 4.0,
    }
    labels = [
        "young\nsensory-motor",
        "young\nhigher-order",
        "older\nsensory-motor",
        "older\nhigher-order",
    ]

    fig, ax = plt.subplots(figsize=(8.0, 5.1))
    for age_group, color in [("young", YOUNG_COLOR), ("older", OLDER_COLOR)]:
        wide = (
            df[df["age_group"] == age_group]
            .pivot(
                index="subject_id", columns="network_type", values="segregation_prop"
            )
            .dropna()
        )
        if {"sensory_motor", "higher_order"}.issubset(wide.columns):
            for _, row in wide.iterrows():
                ax.plot(
                    [
                        positions[(age_group, "sensory_motor")],
                        positions[(age_group, "higher_order")],
                    ],
                    [row["sensory_motor"], row["higher_order"]],
                    color=color,
                    alpha=0.18,
                    linewidth=1.0,
                    zorder=1,
                )

    data = [
        df.loc[
            (df["age_group"] == age_group) & (df["network_type"] == network_type),
            "segregation_prop",
        ].to_numpy()
        for age_group, network_type in order
    ]
    box = ax.boxplot(
        data,
        positions=[positions[key] for key in order],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.8},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
    )
    box_colors = [YOUNG_COLOR, YOUNG_COLOR, OLDER_COLOR, OLDER_COLOR]
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.26)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.8)

    for (age_group, network_type), color in zip(order, box_colors):
        subset = df[
            (df["age_group"] == age_group) & (df["network_type"] == network_type)
        ].reset_index(drop=True)
        offsets = group_offsets(len(subset), width=0.1)
        x_pos = positions[(age_group, network_type)]
        ax.scatter(
            np.full(len(subset), x_pos) + offsets,
            subset["segregation_prop"],
            color=color,
            s=68,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    style_axis(
        ax,
        title="Segregation by Network Type and Age Group",
        ylabel="Mean segregation",
    )
    ax.set_xticks([positions[key] for key in order])
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    ax.axvline(2.0, color=MUTED_REFERENCE_LINE_COLOR, alpha=0.18, linewidth=1.2)
    save_figure(fig, outpath)


def write_summary_markdown(
    outpath: Path,
    sample_summary: pd.DataFrame,
    global_model: pd.DataFrame,
    network_type_model: pd.DataFrame,
) -> None:
    lines = [
        "# Age-Group Analysis Summary",
        "",
        "## Sample",
        "",
    ]
    for _, row in sample_summary.iterrows():
        lines.append(
            f"- {row['age_group']}: n={int(row['n_subjects'])}, mean age={row['age_mean']:.2f}, "
            f"mean FD={row['mean_fd_mean']:.3f}, mean retained volumes={row['retained_volumes_mean']:.1f}"
        )

    def describe_term(table: pd.DataFrame, term: str) -> str | None:
        subset = table[table["term"] == term]
        if subset.empty:
            return None
        row = subset.iloc[0]
        return (
            f"{term}: estimate={row['estimate']:.4f}, "
            f"95% CI [{row['conf_low']:.4f}, {row['conf_high']:.4f}], "
            f"p={row['p_value']:.4g}"
        )

    lines.extend(["", "## Primary Models", ""])
    global_term = describe_term(
        global_model, "C(age_group, Treatment(reference='young'))[T.older]"
    )
    if global_term:
        lines.append(
            f"- Global model older-vs-young effect on proportional global segregation: {global_term}"
        )
    interaction_term = describe_term(
        network_type_model,
        "C(age_group, Treatment(reference='young'))[T.older]:C(network_type, Treatment(reference='sensory_motor'))[T.higher_order]",
    )
    if interaction_term:
        lines.append(
            f"- Network-type interaction on proportional segregation: {interaction_term}"
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

    sample = pd.read_csv(sample_path, sep="\t")
    global_df = pd.read_csv(
        connectivity_dir / "subject_global_segregation.tsv", sep="\t"
    )
    network_df = pd.read_csv(
        connectivity_dir / "subject_network_segregation.tsv", sep="\t"
    )

    global_df = merge_missing_columns(
        global_df,
        sample,
        [
            "age",
            "sex",
            "age_group",
            "mean_fd",
            "pct_fd_gt_0p2",
            "n_retained_after_scrub",
        ],
    )
    network_df = merge_missing_columns(
        network_df,
        sample,
        ["age", "sex", "age_group", "mean_fd", "n_retained_after_scrub"],
    )

    network_df = add_network_type(network_df)
    network_type_df = (
        network_df[network_df["network_type"].isin(["higher_order", "sensory_motor"])]
        .groupby(
            ["subject_id", "age", "sex", "age_group", "mean_fd", "network_type"],
            as_index=False,
        )["segregation_prop"]
        .mean()
    )

    sample_summary = (
        sample.groupby("age_group", as_index=False)
        .agg(
            n_subjects=("subject_id", "count"),
            age_mean=("age", "mean"),
            age_sd=("age", "std"),
            mean_fd_mean=("mean_fd", "mean"),
            mean_fd_sd=("mean_fd", "std"),
            retained_volumes_mean=("n_retained_after_scrub", "mean"),
            retained_volumes_sd=("n_retained_after_scrub", "std"),
        )
        .sort_values("age_group")
    )

    global_model = smf.ols(
        "global_segregation_prop ~ C(age_group, Treatment(reference='young')) + C(sex) + mean_fd",
        data=global_df,
    ).fit(cov_type="HC3")
    network_type_model = smf.ols(
        "segregation_prop ~ C(age_group, Treatment(reference='young')) * C(network_type, Treatment(reference='sensory_motor')) + C(sex) + mean_fd",
        data=network_type_df,
    ).fit(cov_type="cluster", cov_kwds={"groups": network_type_df["subject_id"]})
    network_specific_model = smf.ols(
        "segregation_prop ~ C(age_group, Treatment(reference='young')) * C(network) + C(sex) + mean_fd",
        data=network_df,
    ).fit(cov_type="cluster", cov_kwds={"groups": network_df["subject_id"]})

    global_table = model_to_table(global_model, "global_group_model")
    network_type_table = model_to_table(network_type_model, "network_type_model")
    network_specific_table = model_to_table(
        network_specific_model, "network_specific_model"
    )

    sample_summary.to_csv(output_dir / "sample_summary.tsv", sep="\t", index=False)
    global_df.to_csv(
        output_dir / "subject_global_with_metadata.tsv", sep="\t", index=False
    )
    network_df.to_csv(
        output_dir / "subject_network_with_metadata.tsv", sep="\t", index=False
    )
    network_type_df.to_csv(
        output_dir / "subject_network_type_summary.tsv", sep="\t", index=False
    )
    pd.concat(
        [global_table, network_type_table, network_specific_table],
        ignore_index=True,
    ).to_csv(output_dir / "model_coefficients.tsv", sep="\t", index=False)

    plot_global_by_group(global_df, figures_dir / "global_segregation_by_group.png")
    plot_motion_vs_global(global_df, figures_dir / "motion_vs_global_segregation.png")
    plot_network_type_summary(
        network_type_df, figures_dir / "network_type_by_group.png"
    )
    write_summary_markdown(
        output_dir / "analysis_summary.md",
        sample_summary,
        global_table,
        network_type_table,
    )

    print(f"Wrote age-analysis tables to {output_dir}")
    print(f"Wrote age-analysis figures to {figures_dir}")


if __name__ == "__main__":
    main()
