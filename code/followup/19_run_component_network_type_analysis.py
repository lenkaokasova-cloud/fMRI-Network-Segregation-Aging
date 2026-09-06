#!/usr/bin/env python3

# What this script does:
#   Compares sensory/motor and higher-order networks to see whether the age
#   pattern differs across those broader network types.
# How to run it:
#   Run from the repo root with:
#   python code/followup/19_run_component_network_type_analysis.py
# Main output:
#   data/processed/analysis/network_type_components/

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
    REFERENCE_LINE_COLOR,
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
SEGREGATION_COL = "segregation_prop"
COMPONENT_ORDER = ["within_mean_z", "between_mean_z", SEGREGATION_COL]
COMPONENT_LABELS = {
    "within_mean_z": "Within-network connectivity",
    "between_mean_z": "Between-network connectivity",
    SEGREGATION_COL: "Segregation",
}
NETWORK_TYPE_ORDER = ["sensory_motor", "higher_order"]
NETWORK_TYPE_LABELS = {
    "sensory_motor": "Sensory / motor",
    "higher_order": "Higher-order",
}
TITLE_SIZE = 20
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 14

apply_dissertation_rcparams(
    title_size=TITLE_SIZE,
    label_size=LABEL_SIZE,
    tick_size=TICK_SIZE,
    legend_size=LEGEND_SIZE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether age effects differ between sensory/motor and "
            "higher-order networks for within-network connectivity, between-network "
            "connectivity, and segregation."
        )
    )
    parser.add_argument(
        "--sample",
        default="data/processed/screening/ds005752_final_analysis_sample_tr_3s.tsv",
        help="Final TR = 3 s analysis sample TSV.",
    )
    parser.add_argument(
        "--connectivity-dir",
        default="data/processed/connectivity/metrics",
        help="Directory containing subject_network_segregation.tsv.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/analysis/network_type_components",
        help="Directory for network-type component analysis outputs.",
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
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    ax.title.set_fontsize(20)
    ax.title.set_fontweight("bold")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(14)
    ax.tick_params(axis="x", labelrotation=xrotation)
    for label in ax.get_xticklabels():
        label.set_ha("right" if xrotation else "center")


def save_figure(fig: plt.Figure, outpath: Path) -> None:
    save_dissertation_figure(fig, outpath, dpi=FIG_DPI)


def group_offsets(n_points: int, width: float = 0.12) -> np.ndarray:
    if n_points == 0:
        return np.array([])
    if n_points == 1:
        return np.array([0.0])
    return np.linspace(-width, width, n_points)


def load_tables(
    sample_path: Path, connectivity_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = pd.read_csv(sample_path, sep="\t")
    network_df = pd.read_csv(
        connectivity_dir / "subject_network_segregation.tsv", sep="\t"
    )

    sample_subjects = set(sample["subject_id"])
    network_df = network_df[network_df["subject_id"].isin(sample_subjects)].copy()
    return sample, network_df


def add_network_type(network_df: pd.DataFrame) -> pd.DataFrame:
    out = network_df.copy()
    out["network_type"] = "other"
    out.loc[out["network"].isin(SENSORY_MOTOR), "network_type"] = "sensory_motor"
    out.loc[out["network"].isin(HIGHER_ORDER), "network_type"] = "higher_order"
    out = out[out["network_type"].isin(NETWORK_TYPE_ORDER)].copy()
    out["network_type"] = pd.Categorical(
        out["network_type"], categories=NETWORK_TYPE_ORDER, ordered=True
    )
    return out


def build_subject_network_type_components(network_df: pd.DataFrame) -> pd.DataFrame:
    return (
        network_df.groupby(
            [
                "subject_id",
                "age",
                "sex",
                "age_group",
                "mean_fd",
                "retained_minutes_after_scrub",
                "network_type",
            ],
            as_index=False,
        )
        .agg(
            within_mean_z=("within_mean_z", "mean"),
            between_mean_z=("between_mean_z", "mean"),
            segregation_prop=(SEGREGATION_COL, "mean"),
        )
        .sort_values(["age_group", "subject_id", "network_type"])
        .reset_index(drop=True)
    )


def build_descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby(
        ["component", "network_type", "age_group"], as_index=False
    ).agg(mean_value=("value", "mean"), sd_value=("value", "std"), n=("value", "size"))
    wide = summary.pivot(
        index=["component", "network_type"], columns="age_group", values="mean_value"
    ).reset_index()
    wide = wide.rename(columns={"young": "young_mean", "older": "older_mean"})
    wide["older_minus_young"] = wide["older_mean"] - wide["young_mean"]
    return wide.merge(
        summary, on=["component", "network_type"], how="left"
    ).sort_values(["component", "network_type", "age_group"])


def fit_component_model(df: pd.DataFrame, component: str):
    formula = (
        "value ~ C(age_group, Treatment(reference='young')) * "
        "C(network_type, Treatment(reference='sensory_motor')) + C(sex) + mean_fd"
    )
    subset = df[df["component"] == component].copy()
    return smf.ols(formula, data=subset).fit(
        cov_type="cluster", cov_kwds={"groups": subset["subject_id"]}
    )


def build_effect_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    interaction_term = (
        "C(age_group, Treatment(reference='young'))[T.older]:"
        "C(network_type, Treatment(reference='sensory_motor'))[T.higher_order]"
    )
    older_term = "C(age_group, Treatment(reference='young'))[T.older]"
    higher_order_term = (
        "C(network_type, Treatment(reference='sensory_motor'))[T.higher_order]"
    )

    for component in COMPONENT_ORDER:
        result = fit_component_model(df, component)
        for term_name, interpretation in [
            (older_term, "older_vs_young_within_sensory_motor"),
            (higher_order_term, "higher_order_vs_sensory_motor_within_young"),
            (interaction_term, "additional_older_effect_in_higher_order"),
        ]:
            estimate = float(result.params[term_name])
            std_error = float(result.bse[term_name])
            rows.append(
                {
                    "component": component,
                    "term": term_name,
                    "interpretation": interpretation,
                    "estimate": estimate,
                    "std_error": std_error,
                    "conf_low": estimate - 1.96 * std_error,
                    "conf_high": estimate + 1.96 * std_error,
                    "p_value": float(result.pvalues[term_name]),
                    "nobs": float(result.nobs),
                    "r_squared": float(getattr(result, "rsquared", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def plot_component_panels(df: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.3), sharex=False)
    positions = {
        ("young", "sensory_motor"): 0.0,
        ("young", "higher_order"): 1.3,
        ("older", "sensory_motor"): 3.5,
        ("older", "higher_order"): 4.8,
    }
    xticks = [0.0, 1.3, 3.5, 4.8]
    xlabels = [
        "Young\nsensory/\nmotor",
        "Young\nhigher-\norder",
        "Older\nsensory/\nmotor",
        "Older\nhigher-\norder",
    ]

    for ax, component in zip(axes, COMPONENT_ORDER):
        component_df = df[df["component"] == component].copy()
        box_data = []
        box_positions = []
        box_colors = []
        for age_group in ["young", "older"]:
            for network_type in NETWORK_TYPE_ORDER:
                subset = component_df[
                    (component_df["age_group"] == age_group)
                    & (component_df["network_type"] == network_type)
                ].reset_index(drop=True)
                box_data.append(subset["value"].to_numpy())
                box_positions.append(positions[(age_group, network_type)])
                box_colors.append(YOUNG_COLOR if age_group == "young" else OLDER_COLOR)

        box = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.8},
            whiskerprops={"linewidth": 1.4},
            capprops={"linewidth": 1.4},
        )
        for patch, color in zip(box["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.24)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.7)

        for age_group in ["young", "older"]:
            wide = (
                component_df[component_df["age_group"] == age_group]
                .pivot(index="subject_id", columns="network_type", values="value")
                .reset_index()
            )
            if {"sensory_motor", "higher_order"}.issubset(wide.columns):
                for _, row in wide.iterrows():
                    ax.plot(
                        [
                            positions[(age_group, "sensory_motor")],
                            positions[(age_group, "higher_order")],
                        ],
                        [row["sensory_motor"], row["higher_order"]],
                        color="#9AA3AF",
                        linewidth=0.9,
                        alpha=0.4,
                        zorder=1,
                    )

        for age_group in ["young", "older"]:
            for network_type in NETWORK_TYPE_ORDER:
                subset = component_df[
                    (component_df["age_group"] == age_group)
                    & (component_df["network_type"] == network_type)
                ].reset_index(drop=True)
                offsets = group_offsets(len(subset))
                ax.scatter(
                    np.full(len(subset), positions[(age_group, network_type)])
                    + offsets,
                    subset["value"],
                    color=YOUNG_COLOR if age_group == "young" else OLDER_COLOR,
                    s=58,
                    alpha=0.92,
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=3,
                )

        style_axis(
            ax,
            title=COMPONENT_LABELS[component],
            ylabel=(
                "Mean Fisher z" if component != SEGREGATION_COL else "Mean segregation"
            ),
        )
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.tick_params(axis="x", labelsize=13)
        ax.grid(axis="y", color=GRID_COLOR, alpha=0.26, linewidth=0.9)

    save_figure(fig, outpath)


def plot_interaction_effects(effect_df: pd.DataFrame, outpath: Path) -> None:
    subset = effect_df[
        effect_df["interpretation"] == "additional_older_effect_in_higher_order"
    ].copy()
    subset["component"] = pd.Categorical(
        subset["component"], categories=COMPONENT_ORDER, ordered=True
    )
    subset = subset.sort_values("component")

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    x = np.arange(len(subset))
    estimates = subset["estimate"].to_numpy()
    lower = estimates - subset["conf_low"].to_numpy()
    upper = subset["conf_high"].to_numpy() - estimates

    ax.errorbar(
        x,
        estimates,
        yerr=[lower, upper],
        fmt="o",
        color=REFERENCE_LINE_COLOR,
        ecolor=REFERENCE_LINE_COLOR,
        elinewidth=1.8,
        capsize=4,
        markersize=7,
        zorder=3,
    )
    ax.axhline(
        0, color=MUTED_REFERENCE_LINE_COLOR, linewidth=1.2, linestyle="--", zorder=1
    )
    ax.set_xticks(x)
    ax.set_xticklabels([COMPONENT_LABELS[name] for name in subset["component"]])
    style_axis(
        ax,
        title="Extra age effect in higher-order vs sensory/motor networks",
        ylabel="Older-vs-young interaction estimate",
        xrotation=15,
    )
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.26, linewidth=0.9)
    save_figure(fig, outpath)


def write_summary(
    effect_df: pd.DataFrame, descriptive_df: pd.DataFrame, outpath: Path
) -> None:
    interaction = effect_df[
        effect_df["interpretation"] == "additional_older_effect_in_higher_order"
    ].copy()
    interaction = interaction.set_index("component")
    desc_wide = (
        descriptive_df[["component", "network_type", "age_group", "mean_value"]]
        .pivot(
            index=["component", "network_type"],
            columns="age_group",
            values="mean_value",
        )
        .reset_index()
    )
    desc_wide["older_minus_young"] = desc_wide["older"] - desc_wide["young"]

    lines = [
        "# Sensory/Motor vs Higher-Order Component Analysis",
        "",
        "This analysis asks whether age-related effects are larger in higher-order association networks than in sensory/motor networks.",
        "",
        "Network grouping:",
        "",
        "- Sensory / motor: Vis, SomMot",
        "- Higher-order: DorsAttn, SalVentAttn, Cont, Default",
        "- Limbic was excluded from this grouped contrast because it does not fit cleanly into either set.",
        "",
        "## Interaction Tests",
        "",
        "A negative interaction for within-network connectivity means older participants show a larger drop in within-network coherence in higher-order networks than in sensory/motor networks.",
        "",
        "A positive interaction for between-network connectivity means older participants show more cross-network mixing in higher-order networks than in sensory/motor networks.",
        "",
        "A negative interaction for segregation means older participants show a larger segregation reduction in higher-order networks than in sensory/motor networks.",
        "",
    ]

    for component in COMPONENT_ORDER:
        row = interaction.loc[component]
        lines.append(
            f"- {COMPONENT_LABELS[component]} interaction estimate: {row['estimate']:.4f}, "
            f"95% CI [{row['conf_low']:.4f}, {row['conf_high']:.4f}], p={row['p_value']:.4g}"
        )

    lines.extend(["", "## Descriptive Age Differences", ""])
    for component in COMPONENT_ORDER:
        sensory = desc_wide[
            (desc_wide["component"] == component)
            & (desc_wide["network_type"] == "sensory_motor")
        ].iloc[0]
        higher = desc_wide[
            (desc_wide["component"] == component)
            & (desc_wide["network_type"] == "higher_order")
        ].iloc[0]
        lines.append(
            f"- {COMPONENT_LABELS[component]}: older-young difference in sensory/motor = "
            f"{sensory['older_minus_young']:.4f}; higher-order = {higher['older_minus_young']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If the higher-order interaction is small or non-significant, the current sample does not provide strong evidence that age effects are selectively larger in cognitive networks than in sensory/motor networks.",
            "- If the interaction is in the expected direction but modest, that supports the literature only weakly and should be described as a trend rather than a firm dissociation.",
        ]
    )

    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    sample_path = resolve_project_path(args.sample)
    connectivity_dir = resolve_project_path(args.connectivity_dir)
    output_dir = resolve_project_path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _, network_df = load_tables(sample_path, connectivity_dir)
    network_df = add_network_type(network_df)
    subject_type_df = build_subject_network_type_components(network_df)

    long_df = subject_type_df.melt(
        id_vars=[
            "subject_id",
            "age",
            "sex",
            "age_group",
            "mean_fd",
            "retained_minutes_after_scrub",
            "network_type",
        ],
        value_vars=COMPONENT_ORDER,
        var_name="component",
        value_name="value",
    )
    long_df["component"] = pd.Categorical(
        long_df["component"], categories=COMPONENT_ORDER, ordered=True
    )

    effect_df = build_effect_table(long_df)
    descriptive_df = build_descriptive_table(long_df)

    subject_type_df.to_csv(
        output_dir / "subject_network_type_components.tsv", sep="\t", index=False
    )
    long_df.to_csv(
        output_dir / "subject_network_type_components_long.tsv", sep="\t", index=False
    )
    effect_df.to_csv(
        output_dir / "component_network_type_effects.tsv", sep="\t", index=False
    )
    descriptive_df.to_csv(
        output_dir / "component_network_type_descriptives.tsv", sep="\t", index=False
    )

    settings = pd.DataFrame(
        [
            {"setting": "sample_tsv", "value": str(sample_path)},
            {"setting": "connectivity_dir", "value": str(connectivity_dir)},
            {
                "setting": "sensory_motor_networks",
                "value": ",".join(sorted(SENSORY_MOTOR)),
            },
            {
                "setting": "higher_order_networks",
                "value": ",".join(sorted(HIGHER_ORDER)),
            },
        ]
    )
    settings.to_csv(
        output_dir / "component_network_type_settings.tsv", sep="\t", index=False
    )

    plot_component_panels(long_df, figures_dir / "component_network_type_by_group.png")
    plot_interaction_effects(
        effect_df, figures_dir / "component_network_type_interactions.png"
    )
    write_summary(
        effect_df, descriptive_df, output_dir / "component_network_type_summary.md"
    )

    print(f"Wrote grouped component analysis to {output_dir}")
    print(f"Wrote grouped component figures to {figures_dir}")


if __name__ == "__main__":
    main()
