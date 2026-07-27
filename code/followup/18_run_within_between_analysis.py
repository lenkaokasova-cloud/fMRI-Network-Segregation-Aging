#!/usr/bin/env python3

# What this script does:
#   Splits segregation into within-network and between-network parts and tests
#   whether age effects are coming from one, the other, or both.
# How to run it:
#   Run from the repo root with:
#   python code/followup/18_run_within_between_analysis.py
# Main output:
#   data/processed/analysis/within_between/

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("data/processed/.matplotlib")).resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
SEGREGATION_COL = "segregation_prop"
YOUNG_COLOR = "#4C78A8"
OLDER_COLOR = "#D16A3A"
GRID_COLOR = "#B8BDC7"
REFERENCE_LINE_COLOR = "#1F3A5F"
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
            "Split network segregation into its within-network and between-network "
            "components and test age effects on those components."
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
        help="Directory containing subject_network_segregation.tsv.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/analysis/within_between",
        help="Directory for within/between component analysis outputs.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ordered_networks(networks: list[str]) -> list[str]:
    order_index = {network: idx for idx, network in enumerate(NETWORK_ORDER)}
    unique_networks = list(dict.fromkeys(networks))
    return sorted(unique_networks, key=lambda name: (order_index.get(name, len(order_index)), name))


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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelrotation=xrotation)
    for label in ax.get_xticklabels():
        label.set_ha("right" if xrotation else "center")


def save_figure(fig: plt.Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def group_offsets(n_points: int, width: float = 0.12) -> np.ndarray:
    if n_points == 0:
        return np.array([])
    if n_points == 1:
        return np.array([0.0])
    return np.linspace(-width, width, n_points)


def load_tables(sample_path: Path, connectivity_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = pd.read_csv(sample_path, sep="\t")
    network_df = pd.read_csv(connectivity_dir / "subject_network_segregation.tsv", sep="\t")

    required_sample = {"subject_id", "age", "sex", "age_group"}
    missing_sample = required_sample.difference(sample.columns)
    if missing_sample:
        missing_str = ", ".join(sorted(missing_sample))
        raise ValueError(f"Sample TSV missing columns: {missing_str}")

    required_network = {
        "subject_id",
        "age",
        "sex",
        "age_group",
        "network",
        "within_mean_z",
        "between_mean_z",
        SEGREGATION_COL,
        "mean_fd",
        "retained_minutes_after_scrub",
    }
    missing_network = required_network.difference(network_df.columns)
    if missing_network:
        missing_str = ", ".join(sorted(missing_network))
        raise ValueError(f"Network connectivity TSV missing columns: {missing_str}")

    sample_subjects = set(sample["subject_id"])
    network_df = network_df[network_df["subject_id"].isin(sample_subjects)].copy()
    return sample, network_df


def build_subject_component_table(network_df: pd.DataFrame) -> pd.DataFrame:
    # Here I split segregation into its two ingredients instead of only keeping the final ratio.
    return (
        network_df.groupby(
            ["subject_id", "age", "sex", "age_group", "mean_fd", "retained_minutes_after_scrub"],
            as_index=False,
        )
        .agg(
            overall_within_mean=("within_mean_z", "mean"),
            overall_between_mean=("between_mean_z", "mean"),
            overall_segregation=(SEGREGATION_COL, "mean"),
        )
        .sort_values(["age_group", "age", "subject_id"])
        .reset_index(drop=True)
    )


def fit_subject_level_model(df: pd.DataFrame, outcome: str):
    formula = f"{outcome} ~ C(age_group, Treatment(reference='young')) + C(sex) + mean_fd"
    return smf.ols(formula, data=df).fit(cov_type="HC3")


def build_subject_effect_table(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    term = "C(age_group, Treatment(reference='young'))[T.older]"

    for outcome, label in [
        ("overall_within_mean", "within_mean_z"),
        ("overall_between_mean", "between_mean_z"),
        ("overall_segregation", SEGREGATION_COL),
    ]:
        result = fit_subject_level_model(subject_df, outcome)
        estimate = float(result.params[term])
        std_error = float(result.bse[term])
        rows.append(
            {
                "component": label,
                "estimate_older_vs_young": estimate,
                "std_error": std_error,
                "conf_low": estimate - 1.96 * std_error,
                "conf_high": estimate + 1.96 * std_error,
                "p_value": float(result.pvalues[term]),
                "nobs": float(result.nobs),
                "young_mean": float(subject_df.loc[subject_df["age_group"] == "young", outcome].mean()),
                "older_mean": float(subject_df.loc[subject_df["age_group"] == "older", outcome].mean()),
                "older_minus_young_mean": float(
                    subject_df.loc[subject_df["age_group"] == "older", outcome].mean()
                    - subject_df.loc[subject_df["age_group"] == "young", outcome].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def fit_network_specific_model(network_df: pd.DataFrame, outcome: str, network: str):
    subset = network_df[network_df["network"] == network].copy()
    formula = f"{outcome} ~ C(age_group, Treatment(reference='young')) + C(sex) + mean_fd"
    result = smf.ols(formula, data=subset).fit(cov_type="HC3")
    term = "C(age_group, Treatment(reference='young'))[T.older]"
    estimate = float(result.params[term])
    std_error = float(result.bse[term])
    return {
        "network": network,
        "estimate_older_vs_young": estimate,
        "std_error": std_error,
        "conf_low": estimate - 1.96 * std_error,
        "conf_high": estimate + 1.96 * std_error,
        "p_value": float(result.pvalues[term]),
        "nobs": float(result.nobs),
        "young_mean": float(subset.loc[subset["age_group"] == "young", outcome].mean()),
        "older_mean": float(subset.loc[subset["age_group"] == "older", outcome].mean()),
        "older_minus_young_mean": float(
            subset.loc[subset["age_group"] == "older", outcome].mean()
            - subset.loc[subset["age_group"] == "young", outcome].mean()
        ),
    }


def build_network_effect_table(network_df: pd.DataFrame) -> pd.DataFrame:
    networks = ordered_networks(network_df["network"].tolist())
    rows: list[dict[str, object]] = []
    for component, outcome in [
        ("within_mean_z", "within_mean_z"),
        ("between_mean_z", "between_mean_z"),
        (SEGREGATION_COL, SEGREGATION_COL),
    ]:
        for network in networks:
            row = fit_network_specific_model(network_df, outcome, network)
            row["component"] = component
            rows.append(row)
    out = pd.DataFrame(rows)
    out["network"] = pd.Categorical(out["network"], categories=networks, ordered=True)
    return out.sort_values(["component", "network"]).reset_index(drop=True)


def build_network_component_summary(network_effect_df: pd.DataFrame) -> pd.DataFrame:
    # This brings the within, between, and segregation results for each network back into one readable table.
    keep_cols = [
        "network",
        "young_mean",
        "older_mean",
        "older_minus_young_mean",
        "estimate_older_vs_young",
        "conf_low",
        "conf_high",
        "p_value",
    ]
    within = (
        network_effect_df[network_effect_df["component"] == "within_mean_z"][keep_cols]
        .rename(
            columns={
                "young_mean": "young_within_mean",
                "older_mean": "older_within_mean",
                "older_minus_young_mean": "older_minus_young_within",
                "estimate_older_vs_young": "within_age_effect_estimate",
                "conf_low": "within_conf_low",
                "conf_high": "within_conf_high",
                "p_value": "within_p_value",
            }
        )
        .reset_index(drop=True)
    )
    between = (
        network_effect_df[network_effect_df["component"] == "between_mean_z"][keep_cols]
        .rename(
            columns={
                "young_mean": "young_between_mean",
                "older_mean": "older_between_mean",
                "older_minus_young_mean": "older_minus_young_between",
                "estimate_older_vs_young": "between_age_effect_estimate",
                "conf_low": "between_conf_low",
                "conf_high": "between_conf_high",
                "p_value": "between_p_value",
            }
        )
        .reset_index(drop=True)
    )
    segregation = (
        network_effect_df[network_effect_df["component"] == SEGREGATION_COL][keep_cols]
        .rename(
            columns={
                "young_mean": "young_segregation_mean",
                "older_mean": "older_segregation_mean",
                "older_minus_young_mean": "older_minus_young_segregation",
                "estimate_older_vs_young": "segregation_age_effect_estimate",
                "conf_low": "segregation_conf_low",
                "conf_high": "segregation_conf_high",
                "p_value": "segregation_p_value",
            }
        )
        .reset_index(drop=True)
    )
    return within.merge(between, on="network", how="inner").merge(segregation, on="network", how="inner")


def plot_subject_components(subject_df: pd.DataFrame, outpath: Path) -> None:
    component_specs = [
        ("overall_within_mean", "Overall within-network connectivity"),
        ("overall_between_mean", "Overall between-network connectivity"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=False)

    for ax, (column, title) in zip(axes, component_specs):
        data = [
            subject_df.loc[subject_df["age_group"] == "young", column].to_numpy(),
            subject_df.loc[subject_df["age_group"] == "older", column].to_numpy(),
        ]
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
        for patch, color in zip(box["boxes"], [YOUNG_COLOR, OLDER_COLOR]):
            patch.set_facecolor(color)
            patch.set_alpha(0.28)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.8)

        for x_pos, group, color in zip([0, 1], ["young", "older"], [YOUNG_COLOR, OLDER_COLOR]):
            subset = subject_df[subject_df["age_group"] == group].reset_index(drop=True)
            offsets = group_offsets(len(subset))
            ax.scatter(
                np.full(len(subset), x_pos) + offsets,
                subset[column],
                color=color,
                s=78,
                alpha=0.92,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )

        style_axis(ax, title=title, ylabel="Mean Fisher z connectivity")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["young", "older"])
        ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.9)

    save_figure(fig, outpath)


def plot_network_means(network_summary: pd.DataFrame, outpath: Path) -> None:
    networks = network_summary["network"].astype(str).tolist()
    x = np.arange(len(networks))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), sharex=True)
    specs = [
        ("young_within_mean", "older_within_mean", "Within-network connectivity"),
        ("young_between_mean", "older_between_mean", "Between-network connectivity"),
    ]

    for ax, (young_col, older_col, title) in zip(axes, specs):
        ax.plot(x, network_summary[young_col], color=YOUNG_COLOR, marker="o", linewidth=2.0, label="young")
        ax.plot(x, network_summary[older_col], color=OLDER_COLOR, marker="o", linewidth=2.0, label="older")
        style_axis(ax, title=title, ylabel="Mean Fisher z connectivity")
        ax.set_xticks(x)
        ax.set_xticklabels(networks, rotation=35)
        ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.9)

    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    save_figure(fig, outpath)


def plot_network_age_effects(network_effect_df: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), sharey=True)
    specs = [
        ("within_mean_z", "Age effect on within-network connectivity"),
        ("between_mean_z", "Age effect on between-network connectivity"),
    ]
    networks = ordered_networks(network_effect_df["network"].astype(str).tolist())
    y = np.arange(len(networks))

    for ax, (component, title) in zip(axes, specs):
        subset = (
            network_effect_df[network_effect_df["component"] == component]
            .copy()
            .sort_values("network")
            .reset_index(drop=True)
        )
        estimates = subset["estimate_older_vs_young"].to_numpy()
        lower = subset["estimate_older_vs_young"].to_numpy() - subset["conf_low"].to_numpy()
        upper = subset["conf_high"].to_numpy() - subset["estimate_older_vs_young"].to_numpy()
        colors = [
            OLDER_COLOR if value > 0 else YOUNG_COLOR
            for value in estimates
        ]
        ax.errorbar(
            estimates,
            y,
            xerr=[lower, upper],
            fmt="o",
            color=REFERENCE_LINE_COLOR,
            ecolor=REFERENCE_LINE_COLOR,
            elinewidth=1.8,
            capsize=3,
            markersize=0,
            zorder=2,
        )
        ax.scatter(estimates, y, s=80, c=colors, edgecolors="white", linewidths=0.6, zorder=3)
        ax.axvline(0.0, color=REFERENCE_LINE_COLOR, linewidth=1.4, linestyle="--")
        style_axis(ax, title=title, xlabel="Older minus young estimate")
        ax.grid(axis="x", color=GRID_COLOR, alpha=0.28, linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(networks)

    save_figure(fig, outpath)


def write_summary_markdown(
    outpath: Path,
    subject_effect_df: pd.DataFrame,
    network_summary: pd.DataFrame,
) -> None:
    within_row = subject_effect_df[subject_effect_df["component"] == "within_mean_z"].iloc[0]
    between_row = subject_effect_df[subject_effect_df["component"] == "between_mean_z"].iloc[0]

    largest_within_drop = (
        network_summary.sort_values("older_minus_young_within", ascending=True)
        .head(3)
        .reset_index(drop=True)
    )
    largest_between_increase = (
        network_summary.sort_values("older_minus_young_between", ascending=False)
        .head(3)
        .reset_index(drop=True)
    )

    lines = [
        "# Within/Between Connectivity Analysis Summary",
        "",
        "## Overall Components",
        "",
        (
            f"- Overall within-network connectivity older-vs-young estimate: "
            f"{within_row['estimate_older_vs_young']:.4f}, "
            f"95% CI [{within_row['conf_low']:.4f}, {within_row['conf_high']:.4f}], "
            f"p={within_row['p_value']:.4g}"
        ),
        (
            f"- Overall between-network connectivity older-vs-young estimate: "
            f"{between_row['estimate_older_vs_young']:.4f}, "
            f"95% CI [{between_row['conf_low']:.4f}, {between_row['conf_high']:.4f}], "
            f"p={between_row['p_value']:.4g}"
        ),
        "",
        "## Largest Older Reductions in Within-Network Connectivity",
        "",
    ]
    for _, row in largest_within_drop.iterrows():
        lines.append(
            f"- {row['network']}: older-young within difference={row['older_minus_young_within']:.4f}, "
            f"p={row['within_p_value']:.4g}"
        )

    lines.extend(["", "## Largest Older Increases in Between-Network Connectivity", ""])
    for _, row in largest_between_increase.iterrows():
        lines.append(
            f"- {row['network']}: older-young between difference={row['older_minus_young_between']:.4f}, "
            f"p={row['between_p_value']:.4g}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Lower segregation with age can come from lower within-network connectivity, higher between-network connectivity, or both.",
            "- This analysis separates those two possibilities directly.",
            "- Negative within-network age effects mean older participants show weaker internal coherence in that network.",
            "- Positive between-network age effects mean older participants show more cross-network mixing for that network.",
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

    sample, network_df = load_tables(sample_path, connectivity_dir)
    sample_subjects = set(sample["subject_id"])
    network_df = network_df[network_df["subject_id"].isin(sample_subjects)].copy()
    network_df["network"] = pd.Categorical(
        network_df["network"],
        categories=ordered_networks(network_df["network"].astype(str).tolist()),
        ordered=True,
    )
    network_df = network_df.sort_values(["age_group", "age", "subject_id", "network"]).reset_index(drop=True)

    subject_df = build_subject_component_table(network_df)
    subject_effect_df = build_subject_effect_table(subject_df)
    network_effect_df = build_network_effect_table(network_df)
    network_summary = build_network_component_summary(network_effect_df)

    subject_df.to_csv(output_dir / "subject_component_summary.tsv", sep="\t", index=False)
    network_df.to_csv(output_dir / "subject_network_components.tsv", sep="\t", index=False)
    subject_effect_df.to_csv(output_dir / "overall_component_effects.tsv", sep="\t", index=False)
    network_effect_df.to_csv(output_dir / "network_component_effects.tsv", sep="\t", index=False)
    network_summary.to_csv(output_dir / "network_component_summary.tsv", sep="\t", index=False)

    pd.DataFrame(
        [
            {
                "sample_file": str(sample_path),
                "connectivity_dir": str(connectivity_dir),
                "n_subjects": len(subject_df),
                "n_network_rows": len(network_df),
            }
        ]
    ).to_csv(output_dir / "within_between_settings.tsv", sep="\t", index=False)

    plot_subject_components(subject_df, figures_dir / "overall_within_between_by_group.png")
    plot_network_means(network_summary, figures_dir / "network_within_between_means.png")
    plot_network_age_effects(network_effect_df, figures_dir / "network_within_between_age_effects.png")
    write_summary_markdown(output_dir / "within_between_summary.md", subject_effect_df, network_summary)

    print(f"Wrote within/between tables to {output_dir}")
    print(f"Wrote within/between figures to {figures_dir}")


if __name__ == "__main__":
    main()
