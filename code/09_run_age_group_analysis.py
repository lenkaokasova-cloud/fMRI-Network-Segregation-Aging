#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("data/processed/.matplotlib")).resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIGHER_ORDER = {"Default", "Cont", "DorsAttn", "SalVentAttn"}
SENSORY_MOTOR = {"Vis", "SomMot"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the main age-group analyses on global and network segregation "
            "while controlling for sex and mean FD."
        )
    )
    parser.add_argument(
        "--sample",
        default="data/processed/screening/ds005752_clean_age_sample.tsv",
        help="Clean sample TSV produced by code/07_build_clean_sample.py.",
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


def merge_missing_columns(df: pd.DataFrame, sample: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    missing = [column for column in required_columns if column not in df.columns]
    if not missing:
        return df.copy()
    keep = ["subject_id"] + missing
    return df.merge(sample[keep], on="subject_id", how="inner")


def model_to_table(result, model_name: str) -> pd.DataFrame:
    conf_int = result.conf_int()
    conf_int.columns = ["conf_low", "conf_high"]
    table = pd.DataFrame(
        {
            "term": result.params.index,
            "estimate": result.params.values,
            "std_error": result.bse.values,
            "statistic": result.tvalues.values,
            "p_value": result.pvalues.values,
        }
    )
    table = table.join(conf_int, how="left")
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


def plot_age_vs_global(df: pd.DataFrame, outpath: Path) -> None:
    colors = {"young": "#2b6cb0", "older": "#c05621"}
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, row in df.iterrows():
        ax.scatter(row["age"], row["global_segregation"], color=colors[row["age_group"]], s=75)
    ax.set_xlabel("Age")
    ax.set_ylabel("Global segregation")
    ax.set_title("Age vs global segregation")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_global_by_group(df: pd.DataFrame, outpath: Path) -> None:
    colors = {"young": "#2b6cb0", "older": "#c05621"}
    order = ["young", "older"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for x, group in enumerate(order):
        group_df = df[df["age_group"] == group].reset_index(drop=True)
        offsets = np.linspace(-0.12, 0.12, max(len(group_df), 1))
        for offset, (_, row) in zip(offsets, group_df.iterrows()):
            ax.scatter(x + offset, row["global_segregation"], color=colors[group], s=80)
        if not group_df.empty:
            ax.hlines(
                group_df["global_segregation"].mean(),
                x - 0.18,
                x + 0.18,
                colors=colors[group],
                linewidth=2,
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(order)
    ax.set_ylabel("Global segregation")
    ax.set_title("Global segregation by age group")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_motion_vs_global(df: pd.DataFrame, outpath: Path) -> None:
    colors = {"young": "#2b6cb0", "older": "#c05621"}
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, row in df.iterrows():
        ax.scatter(row["mean_fd"], row["global_segregation"], color=colors[row["age_group"]], s=75)
    ax.set_xlabel("Mean FD (mm)")
    ax.set_ylabel("Global segregation")
    ax.set_title("Motion vs global segregation")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_network_type_summary(df: pd.DataFrame, outpath: Path) -> None:
    colors = {"young": "#2b6cb0", "older": "#c05621"}
    network_order = ["sensory_motor", "higher_order"]
    age_order = ["young", "older"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.35
    for idx, age_group in enumerate(age_order):
        means = []
        for network_type in network_order:
            subset = df[(df["age_group"] == age_group) & (df["network_type"] == network_type)]
            means.append(subset["segregation"].mean())
        x = np.arange(len(network_order)) + (idx - 0.5) * width
        ax.bar(x, means, width=width, color=colors[age_group], label=age_group)
    ax.set_xticks(np.arange(len(network_order)))
    ax.set_xticklabels(network_order)
    ax.set_ylabel("Mean segregation")
    ax.set_title("Higher-order vs sensory-motor segregation by age group")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


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
    global_term = describe_term(global_model, "C(age_group, Treatment(reference='young'))[T.older]")
    if global_term:
        lines.append(f"- Global model older-vs-young effect: {global_term}")
    interaction_term = describe_term(
        network_type_model,
        "C(age_group, Treatment(reference='young'))[T.older]:C(network_type, Treatment(reference='sensory_motor'))[T.higher_order]",
    )
    if interaction_term:
        lines.append(f"- Network-type interaction: {interaction_term}")

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
    global_df = pd.read_csv(connectivity_dir / "subject_global_segregation.tsv", sep="\t")
    network_df = pd.read_csv(connectivity_dir / "subject_network_segregation.tsv", sep="\t")

    global_df = merge_missing_columns(
        global_df,
        sample,
        ["age", "sex", "age_group", "mean_fd", "pct_fd_gt_0p2", "n_retained_after_scrub"],
    )
    network_df = merge_missing_columns(
        network_df,
        sample,
        ["age", "sex", "age_group", "mean_fd", "n_retained_after_scrub"],
    )

    network_df = add_network_type(network_df)
    network_type_df = (
        network_df[network_df["network_type"].isin(["higher_order", "sensory_motor"])]
        .groupby(["subject_id", "age", "sex", "age_group", "mean_fd", "network_type"], as_index=False)["segregation"]
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
        "global_segregation ~ C(age_group, Treatment(reference='young')) + C(sex) + mean_fd",
        data=global_df,
    ).fit(cov_type="HC3")
    continuous_age_model = smf.ols(
        "global_segregation ~ age + C(sex) + mean_fd",
        data=global_df,
    ).fit(cov_type="HC3")
    network_type_model = smf.ols(
        "segregation ~ C(age_group, Treatment(reference='young')) * C(network_type, Treatment(reference='sensory_motor')) + C(sex) + mean_fd",
        data=network_type_df,
    ).fit(cov_type="cluster", cov_kwds={"groups": network_type_df["subject_id"]})
    network_specific_model = smf.ols(
        "segregation ~ C(age_group, Treatment(reference='young')) * C(network) + C(sex) + mean_fd",
        data=network_df,
    ).fit(cov_type="cluster", cov_kwds={"groups": network_df["subject_id"]})

    global_table = model_to_table(global_model, "global_group_model")
    continuous_age_table = model_to_table(continuous_age_model, "global_continuous_age_model")
    network_type_table = model_to_table(network_type_model, "network_type_model")
    network_specific_table = model_to_table(network_specific_model, "network_specific_model")

    sample_summary.to_csv(output_dir / "sample_summary.tsv", sep="\t", index=False)
    global_df.to_csv(output_dir / "subject_global_with_metadata.tsv", sep="\t", index=False)
    network_df.to_csv(output_dir / "subject_network_with_metadata.tsv", sep="\t", index=False)
    network_type_df.to_csv(output_dir / "subject_network_type_summary.tsv", sep="\t", index=False)
    pd.concat(
        [global_table, continuous_age_table, network_type_table, network_specific_table],
        ignore_index=True,
    ).to_csv(output_dir / "model_coefficients.tsv", sep="\t", index=False)

    plot_age_vs_global(global_df, figures_dir / "age_vs_global_segregation.png")
    plot_global_by_group(global_df, figures_dir / "global_segregation_by_group.png")
    plot_motion_vs_global(global_df, figures_dir / "motion_vs_global_segregation.png")
    plot_network_type_summary(network_type_df, figures_dir / "network_type_by_group.png")
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
