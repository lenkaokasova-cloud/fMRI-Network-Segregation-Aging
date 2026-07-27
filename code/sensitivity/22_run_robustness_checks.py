#!/usr/bin/env python3

# What this script does:
#   Runs the main robustness checks for the final sample, including stricter
#   motion subsets and metric-sensitivity analyses.
# How to run it:
#   Run from the repo root with:
#   python code/sensitivity/22_run_robustness_checks.py
# Main output:
#   data/processed/analysis/robustness/

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
YOUNG_COLOR = "#4C78A8"
OLDER_COLOR = "#D16A3A"
REFERENCE_LINE_COLOR = "#1F3A5F"
GRID_COLOR = "#B8BDC7"
FIG_DPI = 300
TITLE_SIZE = 15
LABEL_SIZE = 12
TICK_SIZE = 11

GLOBAL_METRICS = {
    "global_segregation_prop": "Proportional segregation",
    "global_segregation_raw_diff": "Raw difference segregation",
    "global_segregation_prop_posonly": "Positive-only proportional segregation",
}
NETWORK_METRICS = {
    "segregation_prop": "Proportional segregation",
    "segregation_raw_diff": "Raw difference segregation",
    "segregation_prop_posonly": "Positive-only proportional segregation",
}

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
            "Run robustness checks for the final TR=3 s age-split analysis, "
            "including metric sensitivity, leave-one-older-out influence, and "
            "a stricter motion subset."
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
        help="Directory containing connectivity metric TSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/analysis/robustness",
        help="Directory for robustness outputs.",
    )
    parser.add_argument(
        "--strict-max-mean-fd",
        type=float,
        default=0.15,
        help="Stricter motion sensitivity threshold for mean FD.",
    )
    parser.add_argument(
        "--strict-max-pct-fd-gt-0p2",
        type=float,
        default=20.0,
        help="Stricter motion sensitivity threshold for percent of volumes with FD > 0.2 mm.",
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


def load_tables(sample_path: Path, connectivity_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample = pd.read_csv(sample_path, sep="\t")
    global_df = pd.read_csv(connectivity_dir / "subject_global_segregation.tsv", sep="\t")
    network_df = pd.read_csv(connectivity_dir / "subject_network_segregation.tsv", sep="\t")

    subject_ids = set(sample["subject_id"])
    global_df = global_df[global_df["subject_id"].isin(subject_ids)].copy()
    network_df = network_df[network_df["subject_id"].isin(subject_ids)].copy()
    return sample, global_df, network_df


def fit_global_model(df: pd.DataFrame, outcome: str):
    return smf.ols(
        f"{outcome} ~ C(age_group, Treatment(reference='young')) + C(sex) + mean_fd",
        data=df,
    ).fit(cov_type="HC3")


def fit_global_model_with_retained_minutes(df: pd.DataFrame, outcome: str):
    return smf.ols(
        f"{outcome} ~ C(age_group, Treatment(reference='young')) + C(sex) + mean_fd + retained_minutes_after_scrub",
        data=df,
    ).fit(cov_type="HC3")


def fit_network_model(df: pd.DataFrame, outcome: str, network: str):
    subset = df[df["network"] == network].copy()
    return smf.ols(
        f"{outcome} ~ C(age_group, Treatment(reference='young')) + C(sex) + mean_fd",
        data=subset,
    ).fit(cov_type="HC3")


def effect_row(
    result,
    *,
    term: str,
    label: str,
    outcome: str,
    data: pd.DataFrame,
) -> dict[str, object]:
    # I keep one shared formatter for effect rows so every robustness branch is summarized in the same way.
    estimate = float(result.params[term])
    std_error = float(result.bse[term])
    return {
        "outcome": outcome,
        "label": label,
        "estimate_older_vs_young": estimate,
        "std_error": std_error,
        "conf_low": estimate - 1.96 * std_error,
        "conf_high": estimate + 1.96 * std_error,
        "p_value": float(result.pvalues[term]),
        "young_mean": float(data.loc[data["age_group"] == "young", outcome].mean()),
        "older_mean": float(data.loc[data["age_group"] == "older", outcome].mean()),
        "older_minus_young_mean": float(
            data.loc[data["age_group"] == "older", outcome].mean()
            - data.loc[data["age_group"] == "young", outcome].mean()
        ),
        "n_subjects": int(data["subject_id"].nunique()),
        "n_younger": int(data.loc[data["age_group"] == "young", "subject_id"].nunique()),
        "n_older": int(data.loc[data["age_group"] == "older", "subject_id"].nunique()),
    }


def run_metric_sensitivity_global(global_df: pd.DataFrame) -> pd.DataFrame:
    term = "C(age_group, Treatment(reference='young'))[T.older]"
    rows: list[dict[str, object]] = []
    for outcome, label in GLOBAL_METRICS.items():
        result = fit_global_model(global_df, outcome)
        rows.append(effect_row(result, term=term, label=label, outcome=outcome, data=global_df))
    return pd.DataFrame(rows)


def run_metric_sensitivity_network(network_df: pd.DataFrame) -> pd.DataFrame:
    term = "C(age_group, Treatment(reference='young'))[T.older]"
    networks = ordered_networks(network_df["network"].tolist())
    rows: list[dict[str, object]] = []
    for outcome, label in NETWORK_METRICS.items():
        for network in networks:
            subset = network_df[network_df["network"] == network].copy()
            result = fit_network_model(network_df, outcome, network)
            row = effect_row(result, term=term, label=label, outcome=outcome, data=subset)
            row["network"] = network
            rows.append(row)
    out = pd.DataFrame(rows)
    out["network"] = pd.Categorical(out["network"], categories=networks, ordered=True)
    return out.sort_values(["outcome", "network"]).reset_index(drop=True)


def run_leave_one_older_out(global_df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    # This checks whether one particular older participant is carrying too much of the global effect.
    term = "C(age_group, Treatment(reference='young'))[T.older]"
    older_subjects = sorted(global_df.loc[global_df["age_group"] == "older", "subject_id"].unique())
    rows: list[dict[str, object]] = []
    for subject_id in older_subjects:
        subset = global_df[global_df["subject_id"] != subject_id].copy()
        result = fit_global_model(subset, outcome)
        rows.append(
            {
                "left_out_older_subject": subject_id,
                "estimate_older_vs_young": float(result.params[term]),
                "std_error": float(result.bse[term]),
                "conf_low": float(result.params[term] - 1.96 * result.bse[term]),
                "conf_high": float(result.params[term] + 1.96 * result.bse[term]),
                "p_value": float(result.pvalues[term]),
                "n_subjects": int(subset["subject_id"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("estimate_older_vs_young").reset_index(drop=True)


def run_stricter_motion_sensitivity(
    global_df: pd.DataFrame,
    *,
    strict_max_mean_fd: float,
    strict_max_pct_fd_gt_0p2: float,
    outcome: str,
) -> pd.DataFrame:
    term = "C(age_group, Treatment(reference='young'))[T.older]"
    rows: list[dict[str, object]] = []
    subsets = {
        "main_sample": global_df.copy(),
        "stricter_motion_subset": global_df[
            (global_df["mean_fd"] < strict_max_mean_fd)
            & (global_df["pct_fd_gt_0p2"] < strict_max_pct_fd_gt_0p2)
        ].copy(),
    }
    for subset_name, subset in subsets.items():
        if subset["age_group"].nunique() < 2:
            continue
        result = fit_global_model(subset, outcome)
        row = effect_row(result, term=term, label=subset_name, outcome=outcome, data=subset)
        row["subset"] = subset_name
        row["strict_max_mean_fd"] = strict_max_mean_fd
        row["strict_max_pct_fd_gt_0p2"] = strict_max_pct_fd_gt_0p2
        rows.append(row)
    return pd.DataFrame(rows)


def run_retained_minutes_covariate_sensitivity(global_df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    term = "C(age_group, Treatment(reference='young'))[T.older]"
    base_result = fit_global_model(global_df, outcome)
    retained_result = fit_global_model_with_retained_minutes(global_df, outcome)
    rows: list[dict[str, object]] = []
    for label, result in [
        ("base_model", base_result),
        ("with_retained_minutes_covariate", retained_result),
    ]:
        row = effect_row(result, term=term, label=label, outcome=outcome, data=global_df)
        row["model_variant"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def plot_global_metric_sensitivity(results: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    y_pos = np.arange(len(results))[::-1]
    ax.axvline(0.0, color=REFERENCE_LINE_COLOR, linewidth=1.6, linestyle="--")
    ax.errorbar(
        results["estimate_older_vs_young"],
        y_pos,
        xerr=[
            results["estimate_older_vs_young"] - results["conf_low"],
            results["conf_high"] - results["estimate_older_vs_young"],
        ],
        fmt="o",
        color=OLDER_COLOR,
        ecolor=OLDER_COLOR,
        elinewidth=2.0,
        capsize=4,
        markersize=8,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results["label"])
    style_axis(
        ax,
        title="Global Age Effect Across Segregation Metrics",
        xlabel="Older minus younger estimate",
    )
    ax.grid(axis="x", color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    save_figure(fig, outpath)


def plot_leave_one_out(results: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    order = results["left_out_older_subject"].tolist()
    x_pos = np.arange(len(order))
    ax.axhline(0.0, color=REFERENCE_LINE_COLOR, linewidth=1.6, linestyle="--")
    ax.errorbar(
        x_pos,
        results["estimate_older_vs_young"],
        yerr=[
            results["estimate_older_vs_young"] - results["conf_low"],
            results["conf_high"] - results["estimate_older_vs_young"],
        ],
        fmt="o",
        color=OLDER_COLOR,
        ecolor=OLDER_COLOR,
        elinewidth=1.8,
        capsize=3.5,
        markersize=6.5,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(order, rotation=45, ha="right")
    style_axis(
        ax,
        title="Leave-One-Older-Out Global Segregation Effect",
        ylabel="Older minus younger estimate",
    )
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    save_figure(fig, outpath)


def write_summary_markdown(
    outpath: Path,
    global_metric_results: pd.DataFrame,
    leave_one_out_results: pd.DataFrame,
    stricter_motion_results: pd.DataFrame,
    retained_minutes_results: pd.DataFrame,
) -> None:
    main_row = global_metric_results.loc[
        global_metric_results["outcome"] == "global_segregation_prop"
    ].iloc[0]
    raw_row = global_metric_results.loc[
        global_metric_results["outcome"] == "global_segregation_raw_diff"
    ].iloc[0]
    pos_row = global_metric_results.loc[
        global_metric_results["outcome"] == "global_segregation_prop_posonly"
    ].iloc[0]

    leave_one_out_range = (
        float(leave_one_out_results["estimate_older_vs_young"].min()),
        float(leave_one_out_results["estimate_older_vs_young"].max()),
    )

    lines = [
        "# Robustness Checks Summary",
        "",
        "These checks were run to show that the main age-split conclusions are not entirely dependent on one segregation definition or one older subject.",
        "",
        "## Global Metric Sensitivity",
        "",
        f"- Proportional segregation: estimate={main_row['estimate_older_vs_young']:.4f}, p={main_row['p_value']:.4g}",
        f"- Raw difference segregation: estimate={raw_row['estimate_older_vs_young']:.4f}, p={raw_row['p_value']:.4g}",
        f"- Positive-only proportional segregation: estimate={pos_row['estimate_older_vs_young']:.4f}, p={pos_row['p_value']:.4g}",
        "",
        "## Leave-One-Older-Out Influence",
        "",
        f"- Older-vs-young proportional global effect ranged from {leave_one_out_range[0]:.4f} to {leave_one_out_range[1]:.4f} when each older participant was removed in turn.",
        "",
        "## Stricter Motion Sensitivity",
        "",
    ]

    for _, row in stricter_motion_results.iterrows():
        lines.append(
            f"- {row['subset']}: n={int(row['n_subjects'])} "
            f"({int(row['n_younger'])} young, {int(row['n_older'])} older), "
            f"estimate={row['estimate_older_vs_young']:.4f}, p={row['p_value']:.4g}"
        )

    lines.extend(["", "## Retained-Time Covariate Sensitivity", ""])
    for _, row in retained_minutes_results.iterrows():
        lines.append(
            f"- {row['model_variant']}: estimate={row['estimate_older_vs_young']:.4f}, p={row['p_value']:.4g}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If the sign and rough magnitude of the global age effect remain similar across these checks, that supports the stability of the main descriptive result.",
            "- If one variant changes direction or becomes much larger or smaller than the others, that should be discussed as metric dependence rather than ignored.",
            "- These robustness checks do not replace the main permutation/FDR inference; they are meant to show whether the main pattern is structurally fragile or broadly stable.",
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

    _, global_df, network_df = load_tables(sample_path, connectivity_dir)

    global_metric_results = run_metric_sensitivity_global(global_df)
    network_metric_results = run_metric_sensitivity_network(network_df)
    leave_one_out_results = run_leave_one_older_out(global_df, "global_segregation_prop")
    stricter_motion_results = run_stricter_motion_sensitivity(
        global_df,
        strict_max_mean_fd=args.strict_max_mean_fd,
        strict_max_pct_fd_gt_0p2=args.strict_max_pct_fd_gt_0p2,
        outcome="global_segregation_prop",
    )
    retained_minutes_results = run_retained_minutes_covariate_sensitivity(
        global_df,
        "global_segregation_prop",
    )

    global_metric_results.to_csv(output_dir / "global_metric_sensitivity.tsv", sep="\t", index=False)
    network_metric_results.to_csv(output_dir / "network_metric_sensitivity.tsv", sep="\t", index=False)
    leave_one_out_results.to_csv(output_dir / "leave_one_older_out_global.tsv", sep="\t", index=False)
    stricter_motion_results.to_csv(output_dir / "stricter_motion_global.tsv", sep="\t", index=False)
    retained_minutes_results.to_csv(
        output_dir / "retained_minutes_covariate_global.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "sample": str(sample_path),
                "connectivity_dir": str(connectivity_dir),
                "strict_max_mean_fd": args.strict_max_mean_fd,
                "strict_max_pct_fd_gt_0p2": args.strict_max_pct_fd_gt_0p2,
                "primary_global_metric": "global_segregation_prop",
                "primary_network_metric": "segregation_prop",
            }
        ]
    ).to_csv(output_dir / "robustness_settings.tsv", sep="\t", index=False)

    plot_global_metric_sensitivity(
        global_metric_results,
        figures_dir / "global_metric_sensitivity.png",
    )
    plot_leave_one_out(
        leave_one_out_results,
        figures_dir / "leave_one_older_out_global.png",
    )
    write_summary_markdown(
        output_dir / "robustness_summary.md",
        global_metric_results,
        leave_one_out_results,
        stricter_motion_results,
        retained_minutes_results,
    )

    print(f"Wrote robustness tables to {output_dir}")
    print(f"Wrote robustness figures to {figures_dir}")


if __name__ == "__main__":
    main()
