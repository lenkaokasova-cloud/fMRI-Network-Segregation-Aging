#!/usr/bin/env python3

# What this script does:
#   Collects the main analysis outputs and rewrites them into cleaner summary
#   tables and figures that are easier to use in the dissertation.
# How to run it:
#   Run from the repo root with:
#   python code/utilities/16_build_dissertation_summary_outputs.py
# Main output:
#   data/processed/analysis/dissertation_summary/

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
TITLE_SIZE = 15
LABEL_SIZE = 12
TICK_SIZE = 10
GRID_COLOR = "#B8BDC7"
REFERENCE_LINE_COLOR = "#1F3A5F"
OVERALL_COLOR = "#1F3A5F"
NETWORK_COLOR = "#4C78A8"
HIGHLIGHT_COLOR = "#D16A3A"

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
            "ci_95": "95% CI",
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
    ax.grid(axis="x", color=GRID_COLOR, alpha=0.30, linewidth=0.9)
    ax.axvline(0.0, color=REFERENCE_LINE_COLOR, linewidth=1.2, alpha=0.85)


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
            f"q={q:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color="black",
        )


def plot_headline_effects(overall_df: pd.DataFrame, network_df: pd.DataFrame, outpath: Path) -> None:
    overall = clean_overall_table(overall_df).sort_values("display_order").reset_index(drop=True)
    network = clean_network_table(network_df).sort_values("display_order").reset_index(drop=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 5.8),
        gridspec_kw={"width_ratios": [1.0, 1.35]},
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
        elinewidth=1.8,
        capsize=3.5,
        markersize=6.8,
        markerfacecolor="white",
        markeredgewidth=1.8,
        zorder=3,
    )
    ax.set_yticks(y_overall)
    ax.set_yticklabels(overall["result_label"])
    ax.set_xlabel("Older vs younger estimate")
    ax.set_title("Overall outcomes")
    style_axis(ax)
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
            elinewidth=1.7,
            capsize=3.3,
            markersize=6.2,
            markerfacecolor="white",
            markeredgewidth=1.7,
            zorder=3,
        )
    ax.set_yticks(y_network)
    ax.set_yticklabels(network["network"].astype(str))
    ax.set_xlabel("Older vs younger estimate")
    ax.set_title("Network-specific segregation")
    style_axis(ax)
    add_q_labels(
        ax,
        network["conf_high"].to_numpy(dtype=float),
        y_network,
        network["fdr_q_value"].to_numpy(dtype=float),
    )

    fig.suptitle("Main age-effect summary", y=1.02)
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
            "- The confidence intervals in these summary outputs are model-based intervals carried through from the regression summaries.",
            "- The p values and q values are permutation-based from the Freedman-Lane analysis.",
            "- These should therefore not be described as permutation-based confidence intervals in the dissertation text.",
        ]
    )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    sample_path = resolve_project_path(args.sample)
    permutation_dir = resolve_project_path(args.permutation_dir)
    output_dir = resolve_project_path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    sample_df = pd.read_csv(sample_path, sep="\t")
    overall_df = pd.read_csv(permutation_dir / "overall_permutation_summary.tsv", sep="\t")
    network_df = pd.read_csv(permutation_dir / "network_permutation_summary.tsv", sep="\t")

    headline_table = build_headline_table(overall_df, network_df)
    word_table = build_word_friendly_table(headline_table)
    characteristics_table = build_participant_characteristics_table(sample_df)
    headline_table.to_csv(output_dir / "main_age_effects_summary.tsv", sep="\t", index=False)
    word_table.to_csv(output_dir / "table2_main_age_effects_word_friendly.tsv", sep="\t", index=False)
    word_table.to_csv(output_dir / "table2_main_age_effects_word_friendly.csv", index=False)
    characteristics_table.to_csv(output_dir / "table1_participant_characteristics.tsv", sep="\t", index=False)
    characteristics_table.to_csv(output_dir / "table1_participant_characteristics.csv", index=False)

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
            }
        ]
    ).to_csv(output_dir / "dissertation_summary_settings.tsv", sep="\t", index=False)

    plot_headline_effects(
        overall_df,
        network_df,
        figures_dir / "main_age_effects_summary.png",
    )
    write_summary_markdown(headline_table, output_dir / "dissertation_summary.md")

    print(f"Wrote dissertation summary table to {output_dir / 'main_age_effects_summary.tsv'}")
    print(f"Wrote participant characteristics table to {output_dir / 'table1_participant_characteristics.tsv'}")
    print(f"Wrote Word-friendly table to {output_dir / 'table2_main_age_effects_word_friendly.tsv'}")
    print(f"Wrote dissertation summary figure to {figures_dir / 'main_age_effects_summary.png'}")


if __name__ == "__main__":
    main()
