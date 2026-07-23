#!/usr/bin/env python3

# What this script does:
#   Pulls together the main branch and the sensitivity branches so I can compare
#   whether the direction or strength of the results changes much.
# How to run it:
#   Run from the repo root with:
#   python code/sensitivity/24_compare_sensitivity_branches.py
# Main output:
#   data/processed/analysis/literature_sensitivity/

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GLOBAL_TERM = "C(age_group, Treatment(reference='young'))[T.older]"
INTERACTION_TERM = (
    "C(age_group, Treatment(reference='young'))[T.older]:"
    "C(network_type, Treatment(reference='sensory_motor'))[T.higher_order]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the main age-group analysis with literature-style "
            "sensitivity branches such as GSR and atlas-resolution checks."
        )
    )
    parser.add_argument(
        "--main-analysis-dir",
        default="data/processed/analysis",
        help="Main analysis directory containing model_coefficients.tsv.",
    )
    parser.add_argument(
        "--main-permutation-dir",
        default="data/processed/analysis/permutation_inference",
        help="Main permutation directory containing overall_permutation_summary.tsv.",
    )
    parser.add_argument(
        "--gsr-analysis-dir",
        default="data/processed/sensitivity/gsr/analysis",
        help="GSR sensitivity analysis directory.",
    )
    parser.add_argument(
        "--gsr-permutation-dir",
        default="data/processed/sensitivity/gsr/analysis/permutation_inference",
        help="GSR sensitivity permutation directory.",
    )
    parser.add_argument(
        "--atlas100-analysis-dir",
        default="data/processed/sensitivity/atlas100/analysis",
        help="Schaefer-100 sensitivity analysis directory.",
    )
    parser.add_argument(
        "--atlas100-permutation-dir",
        default="data/processed/sensitivity/atlas100/analysis/permutation_inference",
        help="Schaefer-100 sensitivity permutation directory.",
    )
    parser.add_argument(
        "--partial-analysis-dir",
        default="data/processed/sensitivity/partial_correlation/analysis",
        help="Partial-correlation sensitivity analysis directory.",
    )
    parser.add_argument(
        "--partial-permutation-dir",
        default="data/processed/sensitivity/partial_correlation/analysis/permutation_inference",
        help="Partial-correlation sensitivity permutation directory.",
    )
    parser.add_argument(
        "--adjacent-analysis-dir",
        default="data/processed/sensitivity/adjacent_scrub/analysis",
        help="Adjacent-frame censoring sensitivity analysis directory.",
    )
    parser.add_argument(
        "--adjacent-permutation-dir",
        default="data/processed/sensitivity/adjacent_scrub/analysis/permutation_inference",
        help="Adjacent-frame censoring sensitivity permutation directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/analysis/literature_sensitivity",
        help="Directory for the comparison summary.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def extract_model_rows(analysis_dir: Path) -> dict[str, object]:
    # I pull out just the headline effects from each branch so the comparison stays readable.
    coef_path = analysis_dir / "model_coefficients.tsv"
    sample_path = analysis_dir / "sample_summary.tsv"
    coef_df = pd.read_csv(coef_path, sep="\t")
    sample_df = pd.read_csv(sample_path, sep="\t")

    global_row = coef_df[
        (coef_df["model"] == "global_group_model")
        & (coef_df["term"] == GLOBAL_TERM)
    ].iloc[0]
    interaction_row = coef_df[
        (coef_df["model"] == "network_type_model")
        & (coef_df["term"] == INTERACTION_TERM)
    ].iloc[0]

    young_n = int(sample_df.loc[sample_df["age_group"] == "young", "n_subjects"].iloc[0])
    older_n = int(sample_df.loc[sample_df["age_group"] == "older", "n_subjects"].iloc[0])

    return {
        "n_young": young_n,
        "n_older": older_n,
        "global_estimate": float(global_row["estimate"]),
        "global_p": float(global_row["p_value"]),
        "global_conf_low": float(global_row["conf_low"]),
        "global_conf_high": float(global_row["conf_high"]),
        "interaction_estimate": float(interaction_row["estimate"]),
        "interaction_p": float(interaction_row["p_value"]),
        "interaction_conf_low": float(interaction_row["conf_low"]),
        "interaction_conf_high": float(interaction_row["conf_high"]),
    }


def maybe_extract_permutation_rows(permutation_dir: Path) -> dict[str, object]:
    path = permutation_dir / "overall_permutation_summary.tsv"
    if not path.exists():
        return {
            "perm_global_p": float("nan"),
            "perm_global_q": float("nan"),
            "perm_overall_seg_p": float("nan"),
            "perm_overall_seg_q": float("nan"),
        }
    df = pd.read_csv(path, sep="\t")
    global_row = df[df["outcome"] == "global_segregation_prop"]
    overall_seg_row = df[df["outcome"] == "overall_segregation"]
    return {
        "perm_global_p": float(global_row["permutation_p_value"].iloc[0]) if not global_row.empty else float("nan"),
        "perm_global_q": float(global_row["fdr_q_value"].iloc[0]) if not global_row.empty else float("nan"),
        "perm_overall_seg_p": float(overall_seg_row["permutation_p_value"].iloc[0]) if not overall_seg_row.empty else float("nan"),
        "perm_overall_seg_q": float(overall_seg_row["fdr_q_value"].iloc[0]) if not overall_seg_row.empty else float("nan"),
    }


def build_branch_row(label: str, analysis_dir: Path, permutation_dir: Path) -> dict[str, object]:
    # Each branch gets boiled down to one comparable summary row.
    row = {"branch": label}
    row.update(extract_model_rows(analysis_dir))
    row.update(maybe_extract_permutation_rows(permutation_dir))
    return row


def maybe_build_branch_row(label: str, analysis_dir: Path, permutation_dir: Path) -> dict[str, object] | None:
    if not (analysis_dir / "model_coefficients.tsv").exists():
        return None
    return build_branch_row(label, analysis_dir, permutation_dir)


def write_summary_markdown(df: pd.DataFrame, outpath: Path) -> None:
    lines = [
        "# Literature-Style Sensitivity Branch Comparison",
        "",
        "This summary compares the main age-split analysis with the added literature-style sensitivity branches.",
        "",
    ]
    for _, row in df.iterrows():
        lines.extend(
            [
                f"## {row['branch']}",
                "",
                f"- Sample: {int(row['n_young'])} young, {int(row['n_older'])} older",
                f"- Global proportional segregation older-vs-young estimate = {row['global_estimate']:.4f}",
                f"- 95% CI = [{row['global_conf_low']:.4f}, {row['global_conf_high']:.4f}]",
                f"- Asymptotic p = {row['global_p']:.4g}",
                f"- Higher-order vs sensory-motor interaction estimate = {row['interaction_estimate']:.4f}",
                f"- Interaction p = {row['interaction_p']:.4g}",
            ]
        )
        if pd.notna(row["perm_global_p"]):
            lines.extend(
                [
                    f"- Permutation p for global proportional segregation = {row['perm_global_p']:.4g}",
                    f"- Permutation q for global proportional segregation = {row['perm_global_q']:.4g}",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- If the global age effect stays negative across branches, that supports directional robustness.",
            "- If the effect changes sign or becomes much larger only in one branch, that suggests stronger dependence on analytic choice.",
            "- These sensitivity branches are best reported as robustness checks rather than as replacements for the main pipeline.",
        ]
    )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for row in [
        build_branch_row(
            "main",
            resolve_project_path(args.main_analysis_dir),
            resolve_project_path(args.main_permutation_dir),
        ),
        build_branch_row(
            "with_gsr",
            resolve_project_path(args.gsr_analysis_dir),
            resolve_project_path(args.gsr_permutation_dir),
        ),
        build_branch_row(
            "schaefer_100",
            resolve_project_path(args.atlas100_analysis_dir),
            resolve_project_path(args.atlas100_permutation_dir),
        ),
        maybe_build_branch_row(
            "partial_correlation",
            resolve_project_path(args.partial_analysis_dir),
            resolve_project_path(args.partial_permutation_dir),
        ),
        maybe_build_branch_row(
            "adjacent_scrub",
            resolve_project_path(args.adjacent_analysis_dir),
            resolve_project_path(args.adjacent_permutation_dir),
        ),
    ]:
        if row is not None:
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "sensitivity_branch_comparison.tsv", sep="\t", index=False)
    write_summary_markdown(df, output_dir / "sensitivity_branch_summary.md")
    print(f"Wrote sensitivity comparison to {output_dir}")


if __name__ == "__main__":
    main()
