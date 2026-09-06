#!/usr/bin/env python3

# What this script does:
#   Builds the balanced 11 x 11 TR = 3 s sensitivity sample used to compare the
#   main unbalanced branch against a matched alternative.
# How to run it:
#   Run from the repo root with:
#   python code/sensitivity/21_make_balanced_tr3_sensitivity_sample.py
# Main outputs:
#   Balanced sample TSV, pairing TSV, and summary markdown in
#   data/processed/screening/

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATCH_COLUMNS = ["mean_fd", "retained_minutes_after_scrub"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a balanced TR = 3 s sensitivity-analysis sample by matching "
            "younger participants to the older group with exact sex matching and "
            "optimal pairing on motion and retained minutes."
        )
    )
    parser.add_argument(
        "--sample",
        default="data/processed/screening/ds005752_final_analysis_sample_tr_3s.tsv",
        help="Primary final analysis sample TSV.",
    )
    parser.add_argument(
        "--tr-value",
        type=float,
        default=3.0,
        help="TR value to retain before balancing.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/screening/ds005752_balanced_sensitivity_sample_tr_3s_11x11.tsv",
        help="Balanced 11x11 sample TSV to write.",
    )
    parser.add_argument(
        "--pairs-output",
        default="data/processed/screening/ds005752_balanced_sensitivity_sample_tr_3s_11x11_pairs.tsv",
        help="TSV summarizing the younger/older matched pairs.",
    )
    parser.add_argument(
        "--summary-output",
        default="data/processed/screening/ds005752_balanced_sensitivity_sample_tr_3s_11x11_summary.md",
        help="Markdown summary of the matching and resulting sample.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def zscore_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        mean = out[column].mean()
        std = out[column].std(ddof=0)
        if not np.isfinite(std) or np.isclose(std, 0.0):
            std = 1.0
        out[column] = (out[column] - mean) / std
    return out


def pair_within_sex(
    older_df: pd.DataFrame, young_df: pd.DataFrame, pair_start: int
) -> tuple[list[dict[str, object]], list[str], int]:
    if len(young_df) < len(older_df):
        raise ValueError(
            f"Not enough younger participants to match sex='{older_df['sex'].iloc[0]}'."
        )

    combined = pd.concat(
        [older_df[MATCH_COLUMNS], young_df[MATCH_COLUMNS]],
        axis=0,
        ignore_index=True,
    )
    combined_z = zscore_frame(combined, MATCH_COLUMNS)
    older_z = combined_z.iloc[: len(older_df)].to_numpy(dtype=float)
    young_z = combined_z.iloc[len(older_df) :].to_numpy(dtype=float)

    diffs = older_z[:, None, :] - young_z[None, :, :]
    cost_matrix = np.sqrt((diffs**2).sum(axis=2))
    older_idx, young_idx = linear_sum_assignment(cost_matrix)

    pair_rows: list[dict[str, object]] = []
    selected_young: list[str] = []
    pair_id = pair_start
    for oi, yi in zip(older_idx.tolist(), young_idx.tolist(), strict=True):
        older_row = older_df.iloc[oi]
        young_row = young_df.iloc[yi]
        pair_rows.append(
            {
                "balanced_pair_id": pair_id,
                "sex": older_row["sex"],
                "older_id": older_row["subject_id"],
                "young_id": young_row["subject_id"],
                "older_age": int(older_row["age"]),
                "young_age": int(young_row["age"]),
                "older_mean_fd": float(older_row["mean_fd"]),
                "young_mean_fd": float(young_row["mean_fd"]),
                "abs_diff_mean_fd": abs(
                    float(older_row["mean_fd"]) - float(young_row["mean_fd"])
                ),
                "older_retained_minutes_after_scrub": float(
                    older_row["retained_minutes_after_scrub"]
                ),
                "young_retained_minutes_after_scrub": float(
                    young_row["retained_minutes_after_scrub"]
                ),
                "abs_diff_retained_minutes_after_scrub": abs(
                    float(older_row["retained_minutes_after_scrub"])
                    - float(young_row["retained_minutes_after_scrub"])
                ),
                "older_pct_fd_gt_0p2": float(older_row["pct_fd_gt_0p2"]),
                "young_pct_fd_gt_0p2": float(young_row["pct_fd_gt_0p2"]),
                "abs_diff_pct_fd_gt_0p2": abs(
                    float(older_row["pct_fd_gt_0p2"])
                    - float(young_row["pct_fd_gt_0p2"])
                ),
                "match_distance": float(cost_matrix[oi, yi]),
            }
        )
        selected_young.append(str(young_row["subject_id"]))
        pair_id += 1

    return pair_rows, selected_young, pair_id


def write_markdown_summary(
    path: Path,
    source_df: pd.DataFrame,
    balanced_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    tr_value: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    source_counts = source_df["age_group"].value_counts()
    balanced_counts = balanced_df["age_group"].value_counts()
    balanced_stats = (
        balanced_df.groupby("age_group")[
            ["mean_fd", "retained_minutes_after_scrub", "pct_fd_gt_0p2"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    balanced_stats_text = balanced_stats.to_string()

    younger_ids = balanced_df.loc[
        balanced_df["age_group"].eq("young"), "subject_id"
    ].tolist()
    older_ids = balanced_df.loc[
        balanced_df["age_group"].eq("older"), "subject_id"
    ].tolist()

    lines = [
        "# Balanced TR = 3 s Sensitivity Sample",
        "",
        f"- source sample: `{len(source_df)}` participants",
        f"- source counts: `{int(source_counts.get('young', 0))}` young / `{int(source_counts.get('older', 0))}` older",
        f"- TR restriction before matching: `{tr_value:.1f} s`",
        f"- balanced sample: `{len(balanced_df)}` participants",
        f"- balanced counts: `{int(balanced_counts.get('young', 0))}` young / `{int(balanced_counts.get('older', 0))}` older",
        "- matching strategy: exact sex matching + optimal one-to-one pairing on `mean_fd` and `retained_minutes_after_scrub`",
        "",
        "## Selected Younger Participants",
        "",
        ", ".join(younger_ids),
        "",
        "## Older Participants",
        "",
        ", ".join(older_ids),
        "",
        "## Pairing Summary",
        "",
        f"- number of matched pairs: `{len(pairs_df)}`",
        f"- mean absolute FD difference: `{pairs_df['abs_diff_mean_fd'].mean():.4f}` mm",
        f"- mean absolute retained-minutes difference: `{pairs_df['abs_diff_retained_minutes_after_scrub'].mean():.4f}` min",
        f"- mean absolute percent-FD>0.2 difference: `{pairs_df['abs_diff_pct_fd_gt_0p2'].mean():.4f}`",
        "",
        "## Balanced Sample Descriptives",
        "",
        "```text",
        balanced_stats_text,
        "```",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    sample_path = resolve_project_path(args.sample)
    output_path = resolve_project_path(args.output)
    pairs_output_path = resolve_project_path(args.pairs_output)
    summary_output_path = resolve_project_path(args.summary_output)

    df = pd.read_csv(sample_path, sep="\t")
    required = {
        "subject_id",
        "age",
        "sex",
        "age_group",
        "repetition_time_seconds",
        "mean_fd",
        "retained_minutes_after_scrub",
        "pct_fd_gt_0p2",
    }
    missing = required.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Sample file is missing required columns: {missing_str}")

    df = df.copy()
    df["repetition_time_seconds"] = pd.to_numeric(
        df["repetition_time_seconds"], errors="coerce"
    )
    source_df = df[df["repetition_time_seconds"].eq(args.tr_value)].copy()
    if source_df.empty:
        raise ValueError(
            f"No participants with repetition_time_seconds == {args.tr_value:.1f} were found."
        )

    older_df = source_df[source_df["age_group"].eq("older")].sort_values(
        ["sex", "age", "subject_id"]
    )
    young_df = source_df[source_df["age_group"].eq("young")].sort_values(
        ["sex", "age", "subject_id"]
    )
    if older_df.empty or young_df.empty:
        raise ValueError("Both younger and older participants are required.")

    pair_rows: list[dict[str, object]] = []
    selected_young_ids: list[str] = []
    next_pair_id = 1

    for sex in sorted(older_df["sex"].dropna().unique().tolist()):
        older_sex = older_df[older_df["sex"].eq(sex)].reset_index(drop=True)
        young_sex = young_df[young_df["sex"].eq(sex)].reset_index(drop=True)
        rows, ids, next_pair_id = pair_within_sex(
            older_sex,
            young_sex,
            pair_start=next_pair_id,
        )
        pair_rows.extend(rows)
        selected_young_ids.extend(ids)

    pairs_df = (
        pd.DataFrame(pair_rows).sort_values(["balanced_pair_id"]).reset_index(drop=True)
    )
    pair_map = {}
    for row in pairs_df.itertuples(index=False):
        pair_map[row.older_id] = {
            "balanced_pair_id": int(row.balanced_pair_id),
            "balanced_match_role": "older_reference",
            "matching_distance": float(row.match_distance),
        }
        pair_map[row.young_id] = {
            "balanced_pair_id": int(row.balanced_pair_id),
            "balanced_match_role": "younger_match",
            "matching_distance": float(row.match_distance),
        }

    balanced_ids = set(selected_young_ids).union(set(older_df["subject_id"].tolist()))
    balanced_df = source_df[source_df["subject_id"].isin(balanced_ids)].copy()
    balanced_df["balanced_pair_id"] = balanced_df["subject_id"].map(
        lambda sid: pair_map[sid]["balanced_pair_id"]
    )
    balanced_df["balanced_match_role"] = balanced_df["subject_id"].map(
        lambda sid: pair_map[sid]["balanced_match_role"]
    )
    balanced_df["matching_distance"] = balanced_df["subject_id"].map(
        lambda sid: pair_map[sid]["matching_distance"]
    )
    balanced_df["matching_strategy"] = (
        "exact sex + optimal pairing on mean_fd and retained_minutes_after_scrub"
    )
    balanced_df = balanced_df.sort_values(
        ["balanced_pair_id", "age_group", "subject_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(output_path, sep="\t", index=False)
    pairs_df.to_csv(pairs_output_path, sep="\t", index=False)
    write_markdown_summary(
        summary_output_path,
        source_df=source_df,
        balanced_df=balanced_df,
        pairs_df=pairs_df,
        tr_value=args.tr_value,
    )

    print(f"Wrote balanced sample to {output_path}")
    print(f"Wrote pairing table to {pairs_output_path}")
    print(f"Wrote summary to {summary_output_path}")
    print(
        "Balanced counts:",
        balanced_df["age_group"].value_counts().sort_index().to_dict(),
    )


if __name__ == "__main__":
    main()
