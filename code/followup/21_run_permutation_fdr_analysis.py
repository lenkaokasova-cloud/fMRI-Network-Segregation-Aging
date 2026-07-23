#!/usr/bin/env python3

# What this script does:
#   Adds stricter permutation-based inference and FDR correction on top of the
#   main age-group analysis results.
# How to run it:
#   Run from the repo root with:
#   python code/followup/21_run_permutation_fdr_analysis.py
# Main output:
#   data/processed/analysis/permutation_inference/

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
HIGHER_ORDER = {"Default", "Cont", "DorsAttn", "SalVentAttn"}
SENSORY_MOTOR = {"Vis", "SomMot"}
GLOBAL_SEGREGATION_COL = "global_segregation_prop"
SEGREGATION_COL = "segregation_prop"
COMPONENT_ORDER = ["within_mean_z", "between_mean_z", SEGREGATION_COL]
COMPONENT_LABELS = {
    "within_mean_z": "Within",
    "between_mean_z": "Between",
    SEGREGATION_COL: "Segregation",
}
TITLE_SIZE = 15
LABEL_SIZE = 12
TICK_SIZE = 11
FIG_DPI = 300

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
            "Run permutation inference with FDR correction for the final "
            "age-group segregation analyses."
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
        help="Directory containing subject_global_segregation.tsv and subject_network_segregation.tsv.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/analysis/permutation_inference",
        help="Directory for permutation inference outputs.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of Freedman-Lane permutations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=752,
        help="Random seed for reproducible permutations.",
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


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    # I use BH-FDR here because we are testing several related effects and I want a cleaner multiple-testing correction.
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan, dtype=float)
    finite_mask = np.isfinite(p_values)
    if not finite_mask.any():
        return q_values

    finite_p = p_values[finite_mask]
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    n_tests = len(ranked)
    adjusted = ranked * n_tests / np.arange(1, n_tests + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q_values[finite_mask] = restored
    return q_values


def ordered_networks(networks: list[str]) -> list[str]:
    order_index = {network: idx for idx, network in enumerate(NETWORK_ORDER)}
    unique_networks = list(dict.fromkeys(networks))
    return sorted(unique_networks, key=lambda name: (order_index.get(name, len(order_index)), name))


def load_tables(sample_path: Path, connectivity_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample = pd.read_csv(sample_path, sep="\t")
    global_df = pd.read_csv(connectivity_dir / "subject_global_segregation.tsv", sep="\t")
    network_df = pd.read_csv(connectivity_dir / "subject_network_segregation.tsv", sep="\t")

    subject_ids = set(sample["subject_id"])
    global_df = global_df[global_df["subject_id"].isin(subject_ids)].copy()
    network_df = network_df[network_df["subject_id"].isin(subject_ids)].copy()
    return sample, global_df, network_df


def encode_design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # The reduced design keeps the nuisance terms, and the full design adds the age-group effect I want to test.
    sex_male = df["sex"].astype(str).str.lower().eq("male").astype(float).to_numpy()
    age_older = df["age_group"].astype(str).eq("older").astype(float).to_numpy()
    mean_fd = df["mean_fd"].astype(float).to_numpy()

    x_full = np.column_stack([np.ones(len(df)), age_older, sex_male, mean_fd])
    x_reduced = np.column_stack([np.ones(len(df)), sex_male, mean_fd])
    return x_full, x_reduced


def ols_term_stats(y: np.ndarray, x: np.ndarray, term_index: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residuals = y - x @ beta
    dof = x.shape[0] - x.shape[1]
    sigma2 = (residuals**2).sum(axis=0) / dof
    se = np.sqrt(sigma2 * xtx_inv[term_index, term_index])
    t_values = beta[term_index, :] / se
    return beta[term_index, :], se, t_values


def freedman_lane_pvalues(
    y: np.ndarray,
    x_full: np.ndarray,
    x_reduced: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
    term_index: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Freedman-Lane lets me test the age term while still keeping the nuisance covariates in place.
    rng = np.random.default_rng(seed)

    if y.ndim == 1:
        y = y[:, None]

    observed_beta, observed_se, observed_t = ols_term_stats(y, x_full, term_index=term_index)

    xr_inv = np.linalg.inv(x_reduced.T @ x_reduced)
    beta_reduced = xr_inv @ x_reduced.T @ y
    fitted_reduced = x_reduced @ beta_reduced
    residuals_reduced = y - fitted_reduced

    exceed_counts = np.zeros(y.shape[1], dtype=int)
    abs_observed = np.abs(observed_t)

    for _ in range(n_permutations):
        perm_idx = rng.permutation(y.shape[0])
        y_perm = fitted_reduced + residuals_reduced[perm_idx, :]
        _, _, perm_t = ols_term_stats(y_perm, x_full, term_index=term_index)
        exceed_counts += np.abs(perm_t) >= abs_observed

    permutation_p = (exceed_counts + 1) / (n_permutations + 1)
    return observed_beta, observed_se, permutation_p


def build_subject_component_table(network_df: pd.DataFrame) -> pd.DataFrame:
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


def build_interaction_difference_table(network_df: pd.DataFrame) -> pd.DataFrame:
    df = network_df.copy()
    df["network_type"] = "other"
    df.loc[df["network"].isin(SENSORY_MOTOR), "network_type"] = "sensory_motor"
    df.loc[df["network"].isin(HIGHER_ORDER), "network_type"] = "higher_order"
    df = df[df["network_type"].isin(["sensory_motor", "higher_order"])].copy()

    averaged = (
        df.groupby(
            ["subject_id", "age", "sex", "age_group", "mean_fd", "retained_minutes_after_scrub", "network_type"],
            as_index=False,
        )
        .agg(
            within_mean_z=("within_mean_z", "mean"),
            between_mean_z=("between_mean_z", "mean"),
            segregation_prop=(SEGREGATION_COL, "mean"),
        )
    )

    wide = averaged.pivot(
        index=["subject_id", "age", "sex", "age_group", "mean_fd", "retained_minutes_after_scrub"],
        columns="network_type",
        values=["within_mean_z", "between_mean_z", SEGREGATION_COL],
    )
    wide.columns = [f"{component}_{network_type}" for component, network_type in wide.columns]
    wide = wide.reset_index()
    wide["within_diff_higher_minus_sensory"] = wide["within_mean_z_higher_order"] - wide["within_mean_z_sensory_motor"]
    wide["between_diff_higher_minus_sensory"] = (
        wide["between_mean_z_higher_order"] - wide["between_mean_z_sensory_motor"]
    )
    wide["segregation_diff_higher_minus_sensory"] = (
        wide[f"{SEGREGATION_COL}_higher_order"] - wide[f"{SEGREGATION_COL}_sensory_motor"]
    )
    return wide.sort_values(["age_group", "age", "subject_id"]).reset_index(drop=True)


def run_overall_permutation_analysis(
    global_df: pd.DataFrame,
    subject_component_df: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    merged = global_df[
        ["subject_id", "age", "sex", "age_group", "mean_fd", GLOBAL_SEGREGATION_COL]
    ].merge(
        subject_component_df[
            ["subject_id", "overall_within_mean", "overall_between_mean", "overall_segregation"]
        ],
        on="subject_id",
        how="inner",
    )
    merged = merged.sort_values(["age_group", "age", "subject_id"]).reset_index(drop=True)
    x_full, x_reduced = encode_design(merged)
    outcomes = [
        (GLOBAL_SEGREGATION_COL, "Global segregation"),
        ("overall_within_mean", "Overall within-network connectivity"),
        ("overall_between_mean", "Overall between-network connectivity"),
        ("overall_segregation", "Overall segregation"),
    ]
    y = merged[[column for column, _ in outcomes]].to_numpy()
    beta, se, perm_p = freedman_lane_pvalues(
        y,
        x_full,
        x_reduced,
        n_permutations=n_permutations,
        seed=seed,
    )
    q_values = fdr_bh(perm_p)

    rows: list[dict[str, object]] = []
    for idx, (column, label) in enumerate(outcomes):
        rows.append(
            {
                "outcome": column,
                "label": label,
                "estimate_older_vs_young": float(beta[idx]),
                "std_error": float(se[idx]),
                "conf_low": float(beta[idx] - 1.96 * se[idx]),
                "conf_high": float(beta[idx] + 1.96 * se[idx]),
                "permutation_p_value": float(perm_p[idx]),
                "fdr_q_value": float(q_values[idx]),
                "young_mean": float(merged.loc[merged["age_group"] == "young", column].mean()),
                "older_mean": float(merged.loc[merged["age_group"] == "older", column].mean()),
                "older_minus_young_mean": float(
                    merged.loc[merged["age_group"] == "older", column].mean()
                    - merged.loc[merged["age_group"] == "young", column].mean()
                ),
                "n_subjects": int(len(merged)),
            }
        )
    return pd.DataFrame(rows)


def run_network_permutation_analysis(
    network_df: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    networks = ordered_networks(network_df["network"].tolist())
    rows: list[dict[str, object]] = []

    for component_index, component in enumerate(COMPONENT_ORDER):
        wide = (
            network_df.pivot_table(
                index=["subject_id", "age", "sex", "age_group", "mean_fd"],
                columns="network",
                values=component,
            )
            .reset_index()
            .sort_values(["age_group", "age", "subject_id"])
            .reset_index(drop=True)
        )
        outcome_columns = [network for network in networks if network in wide.columns]
        x_full, x_reduced = encode_design(wide)
        y = wide[outcome_columns].to_numpy()

        beta, se, perm_p = freedman_lane_pvalues(
            y,
            x_full,
            x_reduced,
            n_permutations=n_permutations,
            seed=seed + component_index + 1,
        )
        q_values = fdr_bh(perm_p)

        for idx, network in enumerate(outcome_columns):
            rows.append(
                {
                    "component": component,
                    "network": network,
                    "estimate_older_vs_young": float(beta[idx]),
                    "std_error": float(se[idx]),
                    "conf_low": float(beta[idx] - 1.96 * se[idx]),
                    "conf_high": float(beta[idx] + 1.96 * se[idx]),
                    "permutation_p_value": float(perm_p[idx]),
                    "fdr_q_value": float(q_values[idx]),
                    "young_mean": float(wide.loc[wide["age_group"] == "young", network].mean()),
                    "older_mean": float(wide.loc[wide["age_group"] == "older", network].mean()),
                    "older_minus_young_mean": float(
                        wide.loc[wide["age_group"] == "older", network].mean()
                        - wide.loc[wide["age_group"] == "young", network].mean()
                    ),
                    "n_subjects": int(len(wide)),
                }
            )

    out = pd.DataFrame(rows)
    out["network"] = pd.Categorical(out["network"], categories=networks, ordered=True)
    out["component"] = pd.Categorical(out["component"], categories=COMPONENT_ORDER, ordered=True)
    return out.sort_values(["component", "network"]).reset_index(drop=True)


def run_interaction_difference_analysis(
    interaction_df: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    x_full, x_reduced = encode_design(interaction_df)
    outcomes = [
        ("within_diff_higher_minus_sensory", "Within higher-order minus sensory/motor"),
        ("between_diff_higher_minus_sensory", "Between higher-order minus sensory/motor"),
        ("segregation_diff_higher_minus_sensory", "Segregation higher-order minus sensory/motor"),
    ]
    y = interaction_df[[column for column, _ in outcomes]].to_numpy()
    beta, se, perm_p = freedman_lane_pvalues(
        y,
        x_full,
        x_reduced,
        n_permutations=n_permutations,
        seed=seed + 100,
    )
    q_values = fdr_bh(perm_p)

    rows: list[dict[str, object]] = []
    for idx, (column, label) in enumerate(outcomes):
        rows.append(
            {
                "outcome": column,
                "label": label,
                "estimate_older_vs_young": float(beta[idx]),
                "std_error": float(se[idx]),
                "conf_low": float(beta[idx] - 1.96 * se[idx]),
                "conf_high": float(beta[idx] + 1.96 * se[idx]),
                "permutation_p_value": float(perm_p[idx]),
                "fdr_q_value": float(q_values[idx]),
                "young_mean": float(interaction_df.loc[interaction_df["age_group"] == "young", column].mean()),
                "older_mean": float(interaction_df.loc[interaction_df["age_group"] == "older", column].mean()),
                "older_minus_young_mean": float(
                    interaction_df.loc[interaction_df["age_group"] == "older", column].mean()
                    - interaction_df.loc[interaction_df["age_group"] == "young", column].mean()
                ),
                "n_subjects": int(len(interaction_df)),
            }
        )
    return pd.DataFrame(rows)


def plot_network_qvalue_heatmap(network_results: pd.DataFrame, outpath: Path) -> None:
    plot_df = network_results.copy()
    pivot_q = plot_df.pivot(index="component", columns="network", values="fdr_q_value")
    pivot_est = plot_df.pivot(index="component", columns="network", values="estimate_older_vs_young")

    pivot_q = pivot_q.loc[COMPONENT_ORDER, ordered_networks(plot_df["network"].astype(str).tolist())]
    pivot_est = pivot_est.loc[COMPONENT_ORDER, pivot_q.columns]

    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    signed_log_q = np.sign(pivot_est.to_numpy()) * -np.log10(np.clip(pivot_q.to_numpy(), 1e-12, 1.0))
    vmax = max(1.3, float(np.nanmax(np.abs(signed_log_q))))
    im = ax.imshow(signed_log_q, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(pivot_q.columns)))
    ax.set_xticklabels(pivot_q.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot_q.index)))
    ax.set_yticklabels([COMPONENT_LABELS[name] for name in pivot_q.index])
    ax.set_title("Permutation-FDR network summary")
    ax.set_xlabel("Network")
    ax.set_ylabel("Component")

    for row_idx, component in enumerate(pivot_q.index):
        for col_idx, network in enumerate(pivot_q.columns):
            q_value = pivot_q.loc[component, network]
            ax.text(
                col_idx,
                row_idx,
                f"{q_value:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Signed -log10(q)")
    save_figure(fig, outpath)


def write_summary(
    overall_results: pd.DataFrame,
    network_results: pd.DataFrame,
    interaction_results: pd.DataFrame,
    outpath: Path,
) -> None:
    lines = [
        "# Permutation Inference and FDR Summary",
        "",
        "This supplementary analysis re-tests the main age-group effects using Freedman-Lane permutation inference with false-discovery-rate correction for the multi-network families.",
        "",
        "## Overall Effects",
        "",
    ]

    for _, row in overall_results.iterrows():
        lines.append(
            f"- {row['label']}: estimate={row['estimate_older_vs_young']:.4f}, "
            f"95% CI [{row['conf_low']:.4f}, {row['conf_high']:.4f}], "
            f"permutation p={row['permutation_p_value']:.4g}, q={row['fdr_q_value']:.4g}"
        )

    lines.extend(["", "## Network-Specific Results", ""])
    for component in COMPONENT_ORDER:
        subset = network_results[network_results["component"] == component].copy()
        significant = subset[subset["fdr_q_value"] < 0.05].copy()
        if significant.empty:
            lines.append(f"- {COMPONENT_LABELS[component]}: no networks survived FDR correction.")
        else:
            joined = ", ".join(
                f"{row['network']} (q={row['fdr_q_value']:.4g}, estimate={row['estimate_older_vs_young']:.4f})"
                for _, row in significant.iterrows()
            )
            lines.append(f"- {COMPONENT_LABELS[component]}: {joined}")

    lines.extend(["", "## Higher-Order vs Sensory/Motor Difference Scores", ""])
    for _, row in interaction_results.iterrows():
        lines.append(
            f"- {row['label']}: estimate={row['estimate_older_vs_young']:.4f}, "
            f"permutation p={row['permutation_p_value']:.4g}, q={row['fdr_q_value']:.4g}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Permutation inference is more conservative than the original asymptotic p-values and is better suited to the current modest sample size.",
            "- FDR correction addresses the fact that several related network tests were run in parallel.",
            "- If effects do not survive this supplementary step, they should still be described as directional or exploratory rather than as strong confirmatory findings.",
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
    subject_component_df = build_subject_component_table(network_df)
    interaction_df = build_interaction_difference_table(network_df)

    overall_results = run_overall_permutation_analysis(
        global_df,
        subject_component_df,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    network_results = run_network_permutation_analysis(
        network_df,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    interaction_results = run_interaction_difference_analysis(
        interaction_df,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )

    overall_results.to_csv(output_dir / "overall_permutation_summary.tsv", sep="\t", index=False)
    network_results.to_csv(output_dir / "network_permutation_summary.tsv", sep="\t", index=False)
    interaction_results.to_csv(output_dir / "interaction_difference_permutation_summary.tsv", sep="\t", index=False)

    settings = pd.DataFrame(
        [
            {"setting": "sample_tsv", "value": str(sample_path)},
            {"setting": "connectivity_dir", "value": str(connectivity_dir)},
            {"setting": "n_permutations", "value": args.n_permutations},
            {"setting": "seed", "value": args.seed},
            {"setting": "permutation_method", "value": "Freedman-Lane residual permutation"},
            {"setting": "network_fdr_family", "value": "7 networks within each component family"},
            {"setting": "interaction_fdr_family", "value": "3 higher-order minus sensory/motor component difference tests"},
        ]
    )
    settings.to_csv(output_dir / "permutation_settings.tsv", sep="\t", index=False)

    plot_network_qvalue_heatmap(network_results, figures_dir / "network_permutation_qvalues.png")
    write_summary(
        overall_results,
        network_results,
        interaction_results,
        output_dir / "permutation_fdr_summary.md",
    )

    print(f"Wrote permutation inference tables to {output_dir}")
    print(f"Wrote permutation inference figures to {figures_dir}")


if __name__ == "__main__":
    main()
