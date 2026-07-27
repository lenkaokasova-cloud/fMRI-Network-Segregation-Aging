#!/usr/bin/env python3

# What this script does:
#   Runs the small exploratory age-group classification branch using the network
#   segregation features.
# How to run it:
#   Run from the repo root with:
#   python code/exploratory/25_run_classification_analysis.py
# Main output:
#   data/processed/analysis/classification/

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("data/processed/.matplotlib")).resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
GLOBAL_SEGREGATION_COL = "global_segregation_prop"
NETWORK_SEGREGATION_COL = "segregation_prop"
YOUNG_LABEL = "young"
OLDER_LABEL = "older"
YOUNG_COLOR = "#4C78A8"
OLDER_COLOR = "#D16A3A"
REFERENCE_LINE_COLOR = "#1F3A5F"
GRID_COLOR = "#B8BDC7"
POSITIVE_COLOR = "#4C78A8"
NEGATIVE_COLOR = "#D16A3A"
FIG_DPI = 300
TITLE_SIZE = 15
LABEL_SIZE = 12
TICK_SIZE = 11
LEGEND_SIZE = 11
MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "linear_svm": "Linear SVM",
}
FEATURE_SET_LABELS = {
    "networks_only": "Networks only",
    "networks_plus_global": "Networks + global",
}
FEATURE_LABELS = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal attention",
    "SalVentAttn": "Salience / ventral attention",
    "Limbic": "Limbic",
    "Cont": "Control",
    "Default": "Default",
    GLOBAL_SEGREGATION_COL: "Global segregation",
}

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
            "Run an exploratory age-group classification analysis using network segregation "
            "features from the final TR=3 s sample."
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
        help="Directory containing connectivity outputs from code/primary/11_run_connectivity_analysis.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/analysis/classification",
        help="Directory for exploratory classification outputs.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds for stratified cross-validation.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=100,
        help="Number of repeats for repeated stratified cross-validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for reproducible cross-validation splits.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def style_axis(
    ax: plt.Axes,
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
    for label in ax.get_xticklabels():
        label.set_rotation(xrotation)
        if xrotation:
            label.set_ha("right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ordered_networks(networks: list[str]) -> list[str]:
    order_index = {network: idx for idx, network in enumerate(NETWORK_ORDER)}
    unique_networks = list(dict.fromkeys(networks))
    return sorted(unique_networks, key=lambda value: (order_index.get(value, len(order_index)), value))


def load_tables(sample_path: Path, connectivity_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample = pd.read_csv(sample_path, sep="\t")
    global_df = pd.read_csv(connectivity_dir / "subject_global_segregation.tsv", sep="\t")
    network_df = pd.read_csv(connectivity_dir / "subject_network_segregation.tsv", sep="\t")

    required_sample = {"subject_id", "age", "sex", "age_group"}
    missing_sample = required_sample.difference(sample.columns)
    if missing_sample:
        raise ValueError(f"Sample TSV missing columns: {', '.join(sorted(missing_sample))}")

    sample_subjects = set(sample["subject_id"])
    global_df = global_df[global_df["subject_id"].isin(sample_subjects)].copy()
    network_df = network_df[network_df["subject_id"].isin(sample_subjects)].copy()
    return sample, global_df, network_df


def build_feature_matrix(
    sample: pd.DataFrame,
    global_df: pd.DataFrame,
    network_df: pd.DataFrame,
) -> pd.DataFrame:
    # This builds the classifier input table from the same segregation features used elsewhere in the repo.
    network_order = ordered_networks(network_df["network"].tolist())
    network_pivot = (
        network_df.pivot(index="subject_id", columns="network", values=NETWORK_SEGREGATION_COL)
        .reindex(columns=network_order)
        .reset_index()
    )
    meta_cols = ["subject_id", "age", "sex", "age_group"]
    subject_df = sample[meta_cols].drop_duplicates("subject_id").merge(network_pivot, on="subject_id", how="inner")
    global_keep = global_df[
        ["subject_id", GLOBAL_SEGREGATION_COL, "mean_fd", "retained_minutes_after_scrub", "pct_retained_after_scrub"]
    ].drop_duplicates("subject_id")
    subject_df = subject_df.merge(global_keep, on="subject_id", how="left")

    feature_columns = network_order + [GLOBAL_SEGREGATION_COL]
    missing_feature_rows = subject_df[feature_columns].isna().any(axis=1)
    if missing_feature_rows.any():
        missing_subjects = subject_df.loc[missing_feature_rows, "subject_id"].tolist()
        raise ValueError(
            "Feature matrix contains missing values for subjects: "
            + ", ".join(sorted(missing_subjects))
        )

    subject_df["target"] = (subject_df["age_group"] == OLDER_LABEL).astype(int)
    return subject_df


def build_models(random_state: int) -> dict[str, Pipeline]:
    # I keep the models simple and interpretable on purpose because the sample is small.
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="linear",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def feature_sets() -> dict[str, list[str]]:
    network_features = NETWORK_ORDER.copy()
    return {
        "networks_only": network_features,
        "networks_plus_global": network_features + [GLOBAL_SEGREGATION_COL],
    }


def decision_scores(model: Pipeline, x_test: np.ndarray) -> np.ndarray:
    final_estimator = model.named_steps["model"]
    if isinstance(final_estimator, LogisticRegression):
        return model.predict_proba(x_test)[:, 1]
    return model.decision_function(x_test)


def evaluate_model(
    model: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    # Everything here is cross-validated so I am not just reporting an in-sample fit.
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    rows: list[dict[str, object]] = []
    for split_idx, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        fitted = clone(model)
        fitted.fit(x[train_idx], y[train_idx])
        pred = fitted.predict(x[test_idx])
        score = decision_scores(fitted, x[test_idx])

        tn, fp, fn, tp = confusion_matrix(y[test_idx], pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "split": split_idx,
                "n_test": int(len(test_idx)),
                "accuracy": float(accuracy_score(y[test_idx], pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
                "sensitivity": float(recall_score(y[test_idx], pred, pos_label=1)),
                "specificity": float(recall_score(y[test_idx], pred, pos_label=0)),
                "roc_auc": float(roc_auc_score(y[test_idx], score)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows)


def evaluate_single_cv_predictions(
    model: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
    subject_ids: pd.Series,
    age_groups: pd.Series,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows: list[dict[str, object]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        fitted = clone(model)
        fitted.fit(x[train_idx], y[train_idx])
        pred = fitted.predict(x[test_idx])
        score = decision_scores(fitted, x[test_idx])
        for idx, pred_label, raw_score in zip(test_idx, pred, score):
            rows.append(
                {
                    "fold": fold_idx,
                    "subject_id": subject_ids.iloc[idx],
                    "age_group": age_groups.iloc[idx],
                    "target": int(y[idx]),
                    "predicted_target": int(pred_label),
                    "predicted_age_group": OLDER_LABEL if int(pred_label) == 1 else YOUNG_LABEL,
                    "older_score": float(raw_score),
                }
            )
    return pd.DataFrame(rows).sort_values(["fold", "subject_id"]).reset_index(drop=True)


def summarize_metrics(metrics_df: pd.DataFrame, model_name: str, feature_set: str) -> dict[str, object]:
    summary = {
        "model": model_name,
        "feature_set": feature_set,
        "n_splits_total": int(len(metrics_df)),
    }
    for metric in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "roc_auc"]:
        summary[f"{metric}_mean"] = float(metrics_df[metric].mean())
        summary[f"{metric}_sd"] = float(metrics_df[metric].std(ddof=1))
    return summary


def extract_coefficients(
    model: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
    feature_columns: list[str],
    model_name: str,
    feature_set: str,
) -> pd.DataFrame:
    fitted = clone(model)
    fitted.fit(x, y)
    coef = fitted.named_steps["model"].coef_[0]
    return pd.DataFrame(
        {
            "model": model_name,
            "feature_set": feature_set,
            "feature": feature_columns,
            "coefficient": coef.astype(float),
            "abs_coefficient": np.abs(coef).astype(float),
        }
    ).sort_values("abs_coefficient", ascending=False)


def plot_model_comparison(summary_df: pd.DataFrame, outpath: Path) -> None:
    model_order = ["logistic_regression", "linear_svm"]
    feature_order = ["networks_only", "networks_plus_global"]
    offsets = {"networks_only": -0.12, "networks_plus_global": 0.12}
    markers = {"networks_only": "o", "networks_plus_global": "s"}
    colors = {"networks_only": YOUNG_COLOR, "networks_plus_global": OLDER_COLOR}

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    legend_handles: list[Line2D] = []
    for feature_set in feature_order:
        for x_pos, model_name in enumerate(model_order):
            row = summary_df[
                (summary_df["model"] == model_name) & (summary_df["feature_set"] == feature_set)
            ].iloc[0]
            ax.errorbar(
                x_pos + offsets[feature_set],
                row["balanced_accuracy_mean"],
                yerr=row["balanced_accuracy_sd"],
                fmt=markers[feature_set],
                markersize=8.5,
                color=colors[feature_set],
                ecolor=colors[feature_set],
                elinewidth=2.0,
                capsize=4,
                markeredgecolor="white",
                markeredgewidth=0.7,
                zorder=3,
            )
            ax.text(
                x_pos + offsets[feature_set],
                row["balanced_accuracy_mean"] + row["balanced_accuracy_sd"] + 0.012,
                f"{row['balanced_accuracy_mean']:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color=colors[feature_set],
            )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=markers[feature_set],
                color=colors[feature_set],
                linestyle="None",
                markersize=8.5,
                markeredgecolor="white",
                markeredgewidth=0.7,
                label=FEATURE_SET_LABELS[feature_set],
            )
        )

    ax.axhline(0.5, color=REFERENCE_LINE_COLOR, linewidth=1.8, linestyle="--", label="Chance")
    ax.set_xticks(np.arange(len(model_order)))
    ax.set_xticklabels([MODEL_LABELS[name] for name in model_order])
    ax.set_ylim(0.35, 0.75)
    style_axis(
        ax,
        title="Exploratory Classification Performance",
        ylabel="Balanced accuracy",
    )
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    ax.legend(handles=legend_handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    save_figure(fig, outpath)


def plot_logistic_coefficients(coef_df: pd.DataFrame, outpath: Path) -> None:
    ordered = coef_df.sort_values("coefficient").copy()
    ordered["feature_label"] = ordered["feature"].map(FEATURE_LABELS).fillna(ordered["feature"])
    colors = [NEGATIVE_COLOR if value < 0 else POSITIVE_COLOR for value in ordered["coefficient"]]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bars = ax.barh(ordered["feature_label"], ordered["coefficient"], color=colors, alpha=0.9)
    ax.axvline(0.0, color="black", linewidth=1)
    max_abs = float(np.abs(ordered["coefficient"]).max()) if not ordered.empty else 1.0
    ax.set_xlim(-max_abs * 1.28, max_abs * 1.28)
    for bar, value in zip(bars, ordered["coefficient"]):
        x_pos = value + (0.03 * max_abs if value >= 0 else -0.03 * max_abs)
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
            color=POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR,
        )
    style_axis(
        ax,
        title="Logistic Regression Feature Weights",
        xlabel="Standardized coefficient (positive = older)",
        ylabel="Feature",
    )
    ax.grid(axis="x", color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    save_figure(fig, outpath)


def plot_probability_by_group(pred_df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.9, 4.8))
    order = [YOUNG_LABEL, OLDER_LABEL]
    colors = [YOUNG_COLOR, OLDER_COLOR]
    data = [pred_df.loc[pred_df["age_group"] == group, "older_probability"].to_numpy() for group in order]
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
        patch.set_alpha(0.26)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.8)

    for x_pos, group, color in zip([0, 1], order, colors):
        subset = pred_df[pred_df["age_group"] == group].reset_index(drop=True)
        rng = np.random.default_rng(404 + x_pos)
        offsets = rng.uniform(-0.09, 0.09, len(subset))
        ax.scatter(
            np.full(len(subset), x_pos) + offsets,
            subset["older_probability"],
            color=color,
            s=62,
            alpha=0.92,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
    ax.axhline(0.5, color=REFERENCE_LINE_COLOR, linewidth=1.8, linestyle="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([YOUNG_LABEL, OLDER_LABEL])
    ax.set_ylim(0.0, 1.0)
    style_axis(
        ax,
        title="Held-Out Predicted Older-Group Probability",
        ylabel="Predicted probability of older group",
    )
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.28, linewidth=0.9)
    save_figure(fig, outpath)


def write_summary_markdown(
    outpath: Path,
    sample_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    probability_df: pd.DataFrame,
) -> None:
    sample_counts = sample_df["age_group"].value_counts().to_dict()
    best_row = summary_df.sort_values("balanced_accuracy_mean", ascending=False).iloc[0]
    prob_summary = (
        probability_df.groupby("age_group")["older_probability"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("age_group")
    )

    lines = [
        "# Exploratory Classification Summary",
        "",
        f"- Sample: {len(sample_df)} subjects ({sample_counts.get(YOUNG_LABEL, 0)} young, {sample_counts.get(OLDER_LABEL, 0)} older)",
        "- Models: logistic regression and linear SVM",
        "- Feature sets: 7 network segregation values, with and without global segregation",
        "- Evaluation: repeated stratified 5-fold cross-validation",
        "",
        "## Best Model",
        "",
        f"- Model: `{best_row['model']}`",
        f"- Feature set: `{best_row['feature_set']}`",
        f"- Mean balanced accuracy: {best_row['balanced_accuracy_mean']:.4f} (SD {best_row['balanced_accuracy_sd']:.4f})",
        f"- Mean sensitivity: {best_row['sensitivity_mean']:.4f}",
        f"- Mean specificity: {best_row['specificity_mean']:.4f}",
        f"- Mean ROC AUC: {best_row['roc_auc_mean']:.4f}",
        "",
        "## Held-Out Older Probability",
        "",
    ]
    for _, row in prob_summary.iterrows():
        lines.append(
            f"- {row['age_group']}: mean predicted older probability={row['mean']:.4f}, SD={row['std']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is an exploratory classification analysis rather than a main confirmatory analysis.",
            "- Because the sample is small and imbalanced, these results should be interpreted cautiously.",
            "- The most useful value of this analysis is to show whether the segregation features carry age-group signal at all.",
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

    sample, global_df, network_df = load_tables(sample_path, connectivity_dir)
    feature_df = build_feature_matrix(sample, global_df, network_df)
    models = build_models(args.random_state)
    feature_sets_map = feature_sets()

    metrics_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    coefficient_tables: list[pd.DataFrame] = []
    probability_tables: list[pd.DataFrame] = []

    y = feature_df["target"].to_numpy(dtype=int)
    for feature_set_name, feature_columns in feature_sets_map.items():
        x = feature_df[feature_columns].to_numpy(dtype=float)
        for model_name, model in models.items():
            metrics_df = evaluate_model(
                model=model,
                x=x,
                y=y,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                random_state=args.random_state,
            )
            metrics_df["model"] = model_name
            metrics_df["feature_set"] = feature_set_name
            metrics_tables.append(metrics_df)
            summary_rows.append(summarize_metrics(metrics_df, model_name, feature_set_name))

            coef_df = extract_coefficients(
                model=model,
                x=x,
                y=y,
                feature_columns=feature_columns,
                model_name=model_name,
                feature_set=feature_set_name,
            )
            coefficient_tables.append(coef_df)

            if model_name == "logistic_regression" and feature_set_name == "networks_plus_global":
                pred_df = evaluate_single_cv_predictions(
                    model=model,
                    x=x,
                    y=y,
                    subject_ids=feature_df["subject_id"],
                    age_groups=feature_df["age_group"],
                    n_splits=args.n_splits,
                    random_state=args.random_state,
                )
                pred_df["model"] = model_name
                pred_df["feature_set"] = feature_set_name
                # Convert logistic decision values into probabilities by refitting a probability model fold-wise.
                probability_cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
                prob_rows: list[dict[str, object]] = []
                for fold_idx, (train_idx, test_idx) in enumerate(probability_cv.split(x, y), start=1):
                    fitted = clone(model)
                    fitted.fit(x[train_idx], y[train_idx])
                    probabilities = fitted.predict_proba(x[test_idx])[:, 1]
                    predictions = fitted.predict(x[test_idx])
                    for idx, pred_label, prob in zip(test_idx, predictions, probabilities):
                        prob_rows.append(
                            {
                                "fold": fold_idx,
                                "subject_id": feature_df["subject_id"].iloc[idx],
                                "age_group": feature_df["age_group"].iloc[idx],
                                "target": int(y[idx]),
                                "predicted_target": int(pred_label),
                                "predicted_age_group": OLDER_LABEL if int(pred_label) == 1 else YOUNG_LABEL,
                                "older_probability": float(prob),
                                "model": model_name,
                                "feature_set": feature_set_name,
                            }
                        )
                probability_tables.append(pd.DataFrame(prob_rows).sort_values(["fold", "subject_id"]).reset_index(drop=True))

    metrics_all = pd.concat(metrics_tables, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["balanced_accuracy_mean", "roc_auc_mean"], ascending=[False, False]
    ).reset_index(drop=True)
    coefficients_all = pd.concat(coefficient_tables, ignore_index=True)
    probability_df = pd.concat(probability_tables, ignore_index=True)

    feature_df.to_csv(output_dir / "subject_classification_features.tsv", sep="\t", index=False)
    metrics_all.to_csv(output_dir / "crossval_fold_metrics.tsv", sep="\t", index=False)
    summary_df.to_csv(output_dir / "model_summary.tsv", sep="\t", index=False)
    coefficients_all.to_csv(output_dir / "model_coefficients.tsv", sep="\t", index=False)
    probability_df.to_csv(output_dir / "logistic_single_cv_probabilities.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "n_subjects": len(feature_df),
                "n_young": int((feature_df["age_group"] == YOUNG_LABEL).sum()),
                "n_older": int((feature_df["age_group"] == OLDER_LABEL).sum()),
                "n_splits": args.n_splits,
                "n_repeats": args.n_repeats,
                "random_state": args.random_state,
            }
        ]
    ).to_csv(output_dir / "classification_settings.tsv", sep="\t", index=False)

    plot_model_comparison(summary_df, figures_dir / "model_balanced_accuracy.png")
    logistic_coef = coefficients_all[
        (coefficients_all["model"] == "logistic_regression")
        & (coefficients_all["feature_set"] == "networks_plus_global")
    ].copy()
    plot_logistic_coefficients(logistic_coef, figures_dir / "logistic_regression_coefficients.png")
    plot_probability_by_group(probability_df, figures_dir / "heldout_older_probability_by_group.png")
    write_summary_markdown(
        output_dir / "classification_summary.md",
        feature_df,
        summary_df,
        probability_df,
    )

    print(f"Wrote classification tables to {output_dir}")
    print(f"Wrote classification figures to {figures_dir}")


if __name__ == "__main__":
    main()
