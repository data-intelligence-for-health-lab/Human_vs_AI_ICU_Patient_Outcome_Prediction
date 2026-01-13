import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ranksums
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

final_class_df = pd.read_csv('prospective_combined_prediction_df.csv')

outcomes = {
    "delirium": {
        "human_pred": "DELIRIUM_human_Num_CLASS",
        "ai_pred": "DELIRIUM_STATUS_PRED_CLASS",
        "true_label": "DELIRIUM_STATUS",
        "title": "Human vs AI Prediction Performance on Delirium"
    },
    "aki": {
        "human_pred": "AKI_human_Num_CLASS",
        "ai_pred": "AKI_STATUS_PRED_CLASS",
        "true_label": "AKI_STATUS",
        "title": "Human vs AI Prediction Performance on AKI"
    },
    "post_disch_mortality": {
        "human_pred": "30_DAY_MORT_human_Num_CLASS",
        "ai_pred": "30_DAY_MORTALITY_PRED_CLASS",
        "true_label": "30_DAY_POST_DISCHARGE_MORTALITY",
        "title": "Human vs AI Prediction Performance on 30-Day Post Discharge Mortality"
    },
    "in_hospital_mortality": {
        "human_pred": "HOSP_MORT_human_Num_CLASS",
        "ai_pred": "HOSP_MORTALITY_PRED_CLASS",
        "true_label": "IN_HOSPITAL_MORTALITY",
        "title": "Human vs AI Prediction Performance on In-Hospital Mortality"
    }
}

METRICS_NAMES = [
    "Accuracy", "Balanced Accuracy", "Precision",
    "Specificity", "Sensitivity", "F1 Score", "NPV"
]

GROUP_ORDER = ["Subspecialized", "Trainee", "Nurse"]


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "NPV": npv,
    }


def bootstrap_metrics(y_true, y_pred, n_iterations=1000, random_state=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    rng = np.random.default_rng(random_state)
    out = np.empty((n_iterations, len(METRICS_NAMES)), dtype=float)

    for b in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        res = compute_metrics(y_true[idx], y_pred[idx])
        out[b, :] = [res[m] for m in METRICS_NAMES]  # enforce stable order

    return out


def analyze_outcome_by_group(df, outcome_key, outcome_info, output_dir, n_boot=1000, random_state=None):
    print(f"\n{'='*40}\nAnalyzing: {outcome_key} by group\n{'='*40}")

    # Styling
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk", font_scale=1.2)
    sns.set_palette("colorblind")

    plot_rows = []

    # Compute bootstrapped distributions per group
    for group in GROUP_ORDER:
        df_group = (
            df[df["Group"] == group]
            .dropna(subset=[outcome_info["human_pred"], outcome_info["ai_pred"], outcome_info["true_label"]])
            .copy()
        )
        if df_group.empty:
            continue

        y_true = df_group[outcome_info["true_label"]].values
        y_human = df_group[outcome_info["human_pred"]].values
        y_ai = df_group[outcome_info["ai_pred"]].values

        boot_human = bootstrap_metrics(y_true, y_human, n_iterations=n_boot, random_state=random_state)
        boot_ai = bootstrap_metrics(y_true, y_ai, n_iterations=n_boot, random_state=random_state)

        for i, metric in enumerate(METRICS_NAMES):
            plot_rows.extend(
                {"Outcome": outcome_key, "Group": group, "Source": "Human", "Metric": metric, "Value": v}
                for v in boot_human[:, i]
            )
            plot_rows.extend(
                {"Outcome": outcome_key, "Group": group, "Source": "AI", "Metric": metric, "Value": v}
                for v in boot_ai[:, i]
            )

    plot_df = pd.DataFrame(plot_rows)
    if plot_df.empty:
        print(f"No valid data for {outcome_key}. Skipping.")
        return pd.DataFrame()

    # --- Ranksum + medians + stars (using bootstrap distributions) ---
    ranksum_rows = []
    for metric in METRICS_NAMES:
        for group in GROUP_ORDER:
            human_vals = plot_df.query("Metric == @metric and Group == @group and Source == 'Human'")["Value"].values
            ai_vals = plot_df.query("Metric == @metric and Group == @group and Source == 'AI'")["Value"].values

            human_vals = human_vals[~np.isnan(human_vals)]
            ai_vals = ai_vals[~np.isnan(ai_vals)]

            if human_vals.size == 0 or ai_vals.size == 0:
                continue

            stat, pval = ranksums(human_vals, ai_vals)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"

            ranksum_rows.append({
                "Outcome": outcome_key,
                "Metric": metric,
                "Group": group,
                "Statistic": float(stat),
                "P_value": float(pval),
                "Significance": sig,
                "Human_median": float(np.median(human_vals)),
                "AI_median": float(np.median(ai_vals)),
            })

    ranksum_df = pd.DataFrame(ranksum_rows)

    # --- Summary: Median + 95% CI + p-value ---
    summary_rows = []
    for metric in METRICS_NAMES:
        for group in GROUP_ORDER:
            human_vals = plot_df.query("Metric == @metric and Group == @group and Source == 'Human'")["Value"].values
            ai_vals = plot_df.query("Metric == @metric and Group == @group and Source == 'AI'")["Value"].values

            human_vals = human_vals[~np.isnan(human_vals)]
            ai_vals = ai_vals[~np.isnan(ai_vals)]

            if human_vals.size == 0 or ai_vals.size == 0:
                continue

            stat, pval = ranksums(human_vals, ai_vals)

            summary_rows.append({
                "Outcome": outcome_key,
                "Metric": metric,
                "Group": group,
                "Human_Median": float(np.median(human_vals)),
                "Human_CI2.5": float(np.percentile(human_vals, 2.5)),
                "Human_CI97.5": float(np.percentile(human_vals, 97.5)),
                "AI_Median": float(np.median(ai_vals)),
                "AI_CI2.5": float(np.percentile(ai_vals, 2.5)),
                "AI_CI97.5": float(np.percentile(ai_vals, 97.5)),
                "P_value": float(pval),
            })

    summary_out_df = pd.DataFrame(summary_rows)

    # --- Save tables ---
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"{outcome_key}_summary_results.csv")
    ranksum_path = os.path.join(output_dir, f"{outcome_key}_ranksum_results.csv")

    summary_out_df.to_csv(summary_path, index=False)
    ranksum_df.to_csv(ranksum_path, index=False)

    print(f"Saved summary results to {summary_path}")
    print(f"Saved ranksum results to {ranksum_path}")

    # --- Plotting (boxplots of bootstrap distributions) ---
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    fig.suptitle(outcome_info.get("title", outcome_key), fontsize=22, fontweight="bold")

    for ax, metric in zip(axes, METRICS_NAMES):
        metric_data = plot_df[plot_df["Metric"] == metric]

        sns.boxplot(
            data=metric_data,
            x="Group", y="Value", hue="Source",
            order=GROUP_ORDER, ax=ax,
            showfliers=True,
            fliersize=5, whis=1.5, width=0.6,
            linewidth=1
        )

        # Annotate stars per group
        metric_sig = ranksum_df[ranksum_df["Metric"] == metric]
        y_min, y_max = ax.get_ylim()
        y_pos = y_min + (y_max - y_min) * 0.93

        for idx, group in enumerate(GROUP_ORDER):
            row = metric_sig[metric_sig["Group"] == group]
            if not row.empty:
                ax.text(idx, y_pos, row["Significance"].iloc[0],
                        ha="center", va="bottom", fontsize=14, fontweight="bold")

        ax.set_title(metric, fontsize=16, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.tick_params(labelsize=13)

        # Remove per-axis legend (we'll add one global legend instead)
        ax.legend_.remove()

    # Remove unused subplot (8 slots, 7 metrics)
    for j in range(len(METRICS_NAMES), len(axes)):
        fig.delaxes(axes[j])

    fig.text(
        0.99, 0.02,
        "* p<0.05, ** p<0.01, *** p<0.001, ns: not significant",
        ha="right", fontsize=13, style="italic"
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    plot_path = os.path.join(output_dir, f"{outcome_key}_performance.svg")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {plot_path}")

    return summary_out_df


# Run analysis for each outcome

all_summaries = []

for outcome_key, info in outcomes.items():
    out_summary = analyze_outcome_by_group(
        final_class_df,
        outcome_key,
        info,
        output_dir=output_dir,
        n_boot=1000,
        random_state=42
    )
    if out_summary is not None and not out_summary.empty:
        all_summaries.append(out_summary)

if all_summaries:
    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_path = os.path.join(output_dir, "prospective_all_outcomes_summary_results.csv")
    combined_summary.to_csv(combined_path, index=False)
    print(f"\nSaved combined summary to {combined_path}")