import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)
from scipy.stats import ranksums
import matplotlib.pyplot as plt
import seaborn as sns
prosp_df_human = pd.read_csv('data/Prediction_dataset_final.csv')
classification_df = pd.read_csv('data/prosp_final_predictions.csv')
pulled_df = pd.read_csv('data/prosp_pulled_data.csv')
output_dir = ""

human_df = prosp_df_human[
    [
        "MRN",
        "Clinician Role",
        "In-hospital mortality",
        "30-day post-discharge mortality",
        "Delirum during this ICU stay",
        "AKI during this ICU stay",
        "How long patient would be at ICU",
        "How long patient would be at hospital"
    ]
]

human_df = human_df.rename(columns={
    "In-hospital mortality": "HOSP_MORT_human",
    "30-day post-discharge mortality": "30_DAY_MORT_human",
    "Delirum during this ICU stay": "DELIRIUM_human",
    "AKI during this ICU stay": "AKI_human",
    "How long patient would be at ICU": "ICU_LOS_human",
    "How long patient would be at hospital": "HOSP_LOS_human"
})

def map_likelihood_to_binary(value):
    if pd.isna(value):
        return np.nan
    value_str = str(value).strip().lower()

    # Define logic
    if value_str.startswith('unlikely'):
        return 0
    elif value_str.startswith('most unlikely'):
        return 0
    elif value_str.startswith('equally likely'):
        return 0
    elif value_str.startswith('likely'):
        return 1
    elif value_str.startswith('most likely'):
        return 1
    else:
        return np.nan

cols_to_map = ['HOSP_MORT_human', '30_DAY_MORT_human', 'DELIRIUM_human', 'AKI_human']

for col in cols_to_map:
    new_col = col + '_Num'
    human_df[new_col] = human_df[col].apply(map_likelihood_to_binary)

final_human_df = human_df[[
    'MRN',
    'Clinician Role',
    'HOSP_MORT_human_Num',
    '30_DAY_MORT_human_Num',
    'DELIRIUM_human_Num',
    'AKI_human_Num'
]]

# Select only the columns you want
columns_to_keep = [
    'ADMISSION_ID',
    'IN_HOSPITAL_MORTALITY',
    '30_DAY_POST_DISCHARGE_MORTALITY',
    'AKI_STATUS',
    'DELIRIUM_STATUS',
    'HOSP_MORTALITY_PRED_PROB',
    '30_DAY_MORTALITY_PRED_PROB',
    'AKI_STATUS_PRED_PROB',
    'DELIRIUM_STATUS_PRED_PROB'
]
classification_df = classification_df[columns_to_keep]


ai_prob_cols = {
    'AKI_STATUS_PRED_PROB': AKI_threshold,
    'DELIRIUM_STATUS_PRED_PROB': Delirium_threshold,
    'HOSP_MORTALITY_PRED_PROB': Hospital_mortality,
    '30_DAY_MORTALITY_PRED_PROB': Post_discharge_mortality_thres
}

for col, thresh in ai_prob_cols.items():
    # Only create binary column if the probability column exists
    if col in classification_df.columns:
        binary_col = col.replace('_PRED_PROB', '_PRED_CLASS')
        classification_df[binary_col] = (classification_df[col] >= thresh).astype(int)

# Merge final_human_df with pulled_df on 'MRN' to get ADMISSION_ID
merged_df = final_human_df.merge(
    pulled_df[['MRN', 'ADMISSION_ID']],
    on='MRN',
    how='left'
)

# Merge the result with classification_merged_df on 'ADMISSION_ID' (left join)
final_df = merged_df.merge(
    classification_df,
    on='ADMISSION_ID',
    how='left'
)

# Map Clinician Role to Group
role_to_group = {
    "Attending Physician; Fellow": "senior physician",
    "Resident; Clinical clerk": "junior physician",
    "RN": "nurses"
}

def map_role_to_group(role):
    role = str(role).strip()
    if role.startswith("Attending Physician") or role.startswith("Fellow"):
        return "senior physician"
    elif role.startswith("Resident") or role.startswith("Clinical clerk"):
        return "junior physician"
    elif role.startswith("RN"):
        return "nurses"
    else:
        return "other"

final_df['Group'] = final_df['Clinician Role'].apply(map_role_to_group)

# Select relevant columns and drop duplicates
class_data = final_df[["MRN", "IN_HOSPITAL_MORTALITY", "30_DAY_POST_DISCHARGE_MORTALITY", "AKI_STATUS", "DELIRIUM_STATUS"]].drop_duplicates()

# Drop rows where all class columns are NaN
class_data = class_data.dropna(
    subset=["IN_HOSPITAL_MORTALITY", "30_DAY_POST_DISCHARGE_MORTALITY", "AKI_STATUS", "DELIRIUM_STATUS"],
    how="all"
)

def analyze_outcome_by_group(df, outcome_key, outcome_info):
    print(f"\n{'='*40}\nAnalyzing: {outcome_key} by group\n{'='*40}")

    # Styling
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk", font_scale=1.3)
    sns.set_palette("colorblind")

    metrics_names = [
        "Accuracy", "Balanced Accuracy", "Precision",
        "Specificity", "Recall", "F1 Score", "NPV"
    ]

    group_order = ['senior physician', 'junior physician', 'nurses']
    ai_group_order = ['AI_senior', 'AI_junior', 'AI_nurses']

    plot_data = []

    # Compute bootstrapped data
    for group in group_order:
        df_group = df[df['Group'] == group].dropna(
            subset=[outcome_info["human_pred"], outcome_info["ai_pred"], outcome_info["true_label"]]
        ).copy()
        if df_group.empty:
            continue

        y_true = df_group[outcome_info["true_label"]].values
        y_human = df_group[outcome_info["human_pred"]].values
        y_ai = df_group[outcome_info["ai_pred"]].values

        boot_human = bootstrap_metrics(y_true, y_human)
        boot_ai = bootstrap_metrics(y_true, y_ai)

        for i, metric in enumerate(metrics_names):
            for val in boot_human[:, i]:
                plot_data.append({
                    'Group': group,
                    'Source': 'Human',
                    'Metric': metric,
                    'Value': val
                })
            for val in boot_ai[:, i]:
                plot_data.append({
                    'Group': group,
                    'Source': 'AI',
                    'Metric': metric,
                    'Value': val
                })

    plot_df = pd.DataFrame(plot_data)

    # Compute significance and medians
    ranksum_results = []
    for metric in metrics_names:
        metric_df = plot_df[plot_df["Metric"] == metric]
        for group in group_order:
            group_df = metric_df[metric_df["Group"] == group]
            human_vals = group_df[group_df["Source"] == "Human"]["Value"].values
            ai_vals = group_df[group_df["Source"] == "AI"]["Value"].values
            if len(human_vals) == 0 or len(ai_vals) == 0:
                continue

            stat, pval = ranksums(human_vals, ai_vals)
            sig = (
                '***' if pval < 0.001 else
                '**' if pval < 0.01 else
                '*' if pval < 0.05 else 'ns'
            )

            ranksum_results.append({
                "Outcome": outcome_key,
                "Metric": metric,
                "Group": group,
                "Statistic": stat,
                "P-value": pval,
                "Human_median": np.median(human_vals),
                "AI_median": np.median(ai_vals),
                "Significance": sig
            })

    ranksum_df = pd.DataFrame(ranksum_results)

    # Plotting
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    fig.suptitle(outcome_info.get("title", outcome_key), fontsize=22, fontweight='bold')

    for ax, metric in zip(axes, metrics_names):
        metric_data = plot_df[plot_df["Metric"] == metric]

        sns.boxplot(
            data=metric_data,
            x="Group", y="Value", hue="Source",
            ax=ax, showfliers=True, fliersize=5, whis=1.5, width=0.6
        )

        # Annotate significance stars
        metric_sig = ranksum_df[ranksum_df["Metric"] == metric]
        y_min, y_max = ax.get_ylim()
        for idx, group in enumerate(group_order):
            sig_row = metric_sig[metric_sig["Group"] == group]
            if not sig_row.empty:
                sig = sig_row["Significance"].values[0]
                y_pos = y_min + (y_max - y_min) * 0.93
                ax.text(idx, y_pos, sig, ha='center', va='bottom', fontsize=14, fontweight='bold')

        ax.set_title(metric, fontsize=18, fontweight='bold')
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(labelsize=14)
        ax.get_legend().remove()

    # Remove unused subplots
    if len(metrics_names) < len(axes):
        for j in range(len(metrics_names), len(axes)):
            fig.delaxes(axes[j])

    # Add significance legend
    fig.text(0.99, 0.03, '* p<0.05, ** p<0.01, *** p<0.001, ns: not significant',
             ha='right', fontsize=14, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    # Save outputs
    plot_path = os.path.join(output_dir, f"{outcome_key}_performance.svg")
    ranksum_path = os.path.join(output_dir, f"{outcome_key}_ranksum_results.csv")

    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    ranksum_df.to_csv(ranksum_path, index=False)

    plt.close(fig)
    print(f"Saved figure to {plot_path}")
    print(f"Saved ranksum results to {ranksum_path}")


# Map outcomes to columns in final_df
outcomes = {
    "delirium": {
        "human_pred": "DELIRIUM_human_Num",
        "ai_pred": "DELIRIUM_STATUS_PRED_CLASS",
        "true_label": "DELIRIUM_STATUS",
        "title": "Human vs AI Prediction Performance on Delirium"
    },
    "aki": {
        "human_pred": "AKI_human_Num",
        "ai_pred": "AKI_STATUS_PRED_CLASS",
        "true_label": "AKI_STATUS",
        "title": "Human vs AI Prediction Performance on AKI"
    },
    "post_disch_mortality": {
        "human_pred": "30_DAY_MORT_human_Num",
        "ai_pred": "30_DAY_MORTALITY_PRED_CLASS",
        "true_label": "30_DAY_POST_DISCHARGE_MORTALITY",
        "title": "Human vs AI Prediction Performance on 30-Day Post Discharge Mortality"
    },
    "in_hospital_mortality": {
        "human_pred": "HOSP_MORT_human_Num",
        "ai_pred": "HOSP_MORTALITY_PRED_CLASS",
        "true_label": "IN_HOSPITAL_MORTALITY",
        "title": "Human vs AI Prediction Performance on In-Hospital Mortality"
    }
}

# Bootstrapping
def bootstrap_metrics(y_true, y_pred, n_iterations=1000):
    n = len(y_true)
    results = []
    for _ in range(n_iterations):
        idx = np.random.randint(0, n, n)
        res = compute_metrics(y_true[idx], y_pred[idx])
        results.append(list(res.values()))
    return np.array(results)

# Metric calculation function
def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "NPV": npv
    }
    
# Run analysis for each outcome
for outcome_key, info in outcomes.items():
    analyze_outcome_by_group(final_df, outcome_key, info)