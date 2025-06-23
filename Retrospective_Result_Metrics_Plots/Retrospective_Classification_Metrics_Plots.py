# %% Imports
import os
import ast
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, average_precision_score, confusion_matrix
)
from scipy.stats import ranksums
from dotenv import load_dotenv

# %% Load Environment Variables
load_dotenv()

# %% Define Paths
root_folder = os.getenv("ROOT_FOLDER")
source_data = os.getenv("SOURCE_DATA")
output_folder = os.getenv("OUTPUT_FOLDER")

final_df_path = os.path.join(root_folder, source_data, os.getenv("FINAL_DF"))
retro_df_path = os.path.join(root_folder, source_data, os.getenv("RETRO_DF"))
comparison_ids_path = os.path.join(root_folder, source_data, os.getenv("PATIENTS_1000"))

# File mapping for each outcome
ml_files = {
    "hosp_mortality": {
        "prob": os.path.join(root_folder, output_folder, os.getenv("BEST_HOSP_MORT_FILE")),
        "params": os.path.join(root_folder, output_folder, os.getenv("BEST_HOSP_MORT_PARAMS_FILE")),
        "title": "Human vs AI Prediction Performance on In-Hospital Mortality"
    },
    "post_disch_mortality": {
        "prob": os.path.join(root_folder, output_folder, os.getenv("BEST_30DAY_MORT_FILE")),
        "params": os.path.join(root_folder, output_folder, os.getenv("BEST_30DAY_MORT_PARAMS_FILE")),
        "title": "Human vs AI Prediction Performance on 30-Day Post Discharge Mortality"
    },
    "aki": {
        "prob": os.path.join(root_folder, output_folder, os.getenv("BEST_AKI_FILE")),
        "params": os.path.join(root_folder, output_folder, os.getenv("BEST_AKI_PARAMS_FILE")),
        "title": "Human vs AI Prediction Performance on AKI"
    },
    "delirium": {
        "prob": os.path.join(root_folder, output_folder, os.getenv("BEST_DELIRIUM_FILE")),
        "params": os.path.join(root_folder, output_folder, os.getenv("BEST_DELIRIUM_PARAMS_FILE")),
        "title": "Human vs AI Prediction Performance on Delirium"
    }
}

# %% Load and Clean Retro Data
retro_df = pd.read_csv(retro_df_path)
retro_df.drop(columns=['clinician_type_initial', 'set_assigned', 'initial_survey_complete'], inplace=True)
retro_df.drop(columns=[col for col in retro_df.columns if col.startswith('set')], inplace=True)

# %% Reshape Data
df = retro_df.melt(id_vars=['record_id', 'Role'], value_vars=retro_df.columns[2:],
                   var_name='prediction_item', value_name='prediction')
df[['Patient_id', 'Prediction_Category']] = df['prediction_item'].str.split('_', n=1, expand=True)
df.drop(columns='prediction_item', inplace=True)
df['Prediction_Category'] = df['Prediction_Category'].str.replace(r'^\d+_', '', regex=True)

# %% Separate by Category
categories = ["hosp_stay", "icu_stay", "aki", "delirium", "hosp_mortality", "post_disch_mortality"]
df_categories = {cat: df[df["Prediction_Category"] == cat].copy() for cat in categories}

# Clean each category df
for cat, df_cat in df_categories.items():
    df_cat.dropna(subset=["prediction"], inplace=True)
    df_cat.loc[:, "Patient_id"] = df_cat["Patient_id"].str.upper()
    df_categories[cat] = df_cat

# %% Add Group Column
def add_group_column(df):
    role_counts = df.groupby("Patient_id")["Role"].count().reset_index()
    role_counts["Group"] = role_counts["Role"].apply(lambda x: str(min(x, 3)))
    return df.merge(role_counts[["Patient_id", "Group"]], on="Patient_id", how="left")

df_categories = {key: add_group_column(df) for key, df in df_categories.items()}

# %% Load Target Labels
target_df = pd.read_csv(final_df_path).iloc[:, :16]
outcome_label_mapping = {
    "hosp_stay": "HOSP_LOS", 
    "icu_stay": "ICU_LOS",
    "aki": "AKI_STATUS",
    "delirium": "DELIRIUM_STATUS",
    "hosp_mortality": "HOSP_MORTALITY",
    "post_disch_mortality": "30_DAYS_POST_DISCHARGE_MORTALITY"
}

def merge_with_target(df, label_column):
    df_merged = df.merge(target_df, left_on="Patient_id", right_on="PATIENT_ID", how="left")
    return df_merged[["Patient_id", "Role", "prediction", "Group", label_column]]

df_categories_merged = {key: merge_with_target(df, outcome_label_mapping[key])
                        for key, df in df_categories.items()}

# %% Main analysis function for each outcome
def analyze_outcome(outcome_key, ml_files, df_categories_merged, comparison_ids_path):
    print(f"\n{'='*40}\nAnalyzing: {outcome_key}\n{'='*40}")
    
    df = df_categories_merged[outcome_key].copy()
    
    # Map predictions: 1 if prediction < 2 else 0
    df["mapped_prediction"] = df["prediction"].apply(lambda x: 1 if x < 2 else 0)
    test1000 = pd.read_csv(comparison_ids_path)
    ml_best_prob_path = ml_files[outcome_key]["prob"]
    ml_best_params_path = ml_files[outcome_key]["params"]
    ml_best_hosp_mort_df = pd.read_csv(ml_best_prob_path)
    merged_df = df.merge(test1000, left_on='Patient_id', right_on='PATIENT_ID', how='left')
    final_df = merged_df.merge(ml_best_hosp_mort_df, left_on='ADMISSION_ID', right_on='Admission_ID', how='left')
    
    # Load Best Threshold
    best_ml_params = ast.literal_eval(pd.read_csv(ml_best_params_path)["best_params"].iloc[0])
    best_ml_threshold = best_ml_params['best_threshold']
    final_df["prediction_outcome"] = final_df["Predicted_Probability"].apply(lambda x: 1 if x > best_ml_threshold else 0)
    
    # Drop Missing
    final_df.dropna(subset=['True_Label', 'mapped_prediction', 'prediction_outcome'], inplace=True)
    
    # Metric Calculation
    def compute_metrics(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        return {
            "Median Accuracy": accuracy_score(y_true, y_pred),
            "Median Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
            "Median Average Precision": average_precision_score(y_true, y_pred),
            "Median Precision": precision_score(y_true, y_pred, zero_division=0),
            "Median Specificity": specificity,
            "Median Recall": recall_score(y_true, y_pred, zero_division=0),
            "Median F1 Score": f1_score(y_true, y_pred, zero_division=0),
            "Median NPV": npv
        }
    # Bootstrapping
    def bootstrap_metrics(y_true, y_pred, n_iterations=1000):
        n = len(y_true)
        results = []
        for _ in range(n_iterations):
            idx = np.random.randint(0, n, n)
            results.append(list(compute_metrics(y_true[idx], y_pred[idx]).values()))
        return np.array(results)
    
    # Bootstrapped Results
    bootstrap_results = []
    metrics_names = [
        "Median Accuracy",
        "Median Balanced Accuracy",
        "Median Average Precision",
        "Median Precision",
        "Median Specificity",
        "Median Recall",
        "Median F1 Score",
        "Median NPV"
    ]
    
    for group in sorted(final_df['Group'].unique()):
        df_group = final_df[final_df['Group'] == group]
        y_true, y_human, y_ai = df_group['True_Label'].values, df_group['mapped_prediction'].values, df_group['prediction_outcome'].values
        boot_human, boot_ai = bootstrap_metrics(y_true, y_human), bootstrap_metrics(y_true, y_ai)
        for i, metric in enumerate(metrics_names):
            for source, data in zip(['Human', 'AI'], [boot_human, boot_ai]):
                bootstrap_results.append({
                    "Group": f"Group {group}", "Source": source, "Metric": metric,
                    "Score": np.median(data[:, i]),
                    "Q1": np.percentile(data[:, i], 25),
                    "Q3": np.percentile(data[:, i], 75),
                    "Whisker_low": np.percentile(data[:, i], 5),
                    "Whisker_high": np.percentile(data[:, i], 95)
                })
    metrics_df = pd.DataFrame(bootstrap_results)
    plot_data = []
    for group in sorted(final_df['Group'].unique()):
        df_group = final_df[final_df['Group'] == group]
        y_true = df_group['True_Label'].values
        boot_human = bootstrap_metrics(y_true, df_group['mapped_prediction'].values)
        for i, metric in enumerate(metrics_names):
            for sample in boot_human[:, i]:
                plot_data.append({
                    'Group': f'Group {group}',
                    'Source': 'Human',
                    'Metric': metric,
                    'Value': sample
                })
        boot_ai = bootstrap_metrics(y_true, df_group['prediction_outcome'].values)
        for i, metric in enumerate(metrics_names):
            for sample in boot_ai[:, i]:
                plot_data.append({
                    'Group': f'Group {group}',
                    'Source': 'AI',
                    'Metric': metric,
                    'Value': sample
                })
    plot_df = pd.DataFrame(plot_data)
    
    # Compute significance
    def add_significance(ax, data, metric, y_frac=0.96):
        ranksum_results = []
        group_list = list(sorted(data['Group'].unique()))
        for group in group_list:
            group_data = data[data['Group'] == group]
            human_bootstrap = group_data[group_data['Source'] == 'Human']['Value'].values
            ai_bootstrap = group_data[group_data['Source'] == 'AI']['Value'].values
            stat, pval = ranksums(human_bootstrap, ai_bootstrap)
            significance = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
            ranksum_results.append({
                'Metric': metric,
                'Group': group,
                'Statistic': stat,
                'P-value': pval,
                'Human_median': np.median(human_bootstrap),
                'AI_median': np.median(ai_bootstrap),
                'Significance': significance
            })
            x_pos = group_list.index(group)
            # Place marker at a fixed fraction of the axis height
            y_min, y_max = ax.get_ylim()
            y_pos = y_min + (y_max - y_min) * y_frac
            ax.text(x_pos, y_pos, significance, ha='center', va='bottom', fontsize=18, fontweight='bold', color='black')
        return pd.DataFrame(ranksum_results)
    
    # Plot Bootstrapped Results with Outliers
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk", font_scale=1.3)
    sns.set_palette("colorblind")

    fig, axes = plt.subplots(4, 2, figsize=(20, 30))
    fig.suptitle(
        f'Human vs AI Prediction Performance on {outcome_key.replace("_", " ")}',
        y=1.03, fontsize=28, fontweight='bold'
    )
    
    for ax, metric in zip(axes.flatten(), metrics_names):
        metric_data = plot_df[plot_df['Metric'] == metric]
        sns.boxplot(
            data=metric_data, x='Group', y='Value', hue='Source',
            ax=ax, showfliers=True, fliersize=5, whis=1.5
        )
        add_significance(ax, metric_data, metric, y_frac=0.96)
        ax.set_title(metric, fontsize=20, fontweight='bold', pad=10)
        # ax.set_xlabel('Group', fontsize=16, fontweight='bold')
        # ax.set_ylabel('Score', fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(labelsize=14)
        # Move legend outside for the first subplot only
        if ax == axes.flatten()[0]:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14, title_fontsize=16)
        else:
            ax.get_legend().remove()
            
    # Remove extra axes
    if len(metrics_names) < len(axes.flatten()):
        for i in range(len(metrics_names), len(axes.flatten())):
            fig.delaxes(axes.flatten()[i])
            
    # Move annotation below the plot
    fig.text(0.99, -0.01, 
            '* p<0.05, ** p<0.01, *** p<0.001, ns: not significant',
            ha='right', fontsize=16, style='italic')
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    plt.savefig(f"{outcome_key}_performance.png", dpi=400, bbox_inches='tight')
    
    # Save metrics summary as CSV
    metrics_df.to_csv(f"{outcome_key}_metrics_summary.csv", index=False)

    # Collect all ranksum results
    all_ranksum_results = []
    for ax, metric in zip(axes.flatten(), metrics_names):
        metric_data = plot_df[plot_df['Metric'] == metric]
        results_df = add_significance(ax, metric_data, metric)
        all_ranksum_results.append(results_df)
    final_ranksum_df = pd.concat(all_ranksum_results, ignore_index=True)

    # Save ranksum results as CSV
    final_ranksum_df.to_csv(f"{outcome_key}_ranksum_results.csv", index=False)

# %% Run analysis for each outcome
for outcome_key in ["hosp_mortality", "post_disch_mortality", "aki", "delirium"]: 
    analyze_outcome(outcome_key, ml_files, df_categories_merged, comparison_ids_path)