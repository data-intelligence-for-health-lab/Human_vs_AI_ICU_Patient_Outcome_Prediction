import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ranksums

# Styling
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk", font_scale=1.3)
sns.set_palette("colorblind")
    
prosp_df_human = pd.read_csv('data/Prediction_dataset_final.csv')
LOS_df = pd.read_csv('data/prosp_final_LOS_predictions.csv')
pulled_df = pd.read_csv('data/prosp_pulled_data.csv')
output_dir = "/home/lu.kuang/thesis/GithubFolder/jupyterlab/results/scripts/prospective_comparisons/prosp_outputs/svg"

human_df = prosp_df_human[
    [
        "MRN",
        "Clinician Role",
        "How long patient would be at ICU",
        "How long patient would be at hospital"
    ]
]

human_df = human_df.rename(columns={
    "How long patient would be at ICU": "ICU_LOS_human",
    "How long patient would be at hospital": "HOSP_LOS_human"
})

final_human_df = human_df[[
    'MRN',
    'Clinician Role',
    'ICU_LOS_human',
    'HOSP_LOS_human',
]]

LOS_df = LOS_df.merge(pulled_df[['MRN', 'ADMISSION_ID']], on='ADMISSION_ID', how='left')
LOS_merged_df = final_human_df.merge(LOS_df, on='MRN', how='left')
df = LOS_merged_df


# === Step 1: Data Cleaning ===
columns_needed = [
    'ICU_LOS', 'ICU_LOS_human', 'ICU_LOS_PRED',
    'HOSP_LOS', 'HOSP_LOS_human', 'HOSP_LOS_PRED'
]

df_clean = df[columns_needed].dropna()
print(f"Rows after dropping NaNs: {len(df_clean)}")

# === Step 2: Extract arrays ===
y_true_icu = df_clean['ICU_LOS'].values
y_pred_human_icu = df_clean['ICU_LOS_human'].values
y_pred_ai_icu = df_clean['ICU_LOS_PRED'].values

y_true_hosp = df_clean['HOSP_LOS'].values
y_pred_human_hosp = df_clean['HOSP_LOS_human'].values
y_pred_ai_hosp = df_clean['HOSP_LOS_PRED'].values

# === Step 3: Define regression metrics ===
def regression_metrics(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }

# === Step 4: Bootstrap function ===
def bootstrap_metrics(y_true, y_pred, n_iterations=1000):
    n = len(y_true)
    metrics_bootstrap = {'MAE': [], 'RMSE': [], 'R2': []}
    
    for _ in tqdm(range(n_iterations), desc='Bootstrapping'):
        indices = np.random.choice(n, size=n, replace=True)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        metrics = regression_metrics(y_true_sample, y_pred_sample)
        for key in metrics:
            metrics_bootstrap[key].append(metrics[key])
    
    return pd.DataFrame(metrics_bootstrap)

# Map Clinician Role to Group
def map_role_to_group(role):
    role = str(role).strip()
    if role.startswith("Attending Physician") or role.startswith("Fellow"):
        return "Senior physician"
    elif role.startswith("Resident") or role.startswith("Clinical clerk"):
        return "Junior physician"
    elif role.startswith("RN"):
        return "Nurses"
    else:
        return "other"

df['Group'] = df['Clinician Role'].apply(map_role_to_group)

def collect_bootstrap_df(df, los_true_col, los_human_col, los_ai_col):
    plot_data = []
    for group in ['Senior physician', 'Junior physician', 'Nurses']:
        df_group = df[df['Group'] == group].dropna(subset=[los_true_col, los_human_col, los_ai_col])
        if len(df_group) == 0:
            continue
        y_true = df_group[los_true_col].values
        y_human = df_group[los_human_col].values
        y_ai = df_group[los_ai_col].values
        boot_human = bootstrap_metrics(y_true, y_human)
        boot_ai = bootstrap_metrics(y_true, y_ai)
        for metric in boot_human.columns:
            for val in boot_human[metric]:
                plot_data.append({'Group': group, 'Source': 'Human', 'Metric': metric, 'Value': val})
            for val in boot_ai[metric]:
                plot_data.append({'Group': group, 'Source': 'AI', 'Metric': metric, 'Value': val})
    return pd.DataFrame(plot_data)

# ICU LOS metrics for all groups
icu_plot_df = collect_bootstrap_df(df, 'ICU_LOS', 'ICU_LOS_human', 'ICU_LOS_PRED')
# Hospital LOS metrics for all groups
hosp_plot_df = collect_bootstrap_df(df, 'HOSP_LOS', 'HOSP_LOS_human', 'HOSP_LOS_PRED')
def add_significance(ax, data, metric, y_frac=0.96):
    ranksum_results = []
    group_list = list(sorted(data['Group'].unique()))
    for group in group_list:
        group_data = data[data['Group'] == group]
        human_bootstrap = group_data[group_data['Source'] == 'Human']['Value'].values
        ai_bootstrap = group_data[group_data['Source'] == 'AI']['Value'].values
        if len(human_bootstrap) == 0 or len(ai_bootstrap) == 0:
            continue
        stat, pval = ranksums(human_bootstrap, ai_bootstrap)
        significance = (
            '***' if pval < 0.001 else
            '**' if pval < 0.01 else
            '*' if pval < 0.05 else
            'ns'
        )
        ranksum_results.append({
            'Metric': metric,
            'Group': group,
            'Statistic': stat,
            'P-value': pval,
            'Human_median': np.median(human_bootstrap),
            'AI_median': np.median(ai_bootstrap),
            'Significance': significance
        })
        # Position text
        x_pos = group_list.index(group)
        y_min, y_max = ax.get_ylim()
        y_pos = y_min + (y_max - y_min) * y_frac
        ax.text(x_pos, y_pos, significance,
                ha='center', va='bottom',
                fontsize=18, fontweight='bold', color='black')
    return pd.DataFrame(ranksum_results)
    
def plot_all_groups_with_stats(plot_df, title):
    metrics_display = {"MAE": "MAE", "RMSE": "RMSE", "R2": "R-squared"}
    all_results = []  # collect all stats
    
    # fig, axes = plt.subplots(1, 3, figsize=(32, 8))
    fig, axes = plt.subplots(3, 1, figsize=(12, 24))
    
    for ax, metric in zip(axes.flatten(), metrics_display.keys()):
        metric_data = plot_df[plot_df['Metric'] == metric]
        sns.boxplot(data=metric_data, x='Group', y='Value', hue='Source', ax=ax)
        
        # Add significance labels
        metric_results = add_significance(ax, metric_data, metric, y_frac=0.93)
        all_results.append(metric_results)
        
        ax.set_title(metrics_display[metric], fontsize=20, fontweight='bold')
        ax.tick_params(labelsize=14)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend_.remove()
    
    # fig.suptitle(title, fontsize=24)
    fig.suptitle(title, y=1.03, fontsize=24, fontweight='bold')

    # Save figure
    fig_path = os.path.join(output_dir, f"{title.replace(' ', '_')}_with_stats.svg")
    plt.savefig(fig_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    
    # Concatenate and save ranksum results
    final_results_df = pd.concat(all_results, ignore_index=True)
    results_path = os.path.join(output_dir, f"{title.replace(' ', '_')}_ranksum_results.csv")
    final_results_df.to_csv(results_path, index=False)

    return final_results_df

plot_all_groups_with_stats(icu_plot_df, 'Human vs AI Prediction Performance on ICU LOS')
plot_all_groups_with_stats(hosp_plot_df, 'Human vs AI Prediction Performance on Hospital LOS')