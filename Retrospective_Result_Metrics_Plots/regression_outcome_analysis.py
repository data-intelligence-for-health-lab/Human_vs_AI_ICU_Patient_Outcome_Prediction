import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from dotenv import load_dotenv

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk", font_scale=1.3)

def compute_regression_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R-squared": r2}

def bootstrap_regression_metrics(y_true, y_pred, n_iterations=1000):
    n = len(y_true)
    results = []
    for _ in range(n_iterations):
        idx = np.random.randint(0, n, n)
        results.append(list(compute_regression_metrics(y_true[idx], y_pred[idx]).values()))
    return np.array(results)

def add_significance(ax, data, metric, y_frac=0.96):
    ranksum_results = []
    group_list = list(sorted(data['Group'].unique()))
    for group in group_list:
        group_data = data[data['Group'] == group]
        sources = group_data['Source'].unique()
        if len(sources) < 2:
            continue
        data1 = group_data[group_data['Source'] == sources[0]]['Value']
        data2 = group_data[group_data['Source'] == sources[1]]['Value']
        stat, pval = ranksums(data1, data2)
        significance = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        ranksum_results.append({
            'Metric': metric,
            'Group': group,
            'Statistic': stat,
            'P-value': pval,
            f'{sources[0]}_median': np.median(data1),
            f'{sources[1]}_median': np.median(data2),
            'Significance': significance
        })
        x_pos = group_list.index(group)
        y_min, y_max = ax.get_ylim()
        y_pos = y_min + (y_max - y_min) * y_frac
        ax.text(x_pos, y_pos, significance, ha='center', va='bottom', fontsize=18, fontweight='bold', color='black')
    return pd.DataFrame(ranksum_results)

def analyze_regression_outcome(outcome_key, human_col, ai_col, label_col, group_col="Group"):
    # Load environment variables
    load_dotenv()
    root_folder = os.getenv("ROOT_FOLDER")
    source_data = os.getenv("SOURCE_DATA")
    output_folder = os.getenv("OUTPUT_FOLDER")
    final_df_path = os.path.join(root_folder, source_data, os.getenv("FINAL_DF"))
    retro_df_path = os.path.join(root_folder, source_data, os.getenv("RETRO_DF"))
    comparison_ids_path = os.path.join(root_folder, source_data, os.getenv("PATIENTS_1000"))
    ml_best_pred_path = os.path.join(root_folder, output_folder, os.getenv(f"BEST_{outcome_key}_FILE"))

    # Load and process data
    retro_df = pd.read_csv(retro_df_path)
    retro_df.drop(columns=['clinician_type_initial', 'set_assigned', 'initial_survey_complete'], inplace=True)
    retro_df.drop(columns=[col for col in retro_df.columns if col.startswith('set')], inplace=True)
    df = retro_df.melt(id_vars=['record_id', 'Role'], value_vars=retro_df.columns[2:],
                       var_name='prediction_item', value_name='prediction')
    df[['Patient_id', 'Prediction_Category']] = df['prediction_item'].str.split('_', n=1, expand=True)
    df.drop(columns='prediction_item', inplace=True)
    df['Prediction_Category'] = df['Prediction_Category'].str.replace(r'^\d+_', '', regex=True)
    df = df[df["Prediction_Category"] == human_col].copy()
    df.dropna(subset=["prediction"], inplace=True)
    df["Patient_id"] = df["Patient_id"].str.upper()
    # Add group column
    role_counts = df.groupby("Patient_id")["Role"].count().reset_index()
    role_counts["Group"] = role_counts["Role"].apply(lambda x: str(min(x, 3)))
    df = df.merge(role_counts[["Patient_id", "Group"]], on="Patient_id", how="left")
    # Merge with target
    target_df = pd.read_csv(final_df_path).iloc[:, :16]
    df = df.merge(target_df, left_on="Patient_id", right_on="PATIENT_ID", how="left")
    df = df[["Patient_id", "Role", "prediction", "Group", label_col]]
    # Merge with ML predictions
    test1000 = pd.read_csv(comparison_ids_path)
    human_pred_df = df.copy()
    final_human_df = human_pred_df.merge(test1000, left_on='Patient_id', right_on='PATIENT_ID', how='left')
    ml_best_pred_df = pd.read_csv(ml_best_pred_path)
    final_ml_df = ml_best_pred_df.merge(test1000, left_on='Admission_ID', right_on='ADMISSION_ID', how='left')
    final_df = final_human_df.merge(final_ml_df[['ADMISSION_ID', ai_col]], on='ADMISSION_ID', how='inner')
    final_df = final_df.rename(columns={
        "prediction": "Human Prediction",
        label_col: "True Label",
        ai_col: "Model Prediction"
    })
    final_df.dropna(inplace=True)

    # Bootstrap and plot
    plot_data = []
    metrics_names = ["MAE", "MSE", "RMSE", "R-squared"]
    colors = {'Human': '#2ebbf3', 'AI': '#57b02e'}
    for group in sorted(final_df[group_col].unique()):
        df_group = final_df[final_df[group_col] == group]
        y_true = df_group['True Label'].values
        boot_human = bootstrap_regression_metrics(y_true, df_group['Human Prediction'].values)
        boot_ai = bootstrap_regression_metrics(y_true, df_group['Model Prediction'].values)
        for i, metric in enumerate(metrics_names):
            for sample in boot_human[:, i]:
                plot_data.append({'Group': f'Group {group}', 'Source': 'Human', 'Metric': metric, 'Value': sample})
            for sample in boot_ai[:, i]:
                plot_data.append({'Group': f'Group {group}', 'Source': 'AI', 'Metric': metric, 'Value': sample})
    plot_df = pd.DataFrame(plot_data)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Human vs AI Prediction Performance on {outcome_key.replace("_", " ")}', y=1.03, fontsize=24, fontweight='bold')
    for ax, metric in zip(axes.flatten(), metrics_names):
        metric_data = plot_df[plot_df['Metric'] == metric]
        sns.boxplot(data=metric_data, x='Group', y='Value', hue='Source',
                    palette=colors, ax=ax, showfliers=True, fliersize=5, whis=1.5)
        add_significance(ax, metric_data, metric, y_frac=0.95)
        ax.set_title(metric, fontsize=18, fontweight='bold', pad=16)
        # ax.set_xlabel('Group', fontsize=14, fontweight='bold')  # Remove x-label
        # ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(labelsize=12)
        if ax != axes.flatten()[0]:
            ax.get_legend().remove()
        else:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12, title_fontsize=14)
    if len(metrics_names) < len(axes.flatten()):
        for i in range(len(metrics_names), len(axes.flatten())):
            fig.delaxes(axes.flatten()[i])
    fig.text(0.99, -0.04, '* p<0.05, ** p<0.01, *** p<0.001, ns: not significant',
             ha='right', fontsize=14, style='italic')
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    plt.savefig(f"{outcome_key}_performance.png", dpi=400, bbox_inches='tight')
    plt.show()

    # Save summary statistics
    summary_stats = []
    for metric in metrics_names:
        metric_data = plot_df[plot_df['Metric'] == metric]
        for group in sorted(metric_data['Group'].unique()):
            for source in ['Human', 'AI']:
                data = metric_data[(metric_data['Group'] == group) & (metric_data['Source'] == source)]
                if len(data) > 0:
                    summary_stats.append({
                        "Metric": metric,
                        "Group": group,
                        "Source": source,
                        "Median": np.median(data['Value']),
                        "Q1": np.percentile(data['Value'], 25),
                        "Q3": np.percentile(data['Value'], 75)
                    })
    pd.DataFrame(summary_stats).to_csv(f"{outcome_key}_metrics_summary.csv", index=False)

    # Save ranksum results
    all_ranksum_results = []
    for ax, metric in zip(axes.flatten(), metrics_names):
        metric_data = plot_df[plot_df['Metric'] == metric]
        results_df = add_significance(ax, metric_data, metric)
        all_ranksum_results.append(results_df)
    final_ranksum_df = pd.concat(all_ranksum_results, ignore_index=True)
    final_ranksum_df.to_csv(f"{outcome_key}_ranksum_results.csv", index=False)

if __name__ == "__main__":
    # Example usage for ICU_LOS and HOSP_LOS
    analyze_regression_outcome(
        outcome_key="ICU_LOS", 
        human_col="icu_stay", 
        ai_col="Predicted Value", 
        label_col="ICU_LOS"
    )
    analyze_regression_outcome(
        outcome_key="HOSP_LOS", 
        human_col="hosp_stay", 
        ai_col="Predicted Value", 
        label_col="HOSP_LOS"
    )