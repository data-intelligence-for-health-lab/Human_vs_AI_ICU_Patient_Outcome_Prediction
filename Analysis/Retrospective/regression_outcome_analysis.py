import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_env_paths():
    load_dotenv()
    root = Path(os.getenv("ROOT_FOLDER", "."))
    source = Path(os.getenv("SOURCE_DATA", ""))
    paths = {
        "root": root,
        "source": source,
        "final_df": root / source / os.getenv("FINAL_DF", ""),
        "retro_df": root / source / os.getenv("RETRO_DF", ""),
        "comparison_ids": root / source / os.getenv("PATIENTS_1000", ""),
    }
    return paths

def compute_regression_metrics(y_true, y_pred):
    """Return dict of MAE, RMSE, R-squared for 1D numpy arrays."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R-squared": np.nan}
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R-squared": r2}

def bootstrap_regression_metrics(y_true, y_pred, n_iterations=1000, random_state=None):
    n = len(y_true)
    rng = np.random.default_rng(random_state)
    results = np.empty((n_iterations, 3), dtype=float)
    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        vals = compute_regression_metrics(y_true[idx], y_pred[idx])
        results[i, :] = [vals["MAE"], vals["RMSE"], vals["R-squared"]]
    return results

def add_significance(ax, data, metric, y_frac=0.96, source_order=("Human", "AI")):
    """Annotate ax with ranksum significance and return results DataFrame."""
    ranksum_results = []
    group_list = list(sorted(data['Group'].unique()))
    for group in group_list:
        group_data = data[data['Group'] == group]
        # ensure consistent ordering of sources
        sources = [s for s in source_order if s in group_data['Source'].unique()]
        if len(sources) < 2:
            continue
        data1 = group_data[group_data['Source'] == sources[0]]['Value']
        data2 = group_data[group_data['Source'] == sources[1]]['Value']
        if len(data1) == 0 or len(data2) == 0:
            continue
        stat, pval = ranksums(data1, data2)
        significance = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        ranksum_results.append({
            'Metric': metric,
            'Group': group,
            'Statistic': stat,
            'P-value': pval,
            f'{sources[0]}_median': float(np.median(data1)),
            f'{sources[1]}_median': float(np.median(data2)),
            'Significance': significance
        })
        # annotate
        x_pos = group_list.index(group)
        y_min, y_max = ax.get_ylim()
        y_pos = y_min + (y_max - y_min) * y_frac
        ax.text(x_pos, y_pos, significance, ha='center', va='bottom', fontsize=14, fontweight='bold', color='black')
    return pd.DataFrame(ranksum_results)

def analyze_regression_outcome(outcome_key, human_col, ai_col, label_col, group_col="Group", n_boot=1000):
    paths = load_env_paths()
    # construct ml prediction path from env var key
    ml_best_pred_path = paths["root"] / os.getenv(f"BEST_{outcome_key}_FILE", "")

    # Load and process data
    retro_df = pd.read_csv(paths["retro_df"])
    # drop known columns safely
    for col in ['clinician_type_initial', 'set_assigned', 'initial_survey_complete']:
        if col in retro_df.columns:
            retro_df.drop(columns=[col], inplace=True)
    retro_df.drop(columns=[col for col in retro_df.columns if col.startswith('set')], inplace=True)

    df = retro_df.melt(id_vars=['record_id', 'Role'], value_vars=retro_df.columns[2:],
                       var_name='prediction_item', value_name='prediction')
    df[['Patient_id', 'Prediction_Category']] = df['prediction_item'].str.split('_', n=1, expand=True)
    df.drop(columns='prediction_item', inplace=True)
    df['Prediction_Category'] = df['Prediction_Category'].str.replace(r'^\d+_', '', regex=True)
    df = df[df["Prediction_Category"] == human_col].copy()
    df.dropna(subset=["prediction"], inplace=True)
    df["Patient_id"] = df["Patient_id"].str.upper()

    # Add group column based on number of roles per patient (capped at 3)
    role_counts = df.groupby("Patient_id")["Role"].count().reset_index(name="RoleCount")
    role_counts["Group"] = role_counts["RoleCount"].apply(lambda x: str(min(int(x), 3)))
    df = df.merge(role_counts[["Patient_id", "Group"]], on="Patient_id", how="left")

    # Merge with target (take first 16 columns as earlier)
    target_df = pd.read_csv(paths["final_df"]).iloc[:, :16]
    df = df.merge(target_df, left_on="Patient_id", right_on="PATIENT_ID", how="left")
    df = df[["Patient_id", "Role", "prediction", "Group", label_col]]

    # Merge with ML predictions and comparison ids
    test1000 = pd.read_csv(paths["comparison_ids"])
    human_pred_df = df.copy()
    final_human_df = human_pred_df.merge(test1000, left_on='Patient_id', right_on='PATIENT_ID', how='left')

    # Set ML prediction file path based on outcome_key if not found in env
    if outcome_key == "HOSP_LOS":
        ml_best_pred_path = paths["root"] / "output/regression/pred_results_HOSP_LOS_XGB_R_C114.csv"
    elif outcome_key == "ICU_LOS":
        ml_best_pred_path = paths["root"] / "output/regression/pred_results_ICU_LOS_XGB_R_C104.csv"

    if not ml_best_pred_path.exists():
        logging.error(f"ML predictions file not found: {ml_best_pred_path}")
        return
    ml_best_pred_df = pd.read_csv(ml_best_pred_path)
    final_ml_df = ml_best_pred_df.merge(test1000, left_on='Admission_ID', right_on='ADMISSION_ID', how='left')

    # ensure joining keys exist
    if 'ADMISSION_ID' not in final_ml_df.columns:
        logging.error("ADMISSION_ID missing in ML predictions after merge.")
    final_df = final_human_df.merge(final_ml_df[['ADMISSION_ID', ai_col]], on='ADMISSION_ID', how='inner')
    final_df = final_df.rename(columns={
        "prediction": "Human Prediction",
        label_col: "True Label",
        ai_col: "Model Prediction"
    })
    final_df.dropna(inplace=True)

    # Bootstrap metrics per group
    plot_data = []
    metrics_names = ["MAE", "RMSE", "R-squared"]

    for group in sorted(final_df[group_col].unique()):
        df_group = final_df[final_df[group_col] == group]
        if df_group.empty:
            continue
        y_true = df_group['True Label'].values
        boot_human = bootstrap_regression_metrics(y_true, df_group['Human Prediction'].values, n_iterations=n_boot)
        boot_ai = bootstrap_regression_metrics(y_true, df_group['Model Prediction'].values, n_iterations=n_boot)
        for i, metric in enumerate(metrics_names):
            for sample in boot_human[:, i]:
                plot_data.append({'Group': f'Group {group}', 'Source': 'Human', 'Metric': metric, 'Value': sample})
            for sample in boot_ai[:, i]:
                plot_data.append({'Group': f'Group {group}', 'Source': 'AI', 'Metric': metric, 'Value': sample})

    plot_df = pd.DataFrame(plot_data)
    group_label_map = {
        "Group 1": "One Clinician",
        "Group 2": "Two Clinicians",
        "Group 3": "Seven Clinicians"
    }
    plot_df["Group_Label"] = plot_df["Group"].map(group_label_map).fillna(plot_df["Group"])
    
    # Plot
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk", font_scale=1.3)
    sns.set_palette("colorblind")

    fig, axes = plt.subplots(3, 1, figsize=(12, 24))
    fig.suptitle(f'Human vs AI Prediction Performance on {outcome_key.replace("_", " ")}', y=1.03, fontsize=20, fontweight='bold')

    all_ranksum_results = []
    for ax, metric in zip(axes.flatten(), metrics_names):
        metric_data = plot_df[plot_df['Metric'] == metric]
        sns.boxplot(data=metric_data, x='Group_Label', y='Value', hue='Source',
                    ax=ax, showfliers=True, fliersize=5, whis=1.5)
        # call add_significance once and collect results
        results_df = add_significance(ax, metric_data, metric, y_frac=0.95)
        all_ranksum_results.append(results_df)

        ax.set_title(metric, fontsize=20, fontweight='bold', pad=2)
        # ax.set_xlabel('Group', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(labelsize=14)

         # Remove legend from all axes
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    
    # Add a single legend to the right of the first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.92),
        fontsize=16, title="Source", title_fontsize=18, frameon=True
    )
        
    if len(metrics_names) < len(axes.flatten()):
        for i in range(len(metrics_names), len(axes.flatten())):
            fig.delaxes(axes.flatten()[i])

    fig.text(0.99, 0.03, '* p<0.05, ** p<0.01, *** p<0.001, ns: not significant',
             ha='right', fontsize=16, style='italic')
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    # ensure output folder exists
    out_folder = Path("/home/lu.kuang/thesis/GithubFolder/jupyterlab/results/figures&outputs")
    out_folder.mkdir(parents=True, exist_ok=True)
    fig_path = out_folder / f"{outcome_key}_performance.svg"
    fig.savefig(fig_path)
    logging.info(f"Saved plot to {fig_path}")
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
                        "Median": float(np.median(data['Value'])),
                        "Q1": float(np.percentile(data['Value'], 25)),
                        "Q3": float(np.percentile(data['Value'], 75))
                    })
    pd.DataFrame(summary_stats).to_csv(out_folder / f"{outcome_key}_metrics_summary.csv", index=False)
    logging.info("Saved metrics summary")

    # Save ranksum results
    final_ranksum_df = pd.concat(all_ranksum_results, ignore_index=True) if all_ranksum_results else pd.DataFrame()
    final_ranksum_df.to_csv(out_folder / f"{outcome_key}_ranksum_results.csv", index=False)
    logging.info("Saved ranksum results")

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