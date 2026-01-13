import os
import numpy as np
import pandas as pd

from sklearn.metrics import cohen_kappa_score

GROUP_ORDER = ["Subspecialized", "Trainee", "Nurse"]
final_df = pd.read_csv('processed_data/prospective_combined_prediction_df.csv')
output_dir = "/home/lu.kuang/thesis/GithubFolder/jupyterlab/results/scripts/prospective_comparisons/prosp_outputs/1228/agreement"
os.makedirs(output_dir, exist_ok=True)

# Map outcomes to the appropriate columns
outcomes = {
    "delirium": {
        "human_pred": "DELIRIUM_human_Num_CLASS",
        "ai_pred": "DELIRIUM_STATUS_PRED_CLASS",
        "true_label": "DELIRIUM_STATUS",
        "title": "Delirium"
    },
    "aki": {
        "human_pred": "AKI_human_Num_CLASS",
        "ai_pred": "AKI_STATUS_PRED_CLASS",
        "true_label": "AKI_STATUS",
        "title": "Acute Kidney Injury"
    },
    "post_disch_mortality": {
        "human_pred": "30_DAY_MORT_human_Num_CLASS",
        "ai_pred": "30_DAY_MORTALITY_PRED_CLASS",
        "true_label": "30_DAY_POST_DISCHARGE_MORTALITY",
        "title": "30-Day Post-Discharge Mortality"
    },
    "in_hospital_mortality": {
        "human_pred": "HOSP_MORT_human_Num_CLASS",
        "ai_pred": "HOSP_MORTALITY_PRED_CLASS",
        "true_label": "IN_HOSPITAL_MORTALITY",
        "title": "In-Hospital Mortality"
    }
}


def analyze_agreement_by_group(
    df,
    outcome_key,
    outcome_info,
    output_dir,
    n_boot=1000,
    random_state=42
):
    print(f"\n{'='*40}")
    print(f"Agreement (Cohen’s κ): {outcome_key} by group")
    print(f"{'='*40}")

    results = []

    for group in GROUP_ORDER:
        df_group = df[df['Group'] == group].dropna(
            subset=[outcome_info["human_pred"], outcome_info["ai_pred"]]
        ).copy()

        if df_group.empty:
            continue

        y_human = df_group[outcome_info["human_pred"]].astype(int).values
        y_ai = df_group[outcome_info["ai_pred"]].astype(int).values

        # Cohen's kappa
        kappa = cohen_kappa_score(y_human, y_ai)

        results.append({
            "Outcome": outcome_key,
            "Outcome_Label": outcome_info["title"],
            "Group": group,
            "N_pairs": len(df_group),
            "Kappa": kappa,
        })

    results_df = pd.DataFrame(results)
    save_path = os.path.join(output_dir, f"{outcome_key}_agreement_by_group.csv")
    results_df.to_csv(save_path, index=False)

    print(results_df)
    print(f"Saved agreement results to: {save_path}")

    return results_df



# Run agreement analysis for all classification outcomes
all_results = []

for outcome_key, info in outcomes.items():
    res = analyze_agreement_by_group(
        final_df,
        outcome_key,
        info,
        output_dir=output_dir,
        n_boot=1000,
        random_state=42
    )
    if res is not None and not res.empty:
        all_results.append(res)

# Combined summary table
if all_results:
    combined = pd.concat(all_results, ignore_index=True)
    combined_path = os.path.join(output_dir, "all_classification_agreement_by_group.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined agreement table saved to {combined_path}")