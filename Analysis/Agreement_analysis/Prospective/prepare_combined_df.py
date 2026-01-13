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

classification_df = pd.read_csv('data/prosp_final_predictions.csv')
human_df = pd.read_csv('processed_data/prospective_human_prediction_df.csv')
#set likerscale mapping threshold
threshold = 2

pulled_df = pd.read_csv('data/prosp_pulled_data.csv')
output_dir = "/home/lu.kuang/thesis/GithubFolder/jupyterlab/results/scripts/prospective_comparisons/processed_data"

ai_classification_df = classification_df[
    [
        "PATIENT_ID",
        "HOSP_MORTALITY_PRED_PROB",
        "30_DAY_MORTALITY_PRED_PROB",
        "AKI_STATUS_PRED_PROB",
        "DELIRIUM_STATUS_PRED_PROB",
    ]
]

true_labels = classification_df[['PATIENT_ID', 'IN_HOSPITAL_MORTALITY','30_DAY_POST_DISCHARGE_MORTALITY', 'AKI_STATUS', 'DELIRIUM_STATUS', 'ICU_LOS', 'HOSP_LOS']]
true_labels = (
    true_labels
    .sort_values("PATIENT_ID")
    .groupby("PATIENT_ID", as_index=False)
    .last()
)

ai_classification_df = ai_classification_df.copy()

#thresholds
AKI_threshold = 0.5965228
Delirium_threshold = 0.62362075
Hospital_mortality_threshold = 0.5655142
Post_discharge_mortality_threshold = 0.64229524

ai_prob_cols = {
    'AKI_STATUS_PRED_PROB': AKI_threshold,
    'DELIRIUM_STATUS_PRED_PROB': Delirium_threshold,
    'HOSP_MORTALITY_PRED_PROB': Hospital_mortality_threshold,
    '30_DAY_MORTALITY_PRED_PROB': Post_discharge_mortality_threshold
}

for col, thresh in ai_prob_cols.items():
    if col in ai_classification_df.columns:
        binary_col = col.replace('_PRED_PROB', '_PRED_CLASS')
        ai_classification_df.loc[:, binary_col] = (ai_classification_df[col] >= thresh).astype(int)
        
ai_classification_df = ai_classification_df[
    [
        "PATIENT_ID",
        "AKI_STATUS_PRED_CLASS",
        "DELIRIUM_STATUS_PRED_CLASS",
        "HOSP_MORTALITY_PRED_CLASS",
        "30_DAY_MORTALITY_PRED_CLASS",
    ]
]

# Merge final_human_df with pulled_df on 'MRN' to get ADMISSION_ID
ai_merged_df = ai_classification_df.merge(
    pulled_df[['MRN', 'PATIENT_ID']],
    on='PATIENT_ID',
    how='inner'
)
ai_merged_df.drop_duplicates()

ai_merged_df = (
    ai_merged_df
    .sort_values("MRN")
    .groupby("MRN", as_index=False)
    .last()
)


human_cols_to_map = [
    "HOSP_MORT_human_Num",
    "30_DAY_MORT_human_Num",
    "DELIRIUM_human_Num",
    "AKI_human_Num",
]

for col in human_cols_to_map:
    human_df[f"{col}_CLASS"] = (
        human_df[col] <= threshold
    ).astype(int)

human_df = human_df[['MRN', 'Group', 'HOSP_MORT_human_Num_CLASS', '30_DAY_MORT_human_Num_CLASS' , 'DELIRIUM_human_Num_CLASS',	'AKI_human_Num_CLASS']]

# Merge the result with classification_merged_df on 'ADMISSION_ID' (left join)
human_ai_class_df = ai_merged_df.merge(
    human_df,
    on='MRN',
    how='inner'
)

final_class_df = human_ai_class_df.merge(
    true_labels,
    on = 'PATIENT_ID',
    how = 'inner'
)
final_class_df = final_class_df.drop(columns=["MRN"])

save_path = os.path.join(output_dir, "prospective_combined_prediction_df.csv")
final_class_df.to_csv(save_path, index=False)

print(f"Saved: {save_path}")