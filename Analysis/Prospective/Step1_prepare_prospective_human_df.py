import os
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score

# Load data
prosp_df_human = pd.read_csv('data/Prediction_dataset_final.csv')
output_dir = "/home/lu.kuang/thesis/GithubFolder/jupyterlab/results/scripts/prospective_comparisons/processed_data"
os.makedirs(output_dir, exist_ok=True)

# Prepare human predictions
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

def map_likelihood_to_numeric(value):
    if pd.isna(value):
        return np.nan
    value_str = str(value).strip().lower()

    if value_str.startswith('most unlikely'):
        return 5
    elif value_str.startswith('unlikely'):
        return 4
    elif value_str.startswith('equally likely'):
        return 3
    elif value_str.startswith('likely'):
        return 2
    elif value_str.startswith('most likely'):
        return 1
    else:
        return np.nan

cols_to_map = ['HOSP_MORT_human', '30_DAY_MORT_human', 'DELIRIUM_human', 'AKI_human']

for col in cols_to_map:
    new_col = col + '_Num'
    human_df[new_col] = human_df[col].apply(map_likelihood_to_numeric)

human_df = human_df[[
    'MRN',
    'Clinician Role',
    'HOSP_MORT_human_Num',
    '30_DAY_MORT_human_Num',
    'DELIRIUM_human_Num',
    'AKI_human_Num',
    'ICU_LOS_human',
    'HOSP_LOS_human'
]]

def map_role_to_group(role):
    role = str(role).strip()
    if role.startswith("Attending Physician") or role.startswith("Fellow"):
        return "Subspecialized"
    elif role.startswith("Resident") or role.startswith("Clinical clerk"):
        return "Trainee"
    elif role.startswith("RN"):
        return "Nurse"
    else:
        return "other"

human_df = human_df.copy()
human_df.loc[:, "Group"] = human_df["Clinician Role"].apply(map_role_to_group)
human_df.drop(columns=["Clinician Role"], inplace=True)

cols_to_avg = [
    "HOSP_MORT_human_Num",
    "30_DAY_MORT_human_Num",
    "DELIRIUM_human_Num",
    "AKI_human_Num",
    "ICU_LOS_human",
    "HOSP_LOS_human",
]

human_df_avg = (
    human_df
    .groupby(["Group", "MRN"], as_index=False)
    .agg({**{c: "mean" for c in cols_to_avg}})
)


save_path = os.path.join(output_dir, "prospective_human_prediction_df.csv")
human_df_avg.to_csv(save_path, index=False)

print(f"Saved: {save_path}")