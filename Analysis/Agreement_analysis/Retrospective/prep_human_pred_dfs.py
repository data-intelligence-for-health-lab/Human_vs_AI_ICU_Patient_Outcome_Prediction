# %% Imports
import os
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import cohen_kappa_score
from dotenv import load_dotenv

# CONFIG / PATHS
load_dotenv()

root_folder = os.getenv("ROOT_FOLDER")
source_data = os.getenv("SOURCE_DATA")
output_folder = os.getenv("OUTPUT_FOLDER")

final_df_env = os.getenv("FINAL_DF")
retro_df_env = os.getenv("RETRO_DF")

if not all([root_folder, source_data, output_folder, final_df_env, retro_df_env]):
    missing = [k for k, v in {
        "ROOT_FOLDER": root_folder,
        "SOURCE_DATA": source_data,
        "OUTPUT_FOLDER": output_folder,
        "FINAL_DF": final_df_env,
        "RETRO_DF": retro_df_env
    }.items() if not v]
    raise ValueError(f"Missing required environment variables: {missing}")

final_df_path = os.path.join(root_folder, source_data, final_df_env)
retro_df_path = os.path.join(root_folder, source_data, retro_df_env)

# LOAD HUMAN PREDICTIONS
retro_df = pd.read_csv(retro_df_path)

required_cols = ["record_id", "Role"]
for c in required_cols:
    if c not in retro_df.columns:
        raise ValueError(f"Expected column '{c}' not found in retro_df.")

retro_df.drop(
    columns=["clinician_type_initial", "set_assigned", "initial_survey_complete"],
    inplace=True,
    errors="ignore"
)
retro_df.drop(columns=[col for col in retro_df.columns if col.startswith("set")], inplace=True, errors="ignore")

df_long = retro_df.melt(
    id_vars=["record_id", "Role"],
    value_vars=[c for c in retro_df.columns if c not in ["record_id", "Role"]],
    var_name="prediction_item",
    value_name="prediction"
)

df_long[["Patient_id", "Prediction_Category"]] = df_long["prediction_item"].str.split("_", n=1, expand=True)
df_long.drop(columns="prediction_item", inplace=True)
df_long["Prediction_Category"] = df_long["Prediction_Category"].str.replace(r"^\d+_", "", regex=True)

categories = ["hosp_stay", "icu_stay", "aki", "delirium", "hosp_mortality", "post_disch_mortality"]
df_categories = {cat: df_long[df_long["Prediction_Category"] == cat].copy() for cat in categories}

for cat, d in df_categories.items():
    d.dropna(subset=["prediction"], inplace=True)
    d["Patient_id"] = d["Patient_id"].astype(str).str.upper()
    df_categories[cat] = d

# GROUP ASSIGNMENT
def add_group_column(df_in: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df_in.groupby("Patient_id")["record_id"]
        .nunique()
        .reset_index(name="n_raters")
    )
    counts["Group"] = counts["n_raters"].apply(lambda x: str(min(int(x), 3)))
    return df_in.merge(counts[["Patient_id", "Group"]], on="Patient_id", how="left")

df_categories = {k: add_group_column(d) for k, d in df_categories.items()}

human_out_dir = os.path.join(root_folder, output_folder, "prepared_human_predictions")
os.makedirs(human_out_dir, exist_ok=True)

# Save per-category (nice for debugging)
for cat, d in df_categories.items():
    out_path = os.path.join(human_out_dir, f"human_{cat}.csv")
    d.to_csv(out_path, index=False)

# Also save one combined file with an explicit Outcome column
human_df = pd.concat(
    [d.assign(Outcome=cat) for cat, d in df_categories.items()],
    ignore_index=True
)

combined_path = os.path.join(human_out_dir, "human_all_outcomes.csv")
human_df.to_csv(combined_path, index=False)

print(f"Saved combined human_df to: {combined_path}")
print(f"Saved per-category files to: {human_out_dir}")