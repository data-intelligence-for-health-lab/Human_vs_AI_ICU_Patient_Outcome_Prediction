import numpy as np
import pandas as pd
from retrospective_comparisons.secondary_analysis.reg_agreement_utils import bootstrap_icc31, icc_3_1
# Load LOS agreement dataset
df = pd.read_csv("agreement_output_folder/merged_human_ai_HOSP_LOS.csv")

df["Group"] = df["Group"].astype(str)
df = df.dropna(subset=["Patient_id", "Human_Prediction", "AI_Prediction"]).copy()

# ICC(3,1) 
results = []

for grp in ["1", "2", "3"]:
    df_grp = df[df["Group"] == grp].copy()
    n_pat = df_grp["Patient_id"].nunique()

    if n_pat < 10:
        print(f"Skipping Group {grp} (too few patients: {n_pat})")
        continue

    df_grp = (df_grp.groupby("Patient_id", as_index=False)
                    .agg(
                        Human_Prediction=("Human_Prediction", "mean"),
                        AI_Prediction=("AI_Prediction", "mean"),
                        Outcome=("Outcome", "first") if "Outcome" in df_grp.columns else ("Patient_id", "size"),
                    ))

    scores = df_grp[["Human_Prediction", "AI_Prediction"]].to_numpy(float)

    icc_point = icc_3_1(scores)
    boot = bootstrap_icc31(scores, n_boot=1000, seed=42)

    results.append({
        "Outcome": df_grp["Outcome"].iloc[0] if "Outcome" in df_grp.columns else "HOSP_LOS",
        "Group": grp,
        "N_patients": int(scores.shape[0]),
        "ICC_point": icc_point,
        "ICC_boot_mean": boot["mean"],
        "ICC_boot_SD": boot["sd"],
        "ICC_boot_95CI": f'{boot["ci_low"]:.3f}–{boot["ci_high"]:.3f}',
        "N_boot_success": boot["n_ok"],
    })

icc_results = pd.DataFrame(results)
print(icc_results)
