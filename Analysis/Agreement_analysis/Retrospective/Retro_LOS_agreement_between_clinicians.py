import numpy as np
import pandas as pd
from retrospective_comparisons.secondary_analysis.reg_agreement_utils import bootstrap_icc31
df = pd.read_csv("prepared_human_predictions/human_all_outcomes.csv")

df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")


# BUILD DATASET FOR n CLINICIANS PER PATIENT
def build_exact_k_dataset(df, outcome, k):

    df_out = df[df["Outcome"] == outcome].copy()

    # Count clinicians per patient
    clinician_counts = (
        df_out.groupby("Patient_id")["record_id"]
        .nunique()
    )

    keep_patients = clinician_counts[clinician_counts == k].index
    df_k = df_out[df_out["Patient_id"].isin(keep_patients)]

    # Patient × Clinician
    wide = df_k.pivot_table(
        index="Patient_id",
        columns="record_id",
        values="prediction",
        aggfunc="mean"
    )

    wide = wide.dropna(axis=0, how="any")

    return wide


#Run analysis
results = []

for outcome_label, outcome_key in [
    ("Hospital LOS", "hosp_stay"),
    ("ICU LOS", "icu_stay"),
]:
    for k in [2, 12]:
        wide = build_exact_k_dataset(df, outcome_key, k)

        if wide.shape[0] < 2:
            print(f"Skipping {outcome_label}, {k} clinicians (insufficient data)")
            continue

        icc_mean, icc_sd = bootstrap_icc31(wide.values)

        results.append({
            "Outcome": outcome_label,
            "Group": f"{k} Clinicians",
            "Patients, No.": wide.shape[0],
            "ICC (3,1) (SD)": f"{icc_mean:.2f} ({icc_sd:.2f})"
        })


# FINAL TABLE
icc_results = pd.DataFrame(results)
print(icc_results)