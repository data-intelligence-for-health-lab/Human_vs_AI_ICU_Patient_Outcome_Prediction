import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score

df = pd.read_csv("processed_data/final_prosp_compare_df.csv")

# Ensure numeric
cols = [
    "AKI_STATUS_PRED_CLASS",
    "DELIRIUM_STATUS_PRED_CLASS",
    "HOSP_MORTALITY_PRED_CLASS",
    "30_DAY_MORTALITY_PRED_CLASS",
    "AKI_human_Num",
    "DELIRIUM_human_Num",
    "HOSP_MORT_human_Num",
    "30_DAY_MORT_human_Num",
]
for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")


def kappa_bootstrap(y_human, y_ml, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_human)

    if len(np.unique(y_human)) < 2 or len(np.unique(y_ml)) < 2:
        return np.nan, np.nan

    k_hat = cohen_kappa_score(y_human, y_ml)

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        k = cohen_kappa_score(y_human[idx], y_ml[idx])
        if not np.isnan(k):
            boots.append(k)

    return float(k_hat), float(np.std(boots, ddof=1))


outcomes = [
    ("Hospital mortality",
     "HOSP_MORTALITY_PRED_CLASS", "HOSP_MORT_human_Num"),
    ("Post-discharge mortality",
     "30_DAY_MORTALITY_PRED_CLASS", "30_DAY_MORT_human_Num"),
    ("Acute kidney injury",
     "AKI_STATUS_PRED_CLASS", "AKI_human_Num"),
    ("Delirium",
     "DELIRIUM_STATUS_PRED_CLASS", "DELIRIUM_human_Num"),
]

rows = []

for outcome_label, ml_col, human_col in outcomes:

    # number of clinicians per patient
    nclin = df.groupby("PATIENT_ID")["Clinician Role"].transform("nunique")
    df_tmp = df.copy()
    df_tmp["n_clinicians"] = nclin

    for k in [1, 2, 3]:
        d = df_tmp[df_tmp["n_clinicians"] == k][[ml_col, human_col]].dropna()

        if d.empty:
            continue

        y_ml = d[ml_col].astype(int).values
        y_human = (d[human_col] < 2).astype(int).values

        kappa, sd = kappa_bootstrap(y_human, y_ml)

        rows.append({
            "Outcome": outcome_label,
            "Group": f"{k} clinician" if k == 1 else f"{k} clinicians",
            "Patients, No.": d.shape[0],
            "Cohen’s κ (SD)": f"{kappa:.3f} ({sd:.3f})"
        })

final_table = pd.DataFrame(rows)
print(final_table)