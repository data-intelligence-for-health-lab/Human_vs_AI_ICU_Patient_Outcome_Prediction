import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score
from utils import icc_1_1, build_clinician_ml_long, compute_los_icc_table

df = pd.read_csv("processed_data/final_LOS_prosp_compare_df.csv")

PATIENT_COL = "MRN"                 
CLINICIAN_ID_COL = "Clinician Role"
HOSP_HUMAN_COL = "HOSP_LOS_human"  
ICU_HUMAN_COL  = "ICU_LOS_human"   
HOSP_ML_COL    = "HOSP_LOS_PRED"    
ICU_ML_COL     = "ICU_LOS_PRED" 

for c in [HOSP_HUMAN_COL, ICU_HUMAN_COL, HOSP_ML_COL, ICU_ML_COL]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=[PATIENT_COL]).copy()
df[PATIENT_COL] = df[PATIENT_COL].astype(str)



def clinician_count_exact(df_in: pd.DataFrame,
                          patient_col: str,
                          clinician_col: str) -> pd.Series:
    # exact unique clinician count per patient
    return df_in.groupby(patient_col)[clinician_col].nunique()


los_outcomes = [
    ("Hospital LOS", HOSP_HUMAN_COL, HOSP_ML_COL),
    ("ICU LOS", ICU_HUMAN_COL, ICU_ML_COL),
]

los_icc_table = compute_los_icc_table(
    df,
    patient_col=PATIENT_COL,
    clinician_col=CLINICIAN_ID_COL,
    outcomes=los_outcomes,
    n_boot=1000,
    seed=42
)

print(los_icc_table)