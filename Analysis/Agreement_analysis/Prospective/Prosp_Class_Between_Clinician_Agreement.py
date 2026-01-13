import numpy as np
import pandas as pd

from utils import (
    PATIENT_COL,
    CLASS_OUTCOMES,
    build_binary_score_matrix,
    bootstrap_icc_1_1,
    ensure_numeric,
)

# Load and prepare data
df = pd.read_csv("processed_data/final_LOS_prosp_compare_df.csv")

# Ensure numeric
outcome_cols = [col for _, col in CLASS_OUTCOMES]
df = ensure_numeric(df, outcome_cols)

df = df.dropna(subset=[PATIENT_COL]).copy()
df[PATIENT_COL] = df[PATIENT_COL].astype(str)


# Analysis
def compute_classification_icc_by_clinician_count(
    df: pd.DataFrame,
    outcome_col: str,
    min_patients: int = 5,
    n_boot: int = 1000,
    seed: int = 123,
) -> pd.DataFrame:
    # Exact clinician count per patient
    counts = df.groupby(PATIENT_COL).size()
    possible_k = sorted(counts.unique())
    
    out = []
    
    for k in possible_k:
        if k < 2:
            continue
        
        mrns_k = counts[counts == k].index
        df_k = df[df[PATIENT_COL].isin(mrns_k)].copy()
        
        scores = build_binary_score_matrix(df_k, outcome_col, k, patient_col=PATIENT_COL)
        n_patients = scores.shape[0]
        
        if n_patients < min_patients:
            continue
        
        icc_hat, icc_sd = bootstrap_icc_1_1(scores, n_boot=n_boot, seed=seed)
        
        out.append({
            "Outcome": outcome_col,
            "Clinicians_per_patient": int(k),
            "Patients, No.": int(n_patients),
            "ICC(1,1) (SD)": f"{icc_hat:.3f} ({icc_sd:.3f})" if np.isfinite(icc_hat) else "NaN",
        })
    
    return pd.DataFrame(out)


# Main
if __name__ == "__main__":
    tables = []
    
    for outcome_label, outcome_col in CLASS_OUTCOMES:
        res = compute_classification_icc_by_clinician_count(
            df,
            outcome_col=outcome_col,
            min_patients=5,
            n_boot=1000,
            seed=123,
        )
        res.insert(0, "Outcome_Label", outcome_label)
        tables.append(res)
    
    final_class_icc = pd.concat(tables, ignore_index=True)
    print(final_class_icc)