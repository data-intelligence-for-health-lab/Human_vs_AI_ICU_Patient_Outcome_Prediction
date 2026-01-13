import numpy as np
import pandas as pd

df = pd.read_csv("processed_data/final_LOS_prosp_compare_df.csv")
df["n_clinicians"] = df.groupby("MRN")["Clinician Role"].transform("nunique")

def icc_1_1(scores: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    n, k = scores.shape
    if n < 2 or k < 2:
        return np.nan

    mean_subject = scores.mean(axis=1)
    grand_mean = scores.mean()

    # Mean Squares (one-way ANOVA)
    MS_between = (k * np.sum((mean_subject - grand_mean) ** 2)) / (n - 1)
    MS_within = np.sum((scores - mean_subject[:, None]) ** 2) / (n * (k - 1))

    denom = MS_between + (k - 1) * MS_within
    if denom == 0:
        return np.nan

    return (MS_between - MS_within) / denom


def build_score_matrix(df: pd.DataFrame, outcome_col: str, k: int) -> np.ndarray:
    mats = []
    for mrn, g in df.groupby("MRN"):
        vals = g[outcome_col].dropna().to_numpy(dtype=float)
        if len(vals) >= k:
            mats.append(vals[:k])
    if len(mats) == 0:
        return np.empty((0, k), dtype=float)
    return np.vstack(mats)


def icc_by_clinician_count_bootstrap(
    df: pd.DataFrame,
    outcome_col: str,
    min_patients: int = 5,
    n_boot: int = 1000,
    seed: int = 123
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    # number of clinician rows per patient
    counts = df.groupby("MRN").size()
    possible_k = sorted(counts.unique())

    out = []
    for k in possible_k:
        mrns_k = counts[counts == k].index
        df_k = df[df["MRN"].isin(mrns_k)].copy()

        scores = build_score_matrix(df_k, outcome_col, k)
        n_patients = scores.shape[0]

        if n_patients < min_patients or k < 2:
            continue

        icc_hat = icc_1_1(scores)

        boot = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            idx = rng.integers(0, n_patients, size=n_patients)
            boot[b] = icc_1_1(scores[idx, :])

        icc_sd = np.nanstd(boot, ddof=1)

        out.append({
            "Outcome": outcome_col,
            "Clinicians_per_patient": int(k),
            "N_patients": int(n_patients),
            "ICC_1_1": float(icc_hat),
            "ICC_SD_bootstrap": float(icc_sd),
        })

    return pd.DataFrame(out)

res_icu  = icc_by_clinician_count_bootstrap(df, "ICU_LOS_human",  min_patients=5, n_boot=1000, seed=123)
res_hosp = icc_by_clinician_count_bootstrap(df, "HOSP_LOS_human", min_patients=5, n_boot=1000, seed=123)

final_icc_df = pd.concat([res_icu, res_hosp], ignore_index=True)
print(final_icc_df)