
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


# Constants

PATIENT_COL = "MRN"
CLINICIAN_COL = "Clinician Role"
PATIENT_ID_COL = "PATIENT_ID"

# Classification outcome columns
CLASS_OUTCOMES = [
    ("Hospital mortality", "HOSP_MORT_human_Num"),
    ("Post-discharge mortality", "30_DAY_MORT_human_Num"),
    ("Acute kidney injury", "AKI_human_Num"),
    ("Delirium", "DELIRIUM_human_Num"),
]

# LOS outcome columns
LOS_OUTCOMES = [
    ("Hospital LOS", "HOSP_LOS_human", "HOSP_LOS_PRED"),
    ("ICU LOS", "ICU_LOS_human", "ICU_LOS_PRED"),
]

# ML prediction columns for classification
CLASS_ML_COLS = [
    "HOSP_MORTALITY_PRED_CLASS",
    "30_DAY_MORTALITY_PRED_CLASS",
    "AKI_STATUS_PRED_CLASS",
    "DELIRIUM_STATUS_PRED_CLASS",
]


# ICC (Intraclass Correlation Coefficient)

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


def icc_1_1_from_long(df_long: pd.DataFrame,
                       subject_col: str,
                       value_col: str) -> float:
    d = df_long[[subject_col, value_col]].dropna().copy()
    
    # Keep patients with >=2 ratings
    counts = d.groupby(subject_col).size()
    keep = counts[counts >= 2].index
    d = d[d[subject_col].isin(keep)]
    
    n = d[subject_col].nunique()
    N = d.shape[0]
    if n < 2 or N <= n:
        return np.nan
    
    subj_means = d.groupby(subject_col)[value_col].mean()
    grand_mean = d[value_col].mean()
    n_i = d.groupby(subject_col).size().astype(float)
    
    # Between-subject SS
    SS_between = float(np.sum(n_i * (subj_means - grand_mean) ** 2))
    
    # Within-subject SS
    d = d.join(subj_means.rename("mean_i"), on=subject_col)
    SS_within = float(np.sum((d[value_col] - d["mean_i"]) ** 2))
    
    df_between = n - 1
    df_within = N - n
    if df_within <= 0:
        return np.nan
    
    MS_between = SS_between / df_between
    MS_within = SS_within / df_within
    
    # Effective average number of ratings per subject (n0)
    n0 = (N - (np.sum(n_i ** 2) / N)) / (n - 1)
    
    denom = MS_between + (n0 - 1) * MS_within
    if denom == 0:
        return np.nan
    
    return (MS_between - MS_within) / denom


def bootstrap_icc_1_1(scores: np.ndarray,
                      n_boot: int = 1000,
                      seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_patients = scores.shape[0]
    
    icc_hat = icc_1_1(scores)
    
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_patients, size=n_patients)
        boots[b] = icc_1_1(scores[idx, :])
    
    icc_sd = np.nanstd(boots, ddof=1)
    
    return float(icc_hat), float(icc_sd)


def bootstrap_icc_1_1_long(df_long: pd.DataFrame,
                            subject_col: str,
                            value_col: str,
                            n_boot: int = 1000,
                            seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    
    # Only patients with >=2 ratings contribute
    counts = df_long.groupby(subject_col).size()
    keep = counts[counts >= 2].index
    d = df_long[df_long[subject_col].isin(keep)].copy()
    
    patients = d[subject_col].unique()
    n_pat = len(patients)
    if n_pat < 2:
        return (np.nan, np.nan)
    
    icc_hat = icc_1_1_from_long(d, subject_col, value_col)
    
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(patients, size=n_pat, replace=True)
        d_boot = pd.concat([d[d[subject_col] == p] for p in samp], ignore_index=True)
        boots.append(icc_1_1_from_long(d_boot, subject_col, value_col))
    
    icc_sd = np.nanstd(boots, ddof=1)
    
    return float(icc_hat), float(icc_sd)


# Cohen's Kappa

def kappa_bootstrap(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    n_boot: int = 1000,
                    seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return np.nan, np.nan
    
    k_hat = cohen_kappa_score(y_true, y_pred)
    
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        k = cohen_kappa_score(y_true[idx], y_pred[idx])
        if not np.isnan(k):
            boots.append(k)
    
    if len(boots) == 0:
        return float(k_hat), np.nan
    
    return float(k_hat), float(np.std(boots, ddof=1))



# Score Matrix Construction

def build_binary_score_matrix(df: pd.DataFrame,
                              outcome_col: str,
                              k: int,
                              patient_col: str = PATIENT_COL,
                              threshold: float = 2.0) -> np.ndarray:
    mats = []
    for pid, g in df.groupby(patient_col):
        vals = g[outcome_col].dropna().to_numpy(dtype=float)
        vals = (vals < threshold).astype(int)
        if len(vals) == k:
            mats.append(vals)
    
    if len(mats) == 0:
        return np.empty((0, k), dtype=float)
    
    return np.vstack(mats)


def build_score_matrix(df: pd.DataFrame,
                       outcome_col: str,
                       k: int,
                       patient_col: str = PATIENT_COL) -> np.ndarray:
    mats = []
    for pid, g in df.groupby(patient_col):
        vals = g[outcome_col].dropna().to_numpy(dtype=float)
        if len(vals) >= k:
            mats.append(vals[:k])
    
    if len(mats) == 0:
        return np.empty((0, k), dtype=float)
    
    return np.vstack(mats)


def build_clinician_ml_long(df: pd.DataFrame,
                            patient_col: str,
                            clinician_col: str,
                            human_los_col: str,
                            ml_los_col: str) -> pd.DataFrame:
    d = df[[patient_col, clinician_col, human_los_col, ml_los_col]].copy()
    d = d.dropna(subset=[patient_col])
    
    rows = []
    for pid, g in d.groupby(patient_col):
        # ML value (should be 1 per patient)
        ml_vals = g[ml_los_col].dropna().unique()
        if len(ml_vals) == 0:
            continue
        ml_val = ml_vals[0]
        
        # Human ratings (all clinician predictions)
        human_vals = g[human_los_col].dropna().values
        if len(human_vals) == 0:
            continue
        
        # Add clinician ratings
        for v in human_vals:
            rows.append({patient_col: pid, "rating": float(v)})
        
        # Add ML rating
        rows.append({patient_col: pid, "rating": float(ml_val)})
    
    return pd.DataFrame(rows)


# Data Preparation

def ensure_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
