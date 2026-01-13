import numpy as np

# ICC(3,1):
def icc_3_1(scores: np.ndarray) -> float:
    """
    n_patients × k_raters
    """
    X = np.asarray(scores, dtype=float)
    n, k = X.shape

    if n < 2 or k < 2:
        return np.nan

    mean_subject = X.mean(axis=1, keepdims=True)
    mean_rater = X.mean(axis=0, keepdims=True)
    grand_mean = X.mean()

    SS_subject = k * np.sum((mean_subject - grand_mean) ** 2)
    SS_rater   = n * np.sum((mean_rater - grand_mean) ** 2)
    SS_total   = np.sum((X - grand_mean) ** 2)
    SS_error   = SS_total - SS_subject - SS_rater

    MS_subject = SS_subject / (n - 1)
    MS_error   = SS_error / ((n - 1) * (k - 1))

    denom = MS_subject + (k - 1) * MS_error
    if denom == 0:
        return np.nan

    return (MS_subject - MS_error) / denom

def bootstrap_icc31(scores, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = scores.shape[0]

    iccs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        iccs.append(icc_3_1(scores[idx, :]))

    return np.mean(iccs), np.std(iccs, ddof=1)
