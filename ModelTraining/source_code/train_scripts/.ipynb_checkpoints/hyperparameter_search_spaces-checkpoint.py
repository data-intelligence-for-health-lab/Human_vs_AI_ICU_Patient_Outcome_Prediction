import os
from optuna import trial


def get_rf_search_space(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000, step=50),
        'max_depth': trial.suggest_int('max_depth', 10, 1500),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'criterion': trial.suggest_categorical('criterion', ['entropy', 'gini'])}


