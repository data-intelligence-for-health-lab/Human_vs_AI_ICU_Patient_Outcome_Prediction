import optuna
import torch
import numpy as np
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import SelectFromModel
from train_scripts.pipeline_components import make_preprocessor, check_imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import roc_auc_score
from train_scripts.neural_network_models import NeuralNetClassifier, NeuralNetRegressor
from train_scripts.hyperparameter_search_spaces import get_lr_c_search_space, get_svm_c_search_space, get_rf_c_search_space, get_xgb_c_search_space, get_nn_c_search_space, get_en_r_search_space, get_svm_r_search_space, get_rf_r_search_space, get_xgb_r_search_space, get_nn_r_search_space

SEED = 42

classification_study_storage = 'sqlite:///optuna_experiments_classifications.db'
regression_study_storage = 'sqlite:///optuna_experiments_regressions.db'

def tune_c_hyperparameters(X_train, y_train, mtype, target, mname, run_id):
    """Tunes hyperparameters for a classification model using Optuna."""
    print("Tuning classification model...")

    study = optuna.create_study(
        sampler=TPESampler(seed=SEED),
        study_name=f"{run_id}_{mtype}_{target}_{mname}",
        storage=classification_study_storage,
        direction='maximize',
        load_if_exists=True
    )

    study.optimize(lambda trial: c_objective(trial, X_train, y_train, mname), n_trials=200)

    best_params = study.best_trial.params
    return best_params

def tune_r_hyperparameters(X_train, y_train, mtype, target, mname, run_id):
    """Tunes hyperparameters for a regression model using Optuna."""
    print("Tuning regression model...")

    study = optuna.create_study(
        sampler=TPESampler(seed=SEED),
        study_name=f"{run_id}_{mtype}_{target}_{mname}",
        storage=regression_study_storage,
        direction='minimize', 
        load_if_exists=True
    )

    study.optimize(lambda trial: r_objective(trial, X_train, y_train, mname), n_trials=200)

    best_params = study.best_trial.params
    return best_params

# Create custom scoring function using predicted probabilities
def auc_scorer(estimator, X, y):
    # Ensure the estimator supports `predict_proba`
    y_pred_prob = estimator.predict_proba(X)[:, 1]  # Get probabilities for the positive class
    return roc_auc_score(y, y_pred_prob)

# ==================== Classification Objective Function ======================== #
def c_objective(trial, X_train, y_train, mname):
    """Objective function for Optuna hyperparameter tuning."""
    # Define preprocessing steps
    preprocessor = make_preprocessor(X_train)
    over_sampler = SMOTE(random_state=SEED, sampling_strategy=0.5)
    under_sampler = RandomUnderSampler(random_state=SEED, sampling_strategy=1.0)
    feature_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=SEED), 
        max_features=100
    )

    #Define model search spaces
    model_search_spaces = {
        'LR_C': (LogisticRegression, get_lr_c_search_space),
        'SVM_C': (SVC, get_svm_c_search_space),
        'RF_C': (RandomForestClassifier, get_rf_c_search_space),
        'XGB_C': (XGBClassifier, get_xgb_c_search_space),
        'NN_C': (NeuralNetClassifier, get_nn_c_search_space)
    }

    model_class, param_func = model_search_spaces[mname]
    model_params = param_func(trial)
    model = model_class(**model_params)

    # Check if the dataset is imbalanced
    if check_imbalance(y_train):
        pipeline_steps = [
            ('preprocessor', preprocessor),
            ('selectfeatures', feature_selector),
            ('over_sampler', over_sampler),
            ('under_sampler', under_sampler),
            ('model', model)
        ]
    else:
        print("⚠ No significant imbalance detected in training set. Removing samplers.")
        pipeline_steps = [
            ('preprocessor', preprocessor),
            ('selectfeatures', feature_selector),
            ('model', model)
        ]

    pipeline = Pipeline(pipeline_steps)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    
    if mname == "NN_C":
        scorer = auc_scorer
    else:
        scorer = 'roc_auc'

    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scorer, n_jobs=-1, error_score='raise')

    return np.median(scores)
    
# ==================== Regression Objective Function ======================== #
def r_objective(trial, X_train, y_train, mname):
    """Objective function for Optuna hyperparameter tuning for regression models."""

    # Define preprocessing steps
    preprocessor = make_preprocessor(X_train)
    feature_selector = SelectFromModel(
        RandomForestRegressor(n_estimators=100, random_state=SEED),
        max_features=100
    )

    # Define model search spaces
    model_search_spaces = {
        'EN_R': (ElasticNet, get_en_r_search_space),
        'SVM_R': (SVR, get_svm_r_search_space),
        'RF_R': (RandomForestRegressor, get_rf_r_search_space),
        'XGB_R': (XGBRegressor, get_xgb_r_search_space),
        'NN_R': (NeuralNetRegressor, get_nn_r_search_space)
    }

    # Select model and its hyperparameters
    model_class, param_func = model_search_spaces[mname]
    model_params = param_func(trial)
    model = model_class(**model_params)

    # Define pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selectfeatures', feature_selector),
        ('model', model)
    ])

    # Cross-validation setup
    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)

    # Use MSE as the scoring metric
    scoring_metric = 'neg_mean_squared_error'

    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring_metric, n_jobs=-1, error_score='raise')

    return -np.median(scores)