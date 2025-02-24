import optuna
# import random
# import torch
# import json
# import os
import numpy as np
# import pandas as pd
# from datetime import datetime
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from train_scripts.pipeline_components import make_preprocessor, apply_preprocessor
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from train_scripts.hyperparameter_search_spaces import get_lr_c_search_space, get_svm_c_search_space, get_rf_c_search_space, get_xgb_c_search_space, get_keras_c_search_space, get_xgb_r_search_space
# from source_code.modelling.training_scripts.torch_neural_network import TorchNNClassifier
# from source_code.modelling.training_scripts.utils import set_seeds
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



SEED = 42

study_storage = 'sqlite:///optuna_experiments_classifications.db'

def tune_c_hyperparameters(X_train, y_train, mtype, target, mname, run_id):
    print("3. Tunning model...")
    # objectives = {'lr_c': lr_c_objective, 'svm_c': svm_c_objective, 'xgboost_c': xgb_c_objective, 'rf_c':rf_c_objective, 'torch_nn_c': nn_c_objective}
    objectives = {'LR_C': lr_c_objective, 'SVM_C': svm_c_objective, 'RF_C': rf_c_objective, 'XGB_C': xgb_c_objective, 'NN_C': nn_c_objective}
    study = optuna.create_study(sampler=TPESampler(seed=SEED), study_name=f'{run_id}_{mtype}_{target}_{mname}', storage=study_storage, direction='maximize', load_if_exists=True)
    study.optimize(lambda trial: objectives.get(mname)(trial, X_train, y_train), n_trials=1)
    #best_value = study.best_trial.value
    best_params = study.best_trial.params
    return best_params


# def tune_r_hyperparameters(X_train, y_train, outcome: str, days: int, run_id: str, model_algorithm: str = 'random_forest'):
#     #objectives = {'en_r': en_r_objective, 'svm_r': svm_r_objective,'xgboost': xgb_objective, 'logistic_regression': lr_objective, 'torch_nn': nn_objective}
#     objectives = {'en_r': en_r_objective}
#     study = optuna.create_study(sampler=TPESampler(seed=seed), study_name=f'{run_id}_{outcome}_{days}_{model_algorithm}', storage=study_storage, direction='minimize', load_if_exists=True)
#     study.optimize(lambda trial: objectives.get(model_algorithm)(trial, X_train, y_train, groups, selected_features), n_trials=200)
#     #best_value = study.best_trial.value
#     best_params = study.best_trial.params
#     return best_params


# def rf_objective(trial, X_train, y_train):
#     preprocessor = [INSERT PREPROCESSOR]
#     pca = [INSERT PCA]
#     oversampler = SMOTE(random_state=seed, sampling_strategy='minority')
#     model_params = get_rf_search_space(trial)
#     model = RandomForestClassifier(**model_params, n_jobs=-1, random_state=42)
    
#     #if outcome = then pop 
#     pipeline = Pipeline([
#         ('preprocessor', preprocessor),
#         ('pca', pca),
#         ('oversampler', oversampler),
#         ('classifier', model)])
#     cv = StratifiedKFold(n_splits=10 [CHANGE THIS IF YOU WANT], shuffle=True, random_state=42)
#     scoring = make_scorer(average_precision_score) [USE A DIFFERENT METRIC IF YOU WANT]
#     scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
#     median_cv_score = np.median(scores)
#     return median_cv_score

#==================== Classification Objectives ========================#
def lr_c_objective(trial, X_train, y_train):
    preprocessor = make_preprocessor(X_train)
    over_sampler = SMOTE(random_state=SEED, sampling_strategy=0.5)
    under_sampler = RandomUnderSampler(random_state=SEED, sampling_strategy=1.0)
    feature_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=SEED), 
        max_features=100
    )
    
    model_params = get_lr_c_search_space(trial)
    model = LogisticRegression(**model_params, random_state=SEED, n_jobs=-1)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selectfeatures', feature_selector),
        ('over_sampler', over_sampler),
        ('under_sampler', under_sampler),
        ('model', model)
    ])
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED) 
    scoring = make_scorer(average_precision_score)
    
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    median_cv_score = np.median(scores)
    
    return median_cv_score

def svm_c_objective(trial, X_train, y_train):
    # Preprocessing
    preprocessor = make_preprocessor(X_train)
    over_sampler = SMOTE(random_state=SEED, sampling_strategy=0.5)
    under_sampler = RandomUnderSampler(random_state=SEED, sampling_strategy=1.0)

    # Feature selection using RandomForest
    feature_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=SEED), 
        max_features=100
    )

    model_params = get_svm_c_search_space(trial)
    # Define the SVM model with the suggested hyperparameters
    model = SVC(probability=True, random_state=SEED, **model_params)

    # Construct the pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selectfeatures', feature_selector),
        ('over_sampler', over_sampler),
        ('under_sampler', under_sampler),
        ('model', model)
    ])
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED) 
    scoring = make_scorer(average_precision_score)

    # Calculate cross-validation scores
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    median_cv_score = np.median(scores)
    
    return median_cv_score

def xgb_c_objective(trial, X_train, y_train):
    # Preprocessing
    preprocessor = make_preprocessor(X_train)
    over_sampler = SMOTE(random_state=SEED, sampling_strategy=0.5)
    under_sampler = RandomUnderSampler(random_state=SEED, sampling_strategy=1.0)

    # Feature selection using RandomForest
    feature_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=SEED), 
        max_features=100
    )

    model_params = get_svm_c_search_space(trial)
    # Define the SVM model with the suggested hyperparameters
    model = model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **model_params)

    # Construct the pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selectfeatures', feature_selector),
        ('over_sampler', over_sampler),
        ('under_sampler', under_sampler),
        ('model', model)
    ])
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED) 
    scoring = make_scorer(average_precision_score)

    # Calculate cross-validation scores
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    median_cv_score = np.median(scores)
    
    return median_cv_score


def rf_c_objective(trial, X_train, y_train):
    # Preprocessing
    preprocessor = make_preprocessor(X_train)
    over_sampler = SMOTE(random_state=SEED, sampling_strategy=0.5)
    under_sampler = RandomUnderSampler(random_state=SEED, sampling_strategy=1.0)

    # Feature selection using RandomForest
    feature_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=SEED),
        max_features=100
    )

    # Hyperparameter search space for RandomForest
    model_params = get_rf_c_search_space(trial)

    # Define the RandomForest model with the suggested hyperparameters
    model = RandomForestClassifier(random_state=SEED, **model_params)

    # Construct the pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selectfeatures', feature_selector),
        ('over_sampler', over_sampler),
        ('under_sampler', under_sampler),
        ('model', model)
    ])
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    scoring = make_scorer(average_precision_score)

    # Calculate cross-validation scores
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    median_cv_score = np.median(scores)

    return median_cv_score

def nn_c_objective(trial, X_train, y_train):
    X_train, y_train, selected_features = apply_preprocessor(X_train, y_train, num_features=100, random_state=SEED)
    # print(type(X_train))
    print("selected features:", selected_features)

    # Define hyperparameter search space
    input_size = X_train.shape[1]
    hidden_units = trial.suggest_int('hidden_units', 16, 512, log=True)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    epochs = trial.suggest_int('epochs', 10, 100)
    
    # Define neural network model
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_units, num_layers, dropout_rate):
            super(NeuralNet, self).__init__()
            layers = [nn.Linear(input_size, hidden_units), nn.ReLU(), nn.Dropout(dropout_rate)]
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_units, hidden_units))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(hidden_units, 1))
            layers.append(nn.Sigmoid())
            self.model = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.model(x)
    
    # Convert data to tensors
    dataset = TensorDataset(X_train, y_train.view(-1, 1))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    model = NeuralNet(input_size, hidden_units, num_layers, dropout_rate)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    
    # Evaluate model using cross-validation
    def model_scorer(estimator, X, y):
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        with torch.no_grad():
            y_pred = estimator(X_tensor).numpy()
        return average_precision_score(y, y_pred)
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=make_scorer(model_scorer), n_jobs=-1)
    return np.median(scores)








