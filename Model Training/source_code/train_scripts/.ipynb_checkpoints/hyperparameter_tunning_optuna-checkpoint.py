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
from train_scripts.hyperparameter_search_spaces import get_lr_c_search_space, get_svm_c_search_space, get_rf_c_search_space, get_xgb_c_search_space, get_nn_c_search_space, get_xgb_r_search_space
from train_scripts.neural_network_models import NeuralNetClassifier
# from source_code.modelling.training_scripts.torch_neural_network import TorchNNClassifier
# from source_code.modelling.training_scripts.utils import set_seeds
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



SEED = 42

classification_study_storage = 'sqlite:///optuna_experiments_classifications.db'
regression_study_storage = 'sqlite:///optuna_experiments_regression.db'

def tune_c_hyperparameters(X_train, y_train, mtype, target, mname, run_id):
    print("3.1 Tunning classification model...")
    # objectives = {'lr_c': lr_c_objective, 'svm_c': svm_c_objective, 'xgboost_c': xgb_c_objective, 'rf_c':rf_c_objective, 'torch_nn_c': nn_c_objective}
    objectives = {'LR_C': lr_c_objective, 'SVM_C': svm_c_objective, 'RF_C': rf_c_objective, 'XGB_C': xgb_c_objective, 'NN_C': nn_c_objective}
    study = optuna.create_study(sampler=TPESampler(seed=SEED), study_name=f'{run_id}_{mtype}_{target}_{mname}', storage=classification_study_storage, direction='maximize', load_if_exists=True)
    study.optimize(lambda trial: objectives.get(mname)(trial, X_train, y_train), n_trials=200)
    #best_value = study.best_trial.value
    best_params = study.best_trial.params
    return best_params


def tune_r_hyperparameters(X_train, y_train, mtype, target, mname, run_id):
    print("3.2 Tunning regression model...")
    #objectives = {'EN_R': en_r_objective, 'SVM_R': svm_r_objective,'RF_R': rf_r_objective, 'XGB_R': xgb_r_objective, 'NN_R': nn_r_objective}
    objectives = {'EN_R': en_r_objective}
    study = optuna.create_study(sampler=TPESampler(seed=SEED), study_name=f'{run_id}_{mtype}_{target}_{mname}', storage=regression_study_storage, direction='maximize', load_if_exists=True)
    study.optimize(lambda trial: objectives.get(mname)(trial, X_train, y_train), n_trials=200)
    best_params = study.best_trial.params
    return best_params


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

# def nn_c_objective(trial, X_train, y_train):
#     X_train, y_train, selected_features = apply_preprocessor(X_train, y_train, num_features=100, random_state=SEED)
#     print("Selected features:", selected_features)

#     # Get model hyperparameters from search space
#     model_params = get_nn_c_search_space(trial)
    
#     # Convert data to PyTorch tensors
#     print(type(X_train))
#     X_train_tensor = X_train.to(dtype=torch.float32)
#     # X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)

#     y_train_tensor = y_train.to(dtype=torch.float32)

#     # Cross-validation setup
#     cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
#     scores = []

#     for train_idx, val_idx in cv.split(X_train, y_train):
#         X_train_fold, X_val_fold = X_train_tensor[train_idx], X_train_tensor[val_idx]
#         y_train_fold, y_val_fold = y_train_tensor[train_idx], y_train_tensor[val_idx]

#         # Create DataLoader
#         train_dataset = TensorDataset(X_train_fold, y_train_fold)
#         train_loader = DataLoader(train_dataset, batch_size=model_params["batch_size"], shuffle=True)

#         # Initialize model
#         model = NeuralNetClassifier(**model_params)
#         model.train()

#         # Define loss function and optimizer
#         criterion = nn.BCEWithLogitsLoss()  
#         optimizer = optim.Adam(model.parameters(), lr=model_params["learning_rate"])

#         # Training loop
#         for epoch in range(model_params["epochs"]):
#             for X_batch, y_batch in train_loader:
#                 optimizer.zero_grad()
#                 y_pred = model(X_batch)
#                 loss = criterion(y_pred, y_batch)
#                 loss.backward()
#                 optimizer.step()

#         # Validation
#         model.eval()
#         with torch.no_grad():
#             y_val_pred = model(X_val_fold).numpy()
        
#         # Compute Average Precision Score
#         score = average_precision_score(y_val_fold.numpy(), y_val_pred)
#         scores.append(score)

#     return np.median(scores)

def nn_c_objective(trial, X_train, y_train):
    X_train, y_train, selected_features = apply_preprocessor(X_train, y_train, num_features=100, random_state=SEED)
    print("Selected features:", selected_features)

    # Get model hyperparameters from search space
    model_params = get_nn_c_search_space(trial)

    
    # # Cross-validation setup
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    scores = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train_tensor[train_idx], X_train_tensor[val_idx]
        y_train_fold, y_val_fold = y_train_tensor[train_idx], y_train_tensor[val_idx]

        # Create DataLoaders
        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        train_loader = DataLoader(train_dataset, batch_size=model_params["batch_size"], shuffle=True)

        val_dataset = TensorDataset(X_val_fold, y_val_fold)
        val_loader = DataLoader(val_dataset, batch_size=model_params["batch_size"], shuffle=False)

        # Initialize model
        model = NeuralNetClassifier(
            input_size=X_train.shape[1],  # Ensure input_size is correct
            hidden_units=model_params["hidden_units"],
            num_layers=model_params["num_layers"],
            dropout_rate=model_params["dropout_rate"]
        )
        model.train()

        # Define loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=model_params["learning_rate"])

        # Training loop
        for epoch in range(model_params["epochs"]):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

        # Validation loop
        model.eval()
        y_val_pred = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                y_val_pred.extend(preds)

        y_val_pred = np.array(y_val_pred)

        # Compute Average Precision Score
        score = average_precision_score(y_val_fold.numpy(), y_val_pred)
        scores.append(score)

    return np.median(scores)
# import optuna
# # import random
# # import torch
# # import json
# # import os
# import numpy as np
# # import pandas as pd
# # from datetime import datetime
# from optuna.samplers import TPESampler
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import make_scorer, average_precision_score
# from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_selection import SelectFromModel
# from train_scripts.pipeline_components import make_preprocessor, apply_preprocessor
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline
# from imblearn.under_sampling import RandomUnderSampler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier
# from sklearn.svm import SVC
# from train_scripts.hyperparameter_search_spaces import get_lr_c_search_space, get_svm_c_search_space, get_rf_c_search_space, get_xgb_c_search_space, get_nn_c_search_space, get_xgb_r_search_space
# from train_scripts.neural_network_models import NeuralNetClassifier
# # from source_code.modelling.training_scripts.torch_neural_network import TorchNNClassifier
# # from source_code.modelling.training_scripts.utils import set_seeds
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset



# SEED = 42

# classification_study_storage = 'sqlite:///optuna_experiments_classifications.db'
# regression_study_storage = 'sqlite:///optuna_experiments_regression.db'

# def tune_c_hyperparameters(X_train, y_train, mtype, target, mname, run_id):
#     print("3.1 Tunning classification model...")
#     # objectives = {'lr_c': lr_c_objective, 'svm_c': svm_c_objective, 'xgboost_c': xgb_c_objective, 'rf_c':rf_c_objective, 'torch_nn_c': nn_c_objective}
#     objectives = {'LR_C': lr_c_objective, 'SVM_C': svm_c_objective, 'RF_C': rf_c_objective, 'XGB_C': xgb_c_objective, 'NN_C': nn_c_objective}
#     study = optuna.create_study(sampler=TPESampler(seed=SEED), study_name=f'{run_id}_{mtype}_{target}_{mname}', storage=classification_study_storage, direction='maximize', load_if_exists=True)
#     study.optimize(lambda trial: objectives.get(mname)(trial, X_train, y_train), n_trials=200)
#     #best_value = study.best_trial.value
#     best_params = study.best_trial.params
#     return best_params


# def tune_r_hyperparameters(X_train, y_train, mtype, target, mname, run_id):
#     print("3.2 Tunning regression model...")
#     #objectives = {'EN_R': en_r_objective, 'SVM_R': svm_r_objective,'RF_R': rf_r_objective, 'XGB_R': xgb_r_objective, 'NN_R': nn_r_objective}
#     objectives = {'EN_R': en_r_objective}
#     study = optuna.create_study(sampler=TPESampler(seed=SEED), study_name=f'{run_id}_{mtype}_{target}_{mname}', storage=regression_study_storage, direction='maximize', load_if_exists=True)
#     study.optimize(lambda trial: objectives.get(mname)(trial, X_train, y_train), n_trials=200)
#     best_params = study.best_trial.params
#     return best_params


# # def rf_objective(trial, X_train, y_train):
# #     preprocessor = [INSERT PREPROCESSOR]
# #     pca = [INSERT PCA]
# #     oversampler = SMOTE(random_state=seed, sampling_strategy='minority')
# #     model_params = get_rf_search_space(trial)
# #     model = RandomForestClassifier(**model_params, n_jobs=-1, random_state=42)
    
# #     #if outcome = then pop 
# #     pipeline = Pipeline([
# #         ('preprocessor', preprocessor),
# #         ('pca', pca),
# #         ('oversampler', oversampler),
# #         ('classifier', model)])
# #     cv = StratifiedKFold(n_splits=10 [CHANGE THIS IF YOU WANT], shuffle=True, random_state=42)
# #     scoring = make_scorer(average_precision_score) [USE A DIFFERENT METRIC IF YOU WANT]
# #     scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
# #     median_cv_score = np.median(scores)
# #     return median_cv_score

# #==================== Classification Objectives ========================#
# def lr_c_objective(trial, X_train, y_train):
#     preprocessor = make_preprocessor(X_train)
#     over_sampler = SMOTE(random_state=SEED, sampling_strategy=0.5)
#     under_sampler = RandomUnderSampler(random_state=SEED, sampling_strategy=1.0)
#     feature_selector = SelectFromModel(
#         RandomForestClassifier(n_estimators=100, random_state=SEED), 
#         max_features=100
#     )
    
#     model_params = get_lr_c_search_space(trial)
#     model = LogisticRegression(**model_params, random_state=SEED, n_jobs=-1)

#     pipeline = Pipeline([
#         ('preprocessor', preprocessor),
#         ('selectfeatures', feature_selector),
#         ('over_sampler', over_sampler),
#         ('under_sampler', under_sampler),
#         ('model', model)
#     ])
    
#     cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED) 
#     scoring = make_scorer(average_precision_score)
    
#     scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
#     median_cv_score = np.median(scores)
    
#     return median_cv_score

# def svm_c_objective(trial, X_train, y_train):
#     # Preprocessing
#     preprocessor = make_preprocessor(X_train)
#     over_sampler = SMOTE(random_state=SEED, sampling_strategy=0.5)
#     under_sampler = RandomUnderSampler(random_state=SEED, sampling_strategy=1.0)

#     # Feature selection using RandomForest
#     feature_selector = SelectFromModel(
#         RandomForestClassifier(n_estimators=100, random_state=SEED), 
#         max_features=100
#     )

#     model_params = get_svm_c_search_space(trial)
#     # Define the SVM model with the suggested hyperparameters
#     model = SVC(probability=True, random_state=SEED, **model_params)

#     # Construct the pipeline
#     pipeline = Pipeline([
#         ('preprocessor', preprocessor),
#         ('selectfeatures', feature_selector),
#         ('over_sampler', over_sampler),
#         ('under_sampler', under_sampler),
#         ('model', model)
#     ])
    
#     # Cross-validation setup
#     cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED) 
#     scoring = make_scorer(average_precision_score)

#     # Calculate cross-validation scores
#     scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
#     median_cv_score = np.median(scores)
    
#     return median_cv_score

# def xgb_c_objective(trial, X_train, y_train):
#     # Preprocessing
#     preprocessor = make_preprocessor(X_train)
#     over_sampler = SMOTE(random_state=SEED, sampling_strategy=0.5)
#     under_sampler = RandomUnderSampler(random_state=SEED, sampling_strategy=1.0)

#     # Feature selection using RandomForest
#     feature_selector = SelectFromModel(
#         RandomForestClassifier(n_estimators=100, random_state=SEED), 
#         max_features=100
#     )

#     model_params = get_svm_c_search_space(trial)
#     # Define the SVM model with the suggested hyperparameters
#     model = model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **model_params)

#     # Construct the pipeline
#     pipeline = Pipeline([
#         ('preprocessor', preprocessor),
#         ('selectfeatures', feature_selector),
#         ('over_sampler', over_sampler),
#         ('under_sampler', under_sampler),
#         ('model', model)
#     ])
    
#     # Cross-validation setup
#     cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED) 
#     scoring = make_scorer(average_precision_score)

#     # Calculate cross-validation scores
#     scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
#     median_cv_score = np.median(scores)
    
#     return median_cv_score

# def rf_c_objective(trial, X_train, y_train):
#     # Preprocessing
#     preprocessor = make_preprocessor(X_train)
#     over_sampler = SMOTE(random_state=SEED, sampling_strategy=0.5)
#     under_sampler = RandomUnderSampler(random_state=SEED, sampling_strategy=1.0)

#     # Feature selection using RandomForest
#     feature_selector = SelectFromModel(
#         RandomForestClassifier(n_estimators=100, random_state=SEED),
#         max_features=100
#     )

#     # Hyperparameter search space for RandomForest
#     model_params = get_rf_c_search_space(trial)

#     # Define the RandomForest model with the suggested hyperparameters
#     model = RandomForestClassifier(random_state=SEED, **model_params)

#     # Construct the pipeline
#     pipeline = Pipeline([
#         ('preprocessor', preprocessor),
#         ('selectfeatures', feature_selector),
#         ('over_sampler', over_sampler),
#         ('under_sampler', under_sampler),
#         ('model', model)
#     ])
    
#     # Cross-validation setup
#     cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
#     scoring = make_scorer(average_precision_score)

#     # Calculate cross-validation scores
#     scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
#     median_cv_score = np.median(scores)

#     return median_cv_score

# # def nn_c_objective(trial, X_train, y_train):
# #     X_train, y_train, selected_features = apply_preprocessor(X_train, y_train, num_features=100, random_state=SEED)
# #     print("Selected features:", selected_features)

# #     # Get model hyperparameters from search space
# #     model_params = get_nn_c_search_space(trial)
    
# #     # Convert data to PyTorch tensors
# #     print(type(X_train))
# #     X_train_tensor = X_train.to(dtype=torch.float32)
# #     # X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)

# #     y_train_tensor = y_train.to(dtype=torch.float32)

# #     # Cross-validation setup
# #     cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
# #     scores = []

# #     for train_idx, val_idx in cv.split(X_train, y_train):
# #         X_train_fold, X_val_fold = X_train_tensor[train_idx], X_train_tensor[val_idx]
# #         y_train_fold, y_val_fold = y_train_tensor[train_idx], y_train_tensor[val_idx]

# #         # Create DataLoader
# #         train_dataset = TensorDataset(X_train_fold, y_train_fold)
# #         train_loader = DataLoader(train_dataset, batch_size=model_params["batch_size"], shuffle=True)

# #         # Initialize model
# #         model = NeuralNetClassifier(**model_params)
# #         model.train()

# #         # Define loss function and optimizer
# #         criterion = nn.BCEWithLogitsLoss()  
# #         optimizer = optim.Adam(model.parameters(), lr=model_params["learning_rate"])

# #         # Training loop
# #         for epoch in range(model_params["epochs"]):
# #             for X_batch, y_batch in train_loader:
# #                 optimizer.zero_grad()
# #                 y_pred = model(X_batch)
# #                 loss = criterion(y_pred, y_batch)
# #                 loss.backward()
# #                 optimizer.step()

# #         # Validation
# #         model.eval()
# #         with torch.no_grad():
# #             y_val_pred = model(X_val_fold).numpy()
        
# #         # Compute Average Precision Score
# #         score = average_precision_score(y_val_fold.numpy(), y_val_pred)
# #         scores.append(score)

# #     return np.median(scores)

# def nn_c_objective(trial, X_train, y_train):
#     X_train, y_train, selected_features = apply_preprocessor(X_train, y_train, num_features=100, random_state=SEED)
#     print("Selected features:", selected_features)

#     # Get model hyperparameters from search space
#     model_params = get_nn_c_search_space(trial)

#     # Convert data to PyTorch tensors
#     X_train_tensor = torch.FloatTensor(X_train)
#     y_train_tensor = torch.FloatTensor(y_train.values)
    
#     # # Cross-validation setup
#     # cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
#     # scores = []

#     for train_idx, val_idx in cv.split(X_train, y_train):
#         X_train_fold, X_val_fold = X_train_tensor[train_idx], X_train_tensor[val_idx]
#         y_train_fold, y_val_fold = y_train_tensor[train_idx], y_train_tensor[val_idx]

#         # Create DataLoaders
#         train_dataset = TensorDataset(X_train_fold, y_train_fold)
#         train_loader = DataLoader(train_dataset, batch_size=model_params["batch_size"], shuffle=True)

#         val_dataset = TensorDataset(X_val_fold, y_val_fold)
#         val_loader = DataLoader(val_dataset, batch_size=model_params["batch_size"], shuffle=False)

#         # Initialize model
#         model = NeuralNetClassifier(
#             input_size=X_train.shape[1],  # Ensure input_size is correct
#             hidden_units=model_params["hidden_units"],
#             num_layers=model_params["num_layers"],
#             dropout_rate=model_params["dropout_rate"]
#         )
#         model.train()

#         # Define loss function and optimizer
#         criterion = nn.BCEWithLogitsLoss()
#         optimizer = optim.Adam(model.parameters(), lr=model_params["learning_rate"])

#         # Training loop
#         for epoch in range(model_params["epochs"]):
#             for X_batch, y_batch in train_loader:
#                 optimizer.zero_grad()
#                 y_pred = model(X_batch)
#                 loss = criterion(y_pred, y_batch)
#                 loss.backward()
#                 optimizer.step()

#         # Validation loop
#         model.eval()
#         y_val_pred = []
#         with torch.no_grad():
#             for X_batch, y_batch in val_loader:
#                 preds = model(X_batch)
#                 y_val_pred.extend(preds)

#         y_val_pred = np.array(y_val_pred)

#         # Compute Average Precision Score
#         score = average_precision_score(y_val_fold.numpy(), y_val_pred)
#         scores.append(score)

#     return np.median(scores)