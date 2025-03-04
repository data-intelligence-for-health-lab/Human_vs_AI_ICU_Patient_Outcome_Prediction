import optuna
import torch
import numpy as np
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import average_precision_score
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
from train_scripts.neural_network_models import NeuralNetClassifier
from train_scripts.hyperparameter_search_spaces import get_lr_c_search_space, get_svm_c_search_space, get_rf_c_search_space, get_xgb_c_search_space, get_nn_c_search_space

SEED = 42

classification_study_storage = 'sqlite:///optuna_experiments_classifications.db'
regression_study_storage = 'sqlite:///optuna_experiments_regressions.db'

def tune_c_hyperparameters(X_train, y_train, mtype, target, mname, run_id):
    print("3.1 Tuning classification model...")

    study = optuna.create_study(
        sampler=TPESampler(seed=SEED),
        study_name=f"{run_id}_{mtype}_{target}_{mname}",
        storage=classification_study_storage,
        direction='maximize',
        load_if_exists=True
    )

    study.optimize(lambda trial: c_objective(trial, X_train, y_train, mname), n_trials=1)

    best_params = study.best_trial.params
    return best_params

def tune_r_hyperparameters(X_train, y_train, mtype, target, mname, run_id):
    print("3.2 Tunning regression model...")
    #objectives = {'EN_R': en_r_objective, 'SVM_R': svm_r_objective,'RF_R': rf_r_objective, 'XGB_R': xgb_r_objective, 'NN_R': nn_r_objective}
    objectives = {'EN_R': r_objective}
    study = optuna.create_study(sampler=TPESampler(seed=SEED), study_name=f'{run_id}_{mtype}_{target}_{mname}', storage=regression_study_storage, direction='minimizing', load_if_exists=True)
    study.optimize(lambda trial: objectives.get(mname)(trial, X_train, y_train), n_trials=200)
    best_params = study.best_trial.params
    return best_params

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

    # Define model search spaces
    model_search_spaces = {
        'LR_C': (LogisticRegression, get_lr_c_search_space),
        'SVM_C': (SVC, get_svm_c_search_space),
        'RF_C': (RandomForestClassifier, get_rf_c_search_space),
        'XGB_C': (XGBClassifier, get_xgb_c_search_space),
    }

    # Handle Neural Network Classifier separately
    if mname == 'NN_C':
        model_params = get_nn_c_search_space(trial)

        # Preprocess data and apply feature selection
        X_train, y_train, _ = apply_preprocessor(X_train, y_train, num_features=100, random_state=SEED)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train = torch.tensor(y_train, dtype=torch.long, device=device)

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        scores = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            model = NeuralNetClassifier(
                input_size=X_train.shape[1],  
                num_layers=model_params['num_layers'],
                neurons_per_layer=model_params['neurons_per_layer'],
                dropout_rate=model_params['dropout_rate'],
                batch_size=model_params['batch_size'],
                epochs=model_params['epochs'],
                activation_name=model_params['activation_name'],
                optimizer_name=model_params['optimizer_name'],
                learning_rate=model_params['learning_rate']
            )

            model.fit(X_train_fold, y_train_fold)
            y_val_pred_proba = model.predict_proba(X_val_fold)[:, 1]

            if isinstance(y_val_fold, torch.Tensor):
                y_val_fold = y_val_fold.cpu().numpy()

            scores.append(average_precision_score(y_val_fold, y_val_pred_proba))

        return np.median(scores)

    # Handle all other models
    elif mname in model_search_spaces:
        model_class, param_func = model_search_spaces[mname]
        model_params = param_func(trial)
        model = model_class(**model_params)

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('selectfeatures', feature_selector),
            ('over_sampler', over_sampler),
            ('under_sampler', under_sampler),
            ('model', model)
        ])

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

        return np.median(scores)

    else:
        raise ValueError(f"Invalid model name '{mname}'. Please use a valid model.")
    
# ==================== Regression Objective Function ======================== #
def r_objective(trial, X_train, y_train, mname):
    preprocessor = make_preprocessor(X_train)
    over_sampler = SMOTE(random_state=SEED, sampling_strategy=0.5)
    under_sampler = RandomUnderSampler(random_state=SEED, sampling_strategy=1.0)
    feature_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=SEED), 
        max_features=100
    )

    if mname == 'LR_C':
        model_params = get_lr_c_search_space(trial)
        model = LogisticRegression(**model_params, random_state=SEED, n_jobs=-1)
    elif mname == 'SVM_C':
        model_params = get_svm_c_search_space(trial)
        model = SVC(probability=True, random_state=SEED, **model_params)
    elif mname == 'RF_C':
        model_params = get_rf_c_search_space(trial)
        model = RandomForestClassifier(random_state=SEED, **model_params)
    elif mname == 'XGB_C':
        model_params = get_xgb_c_search_space(trial)
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **model_params)
    elif mname == 'NN_C':
        model_params = get_nn_c_search_space(trial)
        # Preprocess data and select features
        X_train, y_train, selected_features = apply_preprocessor(X_train, y_train, num_features=100, random_state=SEED)
        print("Selected features:", selected_features)

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            # Build and train the model
            model = NeuralNetClassifier(
                input_size=X_train.shape[1],  
                num_layers=model_params['num_layers'],
                neurons_per_layer=model_params['neurons_per_layer'],
                dropout_rate=model_params['dropout_rate'],
                batch_size=model_params['batch_size'],
                epochs=model_params['epochs'],
                activation_name=model_params['activation_name'],
                optimizer_name=model_params['optimizer_name'],
                learning_rate=model_params['learning_rate']
            )

            #fit model
            model.fit(X_train_fold, y_train_fold)

            y_val_pred_proba = model.predict_proba(X_val_fold)[:, 1]

            if isinstance(y_val_fold, torch.Tensor):
                y_val_fold = y_val_fold.cpu().numpy()

            score = average_precision_score(y_val_fold, y_val_pred_proba)
            scores.append(score)

        median_cv_score = np.median(scores)
    else:
        raise ValueError(f"Model name '{mname}' is not recognized. Please use a valid model name.")

    if mname in ('LR_C', 'SVM_C', 'RF_C', 'XGB_C'):
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('selectfeatures', feature_selector),
            ('over_sampler', over_sampler),
            ('under_sampler', under_sampler),
            ('model', model)
        ])
    
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        median_cv_score = np.median(scores)
    else:
        raise ValueError(f"Invalid model name '{mname}'. Please use a valid model.")

    return median_cv_score