import os
from optuna import trial
import numpy as np
#=============================== Classification Search Spaces=================================#
'''
Classification Models:
    Logistic Regression
    Support Vector Machine (SVM-C)
    XGBoost Classification
    Random Forest
    Keras Neural Network
'''
# def get_lr_c_search_space(trial):
#     return {
#         "solver": trial.suggest_categorical("solver", ("lbfgs", "liblinear", "saga")),
#         "penalty": trial.suggest_categorical("penalty", ("l1", "l2", "elasticnet")),
#         "C": trial.suggest_float("C", 1e-4, 10, log=True),
#         "max_iter": trial.suggest_int("max_iter", 100, 5000),
#     }

def get_lr_c_search_space(trial):
    return {
        'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
        'solver': trial.suggest_categorical('solver', ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga']),
        'penalty': 'l2', 
        'max_iter': trial.suggest_int('max_iter', 100, 4000, step=100)
    }

def get_rf_c_search_space(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000, step=50),
        'max_depth': trial.suggest_int('max_depth', 10, 1500),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'criterion': trial.suggest_categorical('criterion', ['entropy', 'gini'])}

def get_svm_c_search_space(trial):
    return {
        'C': trial.suggest_float('C', 1e-5, 10, log=True),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid']),
        # 'gamma': trial.suggest_float('gamma', np.exp(0), np.exp(20), log=True),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
    }

def get_xgb_c_search_space(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 10, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'lambda': trial.suggest_float('lambda', 1e-4, 10.0, log=True)
    }


def get_nn_c_search_space(trial):
    return {
        'num_layers': trial.suggest_int('num_layers', 1, 5),  # Number of hidden layers (1 to 5 layers)
        'neurons_per_layer': [trial.suggest_int(f'neurons_layer_{i}', 16, 512, step=16) for i in range(5)],
        'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),  # Dropout rate (0.0 to 0.5)
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),  # Log scale for learning rate
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),  # Batch size options
        'epochs': trial.suggest_int('epochs', 10, 100),  # Number of epochs (10 to 100)
        'activation_name': trial.suggest_categorical('activation_name', ['relu', 'tanh', 'sigmoid']),  # Activation function choice
        'optimizer_name': trial.suggest_categorical('optimizer_name', ['adam', 'rmsprop', 'sgd'])  # Optimizer choice
    }
#=============================== Regression Search Spaces=================================#
'''
Regression Models:
    Elastic Net
    Support Vector Machine for Regression (SVM-R)
    XGBoost Regression
    Random Forest
    Keras Neural Network
'''
def get_en_r_search_space(trial):
    return {
        'alpha': trial.suggest_float('alpha', 1e-5, 10, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0, 1)
    }

def get_svm_r_search_space(trial):
    return {
        'C': trial.suggest_float('C', 1e-5, 10, log=True),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid']),
        'gamma': trial.suggest_float('gamma', 1e-5, 10, log=True)
    }

def get_xgb_r_search_space(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'lambda': trial.suggest_float('lambda', 1e-4, 10.0, log=True)
    }

def get_rf_r_search_space(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000, step=50),
        'max_depth': trial.suggest_int('max_depth', 10, 1500),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'criterion': trial.suggest_categorical('criterion', ['mse', 'mae'])
    }

def get_nn_r_search_space(trial):
    return {
        'num_layers': trial.suggest_int('num_layers', 1, 5),
        'units': trial.suggest_int('units', 16, 512, step=16),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    }
