import pandas as pd
import numpy as np
import argparse
# import sys
import os
from optuna.samplers import TPESampler
# import time
# import json
# import dill
# import random
# import torch
# # from sklearn.feature_selection import SelectFromModel
# from imblearn.pipeline import make_pipeline
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import SMOTE
# from sklearn.compose import make_column_transformer
# from sklearn.impute import SimpleImputer
# # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.decomposition import PCA
# from source_code.modelling.data_cleaning_main import DataCleaner
# from source_code.modelling.training_scripts.data_splitters import split_data_main
# from source_code.modelling.training_scripts.feature_selection import select_features
# from source_code.modelling.training_scripts.generate_metrics import get_quick_metrics, get_bootstrapped_metrics
# from source_code.train_scripts.hyperparameter_tunning_optuna import tune_c_hyperparameters
#, tune_r_hyperparameters
# from source_code.modelling.training_scripts.pipeline_steps import GeneratePreprocessor, FeatureSelector
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier
# from source_code.modelling.training_scripts.static_training_variables import seed
# from source_code.modelling.training_scripts.torch_neural_network import TorchNNClassifier
# from source_code.modelling.training_scripts.utils import set_seeds
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from imblearn.over_sampling import SMOTE
import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# from dotenv import dotenv_values
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
# Construct the file path
root_folder = os.getenv("ROOT_FOLDER")
source_data = os.getenv("SOURCE_DATA")
final_df = os.getenv("FINAL_DF")
comparison_set = os.getenv("PATIENTS_1000")
common_set = os.getenv("COMMON_50")

# Full path to datasource
final_df_path = os.path.join(root_folder, source_data, final_df)
comparison_ids_path = os.path.join(root_folder, source_data, comparison_set)

seed = 42

#Data cleaning
def data_cleaner(target):
    df = pd.read_csv(final_df_path)
    compare1000 = pd.read_csv(comparison_ids_path)
    print("1) Cleaning dataset...")
    key_columns = [
    'PATIENT_ID', 'ADMISSION_ID', 'ICU_LOS', 'HOSP_LOS', 
    'ICU_EXPIRE_FLAG', 'HOSP_EXPIRE_FLAG', '30_DAYS_EXPIRE_FLAG', 
    'DELIRIUM_LABEL','AKI_LABEL'
    ]
    
    compare_ids = compare1000['PATIENT_ID']
    admission_ids = compare1000['ADMISSION_ID']
    compare_set = df[df['ADMISSION_ID'].isin(admission_ids)]
    train_test_set = df[~df['PATIENT_ID'].isin(compare_ids)]
    X = train_test_set.drop(columns=key_columns)
    y = train_test_set[target]
    return compare_set, X, y

# Train test split 
def split_dataset(X, y, seed):
    print("2) Splitting dataset...")
    # First, separate 1000 samples for testing
    X_test, X_train, y_test, y_train = train_test_split(
        X, y, test_size=None, train_size=1000, stratify=y, random_state=seed
    )
    
    return X_train, X_test, y_train, y_test



    
# apply_preporcessor
def apply_preporcessor(X_train, y_train, num_features=100):
    """
    Preprocesses the dataset, applies SMOTE, and selects the top num_features using a RandomForestClassifier.
    
    Parameters:
    X_train (DataFrame): Training feature dataset
    y_train (array-like): Target labels
    num_features (int): Number of features to select
    
    Returns:
    X_train_selected: Transformed dataset with selected features
    y_train_balanced: Resampled target labels
    selected_features: Names of selected features
    """
    print("3) Preprocessing and feature selecting")
    cont_feats = X_train.columns[0:1].tolist() + X_train.columns[2:].tolist()
    gender_transformer = OneHotEncoder(drop='first')
    cont_transformer = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='median'), StandardScaler())

    preprocessor = make_column_transformer(
        (gender_transformer, ['GENDER']),
        (cont_transformer, cont_feats),
        sparse_threshold=0
    )
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Apply SMOTE to balance classes to a 1:1 ratio
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_transformed, y_train)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_balanced, y_train_balanced)
    
    selector = SelectFromModel(clf, max_features=num_features, prefit=True)
    X_train_selected = selector.transform(X_train_balanced)
    selected_features = X_train.columns[selector.get_support()]
    
    return X_train_selected, y_train_balanced, selected_features


####
def get_lr_c_search_space(trial):
    return {
        'C': trial.suggest_loguniform('C', 1e-4, 1e2), 
        'solver': trial.suggest_categorical('solver', ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga']),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None]),
        'l1_ratio': trial.suggest_uniform('l1_ratio', 0.0, 1.0) if trial.params.get('penalty') == 'elasticnet' else None,
        'max_iter': trial.suggest_int('max_iter', 100, 4000, step=100)
    }

study_storage = 'sqlite:///optuna_experiments.db'
def lr_c_objective(trial, X_train_selected, y_train_balanced):
    # C = trial.suggest_loguniform("C", 1e-4, 1e2)  # Regularization strength
    # solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs", "saga"])
    model_params = get_lr_c_search_space(trial)
    model = LogisticRegression(**model_params, n_jobs=-1, random_state=seed)
    score = cross_val_score(model, X_train_selected, y_train_balanced, cv=5, scoring="accuracy").mean()
    return score

def tune_c_hyperparameters(X_train_selected, y_train_balanced, type: str, outcome: str, model: str, run_id: str, model_algorithm: str = 'lr_c'):
    # objectives = {'lr_c': lr_c_objective, 'svm_c': svm_c_objective, 'xgboost_c': xgb_c_objective, 'rf_c':rf_c_objective, 'torch_nn_c': nn_c_objective}
    objectives = {'lr_c': lr_c_objective}
    study = optuna.create_study(sampler=TPESampler(seed=seed), study_name=f'{run_id}_{type}_{outcome}_{model}', storage=study_storage, direction='maximize', show_progress_bar = True, load_if_exists=True)
    study.optimize(lambda trial: objectives.get(model_algorithm)(trial, X_train_selected, y_train_balanced), n_trials=200)
    best_value = study.best_trial.value
    best_params = study.best_trial.params
    # Fit the final model using the best parameters
    final_model = LogisticRegression(**best_params, random_state=seed)
    final_model.fit(X_train_selected, y_train_balanced)

    return final_model, best_value, best_params
####
#Tune lr model
# def tune_logistic_regression(X_train_selected, y_train_balanced, n_trials=5):
#     """
#     Tunes a Logistic Regression model using Optuna.
#     """
    
#     def objective(trial):
#         C = trial.suggest_loguniform("C", 1e-4, 1e2)  # Regularization strength
#         solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs", "saga"])
        
#         model = LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42)
#         score = cross_val_score(model, X_train_selected, y_train_balanced, cv=5, scoring="accuracy").mean()
#         return score
    
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=n_trials)
    
#     best_params = study.best_params
#     best_model = LogisticRegression(**best_params, max_iter=1000, random_state=42)
#     best_model.fit(X_train_selected, y_train_balanced)
    
#     return best_model, best_params


# Train model


#tune

#fit final pipline 

#validation



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('-t','--type', help='one of: <c> <r>', required=True)
    parser.add_argument('-o','--outcome', help='one of: <mortality>', required=True)
    parser.add_argument('-m', '--model', help='one of: <lr_c>', required=True)
    parser.add_argument('-i','--id', help='run descriptor', required=True)
    args = vars(parser.parse_args())

# #now do stuff with your args

# initializetime

# train_test_split(stratified=y)


# preprocessor

compare_set, X, y = data_cleaner(target=args['outcome'])
X_train, X_test, y_train, y_test = split_dataset(X, y, seed)
# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)
X_train_selected, y_train_balanced, selected_features = apply_preporcessor(X_train, y_train)
# print(selected_features)
# best_model, best_params = tune_logistic_regression(X_train_selected, y_train_balanced, n_trials=1)
# print(best_params)
if args['type'] == 'c':
    final_model, best_value, best_params = tune_c_hyperparameters(X_train_selected, y_train_balanced, type=args['type'], outcome=args['outcome'], model=args['model'], run_id=args['run_id'])
elif args['type'] == 'r':
    final_model, best_value, best_params = tune_r_hyperparameters(X_train_selected, y_train_balanced)
else:
    print("Type entered was invalid, enter either c or r")

#fit final pipeline
# Define the final pipeline

# Fit the pipeline on the training data
final_model.fit(X_train_selected, y_train_balanced)

# Predict on test data
y_pred = final_model.predict(X_test)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")