import os
import time
import argparse
# import torch
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from train_scripts.dataset_clean_split import data_cleaner
from train_scripts.pipeline_components import make_preprocessor, check_imbalance
from train_scripts.final_models import classification_model, regression_model


# Load environment variables
load_dotenv()

# Set up paths from environment variables
root_folder = os.getenv("ROOT_FOLDER")
source_data = os.getenv("SOURCE_DATA")
final_df = os.getenv("FINAL_DF")
comparison_set = os.getenv("PATIENTS_1000")
output_folder = os.getenv("OUTPUT_FOLDER")
best_classification_params = os.getenv("CLASSIFICATION_BEST_PARAMS")
best_regression_params = os.getenv("REGRESSION_BEST_PARAMS")

# Full path to data sources
final_df_path = os.path.join(root_folder, source_data, final_df)
comparison_ids_path = os.path.join(root_folder, source_data, comparison_set)
output_path = os.path.join(root_folder, output_folder)
# print("output_path", output_path)

# Set random seed
SEED = 42

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('-t', '--model_type', required=True, choices=['C', 'R'], help='Type of model: Classification (C) or Regression (R)')
    parser.add_argument('-o', '--outcome', required=True, help='Outcome variable')
    parser.add_argument('-m', '--model_name', required=True, choices=['LR_C', 'SVM_C', 'RF_C', 'XGB_C', 'NN_C', 'EN_R', 'SVM_R', 'RF_R', 'XGB_R', 'NN_R'], help='Model examples: Logistic Regression (LR_C) or SVM (SVM_C)')
    parser.add_argument('-i', '--id', required=True, help='Run descriptor')
    args = vars(parser.parse_args())

    start_time = time.time()
    
    target=args['outcome']
    mtype=args['model_type'] 
    mname=args['model_name'] 
    run_id=args['id']


    # Data preparation
    X_train, X_test, y_train, y_test = data_cleaner(final_df_path, comparison_ids_path, target)
    print(X_train, X_test, y_train, y_test)
    # Define preprocessing steps
    preprocessor = make_preprocessor(X_train)
    feature_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=SEED),
        max_features=100
    )
    if mtype == 'C':
        over_sampler=SMOTE(random_state=SEED, sampling_strategy=0.5)
        under_sampler=RandomUnderSampler(random_state=SEED, sampling_strategy=1.0)
        imbalance_status = check_imbalance(y_test)
        if imbalance_status:
            print("Imbalance detected in y_test, perform over and under sampling to mitigate")
        best_params_df = classification_model(
        X_train, y_train, X_test, y_test, preprocessor, over_sampler, 
        under_sampler, feature_selector, imbalance_status, target, mtype, mname, run_id, 
        output_path
    )
    elif mtype == 'R':
        best_params_df = regression_model(
        X_train, y_train, X_test, y_test, preprocessor,
        feature_selector, target, mtype, mname, run_id, output_path
    )
    else:
        raise ValueError(f"Model type '{mtype}' is not recognized. Please use a valid model type.")


    print(f"Completed training {mname} for {target} with best params: {best_params_df}")

    print(f"\nTotal Execution Time: {time.time() - start_time:.2f} seconds")