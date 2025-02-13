### Classification: XGBoost ###
import numpy as np
import pandas as pd
import joblib
import os
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)

### Helper Functions ###

# Create preprocessor function
def make_preprocessor(X_train):
    print("Creating preprocessor...")
    cont_feats = X_train.columns[0:1].tolist() + X_train.columns[2:].tolist()
    gender_transformer = OneHotEncoder(drop='first')
    cont_transformer = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='median'), StandardScaler())

    preprocessor = make_column_transformer(
        (gender_transformer, ['GENDER']),
        (cont_transformer, cont_feats),
        sparse_threshold=0
    )
    print("Preprocessor created.")
    return preprocessor

# Train model and return the best estimator
def train_model(X_train, y_train, preprocessor):
    print("Starting model training...")
    pca = PCA(n_components=100)
    over_sampler = SMOTE(random_state=42, sampling_strategy=0.5)
    under_sampler = RandomUnderSampler(random_state=42, sampling_strategy=1.0)
    
    # Initialize XGBoost Classifier
    xgb_model = XGBClassifier()

    pipeline = make_pipeline(
        preprocessor,  
        over_sampler,  
        under_sampler,  
        pca,  
        xgb_model
    )

    param_grid = {
        'xgbclassifier__n_estimators': [50, 100, 200],
        'xgbclassifier__max_depth': [3, 5, 7],
        'xgbclassifier__learning_rate': [0.01, 0.1, 0.2],
        'xgbclassifier__subsample': [0.5, 0.75, 1.0],
        'xgbclassifier__colsample_bytree': [0.5, 0.75, 1.0],
        'xgbclassifier__gamma': [0, 0.1, 0.2]
    }

    print("Starting hyperparameter search...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='average_precision', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print("Model training complete.")
    return grid_search.best_estimator_

# Evaluate the model and return metrics
def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)  
    recall = recall_score(y_test, y_pred)        
    f1 = f1_score(y_test, y_pred)                

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'True Positives': tp,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
        'AUC': roc_auc,
        'Average Precision': avg_precision
    }

# Save results to CSV
def save_results(results, results_file):
    print("Saving results...")
    results_df = pd.DataFrame([results])

    if os.path.isfile(results_file):
        existing_results = pd.read_csv(results_file)
        updated_results = pd.concat([existing_results, results_df], ignore_index=True)
    else:
        updated_results = results_df

    updated_results.to_csv(results_file, index=False)
    print(f"Results have been saved to {results_file}")

# Main function to run the experiment for each target
def run_experiment(df, target, key_columns):
    print("Running experiment...")
    # Split the data into training and testing sets based on target
    test1000 = pd.read_csv("1000Patients.csv")
    testing_ids = test1000['PATIENT_ID']
    admission_ids = test1000['ADMISSION_ID']
    testing_set = df[df['ADMISSION_ID'].isin(admission_ids)]
    training_set = df[~df['PATIENT_ID'].isin(testing_ids)]

    X_train = training_set.drop(columns=key_columns)
    X_test = testing_set.drop(columns=key_columns)
    y_train = training_set[target]
    y_test = testing_set[target]

    # Create preprocessor
    preprocessor = make_preprocessor(X_train)

    # Train the model
    best_model = train_model(X_train, y_train, preprocessor)

    # Evaluate the model
    evaluation_results = evaluate_model(best_model, X_test, y_test)
    evaluation_results['Target'] = target

    # Save model
    model_filename = f"best_xgboost_model_{target}.joblib"
    joblib.dump(best_model, model_filename)
    print(f"The best model for {target} has been saved as {model_filename}.")

    # Save results
    results_file = f"xgboost_model_({target})_results.csv"
    save_results(evaluation_results, results_file)

### Main Execution Code ###

# Load dataset
df = pd.read_csv("final_df_513.csv")
print("Data Loaded...")

# Define key columns to drop
key_columns = ['PATIENT_ID', 'ADMISSION_ID', 'icu_los', 'hosp_los', 'ICU_EXPIRE_FLAG', 'HOSP_EXPIRE_FLAG', '30_DAYS_EXPIRE_FLAG', 'DELIRIUM_FLAG']

# List of target columns to loop through
target_columns = ['30_DAYS_EXPIRE_FLAG', 'HOSP_EXPIRE_FLAG']

# Loop through each target and run the experiment
for target in target_columns:
    print(f"\nRunning experiment for target: {target}")
    run_experiment(df, target, key_columns)