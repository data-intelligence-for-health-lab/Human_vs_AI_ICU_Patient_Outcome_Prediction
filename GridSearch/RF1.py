### Classification: Random Forest 1 ###
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)

### Helper Functions ###

# Create preprocessor function
def make_preprocessor(X_train):
    print("make preprocessor")
    cont_feats = X_train.columns[0:1].tolist() + X_train.columns[2:].tolist()
    gender_transformer = OneHotEncoder(drop='first')
    cont_transformer = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='median'), StandardScaler())

    preprocessor = make_column_transformer(
        (gender_transformer, ['GENDER']),
        (cont_transformer, cont_feats),
        sparse_threshold=0
    )
    return preprocessor

# Train model and return the best estimator
def train_model(X_train, y_train, preprocessor):
    print("train model")
    pca = PCA(n_components=100)
    over_sampler = SMOTE(random_state=42, sampling_strategy=0.5)
    under_sampler = RandomUnderSampler(random_state=42, sampling_strategy=1.0)
    rf_model = RandomForestClassifier(random_state=42)

    pipeline = make_pipeline(
        preprocessor,
        over_sampler,
        under_sampler,
        pca,
        rf_model
    )

    # Correcting parameter grid for Random Forest
    param_grid = {
        'randomforestclassifier__n_estimators': [50, 100, 200],
        'randomforestclassifier__max_depth': [None, 10, 20, 30],
        'randomforestclassifier__min_samples_split': [2, 5, 10],
        'randomforestclassifier__class_weight': ['balanced', None]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='average_precision', n_jobs=-1, verbose=3)
    grid_search.fit(X_train, y_train)

    print(f"Best Hyperparameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Generate metrics and save predictions
def generate_metrics(pipeline, X_test, y_test, model_name, target):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    metrics = {
        'Target': target,
        'Model': model_name,
        'True Positives': tp,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc,
        'Average Precision': avg_precision
    }
    
    # Create output folder if it doesn't exist
    output_folder = 'Outputs/Classification'
    os.makedirs(output_folder, exist_ok=True)
    
    # Save metrics as a table
    metrics_df = pd.DataFrame([metrics])
    output_filename = f'performance_metrics_{model_name}_{target}.csv'
    output_path = os.path.join(output_folder, output_filename)
    
    if os.path.isfile(output_path):
        # Append to existing file if it exists
        existing_metrics_df = pd.read_csv(output_path)
        metrics_df = pd.concat([existing_metrics_df, metrics_df], ignore_index=True)
    
    metrics_df.to_csv(output_path, index=False)
    print("Metrics saved successfully.")
    
    # Save predicted probabilities
    probabilities_df = pd.DataFrame({'Predicted Probabilities': y_proba})
    probabilities_filename = f'predicted_probabilities_{model_name}_{target}.csv'
    probabilities_path = os.path.join(output_folder, probabilities_filename)
    
    probabilities_df.to_csv(probabilities_path, index=False)
    print("Predicted probabilities saved successfully.")

# Evaluate the model and return metrics
def evaluate_model(model, X_test, y_test, model_name, target):
    print("evaluate_model")
    generate_metrics(model, X_test, y_test, model_name, target)

# Main function to run the experiment for each target
def run_experiment(df, target, key_columns):
    print("run_experiment")
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

    # Evaluate the model and save metrics
    evaluate_model(best_model, X_test, y_test, "RandomForest", target)

    # Save model
    model_dir = "GridSearch/Outputs"
    model_filename = os.path.join(model_dir, f"best_rf_model_{target}.joblib")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, model_filename)
    print(f"The best model for {target} has been saved as {model_filename}.")

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