# Regression: Support Vector Regression (SVR) for ICU_LOS + HOSP_LOS#
import numpy as np
import pandas as pd
import joblib
import os
from imblearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    mean_absolute_percentage_error, median_absolute_error
)

### Helper Functions ###

# Create preprocessor function
def make_preprocessor(X_train):
    cont_feats = X_train.columns[0:1].tolist() + X_train.columns[2:].tolist()
    gender_transformer = OneHotEncoder(drop='first')
    cont_transformer = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='median'), StandardScaler())
    preprocessor = make_column_transformer((gender_transformer, ['GENDER']), (cont_transformer, cont_feats), sparse_threshold=0)
    return preprocessor

# Custom function to generate regression metrics
def generate_regression_metrics(pipeline, X_test, y_test, model_name, target):
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'Model': model_name,
        'Target': target,
        'Mean Squared Error (MSE)': mse,
        'Root Mean Squared Error (RMSE)': rmse,
        'Mean Absolute Error (MAE)': mae,
        'Mean Absolute Percentage Error (MAPE)': mape,
        'Median Absolute Error (MedAE)': medae,
        'R2 Score': r2
    }

    # Print metrics
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

    # Create output folder if it doesn't exist
    output_folder = 'GridSearch/Outputs/Regression'
    os.makedirs(output_folder, exist_ok=True)

    # Save metrics as a table
    metrics_df = pd.DataFrame([metrics])
    output_filename = f'regression_performance_metrics_{model_name}.csv'
    output_path = os.path.join(output_folder, output_filename)

    if os.path.isfile(output_path):
        # Append to existing file if it exists
        existing_metrics_df = pd.read_csv(output_path)
        metrics_df = pd.concat([existing_metrics_df, metrics_df], ignore_index=True)

    metrics_df.to_csv(output_path, index=False)
    print("Metrics saved successfully.")

    # Save predicted values
    predictions_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    predictions_filename = f'regression_predictions_{model_name}.csv'
    predictions_path = os.path.join(output_folder, predictions_filename)
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved successfully to {predictions_path}")

# Train model and return the best estimator
def train_model(X_train, y_train, preprocessor, target):
    pca = PCA(n_components=100, random_state=42)
    svr = SVR()

    pipeline = make_pipeline(
        preprocessor,      # Preprocessing
        pca,               # PCA for dimensionality reduction
        svr                # Support Vector Regression
    )

    param_grid = {
        'svr__C': np.concatenate([np.arange(0, 1.1, 0.1), np.arange(1, 21, 1)]).tolist(),
        'svr__epsilon': [0.01, 0.1, 0.5],
        'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'svr__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }

    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=10, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=3
    )
    grid_search.fit(X_train, y_train)

    # Save best parameters to CSV file
    best_params = grid_search.best_params_
    best_params_df = pd.DataFrame([best_params])
    best_params_filename = f'best_params_{target}.csv'
    best_params_df.to_csv(best_params_filename, index=False)
    print(f"Best parameters for {target} saved to {best_params_filename}")

    print(f"Best Hyperparameters: {best_params}")
    print(f"Best CV Score (Negative MSE): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


# Save results to CSV (Retained for potential additional use)
def save_results(results, results_file):
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
    # Split the data into training and testing sets based on target
    test1000 = pd.read_csv("1000Patients.csv")
    testing_ids = test1000['PATIENT_ID']
    admission_ids = test1000['ADMISSION_ID']
    testing_set = df[df['ADMISSION_ID'].isin(admission_ids)]
    training_set = df[~df['PATIENT_ID'].isin(testing_ids)]

    # Drop any rows with NaN in the target variable
    training_set = training_set.dropna(subset=[target])
    testing_set = testing_set.dropna(subset=[target])

    X_train = training_set.drop(columns=key_columns)
    X_test = testing_set.drop(columns=key_columns)
    y_train = training_set[target]
    y_test = testing_set[target]

    # Create preprocessor
    preprocessor = make_preprocessor(X_train)

    # Train the model
    best_model = train_model(X_train, y_train, preprocessor, target)

    # Evaluate the model using the custom metrics function
    generate_regression_metrics(best_model, X_test, y_test, model_name=f"SVR_{target}", target=target)

    # Save model
    model_filename = f"best_svr_model_{target}.joblib"
    joblib.dump(best_model, model_filename)
    print(f"The best model for {target} has been saved as {model_filename}.")
    
### Main Execution Code ###
df = pd.read_csv("final_df_513.csv")
print("Data Loaded...")

# Define key columns to drop
key_columns = [
    'PATIENT_ID', 'ADMISSION_ID', 'icu_los', 'hosp_los', 
    'ICU_EXPIRE_FLAG', 'HOSP_EXPIRE_FLAG', 
    '30_DAYS_EXPIRE_FLAG', 'DELIRIUM_FLAG'
]

# List of target columns to loop through
target_columns = ['icu_los', 'hosp_los']

# Loop through each target and run the experiment
for target in target_columns:
    print(f"\nRunning experiment for target: {target}")
    run_experiment(df, target, key_columns)