import os
import pandas as pd
from datetime import datetime
import pickle

# Helper function to ensure directories exist
def create_dir_if_not_exists(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

# Helper function to save the model and hyperparameters
def save_model_and_params(pipeline, model_name, target, run_id, output_path, model_params):
    # Save the pipeline
    output_model_pkl = os.path.join(output_path, "classification", f"{model_name}_{target}_{run_id}.pkl")
    print(f"Saving model to {output_model_pkl}")
    create_dir_if_not_exists(output_model_pkl)
    with open(output_model_pkl, 'wb') as model_file:
        pickle.dump(pipeline, model_file)
    print(f"Model saved at: {output_model_pkl}")

    # Save hyperparameters to CSV
    param_data = {
        'date': datetime.now().strftime("%Y-%m-%d"),
        'model_name': model_name,
        'target': target,
        'run_id': run_id,
        'best_params': str(model_params)
    }
    best_params_file = os.path.join(output_path, "classification", f"best_params_{model_name}_{target}_{run_id}.csv")
    create_dir_if_not_exists(best_params_file)
    best_params_df = pd.DataFrame([param_data])
    best_params_df.to_csv(best_params_file, mode='a', header=not os.path.exists(best_params_file), index=False)
    print(f"Best parameters saved in: {best_params_file}")

def save_selected_features(features, model_name, output_path):
    save_feature_path = os.path.join(output_path, "selected_features.csv")
    
    # Create directory if it does not exist
    create_dir_if_not_exists(save_feature_path)

    # Convert features to DataFrame with the model name as a column
    features_df = pd.DataFrame({
        'Model Name': [model_name] * len(features),  # Repeat the model name for each feature
        'Selected Features': features
    })

    # Check if the CSV file exists to determine if we need to include the header
    file_exists = os.path.exists(save_feature_path)

    # Append to the CSV, with header only if the file does not exist
    features_df.to_csv(save_feature_path, mode='a', header=not file_exists, index=False)

# Function to save prediction probabilities and true labels to CSV
def save_classification_pred_probs_and_labels(y_test, y_pred_proba, model_name, target, run_id, output_path):
    # Create a DataFrame with true labels and predicted probabilities
    df = pd.DataFrame({
        'True Label': y_test,
        'Predicted Probability': y_pred_proba[:, 1]  # Assumes binary classification and column 1 is the positive class
    })
    
    # Add the model name as a column
    df['Model Name'] = model_name
    
    # Ensure the directory exists
    create_dir_if_not_exists(output_path)
    
    # Define file path
    output_file = os.path.join(output_path, "classification", f"pred_probs_labels_{target}_{model_name}_{run_id}.csv")
    
    # Append to CSV, creating file if it does not exist
    file_exists = os.path.exists(output_file)
    df.to_csv(output_file, mode='a', header=not file_exists, index=False)
    
    print(f"Prediction probabilities and true labels saved to: {output_file}")

def save_regression_pred_results(y_test, y_pred, model_name, target, run_id, output_path):
    # Create a DataFrame with true labels and predicted values
    df = pd.DataFrame({
        'True Label': y_test,
        'Predicted Value': y_pred
    })
    
    # Add the model name as a column
    df['Model Name'] = model_name
    
    # Ensure the directory exists
    create_dir_if_not_exists(output_path)
    
    # Define file path
    output_file = os.path.join(output_path, "regression", f"pred_results_{target}_{model_name}_{run_id}.csv")
    
    # Append to CSV, creating file if it does not exist
    file_exists = os.path.exists(output_file)
    df.to_csv(output_file, mode='a', header=not file_exists, index=False)
    
    print(f"Prediction results (true labels and predicted values) saved to: {output_file}")