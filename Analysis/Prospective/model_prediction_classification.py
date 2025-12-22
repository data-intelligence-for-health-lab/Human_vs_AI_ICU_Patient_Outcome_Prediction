import sys
import pickle
import pandas as pd

# Add the directory containing train_scripts to your path
sys.path.append('/home/lu.kuang/thesis/GithubFolder/jupyterlab/optuna_code/source_code/')

# Load the data
df = pd.read_csv('/home/lu.kuang/thesis/GithubFolder/jupyterlab/optuna_code/source_data/prospective_data/prosp_final_df_0722.csv')

# Define model paths and output column suffixes
model_info = {
    'HOSP_MORTALITY': 'XGB_C_HOSP_MORTALITY_C004_HOSP_MORTALITY_C004.pkl',
    '30_DAY_MORTALITY': 'XGB_C_30_DAYS_POST_DISCHARGE_MORTALITY_C014_30_DAYS_POST_DISCHARGE_MORTALITY_C014.pkl',
    'AKI_STATUS': 'XGB_C_AKI_STATUS_C024_AKI_STATUS_C024.pkl',
    'DELIRIUM_STATUS': 'XGB_C_DELIRIUM_STATUS_C034_DELIRIUM_STATUS_C034.pkl'
}

base_path = '/home/lu.kuang/thesis/GithubFolder/jupyterlab/optuna_code/source_data/prospective_data/'

# Make a working copy of the DataFrame for prediction
df_pred = df.copy()

# Define columns to exclude from model input
feature_exclude = [
    'PATIENT_ID', 'ADMISSION_ID',
    'IN_HOSPITAL_MORTALITY', '30_DAY_POST_DISCHARGE_MORTALITY',
    'AKI_STATUS', 'DELIRIUM_STATUS',
    'ICU_LOS', 'HOSP_LOS'
]

# Select only feature columns for prediction
features = [col for col in df_pred.columns if col not in feature_exclude]

X = df_pred[features]

# Run predictions for each model
for label, model_file in model_info.items():
    model_path = base_path + model_file
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Predict
    pred_probs = model.predict_proba(X)[:, 1]
    pred_class = model.predict(X)
    
    # Add predictions to DataFrame
    df_pred[f'{label}_PRED_PROB'] = pred_probs
    df_pred[f'{label}_PRED_CLASS'] = pred_class

# Optional: keep only desired columns
output_cols = [
    'PATIENT_ID', 'ADMISSION_ID', 'GENDER', 'AGE_AT_ICU_ADMIT',
    'IN_HOSPITAL_MORTALITY', '30_DAY_POST_DISCHARGE_MORTALITY', 'AKI_STATUS', 'DELIRIUM_STATUS',
    'ICU_LOS', 'HOSP_LOS',
    'HOSP_MORTALITY_PRED_PROB', 'HOSP_MORTALITY_PRED_CLASS',
    '30_DAY_MORTALITY_PRED_PROB', '30_DAY_MORTALITY_PRED_CLASS',
    'AKI_STATUS_PRED_PROB', 'AKI_STATUS_PRED_CLASS',
    'DELIRIUM_STATUS_PRED_PROB', 'DELIRIUM_STATUS_PRED_CLASS'
]

df_pred = df_pred[output_cols]

# Save to CSV
df_pred.to_csv('prosp_final_predictions.csv', index=False)
print("Predictions saved to prosp_final_predictions.csv")