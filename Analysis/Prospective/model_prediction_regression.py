import sys
import pickle
import pandas as pd

# Add your custom code path
sys.path.append('/home/lu.kuang/thesis/GithubFolder/jupyterlab/optuna_code/source_code/')

# Load the prospective data
df = pd.read_csv('/home/lu.kuang/thesis/GithubFolder/jupyterlab/optuna_code/source_data/prospective_data/prosp_final_df_0722.csv')

# Define regression model files for LOS prediction
model_info = {
    'ICU_LOS': 'XGB_R_ICU_LOS_C104_ICU_LOS_C104.pkl',
    'HOSP_LOS': 'XGB_R_HOSP_LOS_C114_HOSP_LOS_C114.pkl'
}

base_path = '/home/lu.kuang/thesis/GithubFolder/jupyterlab/optuna_code/source_data/prospective_data/'

# Make a working copy for prediction
df_pred = df.copy()

# Define columns to exclude from model input
feature_exclude = [
    'PATIENT_ID', 'ADMISSION_ID',
    'IN_HOSPITAL_MORTALITY', '30_DAY_POST_DISCHARGE_MORTALITY',
    'AKI_STATUS', 'DELIRIUM_STATUS',
    'ICU_LOS', 'HOSP_LOS'
]

# Select model input features
features = [col for col in df_pred.columns if col not in feature_exclude]
X = df_pred[features]

# Run predictions for each LOS model
for label, model_file in model_info.items():
    model_path = base_path + model_file
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Predict regression output (e.g., LOS in days)
    predictions = model.predict(X)
    
    # Add predictions to DataFrame
    df_pred[f'{label}_PRED'] = predictions

# Define final columns to keep
output_cols = [
    'PATIENT_ID', 'ADMISSION_ID',
    'ICU_LOS', 'ICU_LOS_PRED',
    'HOSP_LOS', 'HOSP_LOS_PRED'
]

# Filter the DataFrame
df_pred = df_pred[output_cols]

# Save results
df_pred.to_csv('prosp_final_LOS_predictions.csv', index=False)
print("LOS regression predictions saved to prosp_final_LOS_predictions.csv")