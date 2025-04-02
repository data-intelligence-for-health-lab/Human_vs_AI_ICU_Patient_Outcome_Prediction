import os
import re
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from scipy.stats import ranksums
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up paths from environment variables
root_folder = os.getenv("ROOT_FOLDER")
source_data = os.getenv("SOURCE_DATA")
final_df = os.getenv("FINAL_DF")
retro_df = os.getenv("RETRO_DF")
final_df_path = os.path.join(root_folder, source_data, final_df)
retro_df_path = os.path.join(root_folder, source_data, retro_df)

#load data
retro_df = pd.read_csv(retro_df_path)
#drop column
retro_df.drop(columns=['clinician_type_initial', 'set_assigned', 'initial_survey_complete'], inplace=True)
set_columns = [col for col in retro_df.columns if col.startswith('set')]
retro_df.drop(columns=set_columns, inplace=True)
# Reshape the dataset by unpivoting the prediction categories
df = retro_df.melt(id_vars=['record_id', 'Role'], value_vars=retro_df.columns[2:], var_name='prediction_item', value_name='prediction')
# Split the prediction_category into two columns
df[['Patient_id', 'Prediction_Category']] = df['prediction_item'].str.split('_', n=1, expand=True)
df.drop(columns='prediction_item')
# Modify the 'Prediction_Category' column
df['Prediction_Category'] = [re.sub(r'^\d+_', '', name) for name in df['Prediction_Category']]
#Create Group Column for each prediction outcome dataframe based on the number of clinicans predicing the outcome
# Define prediction categories
categories = [
    "hosp_stay", "icu_stay", "aki", "delirium", "hosp_mortality", "post_disch_mortality"
]

# Create a dictionary of filtered DataFrames (using .copy() to ensure a new DataFrame)
df_categories = {cat: df[df["Prediction_Category"] == cat].copy() for cat in categories}

# Apply transformations to all DataFrames
for cat, df_cat in df_categories.items():
    df_cat.drop(columns=["prediction_item"], inplace=True, errors="ignore")  # Drop column safely
    df_cat.dropna(subset=["prediction"], inplace=True)  # Drop rows where 'prediction' is NaN
    df_cat.loc[:, "Patient_id"] = df_cat["Patient_id"].str.upper()  # Use .loc to avoid warnings
    df_categories[cat] = df_cat  # Ensure the dictionary holds the updated DataFrame

# Function to add "Group" column based on Patient_id
def add_group_column(df):
    role_counts = df.groupby("Patient_id")["Role"].count()
    df_role_counts = role_counts.reset_index()
    df_role_counts["Group"] = df_role_counts["Role"].apply(lambda x: "1" if x == 1 else "2" if x == 2 else "3")
    
    # Merge back with the original dataframe to retain all rows
    df = df.merge(df_role_counts[["Patient_id", "Group"]], on="Patient_id", how="left")
    return df

# Apply to all DataFrames in df_categories
df_categories = {key: add_group_column(df) for key, df in df_categories.items()}

# Access cleaned DataFrames
df_hosp_stay = df_categories["hosp_stay"]
df_icu_stay = df_categories["icu_stay"]
df_aki = df_categories["aki"]
df_delirium = df_categories["delirium"]
df_hosp_mortality = df_categories["hosp_mortality"]
df_post_disch_mortality = df_categories["post_disch_mortality"]

#Bring in true labels
target_df = pd.read_csv(final_df_path)
target_df = target_df.iloc[:, :11]

# Define mapping of outcome categories to corresponding label columns in target_df
outcome_label_mapping = {
    "hosp_stay": "HOSP_LOS", 
    "icu_stay": "ICU_LOS",
    "aki": "AKI_LABEL",
    "delirium": "DELIRIUM_LABEL",
    "hosp_mortality": "HOSP_EXPIRE_FLAG",
    "post_disch_mortality": "30_DAYS_EXPIRE_FLAG"
}

# Function to merge outcome DataFrame with target_df based on Patient_id
def merge_with_target(df, label_column):
    # Merge with target_df to bring in true labels
    df_merged = df.merge(target_df, left_on="Patient_id", right_on="PATIENT_ID", how="left")
    
    # If there's a specific label column, keep only relevant columns
    if label_column:
        df_merged = df_merged[["Patient_id", "Role", "prediction", "Group", label_column]]
    
    return df_merged

# Apply function to each DataFrame in df_categories
df_categories_merged = {
    key: merge_with_target(df, outcome_label_mapping[key]) 
    for key, df in df_categories.items()
}

# Access merged DataFrames
# df_hosp_stay_merged = df_categories_merged["hosp_stay"]
# df_icu_stay_merged = df_categories_merged["icu_stay"]
# df_aki_merged = df_categories_merged["aki"]
# df_delirium_merged = df_categories_merged["delirium"]
# df_hosp_mortality_merged = df_categories_merged["hosp_mortality"]
# df_post_disch_mortality_merged = df_categories_merged["post_disch_mortality"]

def bootstrap_ci(y_true, y_pred, metric_func, n_iter=1000, alpha=0.05):
    """Compute bootstrap confidence interval for a given metric function."""
    boot_metrics = []
    n = len(y_true)
    
    for _ in range(n_iter):
        indices = np.random.choice(n, n, replace=True)
        boot_y_true = y_true[indices]
        boot_y_pred = y_pred[indices]
        boot_metrics.append(metric_func(boot_y_true, boot_y_pred))
    
    lower = np.percentile(boot_metrics, 100 * (alpha / 2))
    upper = np.percentile(boot_metrics, 100 * (1 - alpha / 2))
    
    return np.mean(boot_metrics), (lower, upper)

def compute_metrics_by_group(df, target, n_iter=1000):
    """Compute bootstrapped metrics for each group, handling NaNs in y_true."""
    results = []

    for group, group_df in df.groupby("Group"):
        # Drop rows where y_true or y_pred is NaN
        group_df = group_df.dropna(subset=[target, "mapped_prediction"])
        
        if group_df.empty:
            print(f"Skipping group {group} due to no valid data after NaN removal.")
            continue
        
        y_true = group_df[target].values
        y_pred = group_df["mapped_prediction"].values
        
        # Compute bootstrapped metrics
        acc, acc_ci = bootstrap_ci(y_true, y_pred, accuracy_score, n_iter)
        prec, prec_ci = bootstrap_ci(y_true, y_pred, precision_score, n_iter)
        rec, rec_ci = bootstrap_ci(y_true, y_pred, recall_score, n_iter)
        f1, f1_ci = bootstrap_ci(y_true, y_pred, f1_score, n_iter)
        
        # Handle case where confusion matrix may not work due to single class presence
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        except ValueError:  # Happens if only one class is present in y_true or y_pred
            spec, npv = np.nan, np.nan
        
        results.append({
            "Group": group,
            "Accuracy": f"{acc:.3f} ({acc_ci[0]:.3f}, {acc_ci[1]:.3f})",
            "Precision": f"{prec:.3f} ({prec_ci[0]:.3f}, {prec_ci[1]:.3f})",
            "Recall": f"{rec:.3f} ({rec_ci[0]:.3f}, {rec_ci[1]:.3f})",
            "F1 Score": f"{f1:.3f} ({f1_ci[0]:.3f}, {f1_ci[1]:.3f})",
            "Specificity": f"{spec:.3f}" if not np.isnan(spec) else "N/A",
            "NPV": f"{npv:.3f}" if not np.isnan(npv) else "N/A"
        })
    
    return pd.DataFrame(results)

classification_result_label_mapping = {
    "aki": "AKI_LABEL",
    "delirium": "DELIRIUM_LABEL",
    "hosp_mortality": "HOSP_EXPIRE_FLAG",
    "post_disch_mortality": "30_DAYS_EXPIRE_FLAG"
}

results_list = []
threshold = 2
for key, target in classification_result_label_mapping.items():
    df = df_categories_merged[key]  # Extract DataFrame based on key
    df["mapped_prediction"] = df["prediction"].apply(lambda x: 1 if x < threshold else 0)  # Map predictions
    print(df)
    df_results = compute_metrics_by_group(df, target)  # Compute metrics
    df_results["Outcome"] = key  # Add outcome label for tracking
    results_list.append(df_results)

# Combine all results into a single DataFrame
final_results_df = pd.concat(results_list, ignore_index=True)

# Display the results
print(final_results_df)