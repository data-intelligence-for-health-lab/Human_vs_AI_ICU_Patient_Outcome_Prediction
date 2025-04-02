import os
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
import re
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
df_hosp_stay_merged = df_categories_merged["hosp_stay"]
df_icu_stay_merged = df_categories_merged["icu_stay"]
df_aki_merged = df_categories_merged["aki"]
df_delirium_merged = df_categories_merged["delirium"]
df_hosp_mortality_merged = df_categories_merged["hosp_mortality"]
df_post_disch_mortality_merged = df_categories_merged["post_disch_mortality"]

# Grouping and aggregating
df_summary = df_hosp_stay_merged.groupby(["Group", "Role"])["prediction"].agg(["mean", "std", "count"]).reset_index()

# Compute mean and standard deviation for each Group and Role
df_summary = df_hosp_stay_merged.groupby(["Group", "Role"]).agg(
    mean=("prediction", "mean"),
    std=("prediction", "std"),
    count=("prediction", "count")
).reset_index()

# Create the bar plot
plt.figure(figsize=(16, 9))
barplot = sns.barplot(
    data=df_summary,
    x="Group",
    y="mean",
    hue="Role",
    palette="Set2",
    dodge=True,
    capsize=0.1,
    errwidth=1
)

# Add error bars manually
for i, row in df_summary.iterrows():
    group_index = list(df_summary["Group"].unique()).index(row["Group"])
    role_offset = -0.2 if row["Role"] == "Nurse" else 0.2  # Adjust position for hue
    plt.errorbar(x=group_index + role_offset, y=row["mean"], yerr=row["std"], fmt='none', ecolor='black', capsize=5)

# Add count to legend labels
handles, labels = barplot.get_legend_handles_labels()
new_labels = []
for label in labels:
    role_data = df_summary[df_summary["Role"] == label]
    breakdown = [f"Group {row['Group']} (n={row['count']})" for _, row in role_data.iterrows()]
    new_labels.append(f"{label}: " + ", ".join(breakdown))

plt.legend(handles, new_labels, title="Role Breakdown", fontsize=10)

# Formatting
plt.title("Mean Clinician Prediction of Hospital LOS by Clinician Group and Role", fontsize=14)
plt.xlabel("Group", fontsize=12)
plt.ylabel("Mean Prediction", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Show plot
plt.show()
