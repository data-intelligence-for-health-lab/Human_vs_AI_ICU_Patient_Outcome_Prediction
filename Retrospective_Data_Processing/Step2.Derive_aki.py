import pandas as pd
import numpy as np
from alive_progress import alive_bar

# Load the dataset
df = pd.read_csv('AKI_Cr.csv')

# Initialize the AKI status columns as string type
df['aki_status'] = np.nan
df['aki_status_24h'] = np.nan
df['aki_status_24_48h'] = np.nan
df['aki_status_48h'] = np.nan

# Convert measurement_hour to numeric type
df['measurement_hour'] = pd.to_numeric(df['measurement_hour']).round(2)

# Sort the DataFrame by ADMISSION_ID and measurement_hour
df = df.sort_values(by=['ADMISSION_ID', 'measurement_hour'], ascending=True)
df = df.reset_index(drop=True)

def determine_aki_status(current_creatinine, baseline_48h, baseline_7d):
    if pd.isna(baseline_48h) and pd.isna(baseline_7d):
        return np.nan # Set status to NaN if both baseline are missing
    
    aki_criteria_1 = not pd.isna(baseline_48h) and ((current_creatinine - baseline_48h) >= 26.5)
    aki_criteria_2 = not pd.isna(baseline_7d) and (current_creatinine >= 1.5 * baseline_7d)
    
    return aki_criteria_1 or aki_criteria_2

# Get unique admission IDs
admission_ids = df['ADMISSION_ID'].unique()
# admission_ids = admission_ids[:1000]

# Process each patient separately with a progress bar
with alive_bar(len(admission_ids), title="Processing Patients") as bar:
    for admission_id in admission_ids:
        bar.text = f"Processing {admission_id}"  # Update progress text dynamically

        # Extract patient data
        patient_data = df[df['ADMISSION_ID'] == admission_id].reset_index(drop=True)

        # Iterate over each row to determine AKI status
        for i in range(len(patient_data)):
            current_creatinine = patient_data.at[i, 'VALUE_NUM']
            current_hour = patient_data.at[i, 'measurement_hour']
            
            # Define the time windows based on the current hour
            past_48_hours = current_hour - 48
            past_7_days = current_hour - (7 * 24)
            
            # Filter data within the past 48 hours and 7 days for the current patient
            recent_data_48h = patient_data[(patient_data['measurement_hour'] >= past_48_hours) & 
                                           (patient_data['measurement_hour'] < current_hour)]
            recent_data_7d = patient_data[(patient_data['measurement_hour'] >= past_7_days) & 
                                          (patient_data['measurement_hour'] < current_hour)]
            
            # Calculate baseline creatinine levels
            baseline_creatinine_48h = recent_data_48h['VALUE_NUM'].min() if not recent_data_48h.empty else None
            baseline_creatinine_7d = recent_data_7d['VALUE_NUM'].min() if not recent_data_7d.empty else None
                   
            # Determine AKI status
            aki_status = determine_aki_status(current_creatinine, baseline_creatinine_48h, baseline_creatinine_7d)
            
            # Update the aki_status in the original dataframe
            df.loc[(df['ADMISSION_ID'] == admission_id) & (df['measurement_hour'] == current_hour), 'aki_status'] = aki_status

            if current_hour <= 24:
                df.loc[(df['ADMISSION_ID'] == admission_id) & (df['measurement_hour'] == current_hour), 'aki_status_24h'] = aki_status
            elif current_hour <= 48:
                df.loc[(df['ADMISSION_ID'] == admission_id) & (df['measurement_hour'] == current_hour), 'aki_status_24_48h'] = aki_status
            else:
                df.loc[(df['ADMISSION_ID'] == admission_id) & (df['measurement_hour'] == current_hour), 'aki_status_48h'] = aki_status
        
        bar()
        
# Print the updated DataFrame
print(df)

df.to_csv('Check_aki_status_optimized.csv', index=False)

# Aggregate by ADMISSION_ID and update the AKI status columns
aggregated_df = df.groupby('ADMISSION_ID').agg(
    aki_status=('aki_status', lambda x: np.nan if x.isna().all() else x.any()),
    aki_status_24h=('aki_status_24h', lambda x: np.nan if x.isna().all() else x.any()),
    aki_status_24_48h=('aki_status_24_48h', lambda x: np.nan if x.isna().all() else x.any()),
    aki_status_48h=('aki_status_48h', lambda x: np.nan if x.isna().all() else x.any())
).reset_index()

# Print the aggregated DataFrame
print(aggregated_df)
aggregated_df.to_csv('Aggregated_aki_status_optimized.csv', index=False)