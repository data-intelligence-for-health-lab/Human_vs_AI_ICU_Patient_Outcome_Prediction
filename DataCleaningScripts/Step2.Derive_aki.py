import pandas as pd
import os

# Load the dataset
df = pd.read_csv('AKI_Cr.csv')

def determine_aki_status(current_creatinine, baseline_48h, baseline_7d):
    """Determine AKI status based on creatinine values."""
    if baseline_48h is None and baseline_7d is None:
        return None  # No baseline data available

    aki_criteria_1 = (baseline_48h is not None) and ((current_creatinine - baseline_48h) >= 26.5)
    aki_criteria_2 = (baseline_7d is not None) and (current_creatinine >= 1.5 * baseline_7d)

    return aki_criteria_1 or aki_criteria_2

# Initialize AKI status columns with NaN
df['aki_status'] = None
df['aki_status_24h'] = None
df['aki_status_24_48h'] = None
df['aki_status_48h'] = None

# Convert measurement_hour to numeric
df['measurement_hour'] = pd.to_numeric(df['measurement_hour']).round(2)

# Sort DataFrame by patient and time
df = df.sort_values(by=['ADMISSION_ID', 'measurement_hour']).reset_index(drop=True)

# Function to process each patient separately
def process_patient_data(patient_data):
    patient_data = patient_data.copy()  # Avoid modifying original data
    patient_data = patient_data.sort_values(by='measurement_hour').reset_index(drop=True)

    for i in range(len(patient_data)):
        current_creatinine = patient_data.at[i, 'VALUE_NUM']
        current_hour = patient_data.at[i, 'measurement_hour']

        # Define the time windows
        past_24_hours = current_hour - 24
        past_24_48_hours = current_hour - 48
        past_48_hours = current_hour - 48
        past_7_days = current_hour - (7 * 24)

        # Get past measurements
        recent_data_24h = patient_data[(patient_data['measurement_hour'] >= past_24_hours) & 
                                       (patient_data['measurement_hour'] < current_hour)]
        recent_data_24_48h = patient_data[(patient_data['measurement_hour'] >= past_24_48_hours) & 
                                          (patient_data['measurement_hour'] < past_24_hours)]
        recent_data_48h = patient_data[(patient_data['measurement_hour'] >= past_48_hours) & 
                                       (patient_data['measurement_hour'] < current_hour)]
        recent_data_7d = patient_data[(patient_data['measurement_hour'] >= past_7_days) & 
                                      (patient_data['measurement_hour'] < current_hour)]

        # Compute baselines
        baseline_creatinine_48h = recent_data_48h['VALUE_NUM'].min() if not recent_data_48h.empty else None
        baseline_creatinine_7d = recent_data_7d['VALUE_NUM'].min() if not recent_data_7d.empty else None

        # Determine AKI status
        aki_status = determine_aki_status(current_creatinine, baseline_creatinine_48h, baseline_creatinine_7d)

        # Assign AKI status
        patient_data.at[i, 'aki_status'] = aki_status

        # Assign status for different time windows
        patient_data.at[i, 'aki_status_24h'] = aki_status if not recent_data_24h.empty else None
        patient_data.at[i, 'aki_status_24_48h'] = aki_status if not recent_data_24_48h.empty else None
        patient_data.at[i, 'aki_status_48h'] = aki_status if not recent_data_48h.empty else None

    return patient_data

# Apply processing function to each patient group
df = df.groupby('ADMISSION_ID', group_keys=False).apply(process_patient_data)

# Save the processed DataFrame
df.to_csv('Check_aki_status_325.csv', index=False)

# Aggregate by ADMISSION_ID to get final status
aggregated_df = df.groupby('ADMISSION_ID').agg(
    aki_status=('aki_status', lambda x: any(pd.to_numeric(x, errors='coerce').fillna(0))),
    aki_status_24h=('aki_status_24h', lambda x: any(pd.to_numeric(x, errors='coerce').fillna(0))),
    aki_status_24_48h=('aki_status_24_48h', lambda x: any(pd.to_numeric(x, errors='coerce').fillna(0))),
    aki_status_48h=('aki_status_48h', lambda x: any(pd.to_numeric(x, errors='coerce').fillna(0)))
).reset_index()

# Save the aggregated results
aggregated_df.to_csv('Aggregated_aki_status_325.csv', index=False)

print(df.head())
print(aggregated_df.head())