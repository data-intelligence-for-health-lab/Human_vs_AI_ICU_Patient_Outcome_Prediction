import pandas as pd

# Load the dataset
df = pd.read_csv('AKI_Cr.csv')

def determine_aki_status(current_creatinine, baseline_48h, baseline_7d):
    # AKI criteria
    aki_criteria_1 = (current_creatinine - baseline_48h) >= 26.5
    aki_criteria_2 = current_creatinine >= 1.5 * baseline_7d
    
    # print('Current Creatinine:', current_creatinine)
    # print('Baseline Creatinine (48h):', baseline_48h)
    # print('Baseline Creatinine (7d):', baseline_7d)
    # print('AKI Criteria 1:', aki_criteria_1)
    # print('AKI Criteria 2:', aki_criteria_2)
    
    return aki_criteria_1 or aki_criteria_2

# Initialize the AKI status columns as string type
df['aki_status'] = ''

# # Initialize the AKI status columns
df['aki_status_24h'] = ''
df['aki_status_24_48h'] = ''
df['aki_status_48h'] = ''

# Convert measurement_hour to numeric type
df['measurement_hour'] = pd.to_numeric(df['measurement_hour']).round(2)

# Sort the DataFrame by ADMISSION_ID and measurement_hour
df = df.sort_values(by=['ADMISSION_ID', 'measurement_hour'], ascending=True)
df = df.reset_index(drop=True)
#print("df:",df)

# Process each patient separately
for admission_id, patient_data in df.groupby('ADMISSION_ID'):
    print(admission_id)
    
    # Reset index for patient data
    patient_data = patient_data.reset_index(drop=True)
    print("patient_data", patient_data)
    

    # Iterate over each row to determine AKI status
    for i in range(len(patient_data)):
        current_creatinine = patient_data.at[i, 'VALUE_NUM']
        current_hour = patient_data.at[i, 'measurement_hour']
        
        #Define the time windows based on the current hour
        past_48_hours = current_hour - 48
        past_7_days = current_hour - (7 * 24)
        
        # Filter data within the past 48 hours and 7 days for the current patient
        recent_data_48h = patient_data[(patient_data['measurement_hour'] >= past_48_hours) & (patient_data['measurement_hour'] <= current_hour)]
        recent_data_7d = patient_data[(patient_data['measurement_hour'] >= past_7_days) & (patient_data['measurement_hour'] <= current_hour)]
        
        # print(f"Patient {admission_id}, Row {i}")
        # print("Current hour:", current_hour)
        # print("recent_data_48h:\n", recent_data_48h)
        # print("recent_data_7d:\n", recent_data_7d)
        
        # Calculate baseline creatinine levels
        baseline_creatinine_48h = recent_data_48h['VALUE_NUM'].min() if not recent_data_48h.empty else current_creatinine #SET AKI TO ''
        baseline_creatinine_7d = recent_data_7d['VALUE_NUM'].min() if not recent_data_7d.empty else current_creatinine #SET AKI TO ''
        
        # Determine AKI status
        aki_status = determine_aki_status(current_creatinine, baseline_creatinine_48h, baseline_creatinine_7d)
        #print("aki_status: ",aki_status)
        
        # Update the aki_status in the original dataframe
        df.loc[(df['ADMISSION_ID'] == admission_id) & (df['measurement_hour'] == current_hour), 'aki_status'] = aki_status

        if current_hour <= 24:
            df.loc[(df['ADMISSION_ID'] == admission_id) & (df['measurement_hour'] == current_hour), 'aki_status_24h'] = aki_status
        elif 24 < current_hour <= 48:
            df.loc[(df['ADMISSION_ID'] == admission_id) & (df['measurement_hour'] == current_hour), 'aki_status_24_48h'] = aki_status
        else:
            df.loc[(df['ADMISSION_ID'] == admission_id) & (df['measurement_hour'] == current_hour), 'aki_status_48h'] = aki_status
        
# Print the updated DataFrame
print(df)
df.to_csv('check_aki_status.csv', index=False)

# Aggregate by ADMISSION_ID and update the AKI status columns
aggregated_df = df.groupby('ADMISSION_ID').agg(
    aki_status=('aki_status', 'any'),
    aki_status_24h=('aki_status_24h', 'any'),
    aki_status_24_48h=('aki_status_24_48h', 'any'),
    aki_status_48h=('aki_status_48h', 'any')
).reset_index()

# Print the aggregated DataFrame
print(aggregated_df)
aggregated_df.to_csv('aggregated_aki_status.csv', index=False)