import pandas as pd

# Load datasets
df = pd.read_csv("final_df_818.csv")
test1000 = pd.read_csv("1000Patients.csv")

# Define columns
key_columns = [
    'PATIENT_ID', 'ADMISSION_ID', 'ICU_LOS', 'HOSP_LOS', 
    'ICU_EXPIRE_FLAG', 'HOSP_EXPIRE_FLAG', '30_DAYS_EXPIRE_FLAG', 
    'DELIRIUM_LABEL', 'AKI_LABEL'
]

class_columns = [
    'HOSP_EXPIRE_FLAG',
    '30_DAYS_EXPIRE_FLAG',
    'DELIRIUM_LABEL',
    'AKI_LABEL'
]

los_columns = ['ICU_LOS', 'HOSP_LOS']

# Split train/test based on ID matching
testing_ids = test1000['PATIENT_ID']
admission_ids = test1000['ADMISSION_ID']
testing_set = df[df['ADMISSION_ID'].isin(admission_ids)].copy()
training_set = df[~df['PATIENT_ID'].isin(testing_ids)].copy()

### 1. Event Rate Calculation
def calculate_event_rates(data, label_cols):
    results = {}
    for col in label_cols:
        counts = data[col].value_counts().sort_index()
        total = counts.sum()
        rates = (counts / total * 100).round(2)
        results[col] = {
            '0 count': counts.get(0, 0),
            '1 count': counts.get(1, 0),
            '0 %': rates.get(0, 0.0),
            '1 %': rates.get(1, 0.0)
        }
    return pd.DataFrame(results).T

train_event_rates = calculate_event_rates(training_set, class_columns)
test_event_rates = calculate_event_rates(testing_set, class_columns)

print("=== Event Rates in Training Set ===")
print(train_event_rates)
print("\n=== Event Rates in Testing Set ===")
print(test_event_rates)

### 2. Length of Stay Stats
def calculate_los_stats(data, los_cols, label='Set'):
    stats = data[los_cols].agg(['mean', 'std']).round(2).T
    stats.columns = [f'{label} Mean', f'{label} Std']
    return stats

train_los_stats = calculate_los_stats(training_set, los_columns, label='Train')
test_los_stats = calculate_los_stats(testing_set, los_columns, label='Test')

# Combine LOS stats
los_combined = train_los_stats.join(test_los_stats)
print("\n=== Length of Stay Statistics ===")
print(los_combined)