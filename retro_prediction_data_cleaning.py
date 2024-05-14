import re
import pandas as pd

# Load data
retro_df = pd.read_csv('RetrospectivePredictionDataset_raw.csv')

# Drop columns
retro_df.drop(columns=['clinician_type_initial', 'set_assigned', 'initial_survey_complete'], inplace=True)
set_columns = [col for col in retro_df.columns if col.startswith('set')]
retro_df.drop(columns=set_columns, inplace=True)

# Reshape the dataset by un-pivoting the prediction categories
df = retro_df.melt(id_vars=['record_id', 'Role'], value_vars=retro_df.columns[2:], var_name='prediction_item', value_name='prediction')

print(df)

# Split the prediction_category into two columns
df[['Patient_id', 'Prediction_Category']] = df['prediction_item'].str.split('_', n=1, expand=True)

df.drop(columns='prediction_item')

unique_items = df['Prediction_Category'].unique()
df['Prediction_Category'] = [re.sub(r'^\d+_', '', name) for name in df['Prediction_Category']]

# Extract unique patient IDs
unique_ids = df['Patient_id'].unique().tolist()
unique_ids_df = pd.DataFrame(unique_ids, columns=['Patient_id'])
print(unique_ids_df)
unique_ids_df.to_csv('1000_ids_513.csv', index=False)

# Count the occurrences of each ID
id_counts = df['Patient_id'].value_counts()

# Filter IDs that have at least 280 occurrences
repeated_ids = id_counts[id_counts >= 280].index.tolist()
print("Repeated IDs with at least 280 occurrences:")
print(len(repeated_ids))
print(repeated_ids)

repeated_ids_df = pd.DataFrame(repeated_ids, columns=['Patient_id'])
repeated_ids_df.to_csv('50_ids_513.csv', index=False)


