import pandas as pd

# Load all sheets from the Excel file
file_path = 'DataMerged_Prospective_0430.xlsx'
sheets = pd.read_excel(file_path, sheet_name=None)

patient_visit_df = sheets['PATIENT VISIT']

# Create a dictionary with MRN, ICU_ADMIT, RECORD_ID, and GENDER
record_id_mapping = patient_visit_df[['MRN', 'ICU_ADMIT', 'RECORD_ID', 'GENDER']].drop_duplicates()
record_id_mapping_dict = record_id_mapping.dropna().set_index(['MRN', 'ICU_ADMIT']).to_dict('index')

# Process the sheets
sheet_names = ['URINE OUTPUT', 'RASS', 'INV and NIV VENT', 'ICDSC', 'COMMENTS']
processed_dfs = {}

# Fill missing RECORD_ID using the record_id_mapping dictionary
def fill_record_id(row):
    if pd.isna(row['RECORD_ID']):
        key = (row['MRN'], row['ICU_ADMIT'])
        return record_id_mapping_dict.get(key, {}).get('RECORD_ID', row['RECORD_ID'])
    return row['RECORD_ID']

for sheet_name in sheet_names:
    df = sheets[sheet_name]
    df['RECORD_ID'] = df.apply(fill_record_id, axis=1)
    processed_dfs[sheet_name] = df

# Create a new Excel writer
# Create a new Excel writer
output_file = 'DataMerged_Prospective_0430_updated.xlsx'
with pd.ExcelWriter(output_file) as writer:
    # First, write all original sheets that weren't processed
    for sheet_name, df in sheets.items():
        if sheet_name not in processed_dfs:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Then write all processed sheets
    for sheet_name, df in processed_dfs.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)