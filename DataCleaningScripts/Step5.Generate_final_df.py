"""
This script is used to pivot datasets from SQL for intervensions, measurements and prescription, drop value char with missingess > 30% and combine with the mastelist to generate ML compatible dataframe
"""

import pandas as pd
from sqlalchemy import create_engine
engine = create_engine('postgresql+psycopg2://lu.kuang:EFTd66Qa3QhnMeA7\
@10.44.113.30/ABeICU')

# Connect to the database (last updated 2024.08.14)
master_df = pd.read_sql_query('SELECT * FROM "lu.kuang"."FINAL_MASTERTABLE"',con=engine)
interventions_df = pd.read_sql_query('SELECT * FROM "lu.kuang".intervention_24_agg',con=engine)
measurements_df = pd.read_sql_query('SELECT * FROM "lu.kuang".measurements_24_agg',con=engine)
prescription_df = pd.read_sql_query('SELECT * FROM "lu.kuang".final_prescriptions',con=engine)

#=========================Interventions========================#
# Select the desired columns
selected_interventions_columns = interventions_df.columns[4:13]

#pivot columns
pivoted_interventions_df = interventions_df.pivot_table(index='ADMISSION_ID', columns='ITEM_ID', values=selected_interventions_columns)

# Flatten the column names by concatenating the label with the selected columns
new_interventions_columns = []
for column in pivoted_interventions_df.columns:
    new_interventions_column = f'{column[0]}_{column[1]}_{interventions_df.loc[interventions_df["ITEM_ID"] == column[1]]["LABEL"].values[0]}'
    new_interventions_columns.append(new_interventions_column)
    
# Assign the new column names
pivoted_interventions_df.columns = new_interventions_columns
# Reset index
pivoted_interventions_df.reset_index(inplace=True)

# Calculate the NaN percentage for each column
nan_percentage_interventions = pivoted_interventions_df.isna().mean()

threshold = 0.3
selected_interventions_columns = nan_percentage_interventions[nan_percentage_interventions <= threshold].index
# Filter the DataFrame based on selected columns
interventions_final_df = pivoted_interventions_df[selected_interventions_columns]

#interventions_value_char
interventions_value_char = pd.read_sql_query('SELECT * FROM "lu.kuang".interventions_value_char',con=engine)
selected_interventions_vc = interventions_value_char.columns[4]
pivoted_interventions_vc = interventions_value_char.pivot_table(index='ADMISSION_ID', columns=['ITEM_ID','VALUE_CHAR'], values=selected_interventions_vc)

#Intervetion value character
new_interventions_vc_columns = []
for column in pivoted_interventions_vc.columns:
    new_interventions_vc_column = f'{column[0]}_{column[1]}'
    new_interventions_vc_columns.append(new_interventions_vc_column)
    
# Assign the new column names
pivoted_interventions_vc.columns = new_interventions_vc_columns
combined_interventions_df = interventions_final_df.merge(pivoted_interventions_vc,on='ADMISSION_ID', how='outer')

#=========================Measurements========================#
# Select the columns from measurements
selected_measurements_columns = measurements_df.columns[4:13]
pivoted_measurements_df = measurements_df.pivot_table(index='ADMISSION_ID', columns='ITEM_ID', values=selected_measurements_columns)
# Flatten the column names by concatenating the label with the selected columns
new_measurements_columns = []
for column in pivoted_measurements_df.columns:
    new_measurements_column = f'{column[0]}_{column[1]}_{measurements_df.loc[measurements_df["ITEM_ID"] == column[1]]["LABEL"].values[0]}'
    new_measurements_columns.append(new_measurements_column)
# Assign the new column names
pivoted_measurements_df.columns = new_measurements_columns
# Reset index
pivoted_measurements_df.reset_index(inplace=True)
# Calculate the NaN percentage for each column
nan_percentage_measurements = pivoted_measurements_df.isna().mean()
measurements_selected_columns = nan_percentage_measurements[nan_percentage_measurements <= threshold].index
# Filter the DataFrame based on selected columns
measurements_final_df = pivoted_measurements_df[measurements_selected_columns]

#measurements_value_char
measurements_value_char = pd.read_sql_query('SELECT * FROM "lu.kuang".measurements_value_char',con=engine)
# Select the columns from measurements
selected_measurements_vc = measurements_value_char.columns[4]
pivoted_measurements_vc = measurements_value_char.pivot_table(index='ADMISSION_ID', columns=['ITEM_ID','VALUE_CHAR'], values=selected_measurements_vc)

#measurements value character
new_measurements_vc_columns = []
for column in pivoted_measurements_vc.columns:
    new_measurements_vc_column = f'{column[0]}_{column[1]}'
    new_measurements_vc_columns.append(new_measurements_vc_column)

# Assign the new column names
pivoted_measurements_vc.columns = new_measurements_vc_columns
combined_measurements_df = measurements_final_df.merge(pivoted_measurements_vc,on='ADMISSION_ID', how='outer')

#=========================Prescriptions========================#
selected_prescription_columns = prescription_df.columns[5:14]
pivoted_prescription_df = prescription_df.pivot_table(index='ADMISSION_ID', columns='ITEM_ID', values=selected_prescription_columns)
new_prescription_columns = []
for column in pivoted_prescription_df.columns:
     new_prescription_column = f'{column[0]}_{column[1]}'
     new_prescription_columns.append(new_prescription_column)
# Assign the new column names
pivoted_prescription_df.columns = new_prescription_columns
 
# Calculate the NaN percentage for each column
nan_percentage_prescription = pivoted_prescription_df.isna().mean()
nan_percentage_prescription
prescription_selected_columns = nan_percentage_prescription[nan_percentage_prescription <= threshold].index
# Filter the DataFrame based on selected columns
prescription_final_df = pivoted_prescription_df[prescription_selected_columns]

#===========Join df=========#
joined_df = master_df.merge(prescription_final_df, on='ADMISSION_ID', how='left')
joined_df = joined_df.merge(combined_interventions_df, on='ADMISSION_ID', how='left')
final_df = joined_df.merge(combined_measurements_df, on='ADMISSION_ID', how='left')

#=================Impute Categorical Data==============#
# Function to impute 0 for NaN cells if row sum is over 0 for columns with specific prefixes
def impute_zeros(row):
    relevant_columns = [col for col in row.index if any(col.startswith(prefix) for prefix in prefixes)]
    if row[relevant_columns].sum() > 0:
        return row.fillna(0)
    else:
        return row

# Selected Value Characters
prefixes = ['I311', 'I318', 'I323', 'I327', 'I331', 'I004', 'I016', 'I019']

final_df = final_df.apply(impute_zeros, axis=1)
final_df.to_csv('final_df.csv', index=False)


