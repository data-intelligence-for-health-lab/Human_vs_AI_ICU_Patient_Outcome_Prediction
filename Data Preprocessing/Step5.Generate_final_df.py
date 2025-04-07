"""
This script is used to pivot datasets from SQL for intervensions, measurements and prescription, drop value char with missingess > 30% and combine with the mastelist to generate ML compatible dataframe
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
engine = create_engine('postgresql+psycopg2://lu.kuang:EFTd66Qa3QhnMeA7\
@10.44.113.30/ABeICU')

# Connect to the database
master_df = pd.read_sql_query('SELECT * FROM "lu.kuang"."FINAL_MASTERTABLE"',con=engine)
interventions_df = pd.read_sql_query('SELECT * FROM "lu.kuang".interventions_24_agg',con=engine)
measurements_df = pd.read_sql_query('SELECT * FROM "lu.kuang".measurements_24_agg',con=engine)
prescription_df = pd.read_sql_query('SELECT * FROM "lu.kuang".prescriptions_24_agg',con=engine)

#Value characters
measurements_value_char = pd.read_sql_query('SELECT * FROM "lu.kuang".measurements_vc_24hr',con=engine)
interventions_value_char = pd.read_sql_query('SELECT * FROM "lu.kuang".interventions_vc_24hr',con=engine)

# Add derived AKI and Delirium status to master_df
# Add derived AKI_STATUS column
aki_conditions = [
    (master_df['AKI_STATUS_24H'] == 1),  # If AKI_STATUS_24H is 1 then NULL
    (master_df['AKI_STATUS_24_48H'] == 1) | (master_df['AKI_STATUS_48H_PLUS'] == 1),  # If AKI_24 is not 1 (null or 0) and If either AKI_STATUS_24_48H or AKI_STATUS_48H_PLUS is 1
    (master_df['AKI_STATUS_24_48H'] == 0) & (master_df['AKI_STATUS_48H_PLUS'] == 0)  # If both AKI_STATUS_24_48H and AKI_STATUS_48H_PLUS are 0
]
aki_values = [np.nan, 1, 0]  # Corresponding values for the conditions
master_df['AKI_STATUS'] = np.select(aki_conditions, aki_values, default=np.nan)  # Default to NaN for other cases

# Add derived DELIRIUM_STATUS column
delirium_conditions = [
    (master_df['DELIRIUM_STATUS_24H'] == 1),  # If DELIRIUM_STATUS_24H is 1
    (master_df['DELIRIUM_STATUS_24_48H'] == 1) | (master_df['DELIRIUM_STATUS_48H_PLUS'] == 1),  # If either DELIRIUM_STATUS_24_48H or DELIRIUMSTATUS_48H_PLUS is 1
    (master_df['DELIRIUM_STATUS_24_48H'] == 0) & (master_df['DELIRIUM_STATUS_48H_PLUS'] == 0)  # If both DELIRIUM_STATUS_24_48H and DELIRIUMSTATUS_48H_PLUS are 0
]
delirium_values = [np.nan, 1, 0]  # Corresponding values for the conditions
master_df['DELIRIUM_STATUS'] = np.select(delirium_conditions, delirium_values, default=np.nan)  # Default to NaN for other cases

#define functions used to process numeric and categorical data
def pivot_df(df, index_col, columns_col, values_cols):
    """
    Pivot the given DataFrame, flatten multi-index columns, and filter based on NaN threshold.
    """
    # Pivot table
    pivoted_df = df.pivot_table(index=index_col, columns=columns_col, values=values_cols)
    
    # Flatten multi-index columns
    pivoted_df.columns = [f"{col[1]}_{col[0]}" for col in pivoted_df.columns]
    
    # Reset index
    pivoted_df = pivoted_df.reset_index()
    
    return pivoted_df

def one_hot_encode(df, index_col, columns_cols):
    """
    One-hot encode categorical values by pivoting.
    """
    df["count"] = 1
    pivoted_df = df.pivot_table(
        index=index_col, 
        columns=columns_cols, 
        values="count", 
        aggfunc="sum", 
        fill_value= 0
    )
    
    # Flatten multi-index columns
    pivoted_df.columns = [f"{col[0]}_{col[1].replace(' ', '_')}" for col in pivoted_df.columns]
    
    # Reset index
    return pivoted_df.reset_index()
    
# Process numeric data
# Define shared and specific metric lists
shared_metrics = [
    "average_value", "min_value", "max_value", "percentile_25", "median_value", 
    "percentile_75", "range", "standard_deviation", "iqr"
]
prescription_metrics = [
    "total_time", "total_volume", "average_rate", "min_rate", "max_rate", 
    "percentile_25_rate", "median_value_rate", "percentile_75_rate", "rate_range", 
    "standard_deviation_rate", "iqr_rate"
]

#=========================Interventions========================#
interventions_num_df = pivot_df(interventions_df, "ADMISSION_ID", "ITEM_ID", shared_metrics)
#=========================Measurements========================#
measurements_num_df = pivot_df(measurements_df, "ADMISSION_ID", "ITEM_ID", shared_metrics)
#=========================Prescriptions========================#
prescription_num_df = pivot_df(prescription_df, "ADMISSION_ID", "ITEM_ID", prescription_metrics)

# Combine all numeric data into a combined dataframe            
combined_df = interventions_num_df.merge(measurements_num_df, on='ADMISSION_ID', how='outer') \
                                  .merge(prescription_num_df, on='ADMISSION_ID', how='outer')

# Remove columns with NaN values above the threshold (30%)
combined_df_nan_percentage = combined_df.isna().mean()
combined_df_selected_columns = combined_df_nan_percentage[combined_df_nan_percentage <= 0.7].index
combined_df_filtered = combined_df[combined_df_selected_columns]

# Process categorical data
# One-hot encode categorical values
pivoted_interventions_vc = one_hot_encode(interventions_value_char, "ADMISSION_ID", ["ITEM_ID", "VALUE_CHAR"])
pivoted_measurements_vc = one_hot_encode(measurements_value_char, "ADMISSION_ID", ["ITEM_ID", "VALUE_CHAR"])

# Merge interventions and measurements vc with combined_df and master_df to generate final_df
final_df = master_df.merge(combined_df_filtered, on='ADMISSION_ID', how='left') \
                    .merge(pivoted_interventions_vc, on='ADMISSION_ID', how='left') \
                    .merge(pivoted_measurements_vc, on='ADMISSION_ID', how='left')

final_df.to_csv('final_df_0403.csv', index=False)