import pandas as pd
import numpy as np

# Load data from file
df = pd.read_csv("final_df_317_1.csv")
#print(df)

#Extract 1000 testing samples
test1000 = pd.read_csv("1000Patients.csv")
testing_ids = test1000['PATIENT_ID']
admission_ids = test1000['ADMISSION_ID']
testing_set = df[df['ADMISSION_ID'].isin(admission_ids)]
#print(testing_set)
training_set = df[~df['PATIENT_ID'].isin(testing_ids)]
#print(testing_set)

# Define the target column
target_column = 'HOSP_EXPIRE_FLAG' 

# Drop the target columns and patient identifiers to create X_train, X_test
X_train = training_set.drop(columns=['HOSP_EXPIRE_FLAG','PATIENT_ID', 'ADMISSION_ID', 'ICU_LOS', 'HOSP_LOS', 'ICU_EXPIRE_FLAG', 'ICU_EXPIRE_FLAG', '30_DAYS_EXPIRE_FLAG', 'DELIRIUM_FLAG'])
X_test = testing_set.drop(columns=['HOSP_EXPIRE_FLAG','PATIENT_ID', 'ADMISSION_ID', 'ICU_LOS', 'HOSP_LOS', 'ICU_EXPIRE_FLAG', 'ICU_EXPIRE_FLAG', '30_DAYS_EXPIRE_FLAG', 'DELIRIUM_FLAG'])
# Extract the target values to create y_train
y_train = training_set[target_column]
y_test = testing_set[target_column]

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
