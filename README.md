# Human_vs_AI_ICU_Patient_Outcome_Prediction
Data preprocessing steps
1) Retrospective Data (ABeICU)
   1. Create a mastertable with patient demographics and prediction outcomes
   2. Keep first 24 hour data in Measurements, Interventions, Prescriptions
   3. Aggregate value_char and value_num in each table to have unique combination of Admission_ID and ITEM_ID
   4. Pivot table to have a dual index of Admission_ID and ITEM_ID, add ITEM_ID to column names
   5. Drop any column with 30% missing data (drop if 30% of patients do not have this item)
   6. Join the 3 tables back to mastertable to create a dataframe

2) Data Cleaning Steps
   1. Manually select categorical data (value characters that made sense to convert)
   2. Impute 0 for NaN cells if row sum is over 0 for columns with specific prefixes

3) Preprocessing
   1. Extract 1000 testing samples; Extract 50 common patients
   2. Set up X_train, y_train (47781 patients), X_test, y_test (990 patients)
   3. Set up Pipeline (Imblearn Pipeline)
   4. impute missing values numeric (median) and categorical (most frequent)
   5. scale values
   6. Turn gender into binary
