# Human_vs_AI_ICU_Patient_Outcome_Prediction
Data preprocessing steps
1) Retrospective Data (ABeICU)
   1. Create a mastertable with patient demographics and prediction outcomes
   2. Keep first 24 hour data in Measurements, Interventions, Presctiptions
   3. Aggregate value_char and value_num in each table to have unique combination of Admission_ID and ITEM_ID
   4. Pivot table to have a dual index of Admission_ID and ITEM_ID, add ITEM_ID to column names
   5. Drop any column with 30% missing data (drop if 30% of particiapnts do not have this item)
   6. Join the 3 tables back to mastertable to create a dataframe
