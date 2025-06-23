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
   1. Generate a dataset with creatinine measurements for deriving AKI status
   2. Derive AKI status for <24hr, 24-48hr, >48hr after ICU admission based on creatinine measurements
   3. Derive Delirium statusfor <24hr, 24-48hr, >48hr after ICU admission based on delirium measurements
   4. In SQL, generate 4 tables for mastertable, aggregated tables for measuremnts, prescriptions and interventions for further processing in python
   5. In python load the 4 tables and perform pivoting, filtering and imputation and save final df as a csv

3) Preprocessing
   1. Extract 1000 testing samples; Extract 50 common patients
   2. Set up X_train, y_train, X_test, y_test
   3. Set up Pipeline (Imblearn Pipeline)
   4. Impute missing values numeric (median) and categorical (most frequent)
   5. Scale values
   6. Turn gender into binary
   7. Randomforest (100 estimators) to select top 100 features 
   8. SMOTE + RandomUnderSampler to medigate class imbalance

4) Training
   1. Predict Outcomes:
      'ICU_LOS', 'HOSP_LOS', 'ICU_EXPIRE_FLAG', 'HOSP_EXPIRE_FLAG', '30_DAYS_EXPIRE_FLAG', 'DELIRIUM_FLAG', 'AKI'
   2. Regression:
   - 'ICU_LOS',
   - 'HOSP_LOS'
   3. Classification:
   - 'HOSP_EXPIRE_FLAG',
   - '30_DAYS_EXPIRE_FLAG',
   - 'DELIRIUM_FLAG'
   - 'AKI'
   4. Models:
   - classification: logistic regression, SVM, Random Forest, XGBoost, NN
   - regression: Elastic net, SVM, Random Forest, XGBoost, NN

5) Select best model bsed on median F1
6) Predicting on testing set
7) Compare with clinician performances
