import pandas as pd
#import numpy as np

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
#'Extract the target values to create y_train
y_train = training_set[target_column]
y_test = testing_set[target_column]

numerical_features = X_train.columns[0:1].tolist() + X_train.columns[2:524].tolist()
categorical_features = X_train.columns[524:567].tolist()
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler())
])  

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first'), ['GENDER']),
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

pca = PCA()
smt = SMOTE()
logistic = LogisticRegression(max_iter=10000, tol=0.1)
pipeline = Pipeline([('preprocessor', preprocessor), ('smt', smt), ('pca', pca), ('logistic', logistic)])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
