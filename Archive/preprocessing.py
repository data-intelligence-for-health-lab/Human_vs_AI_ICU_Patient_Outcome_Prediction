"""
This module preprocess data into training and testing sets

"""

import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
# from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
# from sklearn import metrics

# import xgboost as xgb

# Load data from file
df = pd.read_csv("final_df_317_1.csv")

print("Data Loaded...")

#----comment out in actual training---#
# Randomly select 10% of the data for testing
test_df = df.sample(frac=0.1, random_state=42)
# Remove the selected rows from the original DataFrame to create the training DataFrame
df = df.drop(test_df.index)
#----comment out in actual training---#

# Define the target column
target_column = "HOSP_EXPIRE_FLAG"

#Extract 1000 testing samples
test1000 = pd.read_csv("1000Patients.csv")
testing_ids = test1000['PATIENT_ID']
admission_ids = test1000['ADMISSION_ID']
testing_set = df[df['ADMISSION_ID'].isin(admission_ids)]
training_set = df[~df['PATIENT_ID'].isin(testing_ids)]
key_columns = ['PATIENT_ID', 'ADMISSION_ID', 'ICU_LOS', 'HOSP_LOS', 'ICU_EXPIRE_FLAG', 'HOSP_EXPIRE_FLAG', '30_DAYS_EXPIRE_FLAG', 'DELIRIUM_FLAG']

# Drop the target columns and patient identifiers to create X_train, X_test
X_train = training_set.drop(columns=key_columns)
X_test = testing_set.drop(columns=key_columns)

#'Extract the target values to create y_train
y_train = training_set[target_column]
y_test = testing_set[target_column]

#define Numeric+Categorical Features
numerical_features = X_train.columns[0:1].tolist() + X_train.columns[2:524].tolist()
categorical_features = X_train.columns[524:567].tolist()
columns = ['GENDER'] + numerical_features + categorical_features
X_train = X_train[columns]
X_test = X_test[columns]

#tansform columns: impute and scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.nan,strategy='most_frequent')),
    ('scaler', StandardScaler())
])

#PCA
pca_transformer = PCA(n_components=100)

preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(drop='first'), ['GENDER']),
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('pca', pca_transformer, numerical_features)
    ],
    remainder='passthrough'
)

preprocess_pipeline = Pipeline([('preprocessor', preprocessor)])

# Preprocess training set
X_train = pd.DataFrame(preprocess_pipeline.fit_transform(X_train), columns=columns)
X_test = pd.DataFrame(preprocess_pipeline.fit_transform(X_test), columns=columns)

print("ready for SMOTE", X_train, X_test)

# Apply random undersampling to the majority class
undersampler = RandomUnderSampler(random_state=42)
X_train_undersampled, y_train_undersampled = undersampler.fit_resample(X_train, y_train)

# Calculate the class imbalance ratio
CIR = len(y_train_undersampled[y_train_undersampled == 0]) / len(y_train_undersampled[y_train_undersampled == 1])

# Determine the desired ratio between the minority and majority class
class_ratio = 1

# Calculate the number of minority class instances to generate using SMOTE
minority_samples = int(class_ratio * len(y_train_undersampled[y_train_undersampled == 0]))

# Perform oversampling using SMOTE
smote = SMOTE(sampling_strategy={1: minority_samples}, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_undersampled, y_train_undersampled)

print(X_train_balanced.describe())
print(y_train_balanced.describe())

#------------------------model begins here--------------------------------#
