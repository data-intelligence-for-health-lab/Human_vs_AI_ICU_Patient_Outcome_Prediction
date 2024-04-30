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
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Load data from file
df = pd.read_csv("final_df_317_1.csv")

print("Data Loaded...")

#----comment out in actual training---#
# Randomly select 10% of the data for testing
test_df = df.sample(frac=0.001, random_state=42)
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
#pca_transformer = PCA(n_components=100)

preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(drop='first'), ['GENDER']),
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
        #('pca', pca_transformer, numerical_features)
    ],
    remainder='passthrough'
)

preprocess_pipeline = Pipeline([('preprocessor', preprocessor)])

# Preprocess training set
X_train = pd.DataFrame(preprocess_pipeline.fit_transform(X_train), columns=columns)
X_test = pd.DataFrame(preprocess_pipeline.fit_transform(X_test), columns=columns)

#PCA
print("start PCA...")
pca = PCA(n_components=100)
X_train = pd.DataFrame(pca.fit_transform(X_train))
X_test = pd.DataFrame(pca.transform(X_test))

#SMOTE
print("start SMOTE")
# Applying smote, first 50% oversampling and then 100% undersampling to achieve a balanced set
over = SMOTE(sampling_strategy=0.5)
under = RandomUnderSampler(sampling_strategy=1.0)
steps = [('o', over), ('u', under)]
smt_pipeline = Pipeline(steps=steps)
X_train, y_train = smt_pipeline.fit_resample(X_train, y_train)

print(X_train.describe())
print(y_train.describe())
print("finished preprocessing!")

#------------------------model begins here--------------------------------#
# parameter grid definition
param_grid = [{'solver':['newton-cg','lbfgs','sag'],'penalty':['l2'],'C':np.arange(0.01,10.0,0.01)}, #0.001, 0.1, 1, 10
                {'solver': ['liblinear','saga'],'penalty': ['l1','l2'],'C':np.arange(0.01,10.0,0.01)},
                {'solver':['newton-cg', 'lbfgs', 'sag', 'saga'], 'penalty': ['none']}]

# grisdearch definition and fitting              
gs = GridSearchCV(LogisticRegression(), param_grid, cv=10,  
                    scoring=['roc_auc','accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy'], refit = 'balanced_accuracy',verbose=2)
gs.fit(X_train, y_train)

# printing check point
print('GridSearch completed')

# Print evaluation metrics
print("Best parameters:", gs.best_params_)
print("Best balanced accuracy:", gs.best_score_)

# Get the best model and make predictions on the test set
best_model = gs.best_estimator_
y_pred = best_model.predict(X_test)

# Print classification report
classification_report = classification_report(y_test, y_pred)
print(classification_report)