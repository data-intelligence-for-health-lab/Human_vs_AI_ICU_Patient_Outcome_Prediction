import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn import metrics

import xgboost as xgb

# Load data from file
df = pd.read_csv("final_df_317_1.csv")
# Randomly select 10% of the data for testing
test_df = df.sample(frac=0.0001, random_state=42)
# Remove the selected rows from the original DataFrame to create the training DataFrame
df = df.drop(test_df.index)

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

#print(X_train)

#define Numeric+Categorical Features
numerical_features = X_train.columns[0:1].tolist() + X_train.columns[2:524].tolist()
categorical_features = X_train.columns[524:567].tolist()
columns = ['GENDER'] + numerical_features + categorical_features
X_train = X_train[columns]
X_test = X_test[columns]
#print(X_test)
#print(numerical_features)
#print(categorical_features)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.nan,strategy='most_frequent')),
    ('scaler', StandardScaler())
])

#print(numerical_features)
preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(drop='first'), ['GENDER']),
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

preprocess_pipeline = Pipeline([('preprocessor', preprocessor)])

# Preprocess training set
X_train = pd.DataFrame(preprocess_pipeline.fit_transform(X_train), columns=columns)
X_test = pd.DataFrame(preprocess_pipeline.fit_transform(X_test), columns=columns)

#print(X_train)


# Apply random undersampling to the majority class
undersampler = RandomUnderSampler(random_state=42)
X_train_undersampled, y_train_undersampled = undersampler.fit_resample(X_train, y_train)

# Calculate the class imbalance ratio
CIR = len(y_train_undersampled[y_train_undersampled == 0]) / len(y_train_undersampled[y_train_undersampled == 1])

# Determine the desired ratio between the minority and majority class
desired_ratio = 1

# Calculate the number of minority class instances to generate using SMOTE
minority_samples = int(desired_ratio * len(y_train_undersampled[y_train_undersampled == 0]))

# Perform oversampling using SMOTE
smote = SMOTE(sampling_strategy={1: minority_samples}, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_undersampled, y_train_undersampled)

#print(y_train_balanced.describe())


# Define parameter grid for grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001]
}

# Create XGBoost classifier
xg_class = xgb.XGBClassifier()

# Perform grid search with cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=xg_class, param_grid=param_grid, cv=cv, scoring='roc_auc')
grid_search.fit(X_train_balanced, y_train_balanced)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Perform cross-validation
cv_scores = cross_val_score(best_model, X_train_balanced, y_train_balanced, cv=cv, scoring='roc_auc')

# Print average cross-validation score
print('Average Cross-Validation ROC AUC: %.4f' % cv_scores.mean())

# Evaluate predictions
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)
print('AUC: %f' % metrics.auc(fpr, tpr))
print('Precision: %f' % metrics.precision_score(y_test, y_pred))
print('Recall: %f' % metrics.recall_score(y_test, y_pred))
print("Accuracy: %.2f%%" % metrics.accuracy_score(y_test, y_pred))
print("F1 Score: %.2f%%" % metrics.f1_score(y_test, y_pred))
