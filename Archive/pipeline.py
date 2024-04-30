import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data from file
df = pd.read_csv("final_df_317_1.csv")

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
# pca = PCA()
# smt = SMOTE()
preprocess_pipeline = Pipeline([('preprocessor', preprocessor)])
#, ('smt', smt), ('pca', pca)])

#print("test print:", X_train)

# Preprocess training set
X_train_preprocessed = preprocess_pipeline.fit_transform(X_train)

print("X_train_preprocessed:", X_train_preprocessed)

# Preprocess testing set
X_test_preprocessed = preprocess_pipeline.transform(X_test)
# print("X_train_preprocessed:", X_train_preprocessed)
# print("X_test", X_test)
# print("y_train", y_train)
# print("y_test", y_test)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

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


print(X_train_balanced.describe())
# model = LogisticRegression(max_iter=10000, tol=0.1)
# pipeline = Pipeline([('preprocessor', preprocessor), ('smt', smt), ('pca', pca), ('model', model)])
# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
# print(classification_report(y_test, y_pred))
