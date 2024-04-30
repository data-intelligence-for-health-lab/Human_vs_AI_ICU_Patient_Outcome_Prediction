import numpy as np
import pandas as pd

from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
# from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Load data from file
df = pd.read_csv("final_df_317_1.csv")

print("Data Loaded...")

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

#define column transformers
def make_preprocessor(X_train):
    cont_feats = X_train.columns[0:1].tolist() + X_train.columns[2:524].tolist()
    cat_feats = X_train.columns[524:567].tolist()
    gender_transformer = OneHotEncoder(drop='first')
    cont_transformer = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='median'), StandardScaler())
    cat_transformer = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='most_frequent'), StandardScaler())
    preprocessor = make_column_transformer((gender_transformer, ['GENDER']), (cont_transformer, cont_feats), (cat_transformer, cat_feats), sparse_threshold=0)
    return preprocessor

preprocessor = make_preprocessor(X_train)
pca = PCA(n_components=100)
over_sampler = SMOTE(sampling_strategy=0.5)
under_sampler = RandomUnderSampler(sampling_strategy=1.0)
model = LogisticRegression()

pipeline = make_pipeline(preprocessor, pca, over_sampler, under_sampler, model)

pipeline.fit(X_train, y_train)
# Make predictions on the test data
y_pred = pipeline.predict(X_test)
# print out the score accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
