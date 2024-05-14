#Classification
from common_scripts.run_models import lr_c_model, svm_c_model, rf_c_model, xgb_c_model, nn_c_model, en_r_model, svm_r_model, rf_r_model, xgb_r_model, nn_r_model

import numpy as np
import pandas as pd

from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier


#Regression Models
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


#define column transformers
def make_preprocessor(X_train):
    cont_feats = X_train.columns[0:1].tolist() + X_train.columns[2:524].tolist()
    cat_feats = X_train.columns[524:567].tolist()
    gender_transformer = OneHotEncoder(drop='first')
    cont_transformer = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='median'), StandardScaler())
    cat_transformer = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='most_frequent'), StandardScaler())
    preprocessor = make_column_transformer((gender_transformer, ['GENDER']), (cont_transformer, cont_feats), (cat_transformer, cat_feats), sparse_threshold=0)
    return preprocessor

# Load data from file
df = pd.read_csv("final_df_513.csv")

#----comment out in actual training---#
# Randomly select 10% of the data for testing
test_df = df.sample(frac=0.001, random_state=42)
# Remove the selected rows from the original DataFrame to create the training DataFrame
df = df.drop(test_df.index)
#----comment out in actual training---#

print("Data Loaded...")

# Define the target column
target_column = "HOSP_EXPIRE_FLAG"

#Extract 1000 testing samples
test1000 = pd.read_csv("1000Patients.csv")
testing_ids = test1000['PATIENT_ID']
admission_ids = test1000['ADMISSION_ID']
testing_set = df[df['ADMISSION_ID'].isin(admission_ids)]
training_set = df[~df['PATIENT_ID'].isin(testing_ids)]
#print(training_set)
key_columns = ['PATIENT_ID', 'ADMISSION_ID', 'icu_los', 'hosp_los', 'ICU_EXPIRE_FLAG', 'HOSP_EXPIRE_FLAG', '30_DAYS_EXPIRE_FLAG', 'DELIRIUM_FLAG']

# Drop the target columns and patient identifiers to create X_train, X_test
X_train = training_set.drop(columns=key_columns)
X_test = testing_set.drop(columns=key_columns)

#Extract the target values to create y_train
y_train = training_set[target_column]
y_test = testing_set[target_column]

#define componets in the pipline
preprocessor = make_preprocessor(X_train)
pca = PCA(n_components=100)
over_sampler = SMOTE(random_state=42, sampling_strategy=0.5)
under_sampler = RandomUnderSampler(random_state=42, sampling_strategy=1.0)

#define models
lrmodel_c = LogisticRegression()
svmmodel_c = svm.SVC()
rfmodel_c = RandomForestClassifier()
xgbmodel_c = xgb.XGBClassifier()
nnmodel_c = MLPClassifier()

enmodel_r = ElasticNet()
svmmodel_r = svm.SVR()
rfmodel_r = RandomForestRegressor()
xgbmodel_r = xgb.XGBRegressor()
nnmodel_r = MLPRegressor()

#------------------------------------------Classification------------------------------------------#
# #Logistic Regression Pipeline
# print("Run Logistic Regression Pipeline...")
# final_lr_pipeline = lr_c_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, lrmodel_c)

# #SVM Pipeline
# print("Run SVM Pipeline...")
# final_svm_pipline = svm_c_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, svmmodel_c)

# #Random Forest Pipeline
# print("Run Random Forest Pipeline...")
# final_rf_pipline = rf_c_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, rfmodel_c)

# #XGBoost
print("Run XGBoost Pipeline...")
final_xgb_pipline = xgb_c_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, xgbmodel_c)

# #Neural Network
# print("Run Neural Network...")
# final_nn_pipline = nn_c_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, nnmodel_c)


# #------------------------------------------Regression------------------------------------------#
# #Elastic Net Pipeline
# final_en_pipeline = en_r_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, enmodel_r)

# #SVM Pipeline
# print("Regression: Run SVM Pipeline...")
# final_svm_pipline = svm_r_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, svmmodel_c)

# #Random Forest Pipeline
# print("Regression: Run Random Forest Pipeline...")
# final_rf_pipline = rf_r_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, rfmodel_c)

# #XGBoost
# print("Regression: Run XGBoost Pipeline...")
# final_xgb_pipline = xgb_r_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, xgbmodel_c)

# #Neural Network
# print("Regression: Run Neural Network...")
# final_nn_pipline = nn_r_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, nnmodel_c)