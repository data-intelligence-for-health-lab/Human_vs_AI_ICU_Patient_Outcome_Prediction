#Classification
from common_scripts.run_models import lr_c_model, svm_c_model, rf_c_model, xgb_c_model, en_r_model, svm_r_model, rf_r_model, xgb_r_model, nn_c_model, nn_r_model
from common_scripts import hyperopt_models
import pickle

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

#Regression Models
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

# Add Keras imports
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam,SGD
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2' # change to available 
# os.environ['OPENBLAS_NUM_THREADS'] = '10' #6

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
test_df = df.sample(frac=0.01, random_state=42)
print("Test dataset size:", len(test_df))
df = test_df.reset_index(drop=True)
print(df)
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

#====================use for delirium=========================#
# # Drop rows where 'delirium_flag_24' is null
# testing_set = testing_set.dropna(subset=['delirium_flag_24'])
# training_set = training_set.dropna(subset=['delirium_flag_24'])

# print(training_set)
# print(testing_set)

# # Count the number of 0 vs 1 in 'delirium_flag_24' for both testing and training sets
# delirium_flag_24_testing_counts = testing_set['delirium_flag_24'].value_counts()
# delirium_flag_24_training_counts = training_set['delirium_flag_24'].value_counts()

# # Print the counts
# print("Testing Set - delirium_flag_24 counts:")
# print(delirium_flag_24_testing_counts)
# print("\nTraining Set - delirium_flag_24 counts:")
# print(delirium_flag_24_training_counts)

# # Display key columns from the training set
# key_columns = ['PATIENT_ID', 'ADMISSION_ID', 'icu_los', 'hosp_los', 'ICU_EXPIRE_FLAG', 'HOSP_EXPIRE_FLAG', '30_DAYS_EXPIRE_FLAG', 'DELIRIUM_FLAG', 'delirium_flag_24', 'delirium_flag_48']
# #print(training_set[key_columns])

# # Drop the target columns and patient identifiers to create X_train, X_test
# X_train = training_set.drop(columns=key_columns)
# X_test = testing_set.drop(columns=key_columns)

# #Extract the target values to create y_train
# y_train = training_set[target_column]
# y_test = testing_set[target_column]

# #define componets in the pipline
# preprocessor = make_preprocessor(X_train)
# pca = PCA(n_components=100)
# # over_sampler = SMOTE(random_state=42, sampling_strategy=0.5)
# # under_sampler = RandomUnderSampler(random_state=42, sampling_strategy=1.0)

# # Increase the count of the minority class to match the majority class
# over_sampler = SMOTE(random_state=42, sampling_strategy={1.0: 23209})
# # Decrease the count of the majority class to match the updated minority class
# under_sampler = RandomUnderSampler(random_state=42, sampling_strategy={0.0: 19682})
#========================================================================================#
input_shape = X_train.shape[1]  # Define the input shape 

#define models
lrmodel_c = LogisticRegression()
svmmodel_c = svm.SVC()
rfmodel_c = RandomForestClassifier()
xgbmodel_c = xgb.XGBClassifier()

enmodel_r = ElasticNet()
svmmodel_r = svm.SVR()
rfmodel_r = RandomForestRegressor()
xgbmodel_r = xgb.XGBRegressor()


#------------------------------------------Classification------------------------------------------#
# #Logistic Regression Pipeline
# print("Run Logistic Regression Pipeline...")
# final_lr_pipeline = lr_c_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, lrmodel_c, target = target_column)

# #SVM Pipeline
print("Run SVM Pipeline...")
final_svm_pipline = svm_c_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, svmmodel_c, target = target_column)

# #Random Forest Pipeline
# print("Run Random Forest Pipeline...")
# final_rf_pipline = rf_c_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, rfmodel_c, target = target_column)

# #XGBoost
# print("Run XGBoost Pipeline...")
# final_xgb_pipline = xgb_c_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, xgcbmodel_c, target = target_column)

#Neural Network
# print("Run Neural Network...")
# nn_c_hyperparameters = nn_c_model(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, input_shape, target = target_column)

# opt = ''
# if nn_c_hyperparameters['optimizer'] == "adam":
#     opt = Adam()
# elif nn_c_hyperparameters['optimizer'] == "sgd":
#     opt = SGD()
# hidden_units = [nn_c_hyperparameters[f'hidden_units_{i}'] for i in range(nn_c_hyperparameters["num_hidden_layers"])]

# def save_nn_r_model():
#     model = Sequential()
#     model.add(Dense(hidden_units[0], input_dim=input_shape, activation=nn_r_hyperparameters["activation"]))
#     for i in range(1, nn_r_hyperparameters["num_hidden_layers"]):
#         model.add(Dense(hidden_units[i], activation=nn_r_hyperparameters["activation"]))
#     model.add(Dense(1, activation="sigmoid"))
#     model.compile(loss='binary_crossentropy', optimizer=opt, metrics=hyperopt_models.average_precision)
#     return model

# model = KerasClassifier(build_fn=save_nn_r_model, epochs=100, batch_size=32, verbose=0)
# nn_pipeline = make_pipeline(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, model)
# nn_pipeline = nn_pipeline.fit(X_train, y_train)
# pickle.dump(nn_pipeline, open('Outputs/Regression/nn_r_model.pkl', 'wb'))


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

# nn_r_hyperparameters = nn_r_model(X_train, y_train, preprocessor, pca, over_sampler, under_sampler)

# opt_r = ''
# if nn_r_hyperparameters['optimizer'] == "adam":
#     opt_r = Adam()
# elif nn_r_hyperparameters['optimizer'] == "sgd":
#     opt_r = SGD()
# hidden_units = [nn_r_hyperparameters[f'hidden_units_{i}'] for i in range(nn_r_hyperparameters["num_hidden_layers"])]

# def save_nn_r_model():
#     model = Sequential()
#     model.add(Dense(hidden_units[0], input_dim=input_shape, activation=nn_r_hyperparameters["activation"]))
#     for i in range(1, nn_r_hyperparameters["num_hidden_layers"]):
#         model.add(Dense(hidden_units[i], activation=nn_r_hyperparameters["activation"]))
#     model.add(Dense(1, activation="sigmoid"))
#     model.compile(loss='binary_crossentropy', optimizer=opt, metrics=hyperopt_models.average_precision)
#     return model

# model = KerasClassifier(build_fn=save_nn_r_model, epochs=100, batch_size=32, verbose=0)
# nn_pipeline = make_pipeline(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, model)
# nn_pipeline = nn_pipeline.fit(X_train, y_train)
# pickle.dump(nn_pipeline, open('Outputs/Regression/nn_r_model.pkl', 'wb'))
