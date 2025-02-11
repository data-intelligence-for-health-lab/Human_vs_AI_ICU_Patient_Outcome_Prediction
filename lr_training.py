#Classification 
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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
#define column transformers
def make_preprocessor(X_train):
    cont_feats = X_train.columns[0:1].tolist() + X_train.columns[2:].tolist()
    gender_transformer = OneHotEncoder(drop='first')
    cont_transformer = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='median'), StandardScaler())
    preprocessor = make_column_transformer((gender_transformer, ['GENDER']), (cont_transformer, cont_feats), sparse_threshold=0)
    return preprocessor

# Load data from file
df = pd.read_csv("final_df_513.csv")

#----comment out in actual training---#
# Randomly select 10% of the data for testing 
test_df = df.sample(frac=0.1, random_state=42)
# Remove the selected rows from the original DataFrame to create the training DataFrame
df = df.drop(test_df.index)
#----comment out in actual training---#

print("Data Loaded...")

# Define the target column
target_column = "30_DAYS_EXPIRE_FLAG"

#Extract 1000 testing samples
test1000 = pd.read_csv("1000Patients.csv")
testing_ids = test1000['PATIENT_ID']
admission_ids = test1000['ADMISSION_ID']
testing_set = df[df['ADMISSION_ID'].isin(admission_ids)]
training_set = df[~df['PATIENT_ID'].isin(testing_ids)]
#print(training_set)
key_columns = ['PATIENT_ID', 'ADMISSION_ID', 'icu_los', 'hosp_los', 'ICU_EXPIRE_FLAG', 'HOSP_EXPIRE_FLAG', '30_DAYS_EXPIRE_FLAG', 'DELIRIUM_FLAG']
print(training_set[key_columns])

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
# Define Logistic Regression model
logreg_model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)

# Build the pipeline
pipeline = make_pipeline(
    preprocessor,  # Preprocessing
    over_sampler,  # SMOTE for oversampling
    under_sampler,  # Undersampling
    pca,  # PCA for dimensionality reduction
    logreg_model  # Logistic Regression
)

# Define hyperparameter grid for tuning
param_grid = {
    'logisticregression__C': [0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
    'logisticregression__penalty': ['l1', 'l2'],  # Regularization type
    'logisticregression__solver': ['liblinear']  # Solver for small datasets
}

# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')

# Train the pipeline with hyperparameter tuning
print("Running GridSearchCV for Logistic Regression with hyperparameter tuning...")
grid_search.fit(X_train, y_train)

# Best hyperparameters found
print("Best Hyperparameters:", grid_search.best_params_)

# Predict on the test set using the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model with different metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Logistic Regression with hyperparameter tuning pipeline completed.")