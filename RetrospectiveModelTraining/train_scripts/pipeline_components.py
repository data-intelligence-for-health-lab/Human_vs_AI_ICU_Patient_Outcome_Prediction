import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

SEED = 42

def make_preprocessor(X_train):
    categorical_features = ['GENDER']
    continuous_features = X_train.columns.difference(categorical_features).tolist()
    
    gender_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    cont_transformer = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())

    preprocessor = make_column_transformer(
        (gender_transformer, categorical_features),
        (cont_transformer, continuous_features),
        sparse_threshold=0
    )

    return preprocessor

# apply_preporcessor for neural net
def apply_preprocessor(X_train, y_train, num_features=100, random_state=SEED):
    """
    Preprocesses the dataset, applies feature selection first using a pre-fitted model, 
    then applies SMOTE for balancing to a 1:1 ratio, and finally applies under-sampling to ensure class balance.
    
    Parameters:
    X_train (DataFrame): Training feature dataset
    y_train (array-like): Target labels
    num_features (int): Number of features to select
    random_state (int): Random seed for reproducibility
    
    Returns:
    X_train_selected (array): Transformed dataset with selected features
    y_train_balanced (array): Resampled target labels after SMOTE
    selected_features (Index): Names of selected features
    """
    print("Preprocessing and feature selecting...")

    # Split into categorical and continuous features
    categorical_features = ['GENDER']
    continuous_features = X_train.columns.difference(categorical_features).tolist()
    
    gender_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    cont_transformer = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())

    # Define column transformer
    preprocessor = make_column_transformer(
        (gender_transformer, categorical_features),
        (cont_transformer, continuous_features),
        sparse_threshold=0
    )
    
    # Transform the features using the column transformer
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Feature selection using RandomForestClassifier (prefit)
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    clf.fit(X_train_transformed, y_train)
    
    # Select the top features
    selector = SelectFromModel(clf, max_features=num_features, prefit=True)
    X_train_selected = selector.transform(X_train_transformed)
    selected_features = X_train.columns[selector.get_support()]
    
    # Apply SMOTE to balance classes to a 1:1 ratio on the selected features
    smote = SMOTE(random_state=random_state, sampling_strategy=1.0)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train)
    
    # Apply under-sampling after SMOTE to ensure balanced classes (if needed)
    under_sampler = RandomUnderSampler(random_state=random_state, sampling_strategy=1.0)
    X_resampled, y_resampled = under_sampler.fit_resample(X_train_balanced, y_train_balanced)
    
    # # Convert to PyTorch tensors
    # X_resampled_tensor = torch.tensor(X_resampled, dtype=torch.float32)
    # y_resampled_tensor = torch.tensor(y_resampled, dtype=torch.long)

    return X_resampled, y_resampled, selected_features
