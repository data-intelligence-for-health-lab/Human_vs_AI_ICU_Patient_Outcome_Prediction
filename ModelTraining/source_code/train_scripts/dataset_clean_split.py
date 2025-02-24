import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

def data_cleaner(final_df_path, comparison_ids_path, target):
    """Loads and cleans dataset, returning training and comparison sets."""
    print("1. Cleaning dataset...")
    df = pd.read_csv(final_df_path)
    compare1000 = pd.read_csv(comparison_ids_path)
    
    key_columns = [
        'PATIENT_ID', 'ADMISSION_ID', 'ICU_LOS', 'HOSP_LOS', 
        'ICU_EXPIRE_FLAG', 'HOSP_EXPIRE_FLAG', '30_DAYS_EXPIRE_FLAG', 
        'DELIRIUM_LABEL', 'AKI_LABEL'
    ]
    
    compare_ids = compare1000['PATIENT_ID']
    admission_ids = compare1000['ADMISSION_ID']
    compare_set = df[df['ADMISSION_ID'].isin(admission_ids)]
    train_test_set = df[~df['PATIENT_ID'].isin(compare_ids)]
    
    X = train_test_set.drop(columns=key_columns)
    y = train_test_set[target]
    
    return compare_set, X, y


def split_dataset(X, y):
    """Splits the dataset into training and testing sets."""
    print("2. Splitting dataset...")
    X_test, X_train, y_test, y_train = train_test_split(
        X, y, train_size=1000, stratify=y, random_state=SEED
    )
    return X_train, X_test, y_train, y_test
