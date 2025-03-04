import pandas as pd

SEED = 42

def data_cleaner(final_df_path, comparison_ids_path, target):
    """Loads and cleans dataset, returning training and comparison sets."""
    print("1. Cleaning dataset...")
    df = pd.read_csv(final_df_path)
    test1000 = pd.read_csv(comparison_ids_path)
    
    key_columns = [
        'PATIENT_ID', 'ADMISSION_ID', 'ICU_LOS', 'HOSP_LOS', 
        'ICU_EXPIRE_FLAG', 'HOSP_EXPIRE_FLAG', '30_DAYS_EXPIRE_FLAG', 
        'DELIRIUM_LABEL', 'AKI_LABEL'
    ]
    
    testing_ids = test1000['PATIENT_ID']
    admission_ids = test1000['ADMISSION_ID']
    testing_set = df[df['ADMISSION_ID'].isin(admission_ids)]
    training_set = df[~df['PATIENT_ID'].isin(testing_ids)]
    
    # Drop the target columns and patient identifiers to create X_train, X_test
    X_train = training_set.drop(columns=key_columns)
    X_test = testing_set.drop(columns=key_columns)

    #Extract the target values to create y_train
    y_train = training_set[target]
    y_test = testing_set[target]
    
    return X_train, X_test, y_train, y_test


# def split_dataset(X, y):
#     """Splits the dataset into training and testing sets."""
#     print("2. Splitting dataset...")
#     X_test, X_train, y_test, y_train = train_test_split(
#         X, y, test_size=1000, stratify=y, random_state=SEED
#     )
#     return X_train, X_test, y_train, y_test
