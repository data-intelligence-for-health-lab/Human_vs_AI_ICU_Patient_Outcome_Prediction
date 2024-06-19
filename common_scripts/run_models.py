from common_scripts.hyperopt_models import lr_hyperopt, svm_c_hyperopt, rf_c_hyperopt, xgb_c_hyperopt, nn_c_hyperopt
from common_scripts.regression_models import elastic_net_hyperopt, svm_r_hyperopt, rf_r_hyperopt, xgb_r_hyperopt, nn_r_hyperopt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import  mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error
import pickle
from imblearn.pipeline import make_pipeline

# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import Adam
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.callbacks import EarlyStopping

import pandas as pd
import os
#=============================Classification models and metrics=============================#
def generate_metrics(pipeline, X_test, y_test, model_name, target):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    metrics = {
        'Target': target,
        'Model': model_name,
        'True Positives': tp,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc,
        'Average Precision': avg_precision
    }
    
    # Create output folder if it doesn't exist
    output_folder = 'Outputs'
    os.makedirs(output_folder, exist_ok=True)
    
    # Save metrics as a table
    metrics_df = pd.DataFrame([metrics])
    output_filename = f'performance_metrics_{model_name}.csv'
    output_path = os.path.join(output_folder, output_filename)
    
    if os.path.isfile(output_path):
        # Append to existing file if it exists
        existing_metrics_df = pd.read_csv(output_path)
        metrics_df = pd.concat([existing_metrics_df, metrics_df], ignore_index=True)
    
    metrics_df.to_csv(output_path, index=False)
    
    print("Metrics saved successfully.")
    
def lr_c_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, model, target):
    lr_hyperparameters = lr_hyperopt(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, model)
    print(lr_hyperparameters)
    tuned_lr_model = LogisticRegression(**lr_hyperparameters)
    lr_pipeline = make_pipeline(preprocessor, pca, over_sampler, under_sampler, tuned_lr_model)
    lr_pipeline = lr_pipeline.fit(X_train, y_train)
    model_name = 'Logistic_regression_Classification'
    generate_metrics(lr_pipeline, X_test, y_test, model_name, target)
    #pikle dump 
    pickle.dump(lr_pipeline, open('Outputs/Classification/lr_model.pkl', 'wb')) #put in outputs folder
    return lr_pipeline

def svm_c_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, model):
    svm_hyperparameters = svm_c_hyperopt(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, model)
    tuned_svm_model = SVC(**svm_hyperparameters)
    svm_pipeline = make_pipeline(preprocessor, pca, over_sampler, under_sampler, tuned_svm_model)
    svm_pipeline = svm_pipeline.fit(X_train, y_train)
    generate_metrics(svm_pipeline, X_test, y_test)
    #pickle dump
    pickle.dump(svm_pipeline, open('Outputs/Classification/svm_model.pkl', 'wb')) #put in outputs folder
    return svm_pipeline

def rf_c_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, model):
    rf_hyperparameters = rf_c_hyperopt(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, model)
    tuned_rf_model = RandomForestClassifier(**rf_hyperparameters)
    rf_pipeline = make_pipeline(preprocessor, pca, over_sampler, under_sampler, tuned_rf_model)
    rf_pipeline = rf_pipeline.fit(X_train, y_train)
    generate_metrics(rf_pipeline, X_test, y_test)
    #pickle dump
    pickle.dump(rf_pipeline, open('Outputs/Classification/rf_model.pkl', 'wb'))  # put in outputs folder
    return rf_pipeline

def xgb_c_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, model):
    xgb_hyperparameters = xgb_c_hyperopt(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, model)
    tuned_xgb_model = xgb.XGBClassifier(**xgb_hyperparameters)
    xgb_pipeline = make_pipeline(preprocessor, pca, over_sampler, under_sampler, tuned_xgb_model)
    xgb_pipeline = xgb_pipeline.fit(X_train, y_train)
    generate_metrics(xgb_pipeline, X_test, y_test)
    # pickle dump
    pickle.dump(xgb_pipeline, open('Outputs/Classification/xgb_model.pkl', 'wb'))  # put in outputs folder
    return xgb_pipeline

# Define the neural network model function
# def create_model(input_dim, output_dim, hidden_layers, activation='relu', optimizer='adam'):
#     model = Sequential()
#     model.add(Dense(hidden_layers[0], input_dim=input_dim, activation=activation))
#     for layer_size in hidden_layers[1:]:
#         model.add(Dense(layer_size, activation=activation))
#     model.add(Dense(output_dim, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model

# def create_model(layers, neurons, learning_rate, dropout_rate):
#     model = Sequential()
#     model.add(Dense(neurons, input_dim=100, activation='relu'))
#     model.add(Dropout(dropout_rate))

#     for _ in range(layers):
#         model.add(Dense(neurons, activation='relu'))
#         model.add(Dropout(dropout_rate))

#     model.add(Dense(1, activation='sigmoid'))
#     optimizer = Adam(learning_rate=learning_rate)
#     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model

def nn_c_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler):
    # Optimize hyperparameters
    nn_hyperparameters = nn_c_hyperopt(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, KerasClassifier(build_fn=create_model))

    # Build the tuned Keras model
    tuned_nn_model = KerasClassifier(
        build_fn=create_model,
        layers=nn_hyperparameters['layers'],
        neurons=nn_hyperparameters['neurons'],
        learning_rate=nn_hyperparameters['learning_rate'],
        dropout_rate=nn_hyperparameters['dropout_rate'],
        epochs=nn_hyperparameters['epochs'],
        batch_size=nn_hyperparameters['batch_size'],
        verbose=0
    )

    # Create and fit the pipeline
    nn_pipeline = make_pipeline(preprocessor, pca, over_sampler, under_sampler, tuned_nn_model)
    nn_pipeline.fit(X_train, y_train)

    # Generate metrics
    generate_metrics(nn_pipeline, X_test, y_test)

    # Save the model
    pickle.dump(nn_pipeline, open('Outputs/Classification/nn_model.pkl', 'wb'))  # put in outputs folder

    return nn_pipeline
#=============================Regression models and metrics=============================#
# def generate_regression_metrics(pipeline, X_test, y_test):
#     y_pred = pipeline.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     print("Mean Squared Error:", mse)
#     print("R2 Score:", r2)
def generate_regression_metrics(pipeline, X_test, y_test, model_name):
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'Model': model_name,
        'Mean Squared Error (MSE)': mse,
        'Root Mean Squared Error (RMSE)': rmse,
        'Mean Absolute Error (MAE)': mae,
        'Mean Absolute Percentage Error (MAPE)': mape,
        'Median Absolute Error (MedAE)': medae,
        'R2 Score': r2
    }

    # Print metrics
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

    # Create output folder if it doesn't exist
    output_folder = 'Outputs'
    os.makedirs(output_folder, exist_ok=True)

    # Save metrics as a table
    metrics_df = pd.DataFrame([metrics])
    output_filename = f'regression_performance_metrics_{model_name}.csv'
    output_path = os.path.join(output_folder, output_filename)

    if os.path.isfile(output_path):
        # Append to existing file if it exists
        existing_metrics_df = pd.read_csv(output_path)
        metrics_df = pd.concat([existing_metrics_df, metrics_df], ignore_index=True)

    metrics_df.to_csv(output_path, index=False)

    print("Metrics saved successfully.")  
def en_r_model(X_train, y_train, X_test, y_test, preprocessor, pca, model):
    en_hyperparameters = elastic_net_hyperopt(X_train, y_train, preprocessor, pca, model)
    tuned_en_model = ElasticNet(**en_hyperparameters)
    en_pipeline = make_pipeline(preprocessor, pca, tuned_en_model)
    en_pipeline = en_pipeline.fit(X_train, y_train)
    generate_regression_metrics(en_pipeline, X_test, y_test)
    # pickle dump
    pickle.dump(en_pipeline, open('Outputs/Regression/en_model.pkl', 'wb'))
    return en_pipeline

def svm_r_model(X_train, y_train, X_test, y_test, preprocessor, pca, model):
    svm_r_hyperparameters = svm_r_hyperopt(X_train, y_train, preprocessor, pca, model)
    tuned_svm_r_model = SVR(**svm_r_hyperparameters)
    svm_r_pipeline = make_pipeline(preprocessor, pca, tuned_svm_r_model)
    svm_r_pipeline = svm_r_pipeline.fit(X_train, y_train)
    generate_regression_metrics(svm_r_pipeline, X_test, y_test)
    # Pickle dump
    pickle.dump(svm_r_pipeline, open('Outputs/Regression/svm_r_model.pkl', 'wb'))
    return svm_r_pipeline

def rf_r_model(X_train, y_train, X_test, y_test, preprocessor, pca, model):
    rf_r_hyperparameters = rf_r_hyperopt(X_train, y_train, preprocessor, pca, model)
    tuned_rf_r_model = RandomForestRegressor(**rf_r_hyperparameters)
    rf_r_pipeline = make_pipeline(preprocessor, pca, tuned_rf_r_model)
    rf_r_pipeline = rf_r_pipeline.fit(X_train, y_train)
    generate_regression_metrics(rf_r_pipeline, X_test, y_test)
    # Pickle dump
    pickle.dump(rf_r_pipeline, open('Outputs/Regression/rf_r_model.pkl', 'wb'))
    return rf_r_pipeline

def xgb_r_model(X_train, y_train, X_test, y_test, preprocessor, pca, model):
    xgb_r_hyperparameters = xgb_r_hyperopt(X_train, y_train, preprocessor, pca, model)
    tuned_xgb_r_model = xgb.XGBRegressor(**xgb_r_hyperparameters)
    xgb_r_pipeline = make_pipeline(preprocessor, pca, tuned_xgb_r_model)
    xgb_r_pipeline = xgb_r_pipeline.fit(X_train, y_train)
    generate_regression_metrics(xgb_r_pipeline, X_test, y_test)
    # Pickle dump
    pickle.dump(xgb_r_pipeline, open('Outputs/Regression/xgb_r_model.pkl', 'wb'))
    return xgb_r_pipeline

def nn_r_model(X_train, y_train, X_test, y_test, preprocessor, pca, model):
    nn_r_hyperparameters = nn_r_hyperopt(X_train, y_train, preprocessor, pca, model)
    tuned_nn_r_model = MLPRegressor(**nn_r_hyperparameters)
    nn_r_pipeline = make_pipeline(preprocessor, pca, tuned_nn_r_model)
    nn_r_pipeline = nn_r_pipeline.fit(X_train, y_train)
    generate_regression_metrics(nn_r_pipeline, X_test, y_test)
    # Pickle dump
    pickle.dump(nn_r_pipeline, open('Outputs/Regression/nn_r_model.pkl', 'wb'))
    return nn_r_pipeline

