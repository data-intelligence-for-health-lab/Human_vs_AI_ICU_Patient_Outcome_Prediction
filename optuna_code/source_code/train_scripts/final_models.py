import os
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, average_precision_score
from datetime import datetime
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from train_scripts.hyperparameter_tunning_optuna import tune_c_hyperparameters
from train_scripts.generate_metrics import get_classification_bootstrapped_metrics

SEED = 42

#******************************Classificaion Models************************************#    
def lr_c_model(X_train, y_train, X_test, y_test, preprocessor, over_sampler, 
            under_sampler, feature_selector, target, mtype, mname, run_id, 
            output_path):

    # Tune hyperparameters
    lr_c_hyperparameters = tune_c_hyperparameters(
        X_train, y_train, mtype, target, mname, run_id
    )
    
    print("Tuned Hyperparameters:", lr_c_hyperparameters)
    
    # Define the Logistic Regression model with tuned hyperparameters
    tuned_lr_c_model = LogisticRegression(**lr_c_hyperparameters)

    # Construct the pipeline
    lr_c_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selectfeatures', feature_selector),
        ('over_sampler', over_sampler),
        ('under_sampler', under_sampler),
        ('model', tuned_lr_c_model)
    ])
    
    # Fit the pipeline on training data
    lr_c_pipeline.fit(X_train, y_train)
    
    # Generate model name
    model_name = f'Logistic_regression_Classification_{target}_{run_id}'
    
    # Evaluate and generate metrics with confidence intervals
    get_classification_bootstrapped_metrics(lr_c_pipeline, X_test, y_test, run_id, mname, target, output_path, n_iterations=1000)
    # Save model using pickle
    output_model_pkl = os.path.join(output_path, "classification", f"lr_c_model_{target}_{run_id}.pkl")
    print("save_output_model", output_model_pkl)

    os.makedirs(os.path.dirname(output_model_pkl), exist_ok=True)  # Ensure directory exists

    with open(output_model_pkl, 'wb') as model_file:
        pickle.dump(lr_c_pipeline, model_file)
    
    print(f"Model saved at: {output_model_pkl}")

    # Save best parameters to CSV with date
    param_data = {
        'date': datetime.now().strftime("%Y-%m-%d"),
        'model_name': model_name,
        'target': target,
        'run_id': run_id,
        'best_params': str(lr_c_hyperparameters)
    }

    best_params_df = pd.DataFrame([param_data])
    
    # Ensure the directory for best_params_file exists
    best_params_file = os.path.join(output_path, "classification", f"best_params_lr_c_model_{target}_{run_id}.csv")
    os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
    # Append to CSV file
    best_params_df.to_csv(best_params_file, mode='a', header=not os.path.exists(best_params_file), index=False)

    print(f"Best parameters saved in: {best_params_file}")
    return best_params_df

def svm_c_model(X_train, y_train, X_test, y_test, preprocessor, over_sampler, 
                under_sampler, feature_selector, target, mtype, mname, run_id, 
                output_path):

    # Tune hyperparameters
    svm_hyperparameters = tune_c_hyperparameters(
        X_train, y_train, mtype, target, mname, run_id
    )
    
    print("Tuned Hyperparameters:", svm_hyperparameters)
    
    # Define the SVM model with tuned hyperparameters
    tuned_svm_model = SVC(probability=True, **svm_hyperparameters)

    # Construct the pipeline
    svc_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selectfeatures', feature_selector),
        ('over_sampler', over_sampler),
        ('under_sampler', under_sampler),
        ('model', tuned_svm_model)
    ])
    
    # Fit the pipeline on training data
    svc_pipeline.fit(X_train, y_train)
    
    # Generate model name
    model_name = f'SVM_Classification_{target}_{run_id}'
    
    # Evaluate and generate metrics with confidence intervals
    get_classification_bootstrapped_metrics(svc_pipeline, X_test, y_test, run_id, mname, target, output_path, n_iterations=1000)
    
    # Save model using pickle
    output_model_pkl = os.path.join(output_path, "classification", f"svm_c_model_{target}_{run_id}.pkl")
    os.makedirs(os.path.dirname(output_model_pkl), exist_ok=True)  # Ensure directory exists

    with open(output_model_pkl, 'wb') as model_file:
        pickle.dump(svc_pipeline, model_file)
    
    print(f"Model saved at: {output_model_pkl}")

    # Save best parameters to CSV with date
    param_data = {
        'date': datetime.now().strftime("%Y-%m-%d"),
        'model_name': model_name,
        'target': target,
        'run_id': run_id,
        'best_params': str(svm_hyperparameters)
    }

    best_params_df = pd.DataFrame([param_data])
    
    # Ensure the directory for best_params_file exists
    best_params_file = os.path.join(output_path, "classification", f"best_params_svm_c_model_{target}_{run_id}.csv")
    os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
    
    # Append to CSV file
    best_params_df.to_csv(best_params_file, mode='a', header=not os.path.exists(best_params_file), index=False)

    print(f"Best parameters saved in: {best_params_file}")
    return best_params_df
    
def rf_c_model(X_train, y_train, X_test, y_test, preprocessor, over_sampler, 
            under_sampler, feature_selector, target, mtype, mname, run_id, 
            output_path):

    # Tune hyperparameters
    rf_hyperparameters = tune_c_hyperparameters(
        X_train, y_train, mtype, target, mname, run_id
    )
    
    print("Tuned Hyperparameters:", rf_hyperparameters)
    
    # Define the Random Forest model with tuned hyperparameters
    tuned_rf_model = RandomForestClassifier(**rf_hyperparameters)

    # Construct the pipeline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selectfeatures', feature_selector),
        ('over_sampler', over_sampler),
        ('under_sampler', under_sampler),
        ('model', tuned_rf_model)
    ])
    
    # Fit the pipeline on training data
    rf_pipeline.fit(X_train, y_train)
    
    # Generate model name
    model_name = f'Random_Forest_Classification_{target}_{run_id}'
    
    # Evaluate and generate metrics with confidence intervals
    get_classification_bootstrapped_metrics(rf_pipeline, X_test, y_test, run_id, mname, target, output_path, n_iterations=1000)
    
    # Save model using pickle
    output_model_pkl = os.path.join(output_path, "classification", f"rf_c_model_{target}_{run_id}.pkl")
    os.makedirs(os.path.dirname(output_model_pkl), exist_ok=True)  # Ensure directory exists

    with open(output_model_pkl, 'wb') as model_file:
        pickle.dump(rf_pipeline, model_file)
    
    print(f"Model saved at: {output_model_pkl}")

    # Save best parameters to CSV with date
    param_data = {
        'date': datetime.now().strftime("%Y-%m-%d"),
        'model_name': model_name,
        'target': target,
        'run_id': run_id,
        'best_params': str(rf_hyperparameters)
    }

    best_params_df = pd.DataFrame([param_data])
    
    # Ensure the directory for best_params_file exists
    best_params_file = os.path.join(output_path, "classification", f"best_params_rf_c_model_{target}_{run_id}.csv")
    os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
    
    # Append to CSV file
    best_params_df.to_csv(best_params_file, mode='a', header=not os.path.exists(best_params_file), index=False)

    print(f"Best parameters saved in: {best_params_file}")
    return best_params_df


def xgb_c_model(X_train, y_train, X_test, y_test, preprocessor, over_sampler, 
                under_sampler, feature_selector, target, mtype, mname, run_id, 
                output_path):
    
    # Tune hyperparameters
    xgb_c_hyperparameters = tune_c_hyperparameters(
        X_train, y_train, mtype, target, mname, run_id
    )
    
    print("Tuned Hyperparameters:", xgb_c_hyperparameters)
    
    # Define the XGBoost Classification model with tuned hyperparameters
    tuned_xgb_c_model = XGBClassifier(**xgb_c_hyperparameters, use_label_encoder=False, eval_metric='logloss')

    # Construct the pipeline
    xgb_c_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selectfeatures', feature_selector),
        ('over_sampler', over_sampler),
        ('under_sampler', under_sampler),
        ('model', tuned_xgb_c_model)
    ])
    
    # Fit the pipeline on training data
    xgb_c_pipeline.fit(X_train, y_train)
    
    # Generate model name
    model_name = f'XGBoost_Classification_{target}_{run_id}'
    
    # Evaluate and generate metrics with confidence intervals
    get_classification_bootstrapped_metrics(xgb_c_pipeline, X_test, y_test, run_id, mname, target, output_path, n_iterations=1000)
    
    # Save model using pickle
    output_model_pkl = os.path.join(output_path, "classification", f"xgb_c_model_{target}_{run_id}.pkl")
    print("save_output_model", output_model_pkl)

    os.makedirs(os.path.dirname(output_model_pkl), exist_ok=True)

    with open(output_model_pkl, 'wb') as model_file:
        pickle.dump(xgb_c_pipeline, model_file)
    
    print(f"Model saved at: {output_model_pkl}")
    
    # Save best parameters to CSV with date
    param_data = {
        'date': datetime.now().strftime("%Y-%m-%d"),
        'model_name': model_name,
        'target': target,
        'run_id': run_id,
        'best_params': str(xgb_c_hyperparameters)
    }

    best_params_df = pd.DataFrame([param_data])
    
    # Ensure the directory for best_params_file exists
    best_params_file = os.path.join(output_path, "classification", f"best_params_xgb_c_model_{target}_{run_id}.csv")
    os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
    
    # Append to CSV file
    best_params_df.to_csv(best_params_file, mode='a', header=not os.path.exists(best_params_file), index=False)

    print(f"Best parameters saved in: {best_params_file}")
    return best_params_df


def nn_c_model(X_train, y_train, X_test, y_test, preprocessor, over_sampler, 
                under_sampler, feature_selector, target, mtype, mname, run_id, output_path):
    # Tune hyperparameters using Optuna or another hyperparameter tuning strategy
    nn_c_hyperparameters = tune_c_hyperparameters(
        X_train, y_train, mtype, target, mname, run_id
    )
    
    print("Tuned Hyperparameters:", nn_c_hyperparameters)
    
    hidden_units = nn_c_hyperparameters['hidden_units']
    num_layers = nn_c_hyperparameters['num_layers']
    dropout_rate = nn_c_hyperparameters['dropout_rate']
    lr = nn_c_hyperparameters['lr']
    batch_size = nn_c_hyperparameters['batch_size']
    epochs = nn_c_hyperparameters['epochs']
    
    # Preprocess the features
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Apply over-sampling (e.g., SMOTE) and under-sampling to balance the classes
    X_train_balanced, y_train_balanced = over_sampler.fit_resample(X_train_transformed, y_train)
    X_train_balanced, y_train_balanced = under_sampler.fit_resample(X_train_balanced, y_train_balanced)
    
    # Feature selection
    X_train_selected = feature_selector.fit_transform(X_train_balanced, y_train_balanced)

    # Convert data to tensors
    X_tensor = torch.tensor(X_train_selected.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_balanced.values, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define the neural network model
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_units, num_layers, dropout_rate):
            super(NeuralNet, self).__init__()
            layers = [nn.Linear(input_size, hidden_units), nn.ReLU(), nn.Dropout(dropout_rate)]
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_units, hidden_units))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(hidden_units, 1))
            layers.append(nn.Sigmoid())
            self.model = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.model(x)
    
    # Initialize model, loss function, and optimizer
    model = NeuralNet(X_train_selected.shape[1], hidden_units, num_layers, dropout_rate)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    
    # Evaluate model using cross-validation
    def model_scorer(estimator, X, y):
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        with torch.no_grad():
            y_pred = estimator(X_tensor).numpy()
        return average_precision_score(y, y_pred)
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X_train_selected, y_train_balanced, cv=cv, scoring=make_scorer(model_scorer), n_jobs=-1)
    
    # Generate model name
    model_name = f'NeuralNet_Classification_{target}_{run_id}'
    
    # Save the model using pickle
    output_model_pkl = os.path.join(output_path, "classification", f"nn_c_model_{target}_{run_id}.pkl")
    print(f"Saving model to {output_model_pkl}")

    os.makedirs(os.path.dirname(output_model_pkl), exist_ok=True)  # Ensure directory exists

    with open(output_model_pkl, 'wb') as model_file:
        pickle.dump(model, model_file)
    
    print(f"Model saved at: {output_model_pkl}")
    
    # Save the best parameters to a CSV with date
    param_data = {
        'date': datetime.now().strftime("%Y-%m-%d"),
        'model_name': model_name,
        'target': target,
        'run_id': run_id,
        'best_params': str(nn_c_hyperparameters)
    }

    best_params_df = pd.DataFrame([param_data])
    
    # Ensure the directory for best_params_file exists
    best_params_file = os.path.join(output_path, "classification", f"best_params_nn_c_model_{target}_{run_id}.csv")
    os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
    
    # Append to the CSV file
    best_params_df.to_csv(best_params_file, mode='a', header=not os.path.exists(best_params_file), index=False)

    print(f"Best parameters saved in: {best_params_file}")
    return best_params_df

# #******************************Regression Models************************************#    
# def en_r_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, model, target):
#     en_hyperparameters = elastic_net_hyperopt(X_train, y_train, preprocessor, pca, model)
#     tuned_en_model = ElasticNet(**en_hyperparameters)
#     en_pipeline = make_pipeline(preprocessor, pca, tuned_en_model)
#     en_pipeline = en_pipeline.fit(X_train, y_train)
#     model_name = 'Elastic_Net_Regression'
#     generate_regression_metrics(en_pipeline, X_test, y_test, model_name, target)
#     # pickle dump
#     pickle.dump(en_pipeline, open(f'Outputs/Regression/en_model_{target}.pkl', 'wb'))
#     return en_pipeline

# def svm_r_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, model, target):
#     svm_r_hyperparameters = svm_r_hyperopt(X_train, y_train, preprocessor, pca, model)
#     tuned_svm_r_model = SVR(**svm_r_hyperparameters)
#     svm_r_pipeline = make_pipeline(preprocessor, pca, tuned_svm_r_model)
#     svm_r_pipeline = svm_r_pipeline.fit(X_train, y_train)
#     model_name = 'SVM_Regression'
#     generate_regression_metrics(svm_r_pipeline, X_test, y_test, model_name, target)
#     # Pickle dump
#     pickle.dump(svm_r_pipeline, open(f'Outputs/Regression/svm_r_model_{target_column}.pkl', 'wb'))
#     return svm_r_pipeline

# # def rf_r_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, model, target):
# #     rf_r_hyperparameters = rf_r_hyperopt(X_train, y_train, preprocessor, pca, model)
# #     tuned_rf_r_model = RandomForestRegressor(**rf_r_hyperparameters)
# #     rf_r_pipeline = make_pipeline(preprocessor, pca, tuned_rf_r_model)
# #     rf_r_pipeline = rf_r_pipeline.fit(X_train, y_train)
# #     model_name = 'RF_Regression'
# #     generate_regression_metrics(rf_r_pipeline, X_test, y_test, model_name, target)
# #     # Pickle dump
# #     pickle.dump(rf_r_pipeline, open(f'Outputs/Regression/rf_r_model_{target_column}.pkl', 'wb'))
# #     return rf_r_pipeline

# def rf_r_model(X_train, y_train, X_test, y_test, preprocessor, pca, model, target):
#     rf_r_hyperparameters = rf_r_hyperopt(X_train, y_train, preprocessor, pca, model)
    
#     # Remove the 'randomforestregressor__' prefix from the hyperparameters
#     cleaned_hyperparameters = {k.split('__')[-1]: v for k, v in rf_r_hyperparameters.items()}
    
#     tuned_rf_r_model = RandomForestRegressor(**cleaned_hyperparameters)
#     rf_r_pipeline = make_pipeline(preprocessor, pca, tuned_rf_r_model)
#     rf_r_pipeline = rf_r_pipeline.fit(X_train, y_train)
#     model_name = 'RF_Regression'
#     generate_regression_metrics(rf_r_pipeline, X_test, y_test, model_name, target)
    
#     # Pickle dump
#     pickle.dump(rf_r_pipeline, open(f'Outputs/Regression/rf_r_model_{target}.pkl', 'wb'))
    
#     return rf_r_pipeline

# def xgb_r_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, model, target):
#     xgb_r_hyperparameters = xgb_r_hyperopt(X_train, y_train, preprocessor, pca, model)
#     tuned_xgb_r_model = xgb.XGBRegressor(**xgb_r_hyperparameters)
#     xgb_r_pipeline = make_pipeline(preprocessor, pca, tuned_xgb_r_model)
#     xgb_r_pipeline = xgb_r_pipeline.fit(X_train, y_train)
#     model_name = 'XGB_Regression'
#     generate_regression_metrics(xgb_r_pipeline, X_test, y_test)
#     # Pickle dump
#     pickle.dump(xgb_r_pipeline, open(f'Outputs/Regression/xgb_r_model_{target_column}.pkl', 'wb'))
#     return xgb_r_pipeline

# def nn_r_model(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, input_dim):
#     nn_r_hyperparameters = nn_r_hyperopt(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, input_dim)
#     return nn_r_hyperparameters