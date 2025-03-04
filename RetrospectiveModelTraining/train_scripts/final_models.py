import torch
from imblearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from train_scripts.neural_network_models import NeuralNetClassifier, NeuralNetRegressor, make_final_neural_network
from train_scripts.hyperparameter_tunning_optuna import tune_c_hyperparameters, tune_r_hyperparameters
from train_scripts.generate_metrics import get_classification_bootstrapped_metrics, get_regression_bootstrapped_metrics
from train_scripts.save_model_and_params import save_model_and_params, save_selected_features


SEED = 42
torch.manual_seed(SEED)

#******************************Classificaion Models************************************#    
def classification_model(X_train, y_train, X_test, y_test, preprocessor, over_sampler, 
            under_sampler, feature_selector, target, mtype, mname, run_id, 
            output_path):
    
    best_c_hyperparameters = tune_c_hyperparameters(
        X_train, y_train, mtype, target, mname, run_id
    )
    print("Tuned Hyperparameters:", best_c_hyperparameters)

    if mname == 'LR_C':
        tuned_c_model = LogisticRegression(**best_c_hyperparameters)
    elif mname == 'SVM_C':
        tuned_c_model = SVC(probability=True, **best_c_hyperparameters)
    elif mname == 'RF_C':
        tuned_c_model = RandomForestClassifier(**best_c_hyperparameters)
    elif mname == 'XGB_C':
        tuned_c_model = XGBClassifier(**best_c_hyperparameters)
    elif mname == 'NN_C':
        tuned_c_model = make_final_neural_network((X_train, best_c_hyperparameters))
    else:
        raise ValueError(f"Model name '{mname}' is not recognized. Please use a valid model name.")

    # Construct the pipeline
    classification_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selectfeatures', feature_selector),
        ('over_sampler', over_sampler),
        ('under_sampler', under_sampler),
        ('model', tuned_c_model)
    ])
    
    # Fit the pipeline on training data
    classification_pipeline.fit(X_train, y_train)

    # Save selected features
    features_mask = classification_pipeline['selectfeatures'].get_support() 
    features = X_train.columns[features_mask].tolist()  # Convert to list

    # Generate model name
    model_name = f'{mname}_{target}_{run_id}'

    # Save features to csv
    save_selected_features(features, model_name, output_path)

    # Evaluate and generate metrics with confidence intervals
    get_classification_bootstrapped_metrics(classification_pipeline, X_test, y_test, run_id, mname, target, output_path, n_iterations=1000)
    
    # Save the model and hyperparameters
    save_model_and_params(classification_pipeline, model_name, target, run_id, output_path, best_c_hyperparameters)

    return best_c_hyperparameters

# #******************************Regression Models************************************#
def regression_model(X_train, y_train, X_test, y_test, preprocessor, 
                    feature_selector, target, mtype, mname, run_id, 
                    output_path):
    
    best_r_hyperparameters = tune_r_hyperparameters(
        X_train, y_train, mtype, target, mname, run_id
    )
    print("Tuned Hyperparameters:", best_r_hyperparameters)

    if mname == 'EN_R':
        tuned_r_model = ElasticNet(**best_r_hyperparameters)
    elif mname == 'SVM_R':
        tuned_r_model = SVR(**best_r_hyperparameters)
    elif mname == 'RF_R':
        tuned_r_model = RandomForestRegressor(**best_r_hyperparameters)
    elif mname == 'XGB_R':
        tuned_r_model = XGBRegressor(**best_r_hyperparameters) 
    elif mname == 'NN_R':
        tuned_r_model = NeuralNetRegressor(**best_r_hyperparameters)
    else:
        raise ValueError(f"Model name '{mname}' is not recognized. Please use a valid model name.")

    # Construct the pipeline
    regression_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selectfeatures', feature_selector),
        ('model', tuned_r_model)
    ])
    
    # Fit the pipeline on training data
    regression_pipeline.fit(X_train, y_train)

    # Generate model name
    model_name = f'{mname}_{target}_{run_id}'

    # Evaluate and generate metrics with confidence intervals
    get_regression_bootstrapped_metrics(regression_pipeline, X_test, y_test, run_id, mname, target, output_path, n_iterations=1000)
    
    # Save the model and hyperparameters
    save_model_and_params(regression_pipeline, model_name, target, run_id, output_path, best_r_hyperparameters)

    return best_r_hyperparameters