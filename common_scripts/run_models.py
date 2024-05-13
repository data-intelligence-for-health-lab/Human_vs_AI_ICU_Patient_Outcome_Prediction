from common_scripts.hyperopt_models import lr_hyperopt, svm_c_hyperopt, rf_c_hyperopt, xgb_c_hyperopt, nn_c_hyperopt
from common_scripts.regression_models import elastic_net_hyperopt, svm_r_hyperopt, rf_r_hyperopt, xgb_r_hyperopt, nn_r_hyperopt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error, r2_score
import pickle
from imblearn.pipeline import make_pipeline

#=============================Classification models and metrics=============================#
def generate_metrics(pipeline, X_test, y_test):
    lr_y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, lr_y_pred)
    print(report)
    print(roc_auc_score(y_test, pipeline.predict_proba(X_test)[:,1]))
    
def lr_c_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, model):
    lr_hyperparameters = lr_hyperopt(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, model)
    tuned_lr_model = LogisticRegression(**lr_hyperparameters)
    lr_pipeline = make_pipeline(preprocessor, pca, over_sampler, under_sampler, tuned_lr_model)
    lr_pipeline = lr_pipeline.fit(X_train, y_train)
    generate_metrics(lr_pipeline, X_test, y_test)
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

def nn_c_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, model):
    nn_hyperparameters = nn_c_hyperopt(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, model)
    tuned_nn_model = MLPClassifier(**nn_hyperparameters)
    nn_pipeline = make_pipeline(preprocessor, pca, over_sampler, under_sampler, tuned_nn_model)
    nn_pipeline = nn_pipeline.fit(X_train, y_train)
    generate_metrics(nn_pipeline, X_test, y_test)
    # pickle dump
    pickle.dump(nn_pipeline, open('Outputs/nn_model.pkl/Classification', 'wb'))  # put in outputs folder
    return nn_pipeline

#=============================Regression models and metrics=============================#
def generate_regression_metrics(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)
    
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