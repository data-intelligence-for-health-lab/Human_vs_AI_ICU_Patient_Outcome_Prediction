import numpy as np
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from statistics import median
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp


#Elastic Net
def elastic_net_hyperopt(X_train, y_train, preprocessor, pca, model):
    def objective(params):
        hyp_en_pipeline = make_pipeline(preprocessor, pca, model)
        hyp_en_pipeline.get_params().keys()
        hyp_en_pipeline.set_params(**params)
        r2_score = median(cross_val_score(hyp_en_pipeline, X_train, y_train, cv=10, scoring="r2"))
        return {'loss': -r2_score, 'status': STATUS_OK}

    search_space = {
        'elasticnet__alpha': hp.uniform('alpha', 0.0001, 1.0),
        'elasticnet__l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0),
    }

    trials = Trials()

    np.random.seed(42)

    en_opt = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=trials,
        verbose=True,
        rstate=np.random.default_rng(42)
    )

    print(en_opt)
    print(trials.best_trial['result']['loss'])

    return en_opt


#SVM_R
def svm_r_hyperopt(X_train, y_train, preprocessor, pca, model):
    def objective(params):
        hyp_svm_pipeline = make_pipeline(preprocessor, pca, model)
        hyp_svm_pipeline.get_params().keys()
        hyp_svm_pipeline.set_params(**params)
        r2_score = median(cross_val_score(hyp_svm_pipeline, X_train, y_train, cv=10, scoring="r2"))
        return {'loss': -r2_score, 'status': STATUS_OK}

    search_space = {
        'svr__C': hp.loguniform('C', 0, 20),
        'svr__kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        'svr__gamma': hp.loguniform('gamma', 0, 20),
    }

    trials = Trials()

    np.random.seed(42)

    svm_opt = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,  # Adjust the number of evaluations to 100 for actual run
        trials=trials,
        verbose=True,
        rstate=np.random.default_rng(42)
    )

    if svm_opt["kernel"] == 0:
        svm_opt["kernel"] = "linear"
    if svm_opt["kernel"] == 1:
        svm_opt["kernel"] = "poly"
    if svm_opt["kernel"] == 2:
        svm_opt["kernel"] = "rbf"
    if svm_opt["kernel"] == 3:
        svm_opt["kernel"] = "sigmoid"

    return svm_opt


#RF_R
def rf_r_hyperopt(X_train, y_train, preprocessor, pca, model):
    def objective(params):
        hyp_rf_r_pipeline = make_pipeline(preprocessor, pca, model)
        hyp_rf_r_pipeline.get_params().keys()
        hyp_rf_r_pipeline.set_params(**params)
        r2_score = median(cross_val_score(hyp_rf_r_pipeline, X_train, y_train, cv=10, scoring="r2"))
        return {'loss': -r2_score, 'status': STATUS_OK}

    search_space = {
        'randomforestregressor__n_estimators': hp.choice('n_estimators', [100, 200, 300]),
        'randomforestregressor__max_depth': hp.choice('max_depth', [None, 5, 10, 20]),
        'randomforestregressor__min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
    }

    trials = Trials()

    np.random.seed(42)

    rf_r_opt = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,  # Adjust the number of evaluations to 100 for actual run
        trials=trials,
        verbose=True,
        rstate=np.random.default_rng(42)
    )

    if rf_r_opt["max_depth"] == 0:
        rf_r_opt["max_depth"] = None

    return rf_r_opt

#XGB_R
def xgb_r_hyperopt(X_train, y_train, preprocessor, pca, model):
    def objective(params):
        hyp_xgb_r_pipeline = make_pipeline(preprocessor, pca, model)
        hyp_xgb_r_pipeline.get_params().keys()
        hyp_xgb_r_pipeline.set_params(**params)
        r2_score = median(cross_val_score(hyp_xgb_r_pipeline, X_train, y_train, cv=10, scoring="r2"))
        return {'loss': -r2_score, 'status': STATUS_OK}

    search_space = {
        'xgbregressor__n_estimators': hp.choice('n_estimators', [100, 200, 300]),
        'xgbregressor__max_depth': hp.choice('max_depth', [None, 5, 10, 20]),
        'xgbregressor__learning_rate': hp.loguniform('learning_rate', -5, 0),
    }

    trials = Trials()

    np.random.seed(42)

    xgb_r_opt = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,  # Adjust the number of evaluations to 100 for actual run
        trials=trials,
        verbose=True,
        rstate=np.random.default_rng(42)
    )

    if xgb_r_opt["max_depth"] == 0:
        xgb_r_opt["max_depth"] = None

    return xgb_r_opt

#NN_R
def nn_r_hyperopt(X_train, y_train, preprocessor, pca, model):
    def objective(params):
        hyp_nn_r_pipeline = make_pipeline(preprocessor, pca, model)
        hyp_nn_r_pipeline.get_params().keys()
        hyp_nn_r_pipeline.set_params(**params)
        r2_score = median(cross_val_score(hyp_nn_r_pipeline, X_train, y_train, cv=10, scoring="r2"))
        return {'loss': -r2_score, 'status': STATUS_OK}

    search_space = {
        'mlpregressor__hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50)]),
        'mlpregressor__activation': hp.choice('activation', ['relu', 'tanh']),
        'mlpregressor__alpha': hp.loguniform('alpha', -5, 0),
    }

    trials = Trials()

    np.random.seed(42)

    nn_r_opt = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,  # Adjust the number of evaluations to 100 for actual run
        trials=trials,
        verbose=True,
        rstate=np.random.default_rng(42)
    )

    return nn_r_opt


