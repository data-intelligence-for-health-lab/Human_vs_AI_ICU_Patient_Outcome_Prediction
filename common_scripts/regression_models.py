import numpy as np
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from statistics import median
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
from sklearn.metrics import average_precision_score

import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense

from hyperopt.pyll.base import scope

#Elastic Net
def elastic_net_hyperopt(X_train, y_train, preprocessor, pca, model):
    def objective(params):
        hyp_en_pipeline = make_pipeline(preprocessor, pca, model)
        hyp_en_pipeline.get_params().keys()
        hyp_en_pipeline.set_params(**params)
        neg_mse = median(cross_val_score(hyp_en_pipeline, X_train, y_train, cv=10, scoring="neg_mean_squared_error"))
        return {'loss': -neg_mse, 'status': STATUS_OK}

    search_space = {
        'elasticnet__alpha': hp.uniform('alpha', 0.001, 10),
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
        neg_mse = median(cross_val_score(hyp_svm_pipeline, X_train, y_train, cv=10, scoring="neg_mean_squared_error"))
        return {'loss': -neg_mse, 'status': STATUS_OK}

    search_space = {
        'svr__C': hp.loguniform('C', np.log(0.001), np.log(1000)),
        'svr__kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        'svr__gamma': hp.choice('gamma', ['scale', 'auto']),
        'svr__degree': hp.quniform('degree', 1, 10, 1)   # Adding degree for polynomial kernel
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

    kernel_map = {0: 'linear', 1: 'poly', 2: 'rbf', 3: 'sigmoid'}
    gamma_map = {0: 'scale', 1: 'auto'}
    
    svm_opt["svr__kernel"] = kernel_map[svm_opt["svr__kernel"]]
    svm_opt["svr__gamma"] = gamma_map[svm_opt["svr__gamma"]]

    return svm_opt


#RF_R
def rf_r_hyperopt(X_train, y_train, preprocessor, pca, model):
    def objective(params):
        hyp_rf_r_pipeline = make_pipeline(preprocessor, pca, model)
        hyp_rf_r_pipeline.get_params().keys()
        hyp_rf_r_pipeline.set_params(**params)
        r2_score = median(cross_val_score(hyp_rf_r_pipeline, X_train, y_train, cv=10, scoring="r2", n_jobs=-1))
        return {'loss': -r2_score, 'status': STATUS_OK}

    search_space = {
        'randomforestregressor__n_estimators': hp.choice('n_estimators', [100, 200, 300]),
        'randomforestregressor__max_depth': hp.choice('max_depth', [None, 5, 10, 20]),
        'randomforestregressor__min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
    }
     #use same as rf_c

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

    return rf_r_opt

#XGB_R
def xgb_r_hyperopt(X_train, y_train, preprocessor, pca, model):
    def objective(params):
        hyp_xgb_r_pipeline = make_pipeline(preprocessor, pca, model)
        hyp_xgb_r_pipeline.set_params(**params)
        neg_mse = median(cross_val_score(hyp_xgb_r_pipeline, X_train, y_train, cv=10, scoring="neg_mean_squared_error"))
        return {'loss': -neg_mse, 'status': STATUS_OK}

    search_space = {
        'xgbregressor__learning_rate': hp.choice('learning_rate', [0.001, 0.01, 0.1, 1, 5, 10]),
        'xgbregressor__n_estimators': hp.quniform('n_estimators', 10, 1000, 1),
        'xgbregressor__max_depth': hp.quniform('max_depth', 1, 50, 1),
        'xgbregressor__min_child_weight': hp.choice('min_child_weight', [1, 3, 5]),
        'xgbregressor__subsample': hp.uniform('subsample', 0.5, 1.0),
        'xgbregressor__gamma': hp.uniform('gamma', 0.0, 1.0),
        'xgbregressor__colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'xgbregressor__reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        'xgbregressor__reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0)
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

    # Mapping indices back to actual values
    learning_rate_mapping = [0.001, 0.01, 0.1, 1, 5, 10]
    min_child_weight_mapping = [1, 3, 5]

    xgb_r_opt['xgbregressor__learning_rate'] = learning_rate_mapping[xgb_r_opt['learning_rate']]
    xgb_r_opt['xgbregressor__min_child_weight'] = min_child_weight_mapping[xgb_r_opt['xgbregressor__min_child_weight']]
    xgb_r_opt['xgbregressor__n_estimators'] = int(xgb_r_opt['xgbregressor__n_estimators'])
    xgb_r_opt['xgbregressor__max_depth'] = int(xgb_r_opt['xgbregressor__max_depth'])

    return xgb_r_opt

#NN_R
def average_precision(y_true, y_pred):
    return tf.py_function(average_precision_score, (y_true, y_pred), tf.float32)


def nn_r_hyperopt(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, input_dim):

    # build model function, pass in input dimension from the main file
    def create_model(num_hidden_layers, hidden_units, activation, optimizer):
        model = Sequential()
        model.add(Dense(units=hidden_units[0], activation=activation))
        for i in range(1, num_hidden_layers):
            model.add(Dense(units=hidden_units[i], activation=activation))
            model.add(Dense(units=1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[average_precision])
        return model

    space = {
        'input_dim': input_dim,
        'num_hidden_layers': scope.int(hp.quniform('num_hidden_layers', 1, 8, 1)),
        'hidden_units': [scope.int(hp.quniform('hidden_units_' + str(i), 16, 256, 16)) for i in range(8)],
        'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh']),
        'optimizer': hp.choice('optimizer', ['adam', 'sgd'])
    }

    # create pipeline in objective function
    def objective(params):
        model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=32, verbose=2, **params)
        pipeline = make_pipeline(preprocessor, pca, over_sampler, under_sampler, model)
        score = median(cross_val_score(pipeline, X_train, y_train, cv=10, scoring="average_precision", error_score='raise',n_jobs=10))
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()

    np.random.seed(42)
    nn_opt = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials,
                rstate=np.random.default_rng(42)
    )

    if nn_opt["activation"] == 0:
        nn_opt["activation"] = "relu"
    elif nn_opt["activation"] == 1:
        nn_opt["activation"] = "sigmoid"
    elif nn_opt["activation"] == 2:
        nn_opt["activation"] = "tanh"
    if nn_opt["optimizer"] == 0:
        nn_opt["optimizer"] = "adam"
    if nn_opt["optimizer"] == 1:
        nn_opt["optimizer"] = "sgd"

    nn_opt["num_hidden_layers"] = int(nn_opt["num_hidden_layers"]) # number hidden layers as integer instead of string

    return nn_opt


