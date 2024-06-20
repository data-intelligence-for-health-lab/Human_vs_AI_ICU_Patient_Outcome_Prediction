import numpy as np
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from statistics import median
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp

# from keras.wrappers.scikit_learn import KerasRegressor
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import Adam

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
# Define a function to create a Keras regression model
# def create_keras_regression_model(hidden_layer_sizes=(100,), activation='relu', dropout_rate=0.0, learning_rate=0.001):
#     model = Sequential()
#     model.add(Dense(hidden_layer_sizes[0], input_dim=100, activation=activation))
#     model.add(Dropout(dropout_rate))
    
#     for units in hidden_layer_sizes[1:]:
#         model.add(Dense(units, activation=activation))
#         model.add(Dropout(dropout_rate))
        
#     model.add(Dense(1, activation='linear'))
#     model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))
#     return model

# def nn_r_hyperopt(X_train, y_train, preprocessor, pca):
#     #input_shape = X_train.shape[1]

#     def objective(params):
#         model = KerasRegressor(build_fn=create_keras_regression_model,
#                                hidden_layer_sizes=params['hidden_layer_sizes'],
#                                activation=params['activation'],
#                                dropout_rate=params['dropout_rate'],
#                                learning_rate=params['learning_rate'],
#                                epochs=200,
#                                batch_size=params['batch_size'],
#                                verbose=0)

#         hyp_nn_r_pipeline = make_pipeline(preprocessor, pca, model)
#         neg_mse = median(cross_val_score(hyp_nn_r_pipeline, X_train, y_train, cv=10, scoring="neg_mean_squared_error"))
#         return {'loss': -neg_mse, 'status': STATUS_OK}

#     search_space = {
#         'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [
#             tuple([hp.choice(f'units_{i}', np.arange(10, 501, 10)) for i in range(h)]) for h in range(1, 21)
#         ]),
#         'activation': hp.choice('activation', ['relu', 'tanh']),
#         'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.5),
#         'learning_rate': hp.choice('learning_rate', [0.001, 0.01, 0.1, 1, 5, 10]),
#         'batch_size': hp.choice('batch_size', [32, 64, 128, 256, 512])
#     }

#     trials = Trials()

#     np.random.seed(42)

#     nn_r_opt = fmin(
#         fn=objective,
#         space=search_space,
#         algo=tpe.suggest,
#         max_evals=10,  # Adjust the number of evaluations to 100 for actual run
#         trials=trials,
#         verbose=True,
#         rstate=np.random.default_rng(42)
#     )

#     hidden_layer_sizes_mapping = [
#         tuple([n for n in range(10, 501, 10) for _ in range(h)]) for h in range(1, 21)
#     ]
#     activation_mapping = ['relu', 'tanh']
#     learning_rate_mapping = [0.001, 0.01, 0.1, 1, 5, 10]
#     batch_size_mapping = [32, 64, 128, 256, 512]

#     nn_r_opt['hidden_layer_sizes'] = hidden_layer_sizes_mapping[nn_r_opt['hidden_layer_sizes']]
#     nn_r_opt['activation'] = activation_mapping[nn_r_opt['activation']]
#     nn_r_opt['learning_rate'] = learning_rate_mapping[nn_r_opt['learning_rate']]
#     nn_r_opt['batch_size'] = batch_size_mapping[nn_r_opt['batch_size']]

#     return nn_r_opt



