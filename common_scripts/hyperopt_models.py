import numpy as np
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from statistics import median
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp

# Add Keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

#========================================= Logistic Regression   ============================================#
def lr_hyperopt(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, model):
    # Define the objective function
    def objective(params):
        print('params:', params)
        
        solver_params = params['logisticregression__solver']
        del params['logisticregression__solver']

        print('solverparams:', solver_params)

        solver_type = solver_params['type']
        penalty = solver_params['penalty']

        params['logisticregression__solver'] = solver_type
        params['logisticregression__penalty'] = penalty
        
        # Create and configure the pipeline
        hyp_lr_pipeline = make_pipeline(preprocessor, pca, over_sampler, under_sampler, model)
        hyp_lr_pipeline.set_params(**params)
        average_precision = median(cross_val_score(hyp_lr_pipeline, X_train, y_train, cv=10, scoring="average_precision", n_jobs=-1))
        return {'loss': -average_precision, 'status': STATUS_OK}
    
    # Define the search space
    search_space = {
        'logisticregression__C': hp.uniform('C', 0.001, 100),
        'logisticregression__solver': hp.choice('solver', [
            {
                'type': 'lbfgs',
                'penalty': hp.choice('lbfgs_penalty', ['l2', None])
            },
            {
                'type': 'newton-cg',
                'penalty': hp.choice('newton_penalty', ['l2', None])
            },
            {
                'type': 'liblinear',
                'penalty': hp.choice('liblinear_penalty', ['l1', 'l2'])
            },
            {
                'type': 'sag',
                'penalty': hp.choice('sag_penalty', ['l2', None])
            },
            {
                'type': 'saga',
                'penalty': hp.choice('saga_penalty', ['l1', 'l2', 'elasticnet', None])
            }
        ]),
        'logisticregression__max_iter': hp.choice('max_iter', range(100, 4000)),
        'logisticregression__class_weight': hp.choice('class_weight', [None, 'balanced'])
    }
    
    trials = Trials()
    
    np.random.seed(42)
    
    # Optimize the hyperparameters
    lr_opt = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100,  # Increase this number to perform a more thorough hyperparameter search
        trials=trials,
        verbose=True,
        rstate=np.random.default_rng(42)
    )

    print('lr_opt:', lr_opt)

    solver_mapping = {
        0: ('lbfgs', ['l2', None]),
        1: ('newton-cg', ['l2', None]),
        2: ('liblinear', ['l1', 'l2']),
        3: ('sag', ['l2', None]),
        4: ('saga', ['l1', 'l2', 'elasticnet', None])
    }

    solver_type, solver_penalties = solver_mapping[lr_opt['solver']]
    lr_opt['logisticregression__solver'] = solver_type

    penalty_index = lr_opt[f'{solver_type}_penalty']
    penalty = solver_penalties[penalty_index]
    lr_opt['logisticregression__penalty'] = penalty

    del lr_opt[f'{solver_type}_penalty']

    if lr_opt['class_weight'] == 0:
        lr_opt['logisticregression__class_weight'] = None
    elif lr_opt['class_weight'] == 1:
        lr_opt['logisticregression__class_weight'] = 'balanced'

    return lr_opt

#=========================================    SVM   ============================================#
def svm_c_hyperopt(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, model):
    def objective(params):
        hyp_svm_c_pipeline = make_pipeline(preprocessor, pca, over_sampler, under_sampler, model)
        hyp_svm_c_pipeline.get_params().keys()
        hyp_svm_c_pipeline.set_params(**params)
        average_precision = median(cross_val_score(hyp_svm_c_pipeline, X_train, y_train, cv=10, scoring="average_precision"))
        return {'loss': -average_precision, 'status': STATUS_OK}

    search_space = {
    'svc__C': hp.loguniform('C', 0, 20),
    'svc__kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
    'svc__degree': hp.choice('degree', range(1, 10)),  # Add degree hyperparameter
    'svc__gamma': hp.loguniform('gamma', 0, 20),
    'svc__class_weight': hp.choice('class_weight', [None, 'balanced'])
}

    trials = Trials()
    
    np.random.seed(42)
    
    svm_c_opt = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,  # Adjust the number of evaluations to 100 for actual run
        trials=trials,
        verbose=True,
        rstate=np.random.default_rng(42)
    )

    if svm_c_opt["kernel"] == 0:
        svm_c_opt["kernel"] = "linear"
    if svm_c_opt["kernel"] == 1:
        svm_c_opt["kernel"] = "poly"
    if svm_c_opt["kernel"] == 2:
        svm_c_opt["kernel"] = "rbf"
    if svm_c_opt["kernel"] == 3:
        svm_c_opt["kernel"] = "sigmoid"

    return svm_c_opt

#================================ Random Forest========================================#
def rf_c_hyperopt(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, model):
    def objective(params):
        hyp_rf_pipeline = make_pipeline(preprocessor, pca, over_sampler, under_sampler, model)
        hyp_rf_pipeline.set_params(**params)
        average_precision = median(cross_val_score(hyp_rf_pipeline, X_train, y_train, cv=10, scoring="average_precision"))
        return {'loss': -average_precision, 'status': STATUS_OK}

    search_space = {
        'randomforestclassifier__n_estimators': hp.choice('n_estimators', range(10,1000)),
        'randomforestclassifier__max_depth': hp.choice('max_depth', range(1,50)),
        'randomforestclassifier__min_samples_split': hp.choice('min_samples_split', range(1, 50)),
        'randomforestclassifier__min_samples_leaf': hp.choice('min_samples_leaf',  range(1, 50)),
        'randomforestclassifier__max_features': hp.choice('max_features', ['log2', 'sqrt', None]),
        'randomforestclassifier__criterion': hp.choice('criterion', ['gini', 'entropy', 'log_loss']) #add criterion 3 choices
    }

    trials = Trials()

    np.random.seed(42)
    
    rf_opt = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,  # Adjust the number of evaluations to the desired value
        trials=trials,
        verbose=True,
        rstate=np.random.default_rng(42)
    )

    return rf_opt


#=========================================    XGBoost   ============================================#
def xgb_c_hyperopt(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, model):
    def objective(params):
        hyp_xgb_pipeline = make_pipeline(preprocessor, pca, over_sampler, under_sampler, model)
        hyp_xgb_pipeline.set_params(**params)
        average_precision = median(cross_val_score(hyp_xgb_pipeline, X_train, y_train, cv=10, scoring="average_precision"))
        return {'loss': -average_precision, 'status': STATUS_OK}

    search_space = {
        'xgbclassifier__learning_rate': hp.choice('learning_rate', [0.001, 0.01, 0.1, 1, 5, 10]), #changed to specified learning rates
        'xgbclassifier__n_estimators': hp.quniform('n_estimators', 10, 1000, 1),
        'xgbclassifier__max_depth': hp.quniform('max_depth', 1, 50, 1),
        'xgbclassifier__min_child_weight': hp.choice('min_child_weight', [1, 3, 5]),
        'xgbclassifier__subsample': hp.uniform('subsample', 0.5, 1.0),
        'xgbclassifier__gamma': hp.uniform('gamma', 0.0, 1.0),
        'xgbclassifier__colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'xgbclassifier__reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        'xgbclassifier__reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0)
    }

    trials = Trials()

    np.random.seed(42)
    xgb_opt = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,  # 100
        trials=trials,
        verbose=True,
        rstate=np.random.default_rng(42)
    )
    
    learning_rate_mapping = [0.001, 0.01, 0.1, 1, 5, 10]
    min_child_weight_mapping = [1, 3, 5]

    if 'learning_rate' in xgb_opt:
        xgb_opt['learning_rate'] = learning_rate_mapping[xgb_opt['learning_rate']]
    
    if 'min_child_weight' in xgb_opt:
        xgb_opt['min_child_weight'] = min_child_weight_mapping[xgb_opt['min_child_weight']]


    return xgb_opt

#================================ Neural Network ========================================#
def nn_c_hyperopt(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, input_shape):
    def create_model(layers, neurons, learning_rate, dropout_rate, input_shape):
        model = Sequential()
        #tune hidden layer 
        model.add(Dense(neurons, input_dim=input_shape, activation='relu'))
        model.add(Dropout(dropout_rate))

        for i in range(layers):
            model.add(Dense(neurons, activation='relu'))
            model.add(Dropout(dropout_rate))

        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
   
    def objective(params):
        model = KerasClassifier(
            build_fn=create_model,
            layers=int(params['layers']),
            neurons=int(params['neurons']),
            learning_rate=params['learning_rate'],
            dropout_rate=params['dropout_rate'],
            epochs=int(params['epochs']),
            batch_size=int(params['batch_size']),
            verbose=0
        )
        hyp_nn_pipeline = make_pipeline(preprocessor, pca, over_sampler, under_sampler, model)
        average_precision = np.median(cross_val_score(hyp_nn_pipeline, X_train, y_train, cv=10, scoring="average_precision"))
        return {'loss': -average_precision, 'status': STATUS_OK}

    search_space = {
        'layers': hp.quniform('layers', 1, 20, 1),
        'neurons': hp.quniform('neurons', 10, 500, 10),
        'learning_rate': hp.choice('learning_rate', [0.001, 0.01, 0.1, 1, 5, 10]),
        'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.5),
        'epochs': hp.choice('epochs', [200]),
        'batch_size': hp.choice('batch_size', [32, 64, 128, 256, 512])
    }

    trials = Trials()

    np.random.seed(42)
    nn_opt = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        verbose=True,
        rstate=np.random.default_rng(42)
    )

    learning_rate_mapping = [0.001, 0.01, 0.1, 1, 5, 10]
    batch_size_mapping = [32, 64, 128, 256, 512]

    if 'learning_rate' in nn_opt:
        nn_opt['learning_rate'] = learning_rate_mapping[nn_opt['learning_rate']]
    
    if 'batch_size' in nn_opt:
        nn_opt['batch_size'] = batch_size_mapping[nn_opt['batch_size']]

    return nn_opt

