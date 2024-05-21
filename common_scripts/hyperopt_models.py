import numpy as np
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from statistics import median
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp

#========================================= Logistic Regression   ============================================#
def lr_hyperopt(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, model):
    # Define the objective function
    def objective(params):
        print('params:', params)
        
        solver_params = params['logisticregression__solver']
        del params['logisticregression__solver']

        print('solverparams:', solver_params)

        if solver_params['type'] == 'lbfgs':
            params['logisticregression__solver'] = 'lbfgs'
            params['logisticregression__penalty'] = solver_params['penalty']
        elif solver_params['type'] == 'newton-cg':
            params['logisticregression__solver'] = 'newton-cg'
            params['logisticregression__penalty'] = solver_params['penalty']
        elif solver_params['type'] == 'liblinear':
            params['logisticregression__solver'] = 'liblinear'
            params['logisticregression__penalty'] = solver_params['penalty']
        elif solver_params['type'] == 'sag':
            params['logisticregression__solver'] = 'sag'
            params['logisticregression__penalty'] = solver_params['penalty']
        elif solver_params['type'] == 'saga':
            params['logisticregression__solver'] = 'saga'
            params['logisticregression__penalty'] = solver_params['penalty']
       
        
        # Create and configure the pipeline
        hyp_lr_pipeline = make_pipeline(preprocessor, pca, over_sampler, under_sampler, model)
        hyp_lr_pipeline.get_params().keys()
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
        max_evals=1, # Increase this number to perform more thorough hyperparameter search
        trials=trials,
        verbose=True,
        rstate=np.random.default_rng(42)
    )

    #print(lr_opt)
    
    if lr_opt['solver'] == 0:
        lr_opt['solver'] = 'lbfgs'
        if lr_opt['lbfgs_penalty'] == 0:
            lr_opt['penalty'] = 'l2'
        if lr_opt['lbfgs_penalty'] == 1:
            lr_opt['penalty'] = None
        del lr_opt['lbfgs_penalty']
    elif lr_opt['solver'] == 1:
        lr_opt['solver'] = 'newton-cg'
        if lr_opt['newton_penalty'] == 0:
            lr_opt['penalty'] = 'l2'
        else:
            lr_opt['penalty'] = None
        del lr_opt['newton_penalty']
    elif lr_opt['solver'] == 2:
        lr_opt['solver'] = 'liblinear'
        if lr_opt['liblinear_penalty'] == 0:
            lr_opt['penalty'] = 'l1'
        else:
            lr_opt['penalty'] = 'l2'
        del lr_opt['liblinear_penalty']
    elif lr_opt['solver'] == 3:
        lr_opt['solver'] = 'sag'
        if lr_opt['sag_penalty'] == 0:
            lr_opt['penalty'] = 'l2'
        else:
            lr_opt['penalty'] = None
        del lr_opt['sag_penalty']
    elif lr_opt['solver'] == 4:
        lr_opt['solver'] = 'saga'
        if lr_opt['saga_penalty'] == 0:
            lr_opt['penalty'] = 'l1'
        elif lr_opt['saga_penalty'] == 1:
            lr_opt['penalty'] = 'l2'
        elif lr_opt['saga_penalty'] == 2:
            lr_opt['penalty'] = 'elasticnet'
        else:
            lr_opt['penalty'] = None
        del lr_opt['saga_penalty']

    
    # elif lr_opt['solver'] == 'newton-cg':
    #     lr_opt['penalty'] = lr_opt['newton_penalty']
    # elif lr_opt['solver'] == 'liblinear':
    #     lr_opt['penalty'] = lr_opt['liblinear_penalty']
    # elif lr_opt['solver'] == 'sag':
    #     lr_opt['penalty'] = lr_opt['sag_penalty']
    # elif lr_opt['solver'] == 'saga':
    #     lr_opt['penalty'] = lr_opt['saga_penalty']

    if lr_opt['class_weight'] == 0:
        lr_opt['class_weight'] = None
    elif lr_opt['class_weight'] == 1:
        lr_opt['class_weight'] = 'balanced'
    
    # class_weight_mapping = {0: None, 1: 'balanced'}
    
   
    print(lr_opt)
    print(trials.best_trial['result']['loss'])

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
    #add degree for poly
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
        'randomforestclassifier__max_features': hp.choice('max_features', ['log2', 'sqrt', None])
        #add criterion 3 choices
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
    'xgbclassifier__learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)), #[0.001, 0.01, 0.1, 1, 5, 10]
    'xgbclassifier__n_estimators': hp.choice('n_estimators', [50, 100, 200]), #10, 1000
    'xgbclassifier__max_depth': hp.choice('max_depth', [3, 5, 7]), #1,50
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

    return xgb_opt

#================================ Neural Network ========================================#
def nn_c_hyperopt(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, model):
    def objective(params):
        hyp_nn_pipeline = make_pipeline(preprocessor, pca, over_sampler, under_sampler, model)
        hyp_nn_pipeline.set_params(**params)
        average_precision = median(cross_val_score(hyp_nn_pipeline, X_train, y_train, cv=10, scoring="average_precision"))
        return {'loss': -average_precision, 'status': STATUS_OK}

    # search_space = {
    #     'mlpclassifier__hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(50,), (100,), (200,)]),
    #     'mlpclassifier__activation': hp.choice('activation', ['relu', 'tanh']),
    #     'mlpclassifier__solver': hp.choice('solver', ['adam', 'lbfgs']),
    #     'mlpclassifier__alpha': hp.loguniform('alpha', np.log(0.0001), np.log(0.01)),
    #     'mlpclassifier__learning_rate_init': hp.loguniform('learning_rate_init', np.log(0.001), np.log(0.1)),
    #     'mlpclassifier__max_iter': hp.choice('max_iter', range(100, 1000)),
    # }
    
    search_space = {
        'mlpclassifier__hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(50,), (100,), (200,)]),
        'mlpclassifier__activation': hp.choice('activation', ['relu', 'tanh']),
        'mlpclassifier__solver': hp.choice('solver', ['adam', 'lbfgs']),
        'mlpclassifier__alpha': hp.loguniform('alpha', np.log(0.0001), np.log(0.01)),
        'mlpclassifier__learning_rate_init': hp.loguniform('learning_rate_init', np.log(0.001), np.log(0.1)),
        'mlpclassifier__max_iter': hp.choice('max_iter', range(100, 1000)),
    }
    
    

    trials = Trials()

    np.random.seed(42)
    
    nn_opt = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,  # 100
        trials=trials,
        verbose=True,
        rstate=np.random.default_rng(42)
    )

    if nn_opt["solver"] == 0:
        nn_opt["solver"] = "adam"
    if nn_opt["solver"] == 1:
        nn_opt["solver"] = "lbfgs"
    
    if nn_opt["activation"] == 0:
        nn_opt["activation"] = "relu"
    if nn_opt["activation"] == 1:
        nn_opt["activation"] = "tanh"
    
    
    return nn_opt