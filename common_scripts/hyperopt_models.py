import numpy as np
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from statistics import median
#from hyperopt.pyll.base import scope
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp


def lr_hyperopt(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, model):
    def objective(params):
        hyp_lr_pipeline = make_pipeline(preprocessor, pca, over_sampler, under_sampler, model)
        hyp_lr_pipeline.get_params().keys()
        hyp_lr_pipeline.set_params(**params)
        average_precision = median(cross_val_score(hyp_lr_pipeline,X_train,y_train, cv = 10, scoring = "average_precision"))
        return {'loss': -average_precision, 'status': STATUS_OK}
    
    search_space = {
        'logisticregression__tol': hp.uniform('tol', 0.00001, 0.1),
        'logisticregression__C': hp.uniform('C', 0.001, 100),
        'logisticregression__solver': hp.choice('solver', ['lbfgs','newton-cg','liblinear','sag','saga']),
        'logisticregression__max_iter': hp.choice('max_iter', range(100, 4000)),
        'logisticregression__class_weight': 'balanced',
        'logisticregression__penalty': 'l2'
                    }
    trials = Trials()
    
    np.random.seed(42)
    lr_opt = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10, #100
        trials = trials,
        verbose=True,
        rstate = np.random.default_rng(42)
    )
    
    if lr_opt["solver"] == 0:
        lr_opt["solver"] = "lbfgs"
    if lr_opt["solver"] == 1:
        lr_opt["solver"] = "newton-cg"
    if lr_opt["solver"] == 2:
        lr_opt["solver"] = "liblinear"
    if lr_opt["solver"] == 3:
        lr_opt["solver"] = "sag"
    if lr_opt["solver"] == 4:
        lr_opt["solver"] = "saga"
    return lr_opt

