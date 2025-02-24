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
        max_evals=100, # Increase this number to perform more thorough hyperparameter search
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


    if lr_opt['class_weight'] == 0:
        lr_opt['class_weight'] = None
    elif lr_opt['class_weight'] == 1:
        lr_opt['class_weight'] = 'balanced'
    
   
    print(lr_opt)
    print(trials.best_trial['result']['loss'])

    return lr_opt