def tune_hyperparameters(X_train, y_train, outcome: str, days: int, run_id: str, model_algorithm: str = 'random_forest'):
    objectives = {'random_forest': rf_objective, 'xgboost': xgb_objective, 'logistic_regression': lr_objective, 'torch_nn': nn_objective}
    study = optuna.create_study(sampler=TPESampler(seed=seed), study_name=f'{run_id}_{outcome}_{days}_{model_algorithm}', storage=study_storage, direction='maximize', load_if_exists=True)
    study.optimize(lambda trial: objectives.get(model_algorithm)(trial, X_train, y_train, groups, selected_features), n_trials=200)
    best_value = study.best_trial.value
    best_params = study.best_trial.params
    return best_params