# def lr_hyperopt(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, model):
#     # Define the objective function
#     def objective(params):
#         print('params:', params)
        
#         solver_params = params['logisticregression__solver']
#         del params['logisticregression__solver']

#         print('solverparams:', solver_params)

#         if solver_params['type'] == 'lbfgs':
#             params['logisticregression__solver'] = 'lbfgs'
#             params['logisticregression__penalty'] = solver_params['penalty']
#         elif solver_params['type'] == 'newton-cg':
#             params['logisticregression__solver'] = 'newton-cg'
#             params['logisticregression__penalty'] = solver_params['penalty']
#         elif solver_params['type'] == 'liblinear':
#             params['logisticregression__solver'] = 'liblinear'
#             params['logisticregression__penalty'] = solver_params['penalty']
#         elif solver_params['type'] == 'sag':
#             params['logisticregression__solver'] = 'sag'
#             params['logisticregression__penalty'] = solver_params['penalty']
#         elif solver_params['type'] == 'saga':
#             params['logisticregression__solver'] = 'saga'
#             params['logisticregression__penalty'] = solver_params['penalty']
       
        
#         # Create and configure the pipeline
#         hyp_lr_pipeline = make_pipeline(preprocessor, pca, over_sampler, under_sampler, model)
#         hyp_lr_pipeline.get_params().keys()
#         hyp_lr_pipeline.set_params(**params)
#         average_precision = median(cross_val_score(hyp_lr_pipeline, X_train, y_train, cv=10, scoring="average_precision", n_jobs=-1))
#         return {'loss': -average_precision, 'status': STATUS_OK}
    
#     # Define the search space
#     search_space = {
#         'logisticregression__C': hp.uniform('C', 0.001, 100),
#         'logisticregression__solver': hp.choice('solver', [
#             {
#                 'type': 'lbfgs',
#                 'penalty': hp.choice('lbfgs_penalty', ['l2', None])
#             },
#             {
#                 'type': 'newton-cg',
#                 'penalty': hp.choice('newton_penalty', ['l2', None])
#             },
#             {
#                 'type': 'liblinear',
#                 'penalty': hp.choice('liblinear_penalty', ['l1', 'l2'])
#             },
#             {
#                 'type': 'sag',
#                 'penalty': hp.choice('sag_penalty', ['l2', None])
#             },
#             {
#                 'type': 'saga',
#                 'penalty': hp.choice('saga_penalty', ['l1', 'l2', 'elasticnet', None])
#             }
#         ]),
#         'logisticregression__max_iter': hp.choice('max_iter', range(100, 4000)),
#         'logisticregression__class_weight': hp.choice('class_weight', [None, 'balanced'])
#     }
    
#     trials = Trials()
    
#     np.random.seed(42)
    
#     # Optimize the hyperparameters
#     lr_opt = fmin(
#         fn=objective,
#         space=search_space,
#         algo=tpe.suggest,
#         max_evals=100, # Increase this number to perform more thorough hyperparameter search
#         trials=trials,
#         verbose=True,
#         rstate=np.random.default_rng(42)
#     )

#     #print(lr_opt)
    
#     if lr_opt['solver'] == 0:
#         lr_opt['solver'] = 'lbfgs'
#         if lr_opt['lbfgs_penalty'] == 0:
#             lr_opt['penalty'] = 'l2'
#         if lr_opt['lbfgs_penalty'] == 1:
#             lr_opt['penalty'] = None
#         del lr_opt['lbfgs_penalty']
#     elif lr_opt['solver'] == 1:
#         lr_opt['solver'] = 'newton-cg'
#         if lr_opt['newton_penalty'] == 0:
#             lr_opt['penalty'] = 'l2'
#         else:
#             lr_opt['penalty'] = None
#         del lr_opt['newton_penalty']
#     elif lr_opt['solver'] == 2:
#         lr_opt['solver'] = 'liblinear'
#         if lr_opt['liblinear_penalty'] == 0:
#             lr_opt['penalty'] = 'l1'
#         else:
#             lr_opt['penalty'] = 'l2'
#         del lr_opt['liblinear_penalty']
#     elif lr_opt['solver'] == 3:
#         lr_opt['solver'] = 'sag'
#         if lr_opt['sag_penalty'] == 0:
#             lr_opt['penalty'] = 'l2'
#         else:
#             lr_opt['penalty'] = None
#         del lr_opt['sag_penalty']
#     elif lr_opt['solver'] == 4:
#         lr_opt['solver'] = 'saga'
#         if lr_opt['saga_penalty'] == 0:
#             lr_opt['penalty'] = 'l1'
#         elif lr_opt['saga_penalty'] == 1:
#             lr_opt['penalty'] = 'l2'
#         elif lr_opt['saga_penalty'] == 2:
#             lr_opt['penalty'] = 'elasticnet'
#         else:
#             lr_opt['penalty'] = None
#         del lr_opt['saga_penalty']


#     if lr_opt['class_weight'] == 0:
#         lr_opt['class_weight'] = None
#     elif lr_opt['class_weight'] == 1:
#         lr_opt['class_weight'] = 'balanced'
    
   
#     print(lr_opt)
#     print(trials.best_trial['result']['loss'])

#     return lr_opt

def get_bootstrapped_metrics(pipeline_list, X_test, y_test):
    '''
    Runs comparative bootstrapped performance analysis.
    '''
    results_list = []
    model_names = ['Logistic Regression', 'Random Forest Classifier', 'XGBoost Classifier', 'Deep Neural Network']
    for index, best_model  in enumerate(pipeline_list):
        print("Calculating scores for bootstrapping for ",  model_names[index])
        # Get model predictions
        y_pred = best_model.predict(X_test)
        # Calculate metrics
        model_accuracy = accuracy_score(y_test, y_pred)
        model_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        model_average_precision = average_precision_score(y_test, y_pred, pos_label=1)
        model_precision = precision_score(y_test, y_pred, pos_label=1)
        model_recall = recall_score(y_test, y_pred, pos_label=1)
        model_f1 = f1_score(y_test, y_pred, pos_label=1)
        model_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1])
        model_specificity = specificity_score(y_test, y_pred, pos_label=1)
        # Initiate empty lists for bootstrapped scores
        accuracy_scores = []
        balanced_accuracy_scores = []
        average_precision_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        roc_auc_scores = []
        specificity_scores = []
        # Select number of iterations for bootstrapping
        n_iterations = 100
        for i in range(n_iterations):
            # Calculate metrics on resampled data
            X_bs, y_bs = resample(X_test, y_test, replace=True)
            y_hat = best_model.predict(X_bs)
            accuracy = accuracy_score(y_bs, y_hat)
            balanced_accuracy = balanced_accuracy_score(y_bs, y_hat)
            average_precision = average_precision_score(y_bs, y_hat, pos_label=1)
            precision = precision_score(y_bs, y_hat, pos_label=1)
            recall = recall_score(y_bs, y_hat, pos_label=1)
            f1 = f1_score(y_bs, y_hat, pos_label=1)
            auc = roc_auc_score(y_bs, best_model.predict_proba(X_bs)[:,1])
            specificity = specificity_score(y_bs, y_hat, pos_label=1)
            accuracy_scores.append(accuracy)
            balanced_accuracy_scores.append(balanced_accuracy)
            average_precision_scores.append(average_precision)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            roc_auc_scores.append(auc)
            specificity_scores.append(specificity)
            accuracy_scores = sorted(accuracy_scores)
            balanced_accuracy_scores = sorted(balanced_accuracy_scores)
            average_precision_scores = sorted(average_precision_scores)
            precision_scores = sorted(precision_scores)
            recall_scores = sorted(recall_scores)
            f1_scores = sorted(f1_scores)
            roc_auc_scores = sorted(roc_auc_scores)
            specificity_scores = sorted(specificity_scores)
        # Add 95% confidence intervals
        results_list.append({
        'Model' : model_names[index],
        'median_accuracy [95% CI]' : [round(model_accuracy,4), [round(num, 4) for num in st.norm.interval(confidence=0.95, loc=np.mean(accuracy_scores), scale=st.sem(accuracy_scores))]],
        'median_balanced_accuracy [95% CI]' : [round(model_balanced_accuracy,4), [round(num, 4) for num in st.norm.interval(confidence=0.95, loc=np.mean(balanced_accuracy_scores), scale=st.sem(balanced_accuracy_scores))]],
        'median_average_precision [95% CI]' : [round(model_average_precision,4), [round(num, 4) for num in st.norm.interval(confidence=0.95, loc=np.mean(average_precision_scores), scale=st.sem(average_precision_scores))]],
        'median_precision [95% CI]' : [round(model_precision,4), [round(num, 4) for num in st.norm.interval(confidence=0.95, loc=np.mean(precision_scores), scale=st.sem(precision_scores))]],
        'median_specificity [95% CI]' : [round(model_specificity,4), [round(num, 4) for num in st.norm.interval(confidence=0.95, loc=np.mean(specificity_scores), scale=st.sem(specificity_scores))]],
        'median_recall [95% CI]' : [round(model_recall,4), [round(num, 4) for num in st.norm.interval(confidence=0.95, loc=np.mean(recall_scores), scale=st.sem(recall_scores))]],
        'median_f1 [95% CI]' : [round(model_f1,4), [round(num, 4) for num in st.norm.interval(confidence=0.95, loc=np.mean(f1_scores), scale=st.sem(f1_scores))]],
        'median_roc_auc [95% CI]' : [round(model_auc,4), [round(num, 4) for num in st.norm.interval(confidence=0.95, loc=np.mean(roc_auc_scores), scale=st.sem(roc_auc_scores))]]
    })
    return pd.DataFrame(results_list)