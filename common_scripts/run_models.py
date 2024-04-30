from common_scripts.hyperopt_models import lr_hyperopt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import pickle
from imblearn.pipeline import make_pipeline

def generate_metrics(pipeline, X_test, y_test):
    lr_y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, lr_y_pred)
    print(report)
    print(roc_auc_score(y_test, pipeline.predict_proba(X_test)[:,1]))


def lr_model(X_train, y_train, X_test, y_test, preprocessor, pca, over_sampler, under_sampler, model):
    lr_hyperparameters = lr_hyperopt(X_train, y_train, preprocessor, pca, over_sampler, under_sampler, model)
    tuned_lr_model = LogisticRegression(**lr_hyperparameters)
    lr_pipeline = make_pipeline(preprocessor, pca, over_sampler, under_sampler, tuned_lr_model)
    lr_pipeline = lr_pipeline.fit(X_train, y_train)
    generate_metrics(lr_pipeline, X_test, y_test)
    #pikle dump 
    pickle.dump(lr_pipeline, open('lr_model.pkl', 'wb')) #put in outputs folder
    return lr_pipeline