import os
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, average_precision_score,
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.utils import resample
from alive_progress import alive_bar

def get_classification_bootstrapped_metrics(pipeline, X_test, y_test, run_id, mname, target, output_path, n_iterations=1000):
    print("getting boostrapped metrics...")

    y_pred = pipeline.predict(X_test)

    # Compute initial model performance metrics
    model_accuracy = accuracy_score(y_test, y_pred)
    model_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    model_average_precision = average_precision_score(y_test, y_pred, pos_label=1)
    model_precision = precision_score(y_test, y_pred, pos_label=1)
    model_recall = recall_score(y_test, y_pred, pos_label=1)
    model_f1 = f1_score(y_test, y_pred, pos_label=1)
    model_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

    # Compute specificity (1 - False Positive Rate)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    model_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Lists to store bootstrap metrics
    accuracy_scores = []
    balanced_accuracy_scores = []
    average_precision_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []
    specificity_scores = []

    # Bootstrapping with progress bar
    with alive_bar(n_iterations, bar='fish', spinner='pulse',
                title=f'Running bootstrapped performance analysis ({n_iterations} iterations)...') as bar:
        for _ in range(n_iterations):
            X_bs, y_bs = resample(X_test, y_test, replace=True)
            y_hat = pipeline.predict(X_bs)

            accuracy_scores.append(accuracy_score(y_bs, y_hat))
            balanced_accuracy_scores.append(balanced_accuracy_score(y_bs, y_hat))
            average_precision_scores.append(average_precision_score(y_bs, y_hat, pos_label=1))
            precision_scores.append(precision_score(y_bs, y_hat, pos_label=1))
            recall_scores.append(recall_score(y_bs, y_hat, pos_label=1))
            f1_scores.append(f1_score(y_bs, y_hat, pos_label=1))
            roc_auc_scores.append(roc_auc_score(y_bs, pipeline.predict_proba(X_bs)[:, 1]))

            tn, fp, fn, tp = confusion_matrix(y_bs, y_hat).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_scores.append(specificity)

            bar()  # Update progress bar

    # Function to compute confidence intervals
    def compute_ci(data):
        return [round(num, 4) for num in st.norm.interval(confidence=0.95, loc=np.mean(data), scale=st.sem(data))]

    # Store results in a structured format
    results_dict = {
        'Model': target,
        'Median Accuracy [95% CI]': [round(model_accuracy, 4), compute_ci(accuracy_scores)],
        'Median Balanced Accuracy [95% CI]': [round(model_balanced_accuracy, 4), compute_ci(balanced_accuracy_scores)],
        'Median Average Precision [95% CI]': [round(model_average_precision, 4), compute_ci(average_precision_scores)],
        'Median Precision [95% CI]': [round(model_precision, 4), compute_ci(precision_scores)],
        'Median Specificity [95% CI]': [round(model_specificity, 4), compute_ci(specificity_scores)],
        'Median Recall [95% CI]': [round(model_recall, 4), compute_ci(recall_scores)],
        'Median F1 Score [95% CI]': [round(model_f1, 4), compute_ci(f1_scores)],
        'Median ROC AUC [95% CI]': [round(model_auc, 4), compute_ci(roc_auc_scores)]
    }

    results_df = pd.DataFrame([results_dict])
    
    # Save results to output folder
    os.makedirs(output_path, exist_ok=True)  # Ensure directory exists
    output_file = os.path.join(output_path, "classification", f"bootstrap_metrics_{target}_{mname}_{run_id}.csv")
    results_df.to_csv(output_file, index=False)
    print(results_df)
    print(f"Metrics saved to: {output_file}")
    return results_df