import os
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, average_precision_score,
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, 
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.utils import resample
from alive_progress import alive_bar
from train_scripts.save_model_and_params import save_classification_pred_probs_and_labels, save_regression_pred_results

def get_classification_bootstrapped_metrics(pipeline, X_test, y_test, run_id, mname, target, output_path, n_iterations=1000):
    print("Getting bootstrapped metrics for classification model...")

    # Get the prediction probabilities for all classes
    y_pred_proba = pipeline.predict_proba(X_test)
    
    # Compute initial model performance metrics
    y_pred = pipeline.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)
    model_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    model_average_precision = average_precision_score(y_test, y_pred, pos_label=1)
    model_precision = precision_score(y_test, y_pred, pos_label=1)
    model_recall = recall_score(y_test, y_pred, pos_label=1)
    model_f1 = f1_score(y_test, y_pred, pos_label=1)
    model_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

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

    # List to store prediction probabilities and true labels for each bootstrap iteration
    all_probs_labels = []

    # Bootstrapping with progress bar
    with alive_bar(n_iterations, bar='fish', spinner='pulse',
                title=f'Running bootstrapped performance analysis ({n_iterations} iterations)...') as bar:
        for _ in range(n_iterations):
            X_bs, y_bs = resample(X_test, y_test, replace=True)
            y_hat = pipeline.predict(X_bs)
            y_hat_proba = pipeline.predict_proba(X_bs)
            
            accuracy_scores.append(accuracy_score(y_bs, y_hat))
            balanced_accuracy_scores.append(balanced_accuracy_score(y_bs, y_hat))
            average_precision_scores.append(average_precision_score(y_bs, y_hat, pos_label=1))
            precision_scores.append(precision_score(y_bs, y_hat, pos_label=1))
            recall_scores.append(recall_score(y_bs, y_hat, pos_label=1))
            f1_scores.append(f1_score(y_bs, y_hat, pos_label=1))
            roc_auc_scores.append(roc_auc_score(y_bs, y_hat_proba[:, 1]))

            tn, fp, fn, tp = confusion_matrix(y_bs, y_hat).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_scores.append(specificity)

            # Save prediction probabilities and true labels for this bootstrap sample
            probs_labels = pd.DataFrame({
                'True Label': y_bs,
                'Predicted Probability': y_hat_proba[:, 1],
                'Model Name': mname
            })
            all_probs_labels.append(probs_labels)

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

    # Save the prediction probabilities and true labels to CSV
    all_probs_labels_df = pd.concat(all_probs_labels, ignore_index=True)
    save_classification_pred_probs_and_labels(y_test, y_pred_proba, mname, target, run_id, output_path)
    
    return results_df, all_probs_labels_df

# Function to get regression bootstrapped metrics
def get_regression_bootstrapped_metrics(pipeline, X_test, y_test, run_id, mname, target, output_path, n_iterations=1000):
    print("Getting bootstrapped metrics for regression model...")

    # Get initial predictions for model performance metrics
    y_pred = pipeline.predict(X_test)
    model_mae = mean_absolute_error(y_test, y_pred)
    model_mse = mean_squared_error(y_test, y_pred)
    model_rmse = np.sqrt(model_mse)
    model_r2 = r2_score(y_test, y_pred)

    # Lists to store bootstrap metrics
    mae_scores = []
    mse_scores = []
    rmse_scores = []
    r2_scores = []

    # List to store prediction results (true labels and predicted values) for each bootstrap iteration
    all_pred_results = []

    # Bootstrapping with progress bar
    with alive_bar(n_iterations, bar='fish', spinner='pulse',
                title=f'Running bootstrapped performance analysis ({n_iterations} iterations)...') as bar:
        for _ in range(n_iterations):
            X_bs, y_bs = resample(X_test, y_test, replace=True)
            y_hat = pipeline.predict(X_bs)
            
            mae_scores.append(mean_absolute_error(y_bs, y_hat))
            mse_scores.append(mean_squared_error(y_bs, y_hat))
            rmse_scores.append(np.sqrt(mean_squared_error(y_bs, y_hat)))
            r2_scores.append(r2_score(y_bs, y_hat))

            # Save prediction results (true labels and predicted values) for this bootstrap sample
            pred_results = pd.DataFrame({
                'True Label': y_bs,
                'Predicted Value': y_hat,
                'Model Name': mname
            })
            all_pred_results.append(pred_results)

            bar()  # Update progress bar

    # Function to compute confidence intervals
    def compute_ci(data):
        return [round(num, 4) for num in st.norm.interval(confidence=0.95, loc=np.mean(data), scale=st.sem(data))]

    # Store results in a structured format
    results_dict = {
        'Model': target,
        'Median MAE [95% CI]': [round(model_mae, 4), compute_ci(mae_scores)],
        'Median MSE [95% CI]': [round(model_mse, 4), compute_ci(mse_scores)],
        'Median RMSE [95% CI]': [round(model_rmse, 4), compute_ci(rmse_scores)],
        'Median R2 [95% CI]': [round(model_r2, 4), compute_ci(r2_scores)]
    }

    results_df = pd.DataFrame([results_dict])
    
    # Save results to output folder
    os.makedirs(output_path, exist_ok=True)  # Ensure directory exists
    output_file = os.path.join(output_path, "regression", f"bootstrap_metrics_{target}_{mname}_{run_id}.csv")
    results_df.to_csv(output_file, index=False)
    print(results_df)
    print(f"Metrics saved to: {output_file}")

    # Save the prediction results (true labels and predicted values) to CSV
    all_pred_results_df = pd.concat(all_pred_results, ignore_index=True)
    save_regression_pred_results(y_test, y_pred, mname, target, run_id, output_path)
    
    return results_df, all_pred_results_df