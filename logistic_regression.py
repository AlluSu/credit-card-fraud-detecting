import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, fbeta_score, roc_curve, auc, precision_recall_curve,
    accuracy_score
)    
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

def logistic_regression(X, y, smote_trigger=False):

    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.05)

    if smote_trigger:
        print("Applying SMOTE...")
        sm = SMOTE(sampling_strategy='minority', random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    model = LogisticRegression(solver="liblinear")
    clf = model.fit(X_train, y_train)

    # Predict
    predicted = model.predict(X_test)

    # Print the classifcation report, confusion matrix, and other metrics
    print("Accuracy from the logistic regression: \n", accuracy_score(y_test, predicted))

    print("\n")

    matrix = confusion_matrix(y_test, predicted)
    print("Confusion matrix: \n", matrix)

    # Do more prettier visualization from the results
    classes = ["Not Fraud", "Fraud"]

    conf_mat = confusion_matrix(y_test, predicted)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

    print('Classification report:\n', classification_report(y_test, predicted))

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_test, predicted).ravel()

    precision = true_positive / (true_positive + false_positive)
    print("Precision: ", precision)

    recall = true_positive / (true_positive + false_negative)
    print("Recall: ", recall)


    # Metriikat bojjiille
    proba = clf.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    metrics_dict = {
        #"roc_auc_cv": roc_auc_cv,                                  # array of CV scores
        "roc_auc": roc_auc_score(y_test, proba),                   # test ROC AUC
        "pr_auc": average_precision_score(y_test, proba),          # test PR AUC (AP)
        "confusion_matrix": confusion_matrix(y_test, pred),
        "classification_report": classification_report(y_test, pred, digits=4),
        "f2_score": fbeta_score(y_test, pred, beta=2),
        "y_test": y_test,      # for external plotting
        "proba": proba         # for external plotting
    }

    print("LogisticRegression built successfully")
    return metrics_dict



def plot_roc_pr(y_true, proba):
    """Match the RF helper for quick plots (optional use)."""
    print('Plotting roc and pr curves..')

    # ROC
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # PR
    precision, recall, _ = precision_recall_curve(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {pr_auc:.4f})')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.show()