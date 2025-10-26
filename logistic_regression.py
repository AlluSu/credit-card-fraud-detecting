import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, fbeta_score)    
from imblearn.over_sampling import SMOTE
from cache import cache_finder, cacher

def logistic_regression(X, y, smote_trigger=False):
    

    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.05)

    if smote_trigger:
        trigger, cache_metrics = cache_finder("LR_smote")
        if trigger:
            return cache_metrics
        print("Applying SMOTE...")
        sm = SMOTE(sampling_strategy='minority', random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    else:
        trigger, cache_metrics = cache_finder("LR_no_smote")
        if trigger:
            return cache_metrics
    model = LogisticRegression(solver="liblinear")
    clf = model.fit(X_train, y_train)


    # Metriikat bojjiille
    proba = clf.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    metrics_dict = {
        "roc_auc": roc_auc_score(y_test, proba),                   # test ROC AUC
        "pr_auc": average_precision_score(y_test, proba),          # test PR AUC (AP)
        "confusion_matrix": confusion_matrix(y_test, pred),
        "classification_report": classification_report(y_test, pred, digits=4),
        "f2_score": fbeta_score(y_test, pred, beta=2),
        "y_test": np.array(y_test),      # for external plotting
        "proba": np.array(proba)         # for external plotting
    }

    print("LogisticRegression built successfully")
    if smote_trigger:
        cacher("LR_smote", metrics_dict)
        print("Logistic regression SMOTE model cached as JSON successfully")
    else:
        cacher("LR_no_smote", metrics_dict)
        print("Logistic regression non SMOTE model cached as JSON successfully")
    return metrics_dict
