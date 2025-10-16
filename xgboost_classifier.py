# xgboost_classifier.py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, fbeta_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from cache import cache_finder, cacher

def xgboost_classifier(
    X, y,
    smote: bool = False,
    threshold: float = 0.50,
    test_size: float = 0.05,
    random_state: int = 42,
    n_estimators: int = 400,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9
):
    """
    Train/evaluate XGBoost like your LR/RF wrappers and return metrics_dict.
    """

    # ---- split (keep test untouched) ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # ---- CV ROC-AUC on training set (SMOTE inside folds if enabled) ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0
    )

    if smote:
        trigger, cache_metrics = cache_finder("XGBoost_smote")
        if trigger:
            return cache_metrics
        print("Creating a XGBoost with SMOTE....")
        cv_pipe = Pipeline([
            ("smote", SMOTE(random_state=random_state)),
            ("xgb", base_model)
        ])
        roc_auc_cv = cross_val_score(cv_pipe, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
        # Fit final model with SMOTE on the training split
        sm = SMOTE(random_state=random_state)
        X_train_fit, y_train_fit = sm.fit_resample(X_train, y_train)
        clf = base_model.fit(X_train_fit, y_train_fit)
    else:
        trigger, cache_metrics = cache_finder("XGBoost_no_smote")
        if trigger:
            return cache_metrics
        roc_auc_cv = cross_val_score(base_model, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
        clf = base_model.fit(X_train, y_train)

    # ---- Predict on untouched test set ----
    proba = clf.predict_proba(X_test)[:, 1]
    pred  = (proba >= threshold).astype(int)

    # ---- Metrics ----
    metrics_dict = {
        "roc_auc_cv": roc_auc_cv,
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred),
        "classification_report": classification_report(y_test, pred, digits=4),
        "f2_score": fbeta_score(y_test, pred, beta=2),
        "y_test": np.array(y_test),
        "proba": np.array(proba)
    }
    if smote:
        cacher("XGBoost_smote", metrics_dict)
        print("XGBoost SMOTE model cached as JSON successfully")
    else:
        cacher("XGBoost_no_smote", metrics_dict)
        print("XGBoost no SMOTE model cached as JSON successfully")

    return metrics_dict
