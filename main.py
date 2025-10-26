from LoadNPrep import data_loader
from random_forest import random_forest
from logistic_regression import logistic_regression
import numpy as np
import os, sys, subprocess
import pandas as pd
from inspect import signature
from xgboost_classifier import xgboost_classifier

def XGB_drive(X, y, smote=False, threshold=0.50):
    return xgboost_classifier(X, y, smote=smote, threshold=threshold)


def RF_drive(X,y, smote_trigger):
    sig = signature(logistic_regression)
    kwargs = {}
    # handle smote / smote_trigger name difference
    if "smote" in sig.parameters:
        kwargs["smote"] = smote_trigger
    elif "smote_trigger" in sig.parameters:
        kwargs["smote_trigger"] = smote_trigger
    return random_forest(X, y, **kwargs)

def LR_drive(X,y, smote_trigger, threshold=0.5):
    sig = signature(logistic_regression)
    kwargs = {}
    # handle smote / smote_trigger name difference
    if "smote" in sig.parameters:
        kwargs["smote"] = smote_trigger
    elif "smote_trigger" in sig.parameters:
        kwargs["smote_trigger"] = smote_trigger
    # only pass threshold if supported
    if "threshold" in sig.parameters:
        kwargs["threshold"] = threshold

    return logistic_regression(X, y, **kwargs)

def build_explore_preview(X, y, n_head: int = 20):
    X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
    y_s  = pd.Series(y, name="Class") if not hasattr(y, "name") else y.rename("Class")
    df_preview = X_df.copy()
    df_preview["Class"] = y_s.values
    counts = pd.Series(y_s).value_counts().sort_index()
    counts = counts.rename({0: "Not Fraud", 1: "Fraud"})  # safe even if labels differ
    return df_preview.head(n_head), counts


def run_streamlit():

    app = os.path.join(os.path.dirname(__file__), "app.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", app])


if __name__ == "__main__":
    sys.exit(run_streamlit())