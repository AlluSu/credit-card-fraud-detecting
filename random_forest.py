from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report, fbeta_score, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from cache import cache_finder, cacher
import numpy as np


def random_forest(X, y, smote_trigger=False):
    # First create train and test sets using 5% for large and imbalanced datasets
    print("Creating spltis where test_size = 0.05")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    

    # Smote will be used if trigger = True
    if smote_trigger:
            trigger, cache_metrics = cache_finder("RF_smote")
            if trigger:
                 return cache_metrics
            print("Creating a RandomForest with SMOTE....")
            # Pipeline applies SMOTE only to training folds during CV
            pipeline = Pipeline([
                ('smote', SMOTE(random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=400, 
                                              max_depth=6, 
                                              random_state=42))])
            
            print("Running Random forest pipeline")
            roc_auc_cv = cross_val_score(pipeline, 
                                         X_train, 
                                         y_train, 
                                         cv=cv, 
                                         scoring='roc_auc', 
                                         n_jobs=-1)
            clf = pipeline.fit(X_train, y_train)
            
    else:
        trigger, cache_metrics = cache_finder("RF_no_smote")
        if trigger:
            return cache_metrics
        print("Creating a RandomForest without SMOTE...")
        clf = RandomForestClassifier(n_estimators=400, 
                                     max_depth=6, 
                                     random_state=42)
        print("Running Random forest pipeline")
        roc_auc_cv = cross_val_score(clf, 
                                     X_train, 
                                     y_train, 
                                     cv=cv, 
                                     scoring='roc_auc', 
                                     n_jobs=-1)
        clf.fit(X_train, y_train)

    print("Calculating metrics...")
    proba = clf.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.9201).astype(int)

    # Compute test set metrics
    metrics_dict = {
        "roc_auc_cv": roc_auc_cv,
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred),
        "classification_report": classification_report(y_test, pred, digits=4),
        "f2_score": fbeta_score(y_test, pred, beta=2),
        "y_test": np.array(y_test),     # Add for plotting
        "proba": np.array(proba)        # Add for plotting
    }

    # return test metrics so we can use these in main
    if smote_trigger:
        cacher("RF_smote", metrics_dict)
        print("Random Forest SMOTE model cached as JSON successfully")
    else:
        cacher("RF_no_smote", metrics_dict)
        print("Random Forest non SMOTE model cached as JSON successfully")

    return  metrics_dict




# Use predicted probabilities from the test set
# You need to modify your random_forest function to also return `proba` if not already:
# proba = clf.predict_proba(X_test)[:, 1]


"""

Call with:

proba = metrics['proba'] if 'proba' in metrics else None
y_test = metrics['y_test'] if 'y_test' in metrics else None


where metrics = random_forest()



"""