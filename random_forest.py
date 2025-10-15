from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report, fbeta_score, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt


def random_forest(X, y, smote_trigger=False):
    # First create train and test sets using 5% for large and imbalanced datasets
    print("Creating spltis where test_size = 0.05")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    

    # Smote will be used if trigger = True
    if smote_trigger:
            print("Creating a RandomForest with SMOTE....")
            # Pipeline applies SMOTE only to training folds during CV
            pipeline = Pipeline([
                ('smote', SMOTE(random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=200, 
                                              max_depth=7, 
                                              random_state=42))])
            
            print("RandomForest created, now running cross-validation 5 fold")
            roc_auc_cv = cross_val_score(pipeline, 
                                         X_train, 
                                         y_train, 
                                         cv=cv, 
                                         scoring='roc_auc', 
                                         n_jobs=-1)
            clf = pipeline.fit(X_train, y_train)
            
    else:
        print("Creating a RandomForest without SMOTE...")
        clf = RandomForestClassifier(n_estimators=200, 
                                     max_depth=7, 
                                     random_state=42)
        
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
        "y_test": y_test,     # Add for plotting
        "proba": proba        # Add for plotting
    }

    # return test metrics so we can use these in main
    print("RandomForest built successfully")
    return  metrics_dict




def plot_roc_pr(y_true, proba):
    print('Plotting roc and pr curves..')
    """
    Plots ROC curve and Precision-Recall curve.
    
    Parameters:
        y_true : array-like, true labels
        proba  : array-like, predicted probabilities for positive class
    """

    # -------- ROC Curve --------
    fpr, tpr, thresholds_roc = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # -------- Precision-Recall Curve --------
    precision, recall, thresholds_pr = precision_recall_curve(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.show()


# Use predicted probabilities from the test set
# You need to modify your random_forest function to also return `proba` if not already:
# proba = clf.predict_proba(X_test)[:, 1]


"""

Call with:

proba = metrics['proba'] if 'proba' in metrics else None
y_test = metrics['y_test'] if 'y_test' in metrics else None


where metrics = random_forest()



"""