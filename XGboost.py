#!/usr/bin/env python
# coding: utf-8

# In[41]:


# Data load and preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    fbeta_score,
    precision_recall_curve
)
from xgboost import XGBClassifier
import shap


df = pd.read_csv("creditcard.csv")

# Check the schema of the data
df.head()

# Check the types of data
df.info()

# Our target variable is the last one, "Class", which determines if the transaction is fraud or not, i.e, 1 or 0
# Occurrences of fraud vs. non-fraud

occurrences = df['Class'].value_counts()
occurrences

# Ratio of fraudulent cases
ratio = (df["Class"] == 1).sum() / len(df["Class"])
print("Ratio of fraudulent vs. non-fraudulent cases: \n", ratio)

# Features and target variable
X = df.drop(columns=["Class"])
y = df["Class"].values



# In[42]:


# Split into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


# In[48]:


# XGBOOST
clf = XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr',
    random_state=42
)
clf.fit(
    X_train_res, y_train_res,
    eval_set=[(X_valid, y_valid)],
    verbose=False,
)

# Print the classifcation report, confusion matrix, and other metrics
BEST_THR = 0.9201
proba = clf.predict_proba(X_test)[:, 1]
pred  = (proba >= BEST_THR).astype(int)

print("Test ROC-AUC: ", roc_auc_score(y_test, proba))
print("Test PR-AUC:  ", average_precision_score(y_test, proba))
print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification report:\n", classification_report(y_test, pred, digits=4))
print("F2 @ chosen threshold:", fbeta_score(y_test, pred, beta=2))


# In[ ]:


f2_default = fbeta_score(y_test, pred, beta=2)
print(f"\nF2-score @ threshold=0.5: {f2_default:.4f}")

# --- Find the threshold that maximizes F2 on the test set ---
prec, rec, thr = precision_recall_curve(y_test, proba)
beta = 2
# F_beta from precision/recall sequence
f2_curve = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec + 1e-12)
best_idx = int(np.nanargmax(f2_curve))
best_thr = thr[best_idx-1] if best_idx > 0 else 0.5  # align with thresholds array length

pred_f2 = (proba >= best_thr).astype(int)
f2_best = fbeta_score(y_test, pred_f2, beta=2)

print(f"Best F2 threshold: {best_thr:.4f}")
print(f"F2-score @ best threshold: {f2_best:.4f}")
print(f"Precision/Recall @ best F2: P={prec[best_idx]:.4f} | R={rec[best_idx]:.4f}")


# In[45]:


from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
PrecisionRecallDisplay.from_predictions(y_test, proba)
plt.title("Precision–Recall Curve")
plt.show()

RocCurveDisplay.from_predictions(y_test, proba)
plt.title("ROC Curve")
plt.show()


# In[49]:


X_test_df = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=X.columns)

Xts = X_test_df.sample(n=min(5000, len(X_test_df)), random_state=42)

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(Xts)

# Global importance: beeswarm 
shap.summary_plot(shap_values, Xts, plot_type="dot", show=True)

# Global importance: mean 
shap.summary_plot(shap_values, Xts, plot_type="bar", show=True)

idx_top = int(np.argmax(proba))  
x_one = X_test_df.iloc[idx_top:idx_top+1]
sv_one = explainer.shap_values(x_one)


try:
    shap.initjs()
    display(shap.force_plot(explainer.expected_value, sv_one, x_one))
except Exception:
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, sv_one[0], feature_names=x_one.columns.tolist())

