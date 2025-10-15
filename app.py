import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from LoadNPrep import data_loader
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from LoadNPrep import data_loader
from main import LR_drive, RF_drive, build_explore_preview

st.set_page_config(page_title="Credit Card Fraud", page_icon="", layout="wide")
st.title("Credit Card Fraud — Lightweight Streamlit App")

# ---------- Data load ----------
def load_data():
    X, y = data_loader()
    return X, y

X, y = load_data()

# ------------ Tabs ------------
tabs = st.tabs(["Introduction", " Explore", " Logistic Regression", "Random Forest"," XGBoost", " Findings"])

with tabs[0]:
    st.subheader("Overview")
    st.write(
        "This mini app explores credit card fraud detection using a highly imbalanced dataset. "
    )
    st.subheader("Research Problem")
    st.markdown(
        """
**Goal.** Build a performant, interpretable classifier for rare fraud events.

**Core questions**
1. 
2. 
3. 

**Evaluation focus**
- Stratified train/test split  
- Metrics: Accuracy, Precision, Recall, Confusion Matrix  
        """
    )

# ---------- Explore ----------
with tabs[1]:
    st.subheader("Preview")
    df_preview, counts = build_explore_preview(X, y, n_head=20)

    st.dataframe(df_preview, width="stretch")
    st.caption(f"Rows (preview): {len(df_preview):,} | Columns: {df_preview.shape[1]}")

    st.subheader("Class Balance")
    st.bar_chart(counts)

# ---------- Logistic Regression ----------
with tabs[2]:
    st.subheader("Logistic Regression Results (No SMOTE vs SMOTE)")
    st.caption("Models are computed once on load and results are displayed below.")

    if "lr_nonce" not in st.session_state:
        st.session_state.lr_nonce = 0

    @st.cache_resource(show_spinner=True)
    def compute_lr_runs(X, y):
        # your LR_drive signature uses smote_trigger + threshold
        res_no = LR_drive(X, y, smote_trigger=False, threshold=0.50)
        res_sm = LR_drive(X, y, smote_trigger=True,  threshold=0.50)
        return res_no, res_sm

    if st.button("Run again", key="lr_rerun"):
        st.session_state.lr_nonce += 1

    res_no, res_sm = compute_lr_runs(X, y)

    def render_lr(col, res, title):
        with col:
            st.markdown(f"### {title}")

            # Metrics summary
            c1, c2, c3, c4 = st.columns(4)
            roc_auc_cv = res.get("roc_auc_cv", None)
            if roc_auc_cv is not None:
                c1.metric("ROC AUC (CV)", f"{np.mean(roc_auc_cv):.4f}")
            c2.metric("ROC AUC (test)", f"{res.get('roc_auc', float('nan')):.4f}")
            c3.metric("PR AUC (AP)", f"{res.get('pr_auc', float('nan')):.4f}")
            c4.metric("F2 Score", f"{res.get('f2_score', float('nan')):.4f}")

            # Classification report
            st.text("Classification report")
            st.code(res.get("classification_report", ""), language="text")

            # Confusion matrix
            cm = res.get("confusion_matrix")
            if cm is not None:
                fig_cm, ax = plt.subplots(figsize=(5.5, 4.5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Not Fraud","Fraud"],
                            yticklabels=["Not Fraud","Fraud"], ax=ax)
                ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                st.pyplot(fig_cm)

            # ROC & PR curves
            y_test, proba = res.get("y_test"), res.get("proba")
            if (y_test is not None) and (proba is not None):
                fig1, ax1 = plt.subplots()
                RocCurveDisplay.from_predictions(y_test, proba, ax=ax1)
                ax1.set_title("ROC Curve")
                st.pyplot(fig1)

                fig2, ax2 = plt.subplots()
                PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax2)
                ax2.set_title("Precision–Recall Curve")
                st.pyplot(fig2)

    left, right = st.columns(2)
    render_lr(left,  res_no, "Without SMOTE")
    render_lr(right, res_sm, "With SMOTE")
"""
# ---------- Random Forest (static + "Run again") ----------
with tabs[3]:
    st.subheader("Random Forest Results (No SMOTE vs SMOTE)")

    # cache-busting nonce
    if "rf_nonce" not in st.session_state:
        st.session_state.rf_nonce = 0

    @st.cache_resource(show_spinner=True)
    def compute_rf_runs(X, y, _nonce: int):
        # uses your wrapper from main.py
        res_no = RF_drive(X, y, smote_trigger=False)
        res_sm = RF_drive(X, y, smote_trigger=True)
        return res_no, res_sm

    # Button to re-train both RF variants
    if st.button("Run again", key="rf_rerun"):
        st.session_state.rf_nonce += 1

    # Compute (cached unless nonce changes)
    rf_no, rf_sm = compute_rf_runs(X, y, st.session_state.rf_nonce)

    def render_rf(col, res, title):
        with col:
            st.markdown(f"### {title}")

            c1, c2, c3, c4 = st.columns(4)
            if "roc_auc_cv" in res:
                c1.metric("ROC AUC (CV)", f"{np.mean(res['roc_auc_cv']):.4f}")
            c2.metric("ROC AUC (test)", f"{res.get('roc_auc', float('nan')):.4f}")
            c3.metric("PR AUC (AP)", f"{res.get('pr_auc', float('nan')):.4f}")
            c4.metric("F2 Score", f"{res.get('f2_score', float('nan')):.4f}")

            st.text("Classification report")
            st.code(res.get("classification_report", ""), language="text")

            # Confusion matrix
            cm = res.get("confusion_matrix")
            if cm is not None:
                fig_cm, ax = plt.subplots(figsize=(5.5, 4.5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Not Fraud","Fraud"],
                            yticklabels=["Not Fraud","Fraud"], ax=ax)
                ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                st.pyplot(fig_cm)

            # ROC & PR
            y_test, proba = res.get("y_test"), res.get("proba")
            if (y_test is not None) and (proba is not None):
                fig1, ax1 = plt.subplots()
                RocCurveDisplay.from_predictions(y_test, proba, ax=ax1)
                ax1.set_title("ROC Curve"); st.pyplot(fig1)

                fig2, ax2 = plt.subplots()
                PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax2)
                ax2.set_title("Precision–Recall Curve"); st.pyplot(fig2)

    left, right = st.columns(2)
    render_rf(left,  rf_no, "Without SMOTE")
    render_rf(right, rf_sm, "With SMOTE")
"""


# ---------- XGBoost (static + "Run again") ----------
with tabs[4]:
    st.subheader("XGBoost Results (No SMOTE vs SMOTE)")

    if "xgb_nonce" not in st.session_state:
        st.session_state.xgb_nonce = 0

    @st.cache_resource(show_spinner=True)
    def compute_xgb_runs(X, y, _nonce: int):
        from main import XGB_drive  # import here to avoid circulars at top-level
        res_no = XGB_drive(X, y, smote=False, threshold=0.50)
        res_sm = XGB_drive(X, y, smote=True,  threshold=0.50)
        return res_no, res_sm

    if st.button("Run again", key="xgb_rerun"):
        st.session_state.xgb_nonce += 1

    xgb_no, xgb_sm = compute_xgb_runs(X, y, st.session_state.xgb_nonce)

    def render_xgb(col, res, title):
        with col:
            st.markdown(f"### {title}")
            c1, c2, c3, c4 = st.columns(4)
            if "roc_auc_cv" in res: c1.metric("ROC AUC (CV)", f"{np.mean(res['roc_auc_cv']):.4f}")
            c2.metric("ROC AUC (test)", f"{res.get('roc_auc', float('nan')):.4f}")
            c3.metric("PR AUC (AP)", f"{res.get('pr_auc', float('nan')):.4f}")
            c4.metric("F2 Score", f"{res.get('f2_score', float('nan')):.4f}")

            st.text("Classification report")
            st.code(res.get("classification_report", ""), language="text")

            cm = res.get("confusion_matrix")
            if cm is not None:
                fig_cm, ax = plt.subplots(figsize=(5.5, 4.5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Not Fraud","Fraud"],
                            yticklabels=["Not Fraud","Fraud"], ax=ax)
                ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                st.pyplot(fig_cm)

            y_test, proba = res.get("y_test"), res.get("proba")
            if (y_test is not None) and (proba is not None):
                fig1, ax1 = plt.subplots(); RocCurveDisplay.from_predictions(y_test, proba, ax=ax1)
                ax1.set_title("ROC Curve"); st.pyplot(fig1)
                fig2, ax2 = plt.subplots(); PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax2)
                ax2.set_title("Precision–Recall Curve"); st.pyplot(fig2)

    left, right = st.columns(2)
    render_xgb(left,  xgb_no, "Without SMOTE")
    render_xgb(right, xgb_sm, "With SMOTE")

with tabs[5]:
    st.subheader("Findings")
    st.write("To be implemented...")
    st.write("Summary of findings, conclusions, next steps, etc.") 