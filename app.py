import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from LoadNPrep import data_loader
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# NEW: SHAP (graceful import)
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

from main import LR_drive, RF_drive, build_explore_preview

st.set_page_config(page_title="Credit Card Fraud", page_icon="", layout="wide")
st.title("Credit Card Fraud — Lightweight Streamlit App")

# ---------- Data load ----------
@st.cache_resource(show_spinner=True)
def load_data():
    X, y = data_loader()
    return X, y

X, y = load_data()

# ------------ Tabs ------------
tabs = st.tabs([
    "Introduction",
    " Explore",
    " Logistic Regression",
    "Random Forest",
    " XGBoost",
    " Overview",
    " Findings"
])

# ---------- Intro ----------
with tabs[0]:
    st.subheader("Overview")
    st.write("This mini app explores credit card fraud detection using a highly imbalanced dataset.")
    st.subheader("Research Problem")
    st.markdown(
        """
**Goal.** Build a performant, interpretable classifier for rare fraud events.

**Core questions**
1. Which models balance recall and precision best for rare fraud?
2. How much does SMOTE help?
3. What thresholds/metrics are most informative for the business?

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

# ----------  render helpers ----------
def render_confusion_matrix(cm, title="Confusion Matrix"):
    fig_cm, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Fraud","Fraud"],
                yticklabels=["Not Fraud","Fraud"], ax=ax)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    st.pyplot(fig_cm)

def render_curves(y_test, proba):
    fig1, ax1 = plt.subplots(); RocCurveDisplay.from_predictions(y_test, proba, ax=ax1)
    ax1.set_title("ROC Curve"); st.pyplot(fig1)
    fig2, ax2 = plt.subplots(); PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax2)
    ax2.set_title("Precision–Recall Curve"); st.pyplot(fig2)

def render_block(res):
    c1, c2, c3, c4 = st.columns(4)
    roc_auc_cv = res.get("roc_auc_cv")
    if roc_auc_cv is not None:
        c1.metric("ROC AUC (CV)", f"{np.mean(roc_auc_cv):.4f}")
    c2.metric("ROC AUC (test)", f"{res.get('roc_auc', float('nan')):.4f}")
    c3.metric("PR AUC (AP)", f"{res.get('pr_auc', float('nan')):.4f}")
    c4.metric("F2 Score", f"{res.get('f2_score', float('nan')):.4f}")

    st.text("Classification report")
    st.code(res.get("classification_report", ""), language="text")

    cm = res.get("confusion_matrix")
    if cm is not None:
        render_confusion_matrix(cm)

    y_test, proba = res.get("y_test"), res.get("proba")
    if (y_test is not None) and (proba is not None):
        render_curves(y_test, proba)

# ---------- Logistic Regression ----------
with tabs[2]:
    st.subheader("Logistic Regression Results (No SMOTE vs SMOTE)")

    @st.cache_resource(show_spinner=True)
    def compute_lr_runs(_key, X, y, threshold=0.50):
        res_no = LR_drive(X, y, smote_trigger=False, threshold=threshold)
        res_sm = LR_drive(X, y, smote_trigger=True,  threshold=threshold)
        return res_no, res_sm

    if "lr_results" not in st.session_state:
        st.session_state.lr_results = None

    thr_col, btn_col = st.columns([3,1])
    with btn_col:
        if st.button("Run / Re-run", key="lr_run_btn"):
            nonce = st.session_state.get("lr_nonce", 0) + 1
            st.session_state.lr_nonce = nonce
            st.session_state.lr_results = compute_lr_runs(nonce, X, y)

    if st.session_state.lr_results is None:
        st.info("Press **Run / Re-run** to train and display Logistic Regression results.")
    else:
        res_no, res_sm = st.session_state.lr_results
        left, right = st.columns(2)
        with left:
            st.markdown("### Without SMOTE")
            render_block(res_no)

        with right:
            st.markdown("### With SMOTE")
            render_block(res_sm)

# ---------- Random Forest ----------
with tabs[3]:
    st.subheader("Random Forest Results (No SMOTE vs SMOTE)")

    @st.cache_resource(show_spinner=True)
    def compute_rf_runs(_key, X, y):
        res_no = RF_drive(X, y, smote_trigger=False)
        res_sm = RF_drive(X, y, smote_trigger=True)
        return res_no, res_sm

    if "rf_results" not in st.session_state:
        st.session_state.rf_results = None

    if st.button("Run / Re-run", key="rf_run_btn"):
        nonce = st.session_state.get("rf_nonce", 0) + 1
        st.session_state.rf_nonce = nonce
        st.session_state.rf_results = compute_rf_runs(nonce, X, y)

    if st.session_state.rf_results is None:
        st.info("Press **Run / Re-run** to train and display Random Forest results.")
    else:
        rf_no, rf_sm = st.session_state.rf_results
        left, right = st.columns(2)
        with left:
            st.markdown("### Without SMOTE")
            render_block(rf_no)

        with right:
            st.markdown("### With SMOTE")
            render_block(rf_sm)


# ---------- XGBoost ----------
with tabs[4]:
    st.subheader("XGBoost Results (No SMOTE vs SMOTE)")

    @st.cache_resource(show_spinner=True)
    def compute_xgb_runs(_key, X, y, threshold=0.50):
        from main import XGB_drive  # local import to avoid circulars
        res_no = XGB_drive(X, y, smote=False, threshold=threshold)
        res_sm = XGB_drive(X, y, smote=True,  threshold=threshold)
        return res_no, res_sm

    if "xgb_results" not in st.session_state:
        st.session_state.xgb_results = None

    if st.button("Run / Re-run", key="xgb_run_btn"):
        nonce = st.session_state.get("xgb_nonce", 0) + 1
        st.session_state.xgb_nonce = nonce
        st.session_state.xgb_results = compute_xgb_runs(nonce, X, y)

    if st.session_state.xgb_results is None:
        st.info("Press **Run / Re-run** to train and display XGBoost results.")
    else:
        xgb_no, xgb_sm = st.session_state.xgb_results
        left, right = st.columns(2)
        with left:
            st.markdown("### Without SMOTE")
            render_block(xgb_no)

        with right:
            st.markdown("### With SMOTE")
            render_block(xgb_sm)


# ---------- Overview (All Models Together) ----------
with tabs[5]:
    st.subheader("All Models — Aggregated Metrics")

    rows = []
    def add_row(res, model_name, variant):
        if not res:
            return
        rows.append({
            "Model": model_name,
            "Variant": variant,
            "ROC AUC (test)": res.get("roc_auc", np.nan),
            "PR AUC (AP)": res.get("pr_auc", np.nan),
            "F2 Score": res.get("f2_score", np.nan)
        })

    # Gather from session state if available
    lr = st.session_state.get("lr_results")
    rf = st.session_state.get("rf_results")
    xgb = st.session_state.get("xgb_results")

    if lr:
        add_row(lr[0], "LogReg", "No SMOTE")
        add_row(lr[1], "LogReg", "SMOTE")
    if rf:
        add_row(rf[0], "RandomForest", "No SMOTE")
        add_row(rf[1], "RandomForest", "SMOTE")
    if xgb:
        add_row(xgb[0], "XGBoost", "No SMOTE")
        add_row(xgb[1], "XGBoost", "SMOTE")

    if not rows:
        st.info("Run at least one model to populate the overview.")
    else:
        df = pd.DataFrame(rows)
        st.dataframe(df, width='stretch')

        # Quick bar charts for each metric
        metric_cols = ["ROC AUC (test)", "PR AUC (AP)", "F2 Score"]
        for m in metric_cols:
            st.markdown(f"**{m}**")
            plot_df = df[["Model", "Variant", m]].copy()
            plot_df["Label"] = plot_df["Model"] + " — " + plot_df["Variant"]
            plot_df = plot_df.set_index("Label")[[m]].sort_values(m, ascending=False)
            st.bar_chart(plot_df)

        # Optional: overlay ROC curves (if multiple y/proba available)
        with st.expander("Overlayed ROC curves (if available)"):
            # Collect (y_test, proba, label)
            curves = []
            def add_curve(res, label):
                y_test, proba = res.get("y_test"), res.get("proba")
                if (y_test is not None) and (proba is not None):
                    curves.append((y_test, proba, label))

            if lr:
                add_curve(lr[0], "LogReg — No SMOTE")
                add_curve(lr[1], "LogReg — SMOTE")
            if rf:
                add_curve(rf[0], "RF — No SMOTE")
                add_curve(rf[1], "RF — SMOTE")
            if xgb:
                add_curve(xgb[0], "XGB — No SMOTE")
                add_curve(xgb[1], "XGB — SMOTE")

            if curves:
                fig, ax = plt.subplots()
                for y_true, y_prob, label in curves:
                    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name=label)
                ax.set_title("ROC Curves — Overlay")
                st.pyplot(fig)
            else:
                st.caption("Run models to see overlayed ROC curves.")

# ---------- Findings ----------
with tabs[6]:
    st.subheader("Findings")
    st.write("To be implemented...")
    st.write("Summary of findings, conclusions, next steps, etc.")
