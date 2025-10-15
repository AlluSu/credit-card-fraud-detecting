import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Credit Card Fraud", page_icon="", layout="wide")
st.title("Credit Card Fraud — Lightweight Streamlit App")

# ---------- Data load ----------
DEFAULT_CSV = Path("creditcard.csv")

@st.cache_data(show_spinner=False)
def load_csv(path: Path):
    return pd.read_csv(path)

if not DEFAULT_CSV.exists():
    st.error("Missing data file: creditcard.csv (place it next to app.py).")
    st.stop()

df = load_csv(DEFAULT_CSV)
st.info(f"Loaded dataset: {DEFAULT_CSV.resolve()}")




# ------------ Tabs ------------
tabs = st.tabs(["Introduction", " Explore", " Logistic Regression", " XGBoost", " Bootstrap CI"])

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
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"Rows: {len(df):,} | Columns: {df.shape[1]}")

    st.subheader("Class Balance")
    counts = df["Class"].value_counts().rename({0: "Not Fraud", 1: "Fraud"})
    st.bar_chart(counts)
