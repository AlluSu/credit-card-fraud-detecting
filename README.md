### Project Overview

This project focuses on **credit card fraud detection** using machine learning algorithms such as **Random Forest**, **Logistic Regression**, and **XGBoost**.

To address the issue of **class imbalance** in the dataset, we implemented **SMOTE (Synthetic Minority Oversampling Technique)**.  
SMOTE generates synthetic samples for the minority class, resulting in a balanced training dataset with an equal number of fraud and non-fraud cases.

The dataset was sourced from [Kaggle’s Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data).  
Some features were already normalized, we decided to normalize the remaining data to improve model performance and overall accuracy.


```bash
# Step 1: Clone the repository
git clone https://github.com/AlluSu/credit-card-fraud-detecting
cd credit-card-fraud-detecting

# Step 2: Create a virtual environment
python3 -m venv .venv
# OR
python -m venv .venv

# Step 3: Activate the virtual environment
# macOS / Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\Activate

# Step 4: Install dependencies
pip install -r requirements.txt
# OR
pip3 install -r requirements.txt

# Step 5: Run the Streamlit app
streamlit run app.py

