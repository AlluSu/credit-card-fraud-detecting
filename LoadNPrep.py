import kagglehub
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, RobustScaler

def data_loader():
    # This function loads in the data from our source "Kaggle"
    # It scales the left out features, Time and Amount to better match the rest of the data
    # and returns our target variable y and dependant variables X as a pandas df

    # Download latest version and load it as a pandas.DataFrame
    print("Loading data from kaggle...")
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_file = os.path.join(path, "creditcard.csv")
    df = pd.read_csv(csv_file)
    print("Data initialized into a df....")

    # Create scalers
    scaler_time = StandardScaler()
    scaler_amount = RobustScaler()

    # Apply scaling
    df['Time_scaled'] = scaler_time.fit_transform(df[['Time']])
    df['Amount_scaled'] = scaler_amount.fit_transform(df[['Amount']])

    
    y = df['Class']
    X = df.drop(['Class','Time', 'Amount'], axis=1)
    print("Data preparation done!")
    return X, y