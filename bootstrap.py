
from __future__ import annotations  
import sys                          
import pathlib                      
import runpy                        
import numpy as np                  
import pandas as pd                 
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression   
from sklearn.metrics import accuracy_score, precision_score, recall_score  


def run_baseline(script_path: pathlib.Path):          
    runpy.run_path(str(script_path))                 


def load_data(csv_path: str):                         
    df = pd.read_csv(csv_path)                        
    X = df.loc[:, "V1":"V28"].values                
    y = df["Class"].values                            
    return X, y                                       


def bootstrap_metrics(X_train, y_train, X_test, y_test, iterations: int, seed: int = 42):
    rng = np.random.default_rng(seed)                 
    n_train = len(X_train)                            
    accs, precs, recalls = [], [], []                 
    for _ in range(iterations):                       
        idx = rng.integers(0, n_train, size=n_train)  
        print(idx)
        Xb = X_train[idx]                             
        yb = y_train[idx]                             
        if len(np.unique(yb)) < 2:                     
            continue
        model = LogisticRegression(solver="liblinear")  
        model.fit(Xb, yb)                             
        yp = model.predict(X_test)                    
        accs.append(accuracy_score(y_test, yp))       
        precs.append(precision_score(y_test, yp, zero_division=0))  
        recalls.append(recall_score(y_test, yp, zero_division=0))   
    return accs, precs, recalls                       


def summarize(name: str, values):                     
    arr = np.array(values)                            
    mean = arr.mean()                                 
    low = np.percentile(arr, 2.5)                     
    high = np.percentile(arr, 97.5)                   
    print(f"{name:9s} mean={mean:.6f}  95%CI=({low:.6f}, {high:.6f})  n={len(arr)}")  


def main():                                           
    iterations = 200                                  
    csv_path = "creditcard.csv"                      

    script_path = pathlib.Path(__file__).parent / "logistic-regression.py"  

    run_baseline(script_path)                       

    X, y = load_data(csv_path)                       
    X_train, X_test, y_train, y_test = train_test_split(  
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size={len(X_train)}, Test size={len(X_test)}, Fraud ratio train={y_train.mean():.6f}")  
    print(f"Running {iterations} bootstrap iterations...")  
    accs, precs, recalls = bootstrap_metrics(X_train, y_train, X_test, y_test, iterations)  

    if not accs:                                      
        print("No valid bootstrap samples (all degenerate). Try increasing iterations.")
        return

    print(" Bootstrap Summary ")  
    summarize("accuracy", accs)                        
    summarize("precision", precs)                      
    summarize("recall", recalls)                       


if __name__ == "__main__":                             
    main()                                             
