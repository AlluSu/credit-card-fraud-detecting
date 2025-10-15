from LoadNPrep import data_loader
from random_forest import random_forest, plot_roc_pr
from logistic_regression import logistic_regression

def RF_drive(X,y, smote_trigger):
    metrics = random_forest(X,y, smote_trigger=False)
    proba = metrics['proba'] if 'proba' in metrics else None
    y_test = metrics['y_test'] if 'y_test' in metrics else None
    plot_roc_pr(y_test, proba)



def driver():

    X,y = data_loader()
    logistic_regression(X, y, smote=False)

    # For none smote
    RF_drive(X,y, False) 

    # For smote
    # RF_drive(X,y, True)



if __name__ == "__main__":
    driver()