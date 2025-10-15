from LoadNPrep import data_loader
from random_forest import random_forest, plot_roc_pr



def driver():
    X,y = data_loader()
    metrics = random_forest(X,y smote_trigger=False)
    proba = metrics['proba'] if 'proba' in metrics else None
    y_test = metrics['y_test'] if 'y_test' in metrics else None
    plot_roc_pr(y_test, proba)



if __name__ == "__main__":
    driver()