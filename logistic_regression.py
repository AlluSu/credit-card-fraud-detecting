import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def logistic_regression(X, y, smote=False):

    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.05)

    model = LogisticRegression(solver="liblinear")
    regression = model.fit(X_train, y_train)

    # Predict
    predicted = model.predict(X_test)

    # Print the classifcation report, confusion matrix, and other metrics
    print("Accuracy from the logistic regression: \n", accuracy_score(y_test, predicted))

    print("\n")

    matrix = confusion_matrix(y_test, predicted)
    print("Confusion matrix: \n", matrix)

    # Do more prettier visualization from the results
    classes = ["Not Fraud", "Fraud"]

    conf_mat = confusion_matrix(y_test, predicted)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

    print('Classification report:\n', classification_report(y_test, predicted))

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_test, predicted).ravel()

    precision = true_positive / (true_positive + false_positive)
    print("Precision: ", precision)

    recall = true_positive / (true_positive + false_negative)
    print("Recall: ", recall)