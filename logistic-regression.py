import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def logistic_regression():
    
    # This assumes that the file exists in the same folder as this script
    # The file is quite large
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

    # Create a feature vector from the columns V1, .., V28
    X = df.loc[:, "V1":"V28"]
    X

    X = df.loc[:, "V1":"V28"].values
    X

    # Set the "Class" column as the target variable
    y = df["Class"].values
    y

    # Split into training and testing sets, use 80/20 ratio
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2)

    # Fit logistic regression model
    # Use liblinear as the solver for now, as it is compatible with a binary classification problem
    # liblinear also supports both L1 and L2 regularization
    # Later on the project we can try with other different parameters
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#logisticregression

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

logistic_regression()