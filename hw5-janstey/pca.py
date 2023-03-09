import argparse
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def normalize(xTrain: pd.DataFrame, xTest: pd.DataFrame) -> tuple():
    """Take the training and test data and normalize it using a sklearn standard scaler

    Args:
        xTrain (pd.DataFrame): The xTrain data
        xTest (pd.DataFrame): The xTest data

    Returns:
        xTrN, xTeN: The normalized datasets
    """

    # make our scaler using the training data
    scaler = sk.preprocessing.StandardScaler().fit(xTrain)

    # return the resulting dataframes through transformation
    return pd.DataFrame(scaler.transform(xTrain), columns=xTrain.columns), pd.DataFrame(scaler.transform(xTest), columns=xTest.columns)

def logisticReg(xTrain: pd.DataFrame, yTrain: pd.DataFrame, xTest: pd.DataFrame) -> np.ndarray:
    """Fit a logistic regression given trainig dataframes and predict the probabilities
    of each label using the test dataset

    Args:
        xTrain (pd.DataFrame): The attribute training dataset
        yTrain (pd.DataFrame): The corresponding training labels 
        xTest (pd.DataFrame): The test dataset

    Returns:
        np.ndarray: The probabilities for a given test row to be a given label. Returns in the format
        [prob of lable 0, prob of label 1]
    """

    return LogisticRegression().fit(xTrain, yTrain['label']).predict_proba(xTest)

def pca(xTrain: pd.DataFrame, xTest: pd.DataFrame) -> tuple():
    """Find the number of components required to have the dataset reach an explained variance ratio of >0.95
    and report that to the user. Then find the weights of the first principal components.
    Lastly, use that number of components value to do PCA transformations on xTrain and xTest.

    Args:
        xTrain (pd.DataFrame): The training dataframe
        xTest (pd.DataFrame): The test dataframe

    Returns:
        pca.transform(xTrain), pca.transform(xTest): The PCA transformation of each dataframe
    """

    componentNum = 1
    while (True):  # keep going until we find a defined optimal result
        pca = PCA(n_components=componentNum)
        pca.fit(xTrain)
        if (pca.explained_variance_ratio_.sum() > 0.95):
            print("Number of components needed for 0.95 or greater variance:", componentNum)
            print("Using the first 3 principal components, these are the average weights for each attribute:")
            print(pd.DataFrame(pca.components_[:3], columns=xTrain.columns).mean().sort_values(ascending=False))
            # TODO make the cutoff for important attributes anything above 0.1
            break  # once we find the appropriate number of components, we break
        componentNum+=1
    
    # return the transformed dataframes
    return pca.transform(xTrain), pca.transform(xTest)


def main():
    """
    Main file to run from the command line.
    """

    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain", default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("yTrain", default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest", default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("yTest", default="q4yTest.csv",
                        help="filename for labels associated with the test data")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()

    # load the data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)

    # get the dataframes through normalization and then through PCA
    xTrN, xTeN = normalize(xTrain, xTest)
    xTrP, xTeP = pca(xTrN, xTeN)

    # find the probabilities post normalization and then post PCA
    probNormalized = logisticReg(xTrN, yTrain, xTeN)
    probPCA = logisticReg(xTrP, yTrain, xTeP)

    # Make the ROC curves
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    # use only the probabilities that the label is 0 (needs a 1D array)
    fprN, tprN, thresholds = sk.metrics.roc_curve(y_true=yTest, y_score=probNormalized[:,1])
    fprPCA, tprPCA, thresholds = sk.metrics.roc_curve(y_true=yTest, y_score=probPCA[:,1])

    # Plot the ROC curves!
    plt.plot(fprN, tprN, label="Normalized")
    plt.plot(fprPCA, tprPCA, label="Principal Component Analysis")
    plt.legend(title="Data Transformation Process")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

if __name__ == "__main__":
    main()