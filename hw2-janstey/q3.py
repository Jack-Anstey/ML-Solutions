import argparse
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from typing import Tuple

def reduceDataset(xFeat: pd.DataFrame, yFeat: pd.DataFrame, percentage: float) -> (Tuple[pd.DataFrame, pd.DataFrame]):
    """Take a given xFeat and yFeat dataframes and reduces them by a given percentage

    Args:
        xFeat (pd.DataFrame): the original dataset
        yFeat (pd.DataFrame): the original labels
        percentage (float): the percentage you want to reduce the dataset by

    Returns:
        A tuple of
        xFeatLabelsReduced: the reduced dataset
        yFeatReduced: the reduced labels
    """

    xFeatLabels = xFeat.copy()
    xFeatLabels['label'] = yFeat
    xFeatLabelsReduced = xFeatLabels.sample(frac=(1-percentage))
    yFeatReduced = xFeatLabelsReduced['label'].to_frame()
    xFeatLabelsReduced.drop(columns=['label'], inplace=True)
    return xFeatLabelsReduced, yFeatReduced

def getAUCandACC(model, xTest: pd.DataFrame, yTest: pd.DataFrame) -> (Tuple[float, float]):
    """Get the AUC and accuracy of your model with a given test dataset

    Args:
        model (sklearn model): The model you want to evalulate
        xTest (pd.DataFrame): _description_
        yTest (pd.DataFrame): _description_

    Returns:
        A tuple of
        auc: the AUC of the model
        acc: the accuracy of the model
    """

    auc = metrics.roc_auc_score(yTest, model.predict_proba(xTest)[:,1])
    acc = metrics.accuracy_score(yTest['label'], model.predict(xTest))
    return auc, acc

def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)

    # Part a
    knnOpt = GridSearchCV(KNeighborsClassifier(),  
        [{'n_neighbors': range(1,10,2), 'metric': ['euclidean','manhattan']}], 
        cv=5, scoring='accuracy')

    knnOpt.fit(xTrain, yTrain.values.ravel())
    knnK = knnOpt.best_params_['n_neighbors'] 
    knnMetric = knnOpt.best_params_['metric']
    
    dtOpt = GridSearchCV(DecisionTreeClassifier(),
       [{'max_depth': range(1, 10), 'min_samples_leaf': range(50, 300)}], cv=5, scoring='accuracy')

    dtOpt.fit(xTrain, yTrain.values.ravel())
    maxDepth = dtOpt.best_params_['max_depth']
    mls = dtOpt.best_params_['min_samples_leaf']

    # print out optimal values
    print("KNN optimal k:", knnK)
    print("KNN optimal metric:", knnMetric)
    print("Decision Tree optimal max depth:", maxDepth)
    print("Decision Tree optimal minimum leaf sample:", mls)
    print()

    # Part b
    xFeat95, yFeat95 = reduceDataset(xTrain, yTrain, 0.05)
    xFeat90, yFeat90 = reduceDataset(xTrain, yTrain, 0.1)
    xFeat80, yFeat80 = reduceDataset(xTrain, yTrain, 0.2)
    
    knn = KNeighborsClassifier(n_neighbors=knnK, metric=knnMetric)  # make a classifer using the optimal hyperparameters

    knn.fit(xTrain, yTrain.values.ravel())
    knnAuc1, knnAcc1 = getAUCandACC(knn, xTest, yTest)
    
    knn.fit(xFeat95, yFeat95.values.ravel())
    knnAuc2, knnAcc2 = getAUCandACC(knn, xTest, yTest)
    
    knn.fit(xFeat90, yFeat90.values.ravel())
    knnAuc3, knnAcc3 = getAUCandACC(knn, xTest, yTest)
    
    knn.fit(xFeat80, yFeat80.values.ravel())
    knnAuc4, knnAcc4 = getAUCandACC(knn, xTest, yTest)

    # Part c
    dt = DecisionTreeClassifier(max_depth=maxDepth, min_samples_leaf=mls)  # make a classifer using the optimal hyperparameters

    dt.fit(xTrain, yTrain.values.ravel())
    dtAuc1, dtAcc1 = getAUCandACC(dt, xTest, yTest)
    
    dt.fit(xFeat95, yFeat95.values.ravel())
    dtAuc2, dtAcc2 = getAUCandACC(dt, xTest, yTest)
    
    dt.fit(xFeat90, yFeat90.values.ravel())
    dtAuc3, dtAcc3 = getAUCandACC(dt, xTest, yTest)
    
    dt.fit(xFeat80, yFeat80.values.ravel())
    dtAuc4, dtAcc4 = getAUCandACC(dt, xTest, yTest)

    # Part D
    # print out the performance results!
    perfDF = pd.DataFrame([['Full Training', knnAuc1, knnAcc1, dtAuc1, dtAcc1],
                           ['95% Training', knnAuc2, knnAcc2, dtAuc2, dtAcc2],
                           ['90% Training', knnAuc3, knnAcc3, dtAuc3, dtAcc3],
                           ['80% Training', knnAuc4, knnAcc4, dtAuc4, dtAcc4],],
                           columns=['Training Set Portion', 'KNN AUC', 'KNN Accuracy', 'DT AUC', 'DT Accuracy'])
    print(perfDF)
    
if __name__ == "__main__":
    main()