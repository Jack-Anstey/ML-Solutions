import argparse
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import time

 
def holdout(model, xFeat, y, testSize):
    """
    Split xFeat into random train and test based on the testSize and
    return the model performance on the training and test set. 

    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : nd-array with shape n x d
        Features of the dataset 
    y : 1-array with shape n x 1
        Labels of the dataset
    testSize : float
        Portion of the dataset to serve as a holdout. 

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    # AUC is area under rock curve -- slide 16
    trainAuc = 0
    testAuc = 0
    timeElapsed = 0
    # TODO fill int
    initialTime = time.time()

    # split everything up (office hours on 9/23/2022)
    xFeatTrain, xFeatTest, yTrain, yTest = sklearn.model_selection.train_test_split(xFeat, y, test_size = testSize)
    # fit the model to the training dataset 
    model.fit(xFeatTrain, yTrain.values.ravel())  # the .values.ravel() was to get rid of a warning when using KNN

    # suggested during office hours and the sktree_train_test in this file 9/23/2022
    trainAuc = metrics.roc_auc_score(yTrain, model.predict_proba(xFeatTrain)[:,1])
    testAuc = metrics.roc_auc_score(yTest, model.predict_proba(xFeatTest)[:,1])

    timeElapsed = time.time() - initialTime  # method end
    return trainAuc, testAuc, timeElapsed


def kfold_cv(model, xFeat, y, k):
    """
    Split xFeat into k different groups, and then use each of the
    k-folds as a validation set, with the model fitting on the remaining
    k-1 folds. Return the model performance on the training and
    validation (test) set. 


    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : nd-array with shape n x d
        Features of the dataset 
    y : 1-array with shape n x 1
        Labels of the dataset
    k : int
        Number of folds or groups (approximately equal size)

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    trainAuc = 0
    testAuc = 0
    timeElapsed = 0
    # TODO FILL IN
    initialTime = time.time()
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    kf = sklearn.model_selection.KFold(n_splits=k)

    # for the indices in each split
    for trainSlice, testSlice in kf.split(xFeat):
        # get the slice and turn it from a nparray back to a dataframe
        xFeatTrain = pd.DataFrame(data = xFeat.to_numpy()[trainSlice].copy(), columns=xFeat.columns)
        xFeatTest = pd.DataFrame(data = xFeat.to_numpy()[testSlice].copy(), columns=xFeat.columns)
        yTrain = pd.DataFrame(data = y.to_numpy()[trainSlice].copy(), columns=y.columns)
        yTest = pd.DataFrame(data = y.to_numpy()[testSlice].copy(), columns=y.columns)

        # model!
        model.fit(xFeatTrain, yTrain.values.ravel())  # fit the model to the training dataset

        # update auc values
        trainAuc += metrics.roc_auc_score(yTrain, model.predict_proba(xFeatTrain)[:,1])
        testAuc += metrics.roc_auc_score(yTest, model.predict_proba(xFeatTest)[:,1])

    trainAuc /= k  # we divide by k to get the average
    testAuc /= k
    timeElapsed = time.time() - initialTime  # method end
    return trainAuc, testAuc, timeElapsed


def mc_cv(model, xFeat, y, testSize, s):
    """
    Evaluate the model using s samples from the
    Monte Carlo cross validation approach where
    for each sample you split xFeat into
    random train and test based on the testSize.
    Returns the model performance on the training and
    test datasets.

    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : nd-array with shape n x d
        Features of the dataset 
    y : 1-array with shape n x 1
        Labels of the dataset
    testSize : float
        Portion of the dataset to serve as a holdout. 

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    trainAuc = 0
    testAuc = 0
    timeElapsed = 0
    # TODO FILL IN
    initialTime = time.time()
    splitsize = int(len(xFeat) * testSize)
    partitions = list()
    
    xFeatTest = pd.DataFrame()
    yTest = pd.DataFrame()
    xFeatTrain = pd.DataFrame()
    yTrain = pd.DataFrame()

    # I couldn't find a way to do the splits using sklearn so I built my own
    while (len(partitions) < s):
        splitstart = np.random.randint(len(xFeat)-splitsize)
        splitend = splitstart+splitsize

        for start, end in partitions:
            if splitstart >= start and splitstart <= end:
                continue

        partitions.append((splitstart, splitend))

    lastEnd = 0
    for start, end in partitions:
        xFeatTest = pd.concat([xFeatTest, xFeat[start:end].copy().reset_index(drop=True)])
        yTest = pd.concat([yTest, y[start:end].copy().reset_index(drop=True)])
        xFeatTrain = pd.concat([xFeatTrain, xFeat[lastEnd:start].copy().reset_index(drop=True)])
        yTrain = pd.concat([yTrain, y[lastEnd:start].copy().reset_index(drop=True)])
        lastEnd = end

    model.fit(xFeatTrain, yTrain.values.ravel())  # fit the model to the training dataset

    trainAuc = metrics.roc_auc_score(yTrain, model.predict_proba(xFeatTrain)[:,1])
    testAuc = metrics.roc_auc_score(yTest, model.predict_proba(xFeatTest)[:,1])

    timeElapsed = time.time() - initialTime  # method end
    return trainAuc, testAuc, timeElapsed


def sktree_train_test(model, xTrain, yTrain, xTest, yTest):
    """
    Given a sklearn tree model, train the model using
    the training dataset, and evaluate the model on the
    test dataset.

    Parameters
    ----------
    model : DecisionTreeClassifier object
        An instance of the decision tree classifier 
    xTrain : nd-array with shape nxd
        Training data
    yTrain : 1d array with shape n
        Array of labels associated with training data
    xTest : nd-array with shape mxd
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    trainAUC : float
        The AUC of the model evaluated on the training data.
    testAuc : float
        The AUC of the model evaluated on the test data.
    """
    # fit the data to the training dataset
    model.fit(xTrain, yTrain.values.ravel())
    # predict training and testing probabilties
    yHatTrain = model.predict_proba(xTrain)
    yHatTest = model.predict_proba(xTest)
    # calculate auc for training
    fpr, tpr, thresholds = metrics.roc_curve(yTrain['label'],
                                             yHatTrain[:, 1])
    trainAuc = metrics.auc(fpr, tpr)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(yTest['label'],
                                             yHatTest[:, 1])
    testAuc = metrics.auc(fpr, tpr)
    return trainAuc, testAuc


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
    # create the decision tree classifier
    dtClass = DecisionTreeClassifier(max_depth=15,
                                     min_samples_leaf=10)
    # use the holdout set with a validation size of 30 of training
    aucTrain1, aucVal1, time1 = holdout(dtClass, xTrain, yTrain, 0.30)  # fixed as asked on piazza
    # use 2-fold validation
    aucTrain2, aucVal2, time2 = kfold_cv(dtClass, xTrain, yTrain, 2)
    # use 5-fold validation
    aucTrain3, aucVal3, time3 = kfold_cv(dtClass, xTrain, yTrain, 5)
    # use 10-fold validation
    aucTrain4, aucVal4, time4 = kfold_cv(dtClass, xTrain, yTrain, 10)
    # use MCCV with 5 samples
    aucTrain5, aucVal5, time5 = mc_cv(dtClass, xTrain, yTrain, 0.30, 5)  # fixed as asked on piazza
    # use MCCV with 10 samples
    aucTrain6, aucVal6, time6 = mc_cv(dtClass, xTrain, yTrain, 0.30, 10)  # fixed as asked on piazza
    # train it using all the data and assess the true value
    trainAuc, testAuc = sktree_train_test(dtClass, xTrain, yTrain, xTest, yTest)
    perfDF = pd.DataFrame([['Holdout', aucTrain1, aucVal1, time1],
                           ['2-fold', aucTrain2, aucVal2, time2],
                           ['5-fold', aucTrain3, aucVal3, time3],
                           ['10-fold', aucTrain4, aucVal4, time4],
                           ['MCCV w/ 5', aucTrain5, aucVal5, time5],
                           ['MCCV w/ 10', aucTrain6, aucVal6, time6],
                           ['True Test', trainAuc, testAuc, 0]],
                           columns=['Strategy', 'TrainAUC', 'ValAUC', 'Time'])
    print(perfDF)


if __name__ == "__main__":
    main()
