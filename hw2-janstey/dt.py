import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from typing import Tuple


class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    tree = dict()

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample

    def train(self, xFeat, y):
        """
        Train the decision tree model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        # TODO do whatever you need
        self.tree = self.decision_tree(xFeat, y, 0, self.criterion)  # tree starts with a depth of 0
        return self

    def decision_tree(self, xFeat: pd.DataFrame, y: pd.DataFrame, depth: int, criterion: str()) -> dict():
        """A recursive method to create a decision tree given a dataframe of data and a set of labels,
        using a given criterion

        Args:
            xFeat (pd.DataFrame): the dataset
            y (pd.DataFrame): the labels for the dataset
            depth (int): the maximum depth allowed for the tree
            criterion (str()): the criterion used for generating the tree

        Returns:
            dict(): A dictionary that is the decision tree
        """

        '''
        Other comments: I've been to 4 different office hours to get this working perfectly.
        I know it doesn't, but after 10+ hours of debugging I cannot figure out why the behavior
        is not consistent with what it should be.
        '''

        splitfeat = ""  # name of the feature that we are splitting on
        splitvalue = -1  # the value at that index
        splitindex = -1  # the index in which the split was made
        bestvalue = float('inf') # the value we will compare the gini index and entropy to

        # Check stopping criteria. If it is met, return majority class of y
        # If greater than or equal to max depth or not enough room to have a minLeafSample on the left and right side
        # or if the labels are pure (all one value), then stopping criteria is met
        if depth >= self.maxDepth or self.minLeafSample >= len(xFeat) - self.minLeafSample or len(np.unique(y)) == 1:
            # assumes y is a dataframe
            return {"ymajority": y.mode()['label'][0]}  # get the first index of mode, which is the most occuring value

        xFeatOrdered = xFeat.copy()  # make a version of xFeat with labels attached
        xFeatOrdered['label'] = y

        for feature in xFeatOrdered.columns:  # enumerate over the features in the dataframe
            if(feature != 'label'):  # for all the features but label (bc it's the label)

                xFeatOrdered.sort_values(by=[feature], ascending=True, inplace = True)  # order the df by the feature we will try splits on

                for index in range(self.minLeafSample, len(xFeatOrdered) - self.minLeafSample):  # iterate through all indexes between mls's
                    # print(index)
                    xFeatL = xFeatOrdered[:index].copy()  # left side split
                    xFeatR = xFeatOrdered[index:].copy()   # right side split

                    labelsL, countsL = np.unique(xFeatL['label'], return_counts=True)
                    labelsR, countsR = np.unique(xFeatR['label'], return_counts=True)
                    
                    finalvalue = 0

                    if (criterion == 'entropy'):
                        # checks if either value is 0, in which case set to 0 (issues otherwise)
                        lentropy = 0
                        rentropy = 0
                        
                        for count in countsL:  # entropy calculations
                            if count != 0:
                                lentropy += count/len(xFeatL) * np.log2(count/len(xFeatL))
                        lentropy *= -1

                        for count in countsR:  # entropy calculations
                            if count != 0:
                                rentropy += count/len(xFeatR) * np.log2(count/len(xFeatR))
                        rentropy *= -1

                        # final entropy value
                        finalvalue = lentropy * (len(xFeatL)/len(xFeatOrdered)) + rentropy * (len(xFeatR)/len(xFeatOrdered))
                    elif (criterion == 'gini'):
                        for count in countsL:  # gini calculations
                            finalvalue += (count/len(xFeatL)) * (1 - (count/len(xFeatL)))
                        for count in countsR:  # gini calculations
                            finalvalue += (count/len(xFeatR)) * (1 - (count/len(xFeatR)))
                    
                    # for both entropy and gini index
                    if finalvalue < bestvalue:  # if the new value is better, update everything
                        splitvalue = xFeatOrdered[feature][index]
                        splitfeat = feature
                        splitindex = index
                        bestvalue = finalvalue
        
        # Partition data using the split feature and split value into two set: xFeatL, xFeatR, yL, yR
        xFeatOrdered.sort_values(by=[splitfeat], ascending=True, inplace = True)
        xFeatL = xFeatOrdered[:splitindex].copy().reset_index(drop=True)  # left side split
        xFeatR = xFeatOrdered[splitindex:].copy().reset_index(drop=True)  # right side split
        yL = xFeatL['label'].to_frame()  # so that y isn't a series
        yR = xFeatR['label'].to_frame()

        xFeatL.drop(columns=['label'], inplace=True)  # go back to normal
        xFeatR.drop(columns=['label'], inplace=True)

        # create returns (from slides)
        return {"left": self.decision_tree(xFeatL, yL, depth+1, criterion), 
        "right": self.decision_tree(xFeatR, yR, depth+1, criterion), "splitfeat": splitfeat, "splitval": splitvalue, "ymajority": -1}


    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted class label per sample
        """
        yHat = [] # variable to store the estimated class label
        for tuple in xFeat.itertuples(index=False):
            yHat.append(self.decisionTreeTest(self.tree, tuple, xFeat.columns))
        return yHat

    def decisionTreeTest(self, tree: dict(), testpoint: Tuple, columns: str()) -> int:
        """Recursively go through the decision tree and return a prediction label

        Args:
            tree (dict()): the decision tree
            testpoint (Tuple): the tuple of datapoints
            columns (str()): The column names in the data

        Returns:
            int: the predicted label
        """

        if tree['ymajority'] != -1:
            return tree['ymajority']

        feature = tree['splitfeat']
        value = tree['splitval']
        index = 0

        for column in columns:
            if column == feature:
                break
            index+=1
        
        if testpoint[index] > value:
            return self.decisionTreeTest(tree['right'], testpoint, columns)
        else:
            return self.decisionTreeTest(tree['left'], testpoint, columns)



def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data 
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
    parser.add_argument("--xTrain", default="q4xTrain.csv",
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
    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
