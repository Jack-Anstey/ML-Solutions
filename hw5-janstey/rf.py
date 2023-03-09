import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import accuracy_score as acc

class RandomForest(object):
    nest = 0           # number of trees
    maxFeat = 0        # maximum number of features
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    forest = None # the eventual forest filled during training and used during prediction

    def __init__(self, nest, maxFeat, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        nest: int
            Number of trees to have in the forest
        maxFeat: int
            Maximum number of features to consider in each tree
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.nest = nest
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.maxFeat = maxFeat  # add this since it was missing from original spec
        self.forest = []

    def train(self, xFeat, y):
        """
        Train the random forest using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the number of trees and
            the values are the out of bag errors
        """

        # Thank you Matt for all the help! Office hours 11/11/2022
        # make a list of DT's as defined by nest
        # divide the features between each DT without replacement
        unusedIndexes = []  # a list of the unused indexes when sampling with replacement. Ends up being a list of lists
        oob = {}  # where we will store the oob results
        for tree in range(self.nest):
            # we get a random sample of the indexes from xFeat w/ replacement
            # taking xFeat.shape[0] aka num rows # of samples
            indexes = np.random.choice(xFeat.shape[0], xFeat.shape[0], replace=True)

            # find the differences and store them
            unusedIndexes.append(np.setdiff1d(np.arange(xFeat.shape[0]), indexes))

            # append the tree to our forest list
            self.forest.append(DT(criterion=self.criterion, 
            max_depth=self.maxDepth, 
            min_samples_leaf=self.minLeafSample,
            max_features=self.maxFeat))

            # fit on the DT classifer we just made
            # the, : is so that we use the indexes we just defined as the data for fitting
            self.forest[tree] = self.forest[tree].fit(xFeat[indexes, :], y[indexes].flatten())  # make sure to flatten y so that it's 1D!

            # calculate OOB
            count = 0  # count and error both start at 0
            error = 0
            for index in range(xFeat.shape[0]):  # for each index in our original dataset
                label0 = 0  # set the label counters to 0
                label1 = 0
                counted = False
                for numTree in range(tree+1): # for each tree
                    if index in unusedIndexes[numTree]:  # if a given index is in the unsusedIndexes list of the current tree
                        if self.forest[numTree].predict(xFeat[index].reshape(1, -1))[0] == 0:  # see what label we get! xFeat is resphaped based on an error that I got before doing so
                            label0 += 1
                        else:
                            label1 += 1
                        if not counted:  # only count a particular row once
                            count += 1  # increase the total count by 1 (this will be used for calculating the error ratio later)
                            counted = True
                if (label0+label1 > 0):
                    if label0 > label1: # if the count of label0 is higher than label1
                        if y[index].flatten()[0] != 0:  # if we a wrong prediction
                            error += 1 # increase error!
                    elif y[index].flatten()[0] != 1:  # same thing here but we don't need to check that label1 >= label0
                        error += 1
            oob[tree+1] = error / count  # calculate the OOB for the given number of trees

        return oob  # return the results!

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
            Predicted response per sample
        """
        yHat = []
        for row in xFeat:  # for each row in xFeat
            labels = []  # reset labels list for every new row
            for tree in self.forest:  # for each tree in the forest
                labels.append(tree.predict(row.reshape(1, -1)))  # reshape(1, -1) since there is only a single sample
            garbage, counts = np.unique(labels, return_counts=True)
            if len(counts) == 1 or counts[0] > counts[1]:  # if there are more 0 labels than 1 (or if 0 is the only label found)
                yHat.append(0)
            else:
                yHat.append(1)

        return yHat  # return the predicted labels


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")

    parser.add_argument("nest")
    parser.add_argument("maxFeat")
    parser.add_argument("criterion")
    parser.add_argument("maxDepth")
    parser.add_argument("minLeafSample")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)   
    model = RandomForest(int(args.nest), int(args.maxFeat), args.criterion, int(args.maxDepth), int(args.minLeafSample))
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)

    print("Accuracy:", acc(yTest.flatten(), yHat))


if __name__ == "__main__":
    main()