import argparse
import numpy as np
import pandas as pd
import time

class Perceptron(object):
    mEpoch = 1000  # maximum epoch size
    w = None       # weights of the perceptron

    def __init__(self, epoch):
        self.mEpoch = epoch

    def train(self, xFeat, y):
        """
        Train the perceptron using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the epochs and values the number of mistakes
        """
        stats = {}
        # TODO implement this
        xFeat = np.append(np.ones((len(xFeat), 1)), xFeat, axis=1)  # add a row of ones to the beginning which will represent the bias
        self.w = np.zeros((len(xFeat[0]), 1))  # set the default weight to be zeros
        for epoch in range(self.mEpoch):
            yHat = np.zeros((len(xFeat), 1))  # initialize yhat
            for iteration in range(len(xFeat)):  # iterate through all of the rows
                yHat[iteration][0] = 1 if np.matmul(xFeat[iteration], self.w)[0] >= 0 else 0  # do the perceptron calculation to create a label
                if yHat[iteration][0] != y[iteration][0]:  # see if the label matches yTrue
                    if yHat[iteration][0] == 1: # if we said 1 instead of 0, subtract
                        self.w = np.subtract(self.w, np.transpose([xFeat[iteration]]))
                    else: # otherwise we said 0 instead of 1, so we add
                        self.w = np.add(self.w, np.transpose([xFeat[iteration]]))
            mistakes = calc_mistakes(yHat, y)  # get the number of total mistakes
            stats.update({epoch: mistakes})  # update the number of mistakes for this iteration
            if mistakes == 0:  # if we have no mistakes we have converged so we end early
                return stats
        return stats

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
        yHat = np.zeros((len(xFeat), 1))  # set yHat to be a bunch of 0s
        xFeat = np.append(np.ones((len(xFeat), 1)), xFeat, axis=1)  # add a bias column
        for iteration in range(len(xFeat)):  # for each row
            # use the percepton with the weights we found in training
            yHat[iteration][0] = 1 if np.matmul(xFeat[iteration], self.w)[0] >= 0 else 0
        return yHat

def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """
    mistakes = 0
    for row in range(len(yHat)):
        if yHat[row][0] != yTrue[row][0]:  # if the y's don't match increase mistakes by 1
            mistakes += 1
            
    return mistakes


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
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)   
    model = Perceptron(args.epoch)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    # print out the number of mistakes
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat, yTest))


if __name__ == "__main__":
    main()