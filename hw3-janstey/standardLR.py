import argparse
from re import S
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy


class StandardLR(LinearRegression):

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """

        # TODO: DO SOMETHING
        # start time
        initialTime = time.time()

        # add ones to x
        xTrain = np.concatenate((np.ones((len(xTrain), 1)), xTrain), axis=1)
        xTest = np.concatenate((np.ones((len(xTest), 1)), xTest), axis=1)

        # define beta
        self.beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(xTrain.transpose(), xTrain)), xTrain.transpose()), yTrain)

        # compute the stats using the mse method
        trainMSE = self.mse(xTrain, yTrain)
        testMSE = self.mse(xTest, yTest)

        timeElapsed = time.time() - initialTime # take the time since we are done generating yHats

        return {0: {"time": timeElapsed, "train-mse": trainMSE, "test-mse": testMSE}}


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

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()
