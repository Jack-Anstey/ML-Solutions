import argparse
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000 # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        
        # Create starting values and variables
        self.beta = np.random.normal(size=(len(xTrain[0])+1, 1))  # Set beta to a random guassian distribution

        initialTime = time.time()  # get the starting time
        B = int(len(xTrain)/self.bs)  # get the value of B

        # add ones to xTrain and xTest
        xTrain = np.concatenate((np.ones((len(xTrain), 1)), xTrain), axis=1)
        xTest = np.concatenate((np.ones((len(xTest), 1)), xTest), axis=1)

        # compute the stats using the mse method
        trainMSE = self.mse(xTrain, yTrain)
        testMSE = self.mse(xTest, yTest)

        # show what the mse is before we do any gradient descent
        trainStats.update({0: {"time": time.time() - initialTime, "train-mse": trainMSE, "test-mse": testMSE}})

        for epoch in range(1, self.mEpoch):
            trainingData = np.concatenate((xTrain, yTrain), axis=1)  # add the yTrain to xTrain 
            np.random.shuffle(trainingData)  # shuffle it
            yTrainShuffled = trainingData[: ,[len(xTrain[0])]].copy()  # get the y by itself
            xTrainShuffled = np.delete(trainingData, len(xTrain[0]), axis=1)  # remove the y's at the end
            # gradient = np.zeros((len(xTrain[0]), 1))  # start the gradient as an array of zeros

            for b in range(B):
                # get the batches (and add ones where it makes sense)
                batchX = xTrainShuffled[b*self.bs:(b+1)*self.bs]
                batchY = yTrainShuffled[b*self.bs:(b+1)*self.bs]

                # update beta by using the average of the gradient
                self.beta += np.multiply(np.divide(self.lr, self.bs), np.matmul(batchX.transpose(), np.subtract(batchY, np.matmul(batchX, self.beta))))
            
            trainMSE = self.mse(xTrain, yTrain)
            testMSE = self.mse(xTest, yTest)

            # compute updates
            trainStats.update({epoch*B-1: {"time": time.time() - initialTime, "train-mse": trainMSE, "test-mse": testMSE}})
           
        return trainStats


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
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # setting the seed for deterministic behavior
    np.random.seed(args.seed)   
    model = SgdLR(args.lr, args.bs, args.epoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()

