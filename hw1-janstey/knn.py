import argparse
import numpy as np
import pandas as pd


class Knn(object):
    k = 0    # number of neighbors to use
    trainingSet = pd.DataFrame
    numLabels = 0

    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k

    def train(self, xFeat, y):
        """
        Train the k-nn model.

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
        # print(xFeat)  # of type class 'pandas.core.frame.DataFrame'>
        # print(y)  # of type <class 'pandas.core.series.Series'>
        self.trainingSet = xFeat.copy()
        self.trainingSet['label'] = y # add the labels to the dataframe
        self.numLabels = int(y.max()+1)  # assumes a label 0
        return self


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

        # TODO
        for dataTuple in xFeat.itertuples():  # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.itertuples.html
            datapoints = []
            for index in range(len(dataTuple)):  # get everything except the column number and label
                if not(index == 0  or index == len(dataTuple)-1):
                    datapoints.append(dataTuple[index])
            distances = []
            labels = [0] * self.numLabels  # creates a list that is of labels length
            for trainTuple in self.trainingSet.itertuples():  # iterate through the memorized dataset
                distance = 0
                index = 1
                # loop to calculate euclidean distance
                for data in datapoints:
                    distance += np.square(data - trainTuple[index])
                    index += 1  # increment index by one
                distances.append((np.sqrt(distance), trainTuple[len(trainTuple)-1]))  # input the sqrt of the distance and the associated label
                # distances.append((np.sqrt((np.square(currX - trainTuple[1]) + np.square(currY - trainTuple[2]))), trainTuple[3]))  # euclidean distance
            distances.sort()  # sort so that the shortest distances are first
            for nn in range(self.k):  # get the k nearest neighbors
                # print("Distance", distances[nn])
                # print("label", distances[nn][1])
                labels[int(distances[nn][1])] += 1
            maxLabel = -1
            maxCount = -1
            for index in range(len(labels)):
                if labels[index] > maxCount:
                    maxCount = labels[index]
                    maxLabel = index
                elif labels[index] == maxCount:  # 50/50 for a tie
                    maxLabel = np.random.choice([maxLabel, index], 1) # https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
                    maxCount = labels[maxLabel[0]] # let the count be updated with whatever the random result is
            yHat.append(float(maxLabel))
        return yHat

def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    # TODO calculate the accuracy
    # print("yHat length: " + str(len(yHat)))
    # print("yHat raw:", yHat)
    # print("yTrue:", yTrue)
    acc = 0
    for index in range(len(yTrue)):
        if yHat[index] == yTrue[index]:
            acc += 1.0  # add one to the count if the result was accurate
    acc /= len(yTrue)  # turn into percentage
    return acc

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()

    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)

    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])

    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)

if __name__ == "__main__":
    main()