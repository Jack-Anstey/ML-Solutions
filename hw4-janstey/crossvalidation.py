import numpy as np
import pandas as pd
import perceptron

def kCV(xTrainName: str, yTrainName: str, k: int, mEpoch: int) -> int:
    """Takes training and test inputs and finds the optimal epoch

    Args:
        xTrainName (str): the x training input name
        yTrainName (str): the y training input name
        k (int): the number of folds
        mEpoch (int): the maximum epoch value to test

    Returns:
        int: the optimal epoch
    """

    # load the files
    xTrain = pd.read_csv(xTrainName)
    yTrain = pd.read_csv(yTrainName)

    # set the seed
    np.random.seed(334) 
    
    # define default optimal epoch
    oEpoch = 0

    # define the lowest number of average mistakes
    smallestMistakes = float('inf')
    
    # the size of the folds
    pSize = int(len(xTrain)/k)
    
    # cross validation!
    for epoch in range(1, mEpoch):
        avgMistakes = 0  # initalize avgMistakes
        for fold in range(k):
            # break up into k-folds
            xTrainPortion = pd.concat([xTrain[: pSize*(fold)], xTrain[pSize*(fold+1):]], ignore_index=True).to_numpy()
            xTest = xTrain[pSize*(fold): pSize*(fold+1)].to_numpy()
            yTrainPortion = pd.concat([yTrain[: pSize*(fold)], yTrain[pSize*(fold+1):]], ignore_index=True).to_numpy()
            yTest = yTrain[pSize*(fold): pSize*(fold+1)].to_numpy()
            
            # model!
            model = perceptron.Perceptron(epoch)
            trainStats = model.train(xTrainPortion, yTrainPortion)
            yHat = model.predict(xTest)

            # get the number of mistakes
            avgMistakes += perceptron.calc_mistakes(yHat, yTest)

        avgMistakes /= k

        if smallestMistakes > avgMistakes:
            smallestMistakes = avgMistakes
            oEpoch = epoch

    return oEpoch  #returns the average which is the optimal

mEpoch = 20

oEpochBinary = kCV("binaryTrain.csv", "yTrain.csv", 5, mEpoch)
print("Optimal epoch for the binary dataset:", oEpochBinary)

oEpochCount = kCV("countTrain.csv", "yTrain.csv", 5, mEpoch)
print("Optimal epoch for the count dataset:", oEpochCount)

def useOptEpoch(xTrainName: str, yTrainName: str, xTestName: str, yTestName: str, oEpoch: int) -> list():
    """See the number of mistakes on the full dataset using the optimal epoch value

    Args:
        xTrainName (str): the x training input name
        yTrainName (str): the y training input name
        xTestName (str): the x test input name
        yTestName (str): the y test input name
        oEpoch (int): the optimal epoch value
    """

    xTrain = perceptron.file_to_numpy(xTrainName)
    yTrain = perceptron.file_to_numpy(yTrainName)
    xTest = perceptron.file_to_numpy(xTestName)
    yTest = perceptron.file_to_numpy(yTestName)

    # model!
    model = perceptron.Perceptron(oEpoch)
    trainStats = model.train(xTrain, yTrain)
    yHatTrain = model.predict(xTrain)
    yHatTest = model.predict(xTest)

    # get the number of mistakes
    trainMistakes = perceptron.calc_mistakes(yHatTrain, yTrain)
    testMistakes = perceptron.calc_mistakes(yHatTest, yTest)

    # Print the results
    print("Number of training mistakes:", trainMistakes)
    print("Number of test mistakes:", testMistakes)

    # return the model for solving part c
    return model.w

# print out the number of mistakes using the optimal epoch and save the weights from the model
print("Using the binary dataset, the results are as follows:")
weightsBinary = useOptEpoch("binaryTrain.csv", "yTrain.csv", "binaryTest.csv", "yTest.csv", oEpochBinary)

# print out the number of mistakes using the optimal epoch and save the weights from the model
print("Using the count dataset, the results are as follows:")
weightsCount = useOptEpoch("countTrain.csv", "yTrain.csv", "countTest.csv", "yTest.csv", oEpochCount)

def posAndNegWeights(weights: np.array, filename: str) -> tuple:
    """Get the most positive and negative weighted names from a perceptron model

    Args:
        weights (np.array): the weights from the model
        filename (str): the .csv that has all the column names you want to reference

    Returns:
        tuple: a tuple of lists that have the names which have the most positive and negative weights
    """

    # create and fill arrays with 0's to start
    posWeight = [0]*15
    negWeight = [0]*15

    # we remove the bias since that will throw our results off
    weightsPos = weights[1:].copy()  
    weightsNeg = weights[1:].copy()

    # load the file to get the word names
    xTrain = pd.read_csv(filename)
    
    # find the index's of the most positive weights and then convert to column name
    for index in range(len(posWeight)):
        posWeight[index] = np.where(weightsPos == max(weightsPos))[0][0]  # get the first index of the largest weight
        weightsPos[posWeight[index]] = float('-inf')  # set to neg infinity to make sure we don't see it again
        posWeight[index] = xTrain.columns[posWeight[index]]  # set the name instead of the index

    # find the index's of the most negative weights
    for index in range(len(negWeight)):
        negWeight[index] = np.where(weightsNeg == min(weightsNeg))[0][0]  # get the first index of the largest weight
        weightsNeg[negWeight[index]] = float('inf')  # set to infinity to make sure we don't see it again
        negWeight[index] = xTrain.columns[negWeight[index]]  # set the name instead of the index

    # return the lists of the names
    return posWeight, negWeight

# get and print the names with the most positive and negative weights
print("Using the binary dataset, the results are as follows:")
posBinary, negBinary = posAndNegWeights(weightsBinary, "binaryTrain.csv")
print("The words with the most positive weights:", posBinary)
print("The words with the most negative weights:", negBinary)

# get and print the names with the most positive and negative weights
print("Using the count dataset, the results are as follows:")
posCount, negCount = posAndNegWeights(weightsCount, "countTrain.csv")
print("The words with the most positive weights:", posCount)
print("The words with the most negative weights:", negCount)