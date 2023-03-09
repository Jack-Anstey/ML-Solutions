import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score as acc
import rf

def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()

def plotClassError(errors, title, labels):
    # Plot!
    count = 0
    for error in errors:
        plt.plot(error.keys(), error.values(), label=labels[count])
        count += 1
    plt.legend(title=title)
    plt.xlabel("Number of Trees (Nest)")
    plt.ylabel("OOB Error")
    plt.tight_layout()
    plt.show()

def iterateTrees(xTrain, yTrain, criterion):
    errors = []
    nest = 15

    model = rf.RandomForest(nest, 2, criterion, 3, 50)
    errors.append(model.train(xTrain, yTrain))

    model = rf.RandomForest(nest, 3, criterion, 4, 50)
    errors.append(model.train(xTrain, yTrain))

    model = rf.RandomForest(nest, 4, criterion, 4, 50)
    errors.append(model.train(xTrain, yTrain))

    model = rf.RandomForest(nest, 5, criterion, 4, 50)
    errors.append(model.train(xTrain, yTrain))

    model = rf.RandomForest(nest, 6, criterion, 5, 50)
    errors.append(model.train(xTrain, yTrain))
    
    plotClassError(errors, "OOB Error Using "+str(criterion).capitalize()+" Criterion and the Following Hyperparameters:", 
    ['MaxFeat: 2, MaxDepth: 3, MinLeafSample: 50', 'MaxFeat: 3, MaxDepth: 4, MinLeafSample: 50',
    'MaxFeat: 4, MaxDepth: 4, MinLeafSample: 50', 'MaxFeat: 5, MaxDepth: 4, MinLeafSample: 50',
    'MaxFeat: 6, MaxDepth: 5, MinLeafSample: 50'])

def main():
    """
    Main file to run from the command line.
    """

    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy("q4xTrain.csv")
    yTrain = file_to_numpy("q4yTrain.csv")
    xTest = file_to_numpy("q4xTest.csv")
    yTest = file_to_numpy("q4yTest.csv")

    # iterateTrees(xTrain, yTrain, "entropy")
    # iterateTrees(xTrain, yTrain, "gini")

    model = rf.RandomForest(10, 3, "entropy", 4, 50)  # using optimal parameters
    trainStats = model.train(xTrain, yTrain)  # train the model

    # show the accuracy against the test data
    print("Accuracy:", acc(yTest.flatten(), model.predict(xTest)))
    print()  # spacer to make this more readable

    # show the OOB
    print("OOB during training:", trainStats)


if __name__ == "__main__":
    main()