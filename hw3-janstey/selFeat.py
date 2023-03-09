import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing 


def extract_features(df):
    """
    Given a pandas dataframe, extract the relevant features
    from the date column

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with the new features
    """

    # get only the hour and update date accordingly
    df['date'] = pd.to_datetime(df['date']).transform(lambda x: x.hour)
    
    # one hot encoding -- break up the time of day into distint groups: morning afternoon evening night
    df['morning'] = df['afternoon'] = df['evening'] = df['night'] = df['date']
    df['morning'] = df['morning'].transform(lambda hour: 1 if hour >= 6 and hour < 12 else 0)
    df['afternoon'] = df['afternoon'].transform(lambda hour: 1 if hour >= 12 and hour < 17 else 0)
    df['evening'] = df['evening'].transform(lambda hour: 1 if hour >= 17 and hour < 23 else 0)
    df['night'] = df['night'].transform(lambda hour: 1 if hour >= 23 or hour < 6 else 0)
    
    df = df.drop(columns=['date'])  # remove date from the dataframe
    return df


def select_features(df):
    """
    Select the features to keep

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with a subset of the columns
    """
    
    keptFeatures = []
    pc = pearson_correlation(df)  # get the pearson correlation matrix
    for feature in pc.columns:
        if not ((pc[feature].drop(feature) > 0.5).any() or (pc[feature].drop(feature) < -0.5).any()):
            keptFeatures.append(feature)

    return df[keptFeatures]  # use only the features that we chose to keep


def pearson_correlation(df):
    pc = df.corr(method='pearson')  # calculate the pearson correlation

    # make a heatmap with the x labels on top and only show every other label
    heatmap = sns.heatmap(pc, xticklabels=1, yticklabels=1)
    heatmap.xaxis.tick_top()  # move the x axis labels to the top
    heatmap.set_xticklabels(labels=df.columns, rotation=90)  # rotate the text 90 degrees so it's readable
    plt.tight_layout()
    plt.show()  # render the heatmap
    return pc  # return the pearson correlation matrix


def preprocess_data(trainDF, testDF):
    """
    Preprocess the training data and testing data

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data 
    testDF : pandas dataframe
        Test data 
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """
    # TODO do something

    # make sure columns match
    index = 0
    while index < (len(trainDF.columns) if len(trainDF.columns) > len(testDF.columns) else len(testDF.columns)):
        if trainDF.columns[index] != testDF.columns[index]:
            if len(trainDF.columns) > len(testDF.columns):
                trainDF = trainDF.drop([trainDF.columns[index]], axis=1)
            else:
                testDF = testDF.drop([testDF.columns[index]], axis=1)
        index += 1

    # standard scale
    trainCols = trainDF.columns

    scaler = preprocessing.StandardScaler().fit(trainDF)
    trainDF = scaler.transform(trainDF)
    testDF = scaler.transform(testDF)

    trainDF = pd.DataFrame(data=trainDF, columns=trainCols)
    testDF = pd.DataFrame(data=testDF, columns=trainCols)

    return trainDF, testDF


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)
    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest = extract_features(xTest)
    # select the features
    xNewTrain = select_features(xNewTrain)
    xNewTest = select_features(xNewTest)
    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)
    # save it to csv
    xTrainTr.to_csv(args.outTrain, index=False)
    xTestTr.to_csv(args.outTest, index=False)


if __name__ == "__main__":
    main()
