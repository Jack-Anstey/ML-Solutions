import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing

def model_assessment(filename):
    """
    Given the entire data, split it into training and test set 
    so you can assess your different models 
    to compare perceptron, logistic regression,
    and naive bayes. 
    """

    data = pd.read_csv(filename, names=['email', 'label'])
    splits = data['email'].str.split(' ', 1)
    data['email'] = splits.str[1]
    data['label'] = splits.str[0]
    return data[:int(len(data)*0.7)], data[int(len(data)*0.7):]  #70/30 split


def build_vocab_map(train: pd.Series) -> dict():
    """Given a dataset, build a dictionary of words that appear more than 30 times, 
    along with the number of times that they appear

    Args:
        train (pd.Series): a series of the contents of the emails

    Returns:
        dict(): a dictionary that contains the words that appear more than 30 times, 
        along with the number of times that appeared
    """
    
    full_map = dict()  # this is a dictionary that will hold everything
    vocab_map = dict()  # the pruned dictionary that we return
    for email in train:  # for each email
        for word in email.split():  # split into individual words
            if word in full_map:  # if its in the dictionary add one to it
                full_map[word] += 1
            else:  # otherwise create a key:value pair and set the value to 1
                full_map[word] = 1
    
    for key in full_map.keys():  # for each key
        if full_map[key] >= 30:  # if the value is greater than 30, add it to the pruned dictionary
            vocab_map[key] = full_map[key]

    # we're done!
    return vocab_map


def construct_binary(train, test, vocab_map):
    """
    Construct email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """

    # create two dataframes for the training and test datasets and fill them all with 0's
    # where the columns are the keys of the dictionary
    binaryTrain = pd.DataFrame(data=np.zeros((len(train), len(vocab_map.keys()))), columns=vocab_map.keys())
    binaryTest = pd.DataFrame(data=np.zeros((len(test), len(vocab_map.keys()))), columns=vocab_map.keys())

    index = 0
    for email in train:  # for each email in the training dataset
        for key in vocab_map:  # for each key in the dictionary
            if key in email:  # if there's a match
                binaryTrain[key][index] = 1  # set the corresponding column in the dataframe that has that word to 1
        index+=1

    # do the exact same thing but for the test dataset
    index = 0
    for email in test:
        for key in vocab_map:
            if key in email:
                binaryTest[key][index] = 1
        index+=1

    # return the constructed dataframes
    return binaryTrain, binaryTest


def construct_count(train, test, vocab_map):
    """
    Construct email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    
    # start with making two dataframes full of zeros whose columns are that of the dictionary
    countTrain = pd.DataFrame(data=np.zeros((len(train), len(vocab_map.keys()))), columns=vocab_map.keys())
    countTest = pd.DataFrame(data=np.zeros((len(test), len(vocab_map.keys()))), columns=vocab_map.keys())

    index = 0
    for email in train:  # for each email in the dataset
        # get the number of unique words and the number of times they appear in a email
        words, counts = np.unique(email.split(), return_counts=True)
        wordcount = dict(zip(words, counts))  # make the two seperate lists a dictionary
        for key in vocab_map:  # for each key in the vocab dictionary
            if key in wordcount:  # for each word in the unique words list
                countTrain[key][index] = wordcount[key]  # set the appropriate value in the df to be the the number of times it appeared in the email
        index+=1

    # do the exact same thing again for the test dataset
    index = 0
    for email in test:
        words, counts = np.unique(email, return_counts=True)
        wordcount = dict(zip(words, counts))
        for key in vocab_map:
            if key in wordcount:
                countTest[key][index] = wordcount[key]
        index+=1

    # return the resulting dataframes!
    return countTrain, countTest


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()
    train, test = model_assessment(args.data)
    vocab_map = build_vocab_map(train['email'])
    binaryTrain, binaryTest = construct_binary(train['email'], test['email'], vocab_map)
    countTrain, countTest = construct_count(train['email'], test['email'], vocab_map)

    # make all of the dataframes csv's to use in perceptron.py
    binaryTrain.to_csv("binaryTrain.csv", index=False)
    binaryTest.to_csv("binaryTest.csv", index=False)
    countTrain.to_csv("countTrain.csv", index=False)
    countTest.to_csv("countTest.csv", index=False)

    # make the label dataframes
    train['label'].to_frame().to_csv("yTrain.csv", index=False)
    test['label'].to_frame().to_csv("yTest.csv", index=False)

if __name__ == "__main__":
    main()