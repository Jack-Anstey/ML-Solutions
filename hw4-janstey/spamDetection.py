import numpy as np
import pandas as pd
import perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load the datasets into dataframes
xTrainB = pd.read_csv("binaryTrain.csv")
xTestB = pd.read_csv("binaryTest.csv")
xTrainC = pd.read_csv("countTrain.csv")
xTestC = pd.read_csv("countTest.csv")

yTrain = pd.read_csv("yTrain.csv")
yTest = pd.read_csv("yTest.csv")

# Try perceptron
mnb = MultinomialNB()  # make a multinomial naive bayes object
yHatMNBB = mnb.fit(xTrainB, yTrain['label']).predict(xTestB)
yHatMNBC = mnb.fit(xTrainC, yTrain['label']).predict(xTestC)

# Try logistic regression
regressionB = LogisticRegression().fit(xTrainB, yTrain['label'])
regressionC = LogisticRegression().fit(xTrainC, yTrain['label'])
yHatLRB = regressionB.predict(xTestB)
yHatLRC = regressionC.predict(xTestC)

# print results for mnb
print("Number of mistakes when using the binary dataset and multinomial naive bayes:", perceptron.calc_mistakes(np.transpose([yHatMNBB]), yTest.to_numpy()))
print("Number of mistakes when using the count dataset and multinomial naive bayes:", perceptron.calc_mistakes(np.transpose([yHatMNBC]), yTest.to_numpy()))

# print results for logistic regression
print("Number of mistakes when using the binary dataset and logistic regression:", perceptron.calc_mistakes(np.transpose([yHatLRB]), yTest.to_numpy()))
print("Number of mistakes when using the count dataset and logistic regression:", perceptron.calc_mistakes(np.transpose([yHatLRC]), yTest.to_numpy()))