import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import dt

"""This is a .py version of calculateacc.ipynb, which creates accuracy graphs for question 1
"""

xTrain = pd.read_csv("q4xTrain.csv")
yTrain = pd.read_csv("q4yTrain.csv")
xTest = pd.read_csv("q4xTest.csv")
yTest = pd.read_csv("q4yTest.csv")

datapoints1 = []
maxdepth = 4
for mls in range(50, 501, 25):
    dt1 = dt.DecisionTree('entropy', maxdepth, mls)
    trainAcc, testAcc = dt.dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    datapoints1.append((testAcc, mls))

datapoints2 = []
mls = 100
for maxdepth in range(1, 100):
    dt2 = dt.DecisionTree('entropy', maxdepth, mls)
    trainAcc, testAcc = dt.dt_train_test(dt2, xTrain, yTrain, xTest, yTest)
    datapoints2.append((testAcc, maxdepth))

df1 = pd.DataFrame(datapoints1, columns=["Accuracy", "Minimum Leaf Split"])
sns.scatterplot(data=df1, x="Minimum Leaf Split", y="Accuracy", legend="full")
plt.show()

df2 = pd.DataFrame(datapoints2, columns=["Accuracy", "Depth"])
sns.scatterplot(data=df2, x="Depth", y="Accuracy", legend="full")
plt.show()