# Converted from q2.ipynb using "jupyter nbconvert --to script q2.ipynb"
import sklearn
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # as recommended during office hours with Professor Xiong

"""
Part a
"""
iris = sns.load_dataset("iris")  # instrumental for the box plots

"""
Part B -- make a box plot for each feature. Each plot should have 3 box plots,
one for each species type
"""
ax = sns.boxplot(x=iris["species"], y=iris["sepal_length"])  # figured out from the seaborn docs: https://seaborn.pydata.org/generated/seaborn.boxplot.html
plt.show()  # from Lukas office hours

ax = sns.boxplot(x=iris["species"], y=iris["sepal_width"])
plt.show()

ax = sns.boxplot(x=iris["species"], y=iris["petal_width"])
plt.show()

ax = sns.boxplot(x=iris["species"], y=iris["petal_length"])
plt.show() 

"""
Part C -- Plot a scatterplot of petal and sepal separately 
"""
# Figured out from the seaborn docs: https://seaborn.pydata.org/generated/seaborn.scatterplot.html
sns.scatterplot(data=iris, x="sepal_length", y="sepal_width", hue="species")
plt.show()

sns.scatterplot(data=iris, x="petal_length", y="petal_width", hue="species")
plt.show()