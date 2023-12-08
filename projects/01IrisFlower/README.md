# Goal

The iris species have been classified into three categories that are setosa, versicolor, or virginica.

The goal is to create a machine learning model that can learn from the measurements of these irises whose species are already known so that we can predict the species for the new irises that she has founded or in other words if she will find any new iris flower.

## Steps to follow

1. [Installing the Python and SciPy platform](#installing-the-python-and-scipy-platform)
2. [Loading the dataset](#load-the-data)
    1. [Import Libraries](#import-libraries)
    2. [Load Dataset](#load-dataset)
3. [Summarizing the dataset](#summarize-the-dataset)
    1. [Dimensions of Dataset](#dimensions-of-dataset)
    2. [Peek at the Data](#peek-at-the-data)
    3. [Statistical Summary](#statistical-summary)
4. [Visualizing the dataset]
5. [Evaluating some algorithms]
6. [Making some predictions]


### Installing the Python and SciPy platform

The following libraries are required for this project:
* scipy
* numpy
* matplotlib
* pandas
* sklearn


### Load The Data

Iris flowers dataset will be used.

This dataset is famous because it is used as the “hello world” dataset in machine learning and statistics by pretty much everyone.

The dataset contains 150 observations of iris flowers. There are four columns of measurements of the flowers in centimeters. The fifth column is the species of the flower observed. All observed flowers belong to one of three species.

#### Import Libraries

``` python
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
```

#### Load Dataset

Pandas will be used to load the data, also pandas next will be used to explore the data both with descriptive statistics and data visualization.

```python
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
```

### Summarize the Dataset

In this step we are going to take a look at the data a few different ways:

1. Dimensions of the dataset.
2. Peek at the data itself.
3. Statistical summary of all attributes.
4. Breakdown of the data by the class variable.

#### Dimensions of Dataset

A quick idea of how many instances (rows) and how many attributes (columns) the data contains can be get using the shape property.

```python
# shape
print(dataset.shape)
```

Outpu will be (150,5) => 150 instances and 5 attributes

#### Peek at the Data

It can be achieved using 
```python
# head
print(dataset.head(20))
```

![Peek at the data](./images/peek.png)


#### Statistical Summary

This includes the count, mean, the min and max values as well as some percentiles.
```pythons
# descriptions
print(dataset.describe())
```

![Statistical Summary](./images/statisticalSummary.png)




Thanks for this exercise to:
https://machinelearningmastery.com/machine-learning-in-python-step-by-step/