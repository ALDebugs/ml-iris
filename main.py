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

# Data Set
iris_data = "iris/iris.data"
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset = read_csv(iris_data, names=names)
#print(dataset)

# Shape -> how many instances (rows) and attributes (columns) in the data
print(dataset.shape)

# Print the first 20 rows
print(dataset.head(20))

# Descriptions
print(dataset.describe())

# Check how many of each row belongs to each class
print(dataset.groupby("class").size())

############################################################
# Univariate Plots - plots of each individual variable
# Box and whisker plots
dataset.plot(kind="box", subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# Histograms - for an idea of the distribution
dataset.hist()
plt.show()
###########################################################

###########################################################
# Multivariate Plots - interactions between the variables
# Scatter plot matrix
scatter_matrix(dataset)
plt.show()
# Diagonal grouping of pairs of attributes -> high correlation and predictable relationship
