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
from sklearn.multiclass import OneVsRestClassifier
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
############################################################

# Validation Dataset
# Split-out validation dataset: split into two -> 80% used for training, eval, select among the models
# 20% held back as a validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

"""
	Stratified 10-fold cross validation estimate model accuracy.
	Dataset split in 10, 9 for training, 1 for testing -> repeated for all combinations of train-test splits.
	Stratified = each fold/split of the dataset aims for the same distribution of example by class as exists in the whole training dataset.
	Fixed random seed, so all algorithms train on the same dataset splits.
"""

# Spot Check Algorithms
models = [
	("LR", OneVsRestClassifier(LogisticRegression(solver="liblinear"))),
	("LDA", LinearDiscriminantAnalysis()),
	("KNN", KNeighborsClassifier()),
	("CART", DecisionTreeClassifier()),
	("NB", GaussianNB()),
	("SVM", SVC(gamma="auto"))
		  ]

# Evaluate each model
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")
	results.append(cv_results)
	names.append(name)
	print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms - using box and whisker plots
plt.boxplot(results, tick_labels=names)
plt.title("Algorithm Comparison")
plt.show()

# Make predictions on validation dataset
model = SVC(gamma="auto")
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))