from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
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

# Univariate plot to better understand each attribute
# Multivariate plot to better understand the relationships between attribute

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length' , 'sepal-width' , 'petal-length' , 'petal-width' , 'class']
dataset = read_csv(url , names= names)

 #Univariate
# box and whisker plots
#dataset.plot(kind="box" , subplots=True , layout=(2,2) , sharex = False , sharey = False)
#pyplot.show()

#histogram
#dataset.hist()
#pyplot.show()

    #Multivariate

scatter_matrix(dataset)
pyplot.show()