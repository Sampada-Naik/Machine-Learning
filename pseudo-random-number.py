//Import packages
import csv
import numpy
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

//Load the dataset and print the first few datas
filename = "DS.csv"
x = []
y = []
with open(filename, 'r') as csvfile:
 csvreader = csv.reader(csvfile)
 for row in csvreader:
 	x.append(float(row[0]))
 	y.append(float(row[1]))

 print(x)
 print(y)
 print(len(x), len(y))
 print(type(x), type(y))

//Line plot
plt.scatter(x, y)
plt.show()

//Train and test values
trainx = x[:1600]
trainy = y[:1600]

testx = x[1600:]
testy = y[1600:]

//Linear regression
slope, intercept, r, p, std_err = stats.linregress(trainx, trainy)

def myfunc(x):
 return slope * x + intercept

mymodel = list(map(myfunc, trainx))

//Scatter plot
plt.scatter(trainx, trainy)
plt.plot(trainx, mymodel, color='r')
plt.title('Linear Regression')
plt.show()

//Print values
print("s: ",slope)
print("i: ", intercept)
print("r: ",r)
print("r2: ", r**2)
print(myfunc(15))

//Polynomial Regression
mymodel = numpy.poly1d(numpy.polyfit(trainx, trainy, 3))
mylineplots= numpy.linspace(-13,13, 2000)

//Scatter plots
plt.scatter(xval, yval)
plt.plot(mylineplots, mymodel(mylineplots), color = 'r')
plt.title('Polynomial Regression')
plt.show()
print(mymodel)

//Print r2 score
r2 = r2_score(testy, mymodel(testx))
print(r2)

mymodel(15)

//Import packages
import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeRegressor
import matplotlib.image as pltimg

df = pandas.read_csv(filename, header=None)

print(df)
x = [[i] for i in df[0]]
y = [i for i in df[1]]

print(x)
print(y)

rng = numpy.random.RandomState(1)
X = numpy.sort(5 * rng.rand(80, 1), axis=0)
y = numpy.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))
X = df[0].values
y = df[1].values

//Split test and train data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

treeRegr = DecisionTreeRegressor(max_depth=10)
treeRegr.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

y_pred = treeRegr.predict(X_test.reshape(-1,1))

//Print r2 score
r2 = r2_score(y_test, y_pred)
print(r2)

//Decision tree regression
X_grid = numpy.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test, color = 'r')
plt.scatter(X_test, y_pred, color = 'g')
plt.title('Decision Tree Regression')
plt.show()

plt.plot(X_grid, treeRegr.predict(X_grid), color = 'k')
plt.title('Decision Tree Regression')
plt.show()

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X.reshape(-1,1))
y = sc_y.fit_transform(y.reshape(-1,1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)

r2 = r2_score(y_test, y_pred)
print(r2)

//SVR regression
X_grid = numpy.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X_test), sc_y.inverse_transform(y_test.reshape(-
1)), color = 'red')
plt.scatter(sc_X.inverse_transform(X_test), y_pred, color = 'green')
plt.title('SVR Regression')
plt.show()

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X.reshape(-1,1))
y = sc_y.fit_transform(y.reshape(-1,1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)

r2 = r2_score(y_test, y_pred)
print(r2)

X_grid = numpy.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X_test), sc_y.inverse_transform(y_test.reshape(-
1)), color = 'red')
plt.scatter(sc_X.inverse_transform(X_test), y_pred, color = 'green')
plt.title('SVR Regression')
plt.show()
