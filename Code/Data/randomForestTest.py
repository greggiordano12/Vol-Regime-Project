# Just using this for practice for the random forest classifier
# This is from https://www.datacamp.com/community/tutorials/random-forests-classifier-python
from Data.Inputs import volClass
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

fred_s = ["DCOILBRENTEU","BAMLH0A0HYM2", "GOLDAMGBD228NLBM","DAAA","RIFSPPFAAD01NB","BAMLHE00EHYIOAS"]
trial_vol = volClass.Vol_Data("2000-01-01", fred_strings = fred_s)
x = trial_vol.weekly_fred_data()
x.shape
y = trial_vol.weekly_vix() #weekly_vix should be the target data set for when we run our tests.
y.shape
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

x.isnull()
print(x.head())

#Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=10)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))



iris = datasets.load_iris()

# Getting an idea for what the data looks like
print(iris.target_names)
print(iris.feature_names)


print(iris.data[0:5])
print(iris.target)


# Creating a DataFrame of given iris dataset.
data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
data.head()

X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y=data['species']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


#Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=10)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)



#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
