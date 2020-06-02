# Just using this for practice for the random forest classifier
# This is from https://www.datacamp.com/community/tutorials/random-forests-classifier-python
import os
os.getcwd()

from Code.Data.Inputs import volClass
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

fred_s = ["DCOILBRENTEU","BAMLH0A0HYM2", "GOLDAMGBD228NLBM","DAAA","RIFSPPFAAD01NB","BAMLHE00EHYIOAS"]
trial_vol = volClass.Vol_Data("2000-01-01", fred_strings = fred_s)
x = trial_vol.weekly_fred_data()
x.shape
y = trial_vol.weekly_vix() #weekly_vix should be the target data set for when we run our tests.
y.shape


x.tail()
y.head()

x_lag = x.drop(pd.to_datetime('2020-05-26'))
y_lag = y.drop(pd.to_datetime('2000-01-03'))

X_train, X_test, y_train, y_test = train_test_split(x_lag, y_lag, test_size=0.3)

#Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=10)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train.ravel())


y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Drop last row of input data and drop first row target data to create a lag
# Feature importance

clf.fit(X_train, y_train.ravel())
importances = clf.feature_importances_
clf.estimators_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the impurity-based feature importances of the forest
col = ["DCOILBRENTEU","BAMLH0A0HYM2", "GOLDAMGBD228NLBM","DAAA","RIFSPPFAAD01NB","BAMLHE00EHYIOAS", "VIX_Regime"]
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()



clf.feature_importances_


# Need to doulbe check how feature_importances_ works and which features it corresponds to

col = ["DCOILBRENTEU","BAMLH0A0HYM2", "GOLDAMGBD228NLBM","DAAA","RIFSPPFAAD01NB","BAMLHE00EHYIOAS"]
clf.feature_importances_
y = clf.feature_importances_
y = np.delete(y, -1)
#plot
fig, ax = plt.subplots()
width = 0.4 # the width of the bars
ind = np.arange(len(y)) # the x locations for the groups
ax.barh(ind, y, width, color = "green")
ax.set_yticks(ind + width / 10)
ax.set_yticklabels(col, minor=False)
plt.title('Feature Importance in RandomForest Classifier')
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
plt.figure(figsize=(5,5))
fig.set_size_inches(6.5, 4.5, forward=True)
