from Code.Data.Inputs import volClass
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

fred_s = ["DCOILBRENTEU" ,"BAMLH0A0HYM2", "GOLDAMGBD228NLBM","DAAA","RIFSPPFAAD01NB","BAMLHE00EHYIOAS", "DEXCHUS", "DEXUSEU", "T10Y3M", "BAMLEMFSFCRPITRIV"]
trial_vol = volClass.Vol_Data("2000-01-01", fred_strings = fred_s)
x = trial_vol.weekly_fred_data()
x.shape
y = trial_vol.weekly_vix() #weekly_vix should be the target data set for when we run our tests.
y.shape
x.corr()

x.tail()
y.tail()

x_lag = x.drop(pd.to_datetime('2020-06-01'))
y_lag = y.drop(pd.to_datetime('2000-01-03'))
x_lag.shape
y_lag.shape

x_lag
X_train, X_test, y_train, y_test = train_test_split(x_lag, y_lag, test_size=0.3)

#Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=200, max_depth = 140, max_features = 'sqrt')

# Convert Y to an array because that is how the RF needs the data to be
y_train = y_train.to_numpy()

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train.ravel())
y_prob = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

y_prob
y_pred

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(rfc_cv_score.mean())
# Drop last row of input data and drop first row target data to create a lag
# Feature importance

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the impurity-based feature importances of the forest
col = x.columns
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()



# Need to doulbe check how feature_importances_ works and which features it corresponds to
col = x.columns
y = clf.feature_importances_
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

# Comparing actual results to the Predicted
df = pd.DataFrame({'Actual': y_test["Weekly_Vol"], 'Predicted':y_pred.flatten()})
df1 = df.head(25)
df.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

y_lag = y_lag.to_numpy()
y_lag.shape
y_lag = y_lag.ravel()
type(y_lag)
rfc_cv_score = cross_val_score(clf, x_lag, y_lag, cv=10)


print(confusion_matrix(y_test, y_pred))


print(classification_report(y_test, y_pred))


print(rfc_cv_score.mean())




from sklearn.model_selection import RandomizedSearchCV
# number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# number of features at every split
max_features = ['auto', 'sqrt']

# max depth
max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
max_depth.append(None)
# create random grid
random_grid = {
 'n_estimators': n_estimators,
 'max_features': max_features,
 'max_depth': max_depth
 }
# Random search of parameters
rfc_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the model
y_train = y_train.to_numpy()

rfc_random.fit(X_train, y_train.ravel())
# print results
print(rfc_random.best_params_)
