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
import pandas_datareader as pdr
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_reportfrom
from sklearn.model_selection import RandomizedSearchCV


class Regime_Predict:
    #### takes input vol_data which is expected to be an object from Vol_Data class
    def __init__(self, vol_data, test_size = .3, n_estimators = 100, max_features = 'sqrt', max_depth = None):
        self.vol_data = vol_data
        self.test_size = test_size
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.inputs = vol_data.weekly_fred_data()
        self.target = vol_data.weekly_vix()
        self.inputs_lag = self.inputs.iloc[1:]
        self.target_lag = self.target.iloc[:len(self.target["Weekly_Vol"])-1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.inputs_lag, self.target_lag, test_size = self.test_size )
        self.clf = RandomForestClassifier(n_estimators = self.n_estimators, max_features = self.max_features, max_depth = self.max_depth)
        self.y_train = self.y_train.to_numpy()
        self.clf.fit(self.X_train, self.y_train.ravel())
        self.y_pred = self.clf.predict(self.X_test)
        self.y_prob = self.clf.predict_proba(self.X_test)

    def Regime_Accuracy(self):
        print("Accuracy: ", metrics.accuracy_score(self.y_test, self.y_pred))
        print('\n')
        self.target_lag = self.target_lag.to_numpy()
        rfc_cv_score = cross_val_score(self.clf, self.inputs_lag, self.target_lag.ravel(), cv=10)
        print("=== Confusion Matrix ===")
        print(confusion_matrix(self.y_test, self.y_pred))
        print('\n')
        print("=== Classification Report ===")
        print(classification_report(self.y_test, self.y_pred))
        print('\n')
        print("=== All AUC Scores ===")
        print(rfc_cv_score)
        print('\n')
        print("=== Mean AUC Score ===")
        print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

    def plot_feature_importances(self):

        col = self.inputs.columns
        y = self.clf.feature_importances_
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
        plt.show()


    def tune_parameters(self): # Takes a while to run
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
        rfc_random = RandomizedSearchCV(estimator = self.clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        # Fit the model
        rfc_random.fit(self.X_train, self.y_train.ravel())
        print(rfc_random.best_params_)


<<<<<<< HEAD
# fred_s = ["DCOILBRENTEU","BAMLH0A0HYM2", "GOLDAMGBD228NLBM","DAAA","RIFSPPFAAD01NB","BAMLHE00EHYIOAS"]
# trial_vol = volClass.Vol_Data("2000-01-01", fred_strings = fred_s)
# #
# trial_regime_predict = Regime_Predict(trial_vol)
=======


fred_s = ["DCOILBRENTEU"]
#,"BAMLH0A0HYM2", "GOLDAMGBD228NLBM","DAAA","RIFSPPFAAD01NB","BAMLHE00EHYIOAS"]
trial_vol = volClass.Vol_Data("2000-01-01", fred_strings = fred_s)
trial_regime_predict = Regime_Predict(trial_vol)
trial_regime_predict.plot_feature_importances()

trial_regime_predict.Regime_Accuracy()

trial_regime_predict.tune_parameters()


>>>>>>> 5207311ad854a903b78465fadba2271d217a2e77
#
# trial_regime_predict.Regime_Accuracy()
# trial_regime_predict.plot_feature_importances()
