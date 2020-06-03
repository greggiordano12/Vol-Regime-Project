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

class Regime_Predict:
    #### takes input vol_data which is expected to be an object from Vol_Data class
    def __init__(self, vol_data, test_size = .3, n_estimators = 10):
        self.vol_data = vol_data
        self.test_size = test_size
        self.n_estimators = n_estimators
        self.inputs = vol_data.weekly_fred_data()
        self.target = vol_data.weekly_vix()
        self.inputs_lag = self.inputs.iloc[1:]
        self.target_lag = self.target.iloc[:len(self.target["Weekly_Vol"])-1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.inputs_lag, self.target_lag, test_size = self.test_size )
        self.clf = RandomForestClassifier(n_estimators = self.n_estimators)
        self.y_train = self.y_train.to_numpy()
        self.clf.fit(self.X_train, self.y_train.ravel())
        self.y_pred = self.clf.predict(self.X_test)

    def Regime_Accuracy(self):
        return metrics.accuracy_score(self.y_test, self.y_pred)

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





fred_s = ["DCOILBRENTEU","BAMLH0A0HYM2", "GOLDAMGBD228NLBM","DAAA","RIFSPPFAAD01NB","BAMLHE00EHYIOAS"]
trial_vol = volClass.Vol_Data("2000-01-01", fred_strings = fred_s)
#
trial_regime_predict = Regime_Predict(trial_vol)
#
# trial_regime_predict.plot_feature_importances()
