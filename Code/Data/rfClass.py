import os
os.getcwd()

from Code.Data.Inputs import volClass
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix

class Regime_Predict:
    #### takes input vol_data which is expected to be an object from Vol_Data class
    def __init__(self, vol_data, test_size = .3, n_estimators = 10):
        self.vol_data = vol_data
        self.test_size = test_size
        self.n_estimators = n_estimators
        # self.fred_strings = fred_strings
        self.inputs = vol_data.weekly_fred_data()
        self.target = vol_data.weekly_vix()
        self.inputs_lag = self.inputs.iloc[:len(self.target["Weekly_Vol"])-1]
        self.target_lag = self.target.iloc[1:]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.inputs_lag, self.target_lag, test_size = self.test_size )
        self.clf = RandomForestClassifier(n_estimators = self.n_estimators)
        self.y_train = self.y_train.to_numpy()
        self.clf.fit(self.X_train, self.y_train.ravel())

        self.y_pred = self.clf.predict(self.X_test)
        self.all_predictions = self.clf.predict(self.inputs_lag)
        all_predictions_df = pd.DataFrame({"Week":self.target_lag.index, "Vol_Regime":self.all_predictions})
        self.all_predictions_df = all_predictions_df.set_index("Week")
        self.all_prob = pd.DataFrame(self.clf.predict_proba(self.inputs_lag))
        self.all_prob.columns = ["Low_Vol", "Med_Vol", "High_Vol"]
        self.all_prob["Week"] = self.target_lag.index
        self.all_prob = self.all_prob.set_index("Week")

    def Regime_Accuracy(self):
        x = metrics.accuracy_score(self.y_test, self.y_pred)
        print("The accuracy of our model to predict the right category is: ")
        print("%.3f" % x + "%")

        plot_confusion_matrix(self.clf, self.X_test, self.y_test)

        print(classification_report(self.y_test, self.y_pred))
        y_lag = self.target_lag.to_numpy()
        y_lag = y_lag.ravel()
        rfc_cv_score = cross_val_score(self.clf, self.inputs_lag, y_lag, cv=10)
        print("%.3f" % rfc_cv_score + "%")


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

    def plot_feature_rankings(self):
        data = self.X_train
        importances = self.clf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.clf.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")
        for f in range(data.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

            # Plot the impurity-based feature importances of the forest
        plt.figure()
        plt.title("Feature importance rankings")
        plt.bar(range(data.shape[1]), importances[indices],
                color="green", yerr=std[indices], align="center")
        plt.xticks(range(data.shape[1]), indices)
        plt.xlim([-1, data.shape[1]])
        plt.show()


# Get more outputs descrbing accuracy of the random forest

fred_s = ["DCOILBRENTEU","BAMLH0A0HYM2", "GOLDAMGBD228NLBM","DAAA","RIFSPPFAAD01NB","BAMLHE00EHYIOAS", "T10Y3M", "BAMLEMFSFCRPITRIV"]
trial_vol = volClass.Vol_Data("2007-01-01", fred_strings = fred_s)
#
trial_regime_predict = Regime_Predict(trial_vol)
trial_regime_predict.plot_feature_importances()
trial_regime_predict.plot_feature_rankings()

trial_regime_predict.all_prob
trial_regime_predict.inputs_lag
trial_regime_predict.target_lag

trial_regime_predict.all_prob.loc[pd.to_datetime("2007-01-08")][2]

trial_regime_predict.Regime_Accuracy()
trial_regime_predict.plot_feature_importances()
regime_data = trial_regime_predict.all_predictions_df


regime_data
