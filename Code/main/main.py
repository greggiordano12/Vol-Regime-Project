import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
from Code.Trading_Strats import portfolioClass
from Code.Trading_Strats.portfolioClass import returns_matrix, portfolio_return, portfolio_variance, vol, returns_vector
from Code.Data.Inputs import volClass
from Code.Data import rfClass
import scipy.stats as st

'''
Implement a class that takes in params which will run all classes and produce output.

Portfolio with just PNQI, SPXL, and GLD produces best results
'''
def VaR(rvec,conf_level = .95):
    '''
    takes in a daily returns vector and confidence level for VaR and returns the annual VaR
    '''
    vol = np.std(rvec)
    final = -(st.norm.ppf(.95))*vol*np.sqrt(252)
    return float(final)

class Vol_Outputs:
    def __init__(self, start_date="2008-01-01", end_date= None, bull_tickers = ["PNQI","GLD","SPXL"], fred_strings=["DCOILBRENTEU" ,"BAMLH0A0HYM2", "GOLDAMGBD228NLBM","DAAA","RIFSPPFAAD01NB","BAMLHE00EHYIOAS", "DEXCHUS", "DEXUSEU", "T10Y3M", "BAMLEMFSFCRPITRIV"], n_estimators = 100, max_features = 'sqrt', max_depth = None):
        self.vol_data = volClass.Vol_Data(start_date, end_date, fred_strings)
        self.regime_predict = rfClass.Regime_Predict(self.vol_data)
        self.portfolio = portfolioClass.Portfolio(start_date, end_date, bull_tickers, regime_predict = self.regime_predict)
        self.opt_daily = self.portfolio.weekly_optimization()

        self.start_date = start_date
        self.end_date = end_date
        self.bull_tickers = bull_tickers
        self.fred_strings = fred_strings
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth

    def compare_equal(self):
        '''
        Compares the return and vol of an equally weighted portfolio with tickers above with the optimized portfolio. Returns a data frame
        '''
        num_assets = len(self.bull_tickers)
        rmat = returns_matrix(self.bull_tickers, self.start_date, self.end_date)
        p_vec = returns_vector(rmat, np.array(num_assets*[1./num_assets,]))
        p_ret_equal = float(np.mean(p_vec)*252)
        p_vol_equal = float(np.std(p_vec)*np.sqrt(252))

        opt_ret = float(np.mean(self.opt_daily)*252)
        opt_vol = float(np.std(self.opt_daily)*np.sqrt(252))

        final_df = pd.DataFrame({"Metric (Annual)":["Return", "Volatiltiy"], "Optimized_Port":[opt_ret, opt_vol], "Equal_Weight":[p_ret_equal, p_vol_equal]})
        final_df=final_df.set_index("Metric (Annual)")
        return final_df

    def compare_spy(self):
        '''
        compares the return and vol of spy to optimized portfolio
        '''
        spy_returns = returns_matrix(["SPY"], self.start_date, self.end_date)["SPY"]
        spy_annual_ret = float(np.mean(spy_returns)*252)
        spy_annual_vol = float(np.std(spy_returns)*np.sqrt(252))

        opt_ret = float(np.mean(self.opt_daily)*252)
        opt_vol = float(np.std(self.opt_daily)*np.sqrt(252))

        final_df = pd.DataFrame({"Metric (Annual)":["Return", "Volatiltiy"], "Optimized_Port":[opt_ret, opt_vol], "SPY":[spy_annual_ret, spy_annual_vol]})
        final_df=final_df.set_index("Metric (Annual)")
        return final_df

    def analyze_volatile_periods(self):
        all_dfs = {}
        dates = ["2009-01-26","2009-03-26","2010-05-26","2010-07-26", "2011-07-26", "2011-09-26", "2018-01-29", "2018-03-29", "2018-09-27","2018-12-27", "2020-02-15", "2020-04-15"]
        num_assets = len(self.bull_tickers)
        rmat = returns_matrix(self.bull_tickers, self.start_date, self.end_date)
        p_vec_equal = returns_vector(rmat, np.array(num_assets*[1./num_assets,]))
        spy_returns = returns_matrix(["SPY"], self.start_date, self.end_date)["SPY"]
        count = 0
        for i in range(0,len(dates),2):
            temp_opt,temp_pvec_equal,temp_spy = self.opt_daily.loc[pd.to_datetime(dates[i]):pd.to_datetime(dates[i+1])], p_vec_equal.loc[pd.to_datetime(dates[i]):pd.to_datetime(dates[i+1])], spy_returns.loc[pd.to_datetime(dates[i]):pd.to_datetime(dates[i+1])]
            temp_opt_mean, temp_pvec_mean, temp_spy_mean = float(np.mean(temp_opt)*252), float(np.mean(temp_pvec_equal)*252), float(np.mean(temp_spy)*252)
            temp_opt_vol, temp_pvec_vol, temp_spy_vol = float(np.std(temp_opt)*np.sqrt(252)), float(np.std(temp_pvec_equal)*np.sqrt(252)), float(np.std(temp_spy)*np.sqrt(252))
            temp_opt_var, temp_pvec_var, temp_spy_var = VaR(temp_opt), VaR(temp_pvec_equal), VaR(temp_spy)
            df_ind = "Metrics "+str(dates[i]) + " - " + str(dates[i+1])
            temp_df = pd.DataFrame({df_ind:["Return", "Volatilty", "VaR"], "Optimized_Port":[temp_opt_mean, temp_opt_vol, temp_opt_var], "Equal_Weight_Port":[temp_pvec_mean, temp_pvec_vol, temp_pvec_var], "SPY":[temp_spy_mean, temp_spy_vol, temp_spy_var]})
            temp_df = temp_df.set_index(df_ind)
            all_dfs[count] = temp_df
            count+=1
        return all_dfs



v_trial = Vol_Outputs(end_date="2020-06-20")

#
#
equal_compare = v_trial.compare_equal()
spy_compare = v_trial.compare_spy()

equal_compare
spy_compare
v_trial.analyze_volatile_periods()


# VaR(v_trial.opt_daily)
#
# all_dfs = v_trial.analyze_volatile_periods()
# all_dfs[5]

# ############ Finding most volatiltiy in 2 months. 21 trading days in 2 months ###############
# SPY_data=pdr.DataReader("SPY", "yahoo", "2008-11-20")
# SPY_returns = (np.log(SPY_data.Close) - np.log(SPY_data.Close.shift(1)))[1:]
# top_10_greatest_vols = []
# for i in range(0,len(SPY_returns), int(252/6)):
#     temp_2month = SPY_returns[i:int(i+(252/6))]
#     temp_annual_vol = np.std(temp_2month)*np.sqrt(252)
#     temp_tup = (temp_annual_vol, temp_2month.index[0])
#     if len(top_10_greatest_vols) < 10:
#         top_10_greatest_vols.append(temp_tup)
#     else:
#         for j in range(len(top_10_greatest_vols)):
#             if top_10_greatest_vols[j][0]<temp_tup[0]:
#                 top_10_greatest_vols[j] = temp_tup
#                 break
#
#
#
# top_10_greatest_vols
# i
# temp_2month
#  Timestamp('2009-01-26 00:00:00'))
