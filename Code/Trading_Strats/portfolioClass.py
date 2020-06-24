import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import scipy.optimize as sco
from Code.Data.Inputs import volClass
from Code.Data import rfClass


def returns_matrix(tickers, start_date, end_date):
    #NEED TO CHANGE FOR FACT THAT TICKERS HAVE VERY DIFFERENT DATA THAT THEY GO BACK TO#
    '''
    Creates a daily returs dataframe where each column is a different stock
    '''
    min_week = pd.to_datetime("1900-01-01") #sets initial week as today
    rmat = []
    data_dic = {}
    count = 0
    for tick in tickers:
        temp_df = pdr.DataReader(tick, "yahoo", start_date, end_date)
        temp_dates = temp_df.index
        data_dic[count] = temp_df

        if temp_dates[0] > min_week:
            min_week = temp_dates[0]
        count +=1
    for i in range(len(tickers)):
        temp_df = data_dic[i]
        temp_dates = temp_df.index
        if temp_dates[0] != min_week:
            drop_index = list(temp_dates).index(min_week)
            temp_df = temp_df.drop(temp_dates[:drop_index])
        temp_returns = np.log(temp_df.Close) - np.log(temp_df.Close.shift(1))
        temp_returns = list(temp_returns[1:])
        rmat = rmat + [temp_returns]
    rmat = np.array(rmat)
    dates = temp_df.index
    rdf=pd.DataFrame({tickers[i]:rmat[i] for i in range(len(tickers))})
    rdf.index = dates[1:]
    return rdf


def returns_vector(rmat,weights):
    '''
    Returns a portfolio's return given rmat that is a returns matrix with each column a different ticker and weights is
    a standard weight vector (not transposed)
    '''
    rvec = rmat.dot(weights)
    return rvec

def portfolio_return(rmat, weights):
    rmeans = np.array(rmat.mean(axis=0))
    return float(rmeans.dot(weights))

def portfolio_variance(rmat, weights):
    '''
    Takes in a returns matrix with each column as a different ticker and weights vector
    returns: wt * cov_mat * w
    '''
    cov_mat = rmat.cov()
    wt = weights.transpose()
    wt_cov = wt.dot(cov_mat)
    return float(wt_cov.dot(weights))

def neg_sharpe_ratio(weights, rmat):
    '''
    Returns weekly sharpe ratio. Makes negative for minimization function
    '''
    p_ret = portfolio_return(rmat, weights) * len(rmat.iloc[0:,0])
    p_sd = np.sqrt(portfolio_variance(rmat, weights)) * np.sqrt(len(rmat.iloc[0:,0]))
    return -(p_ret/p_sd)

def vol(weights, rmat):
    p_sd = np.sqrt(portfolio_variance(rmat, weights)) * np.sqrt(len(rmat.iloc[0:,0]))
    return p_sd

def ret(weights, rmat):
    p_ret = portfolio_return(rmat, weights) * len(rmat.iloc[0:,0])
    return -p_ret

def max_sharpe_ratio(rmat):
    '''
    Uses optimization techniques to output the optimal weight vector for max sharpe ratio
    rmat: returns matrix where each column is a different stock's returns
    Output: A dictionary of different values. The x key contains the weights
    '''
    num_assets = len(rmat.columns)
    args = (rmat)
    constraints = constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) #make sure weights sum to 1
    bound = (-1,1)
    bounds = tuple(bound for asset in range(num_assets))
    initial_guess = np.array(num_assets*[1./num_assets,]) #first guess is equal weight
    result = sco.minimize(vol, initial_guess , args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints) #just need to change first paramter
    return result

def weekly_data(returns_vec):
    '''
    Takes in a returns vector indexed with daily dates and returns a weekly dataframe that has the weekly return and vol

    '''
    weekly_returns, weekly_sd, week_start_dates= [], [], []
    dates = returns_vec.index
    i = 0
    while i<len(dates):
        temp_week = []
        temp_week_day = []
        temp_week_day.append(dates[i].weekday()) # adding start of the week day
        week_start_dates.append(dates[i]) # adding the start date to dates list
        temp_week.append(returns_vec[i]) # add the return of the start of the week
        i+=1 # increment up
        if i < len(dates):
            while (dates[i].weekday() not in temp_week_day) and (dates[i].weekday() != 0):
                # This while loop gathers data for current week. Adjusts for weeks that have Mondays off.
                temp_week.append(returns_vec[i])
                temp_week_day.append(dates[i].weekday())
                i+=1
                if i >= len(dates):
                    break
        temp_data = np.array(temp_week)
        temp_mean_return = temp_data.mean() * len(temp_data)
        temp_sd = np.sqrt(temp_data.var()) * np.sqrt(len(temp_data))
        weekly_returns.append(temp_mean_return)
        weekly_sd.append(temp_sd)
    weekly_df = pd.DataFrame({"Week":week_start_dates, "Returns":weekly_returns, "Volatiltiy":weekly_sd})
    weekly_df = weekly_df.set_index("Week")
    return weekly_df



class Portfolio:
    def __init__(self, start_date, end_date = None, bull_tickers = ["PNQI", "SPY", "XLK", "SPXL", "TQQQ", "XLY", "XLF"],
    bear_tickers = ["GDX", "^VIX", "SHY", "UUP", "PNQI"], regime_predict = None):
        self.bull_tickers = bull_tickers
        self.bear_tickers = bear_tickers
        self.start_date = start_date
        self.end_date = end_date
        self.regime_predict = regime_predict
        ####### Make bull and bear time frames equal #############
        daily_bull_rmat = returns_matrix(self.bull_tickers, self.start_date, self.end_date)
        daily_bear_rmat = returns_matrix(self.bear_tickers,self.start_date, self.end_date)
        bull_dates = daily_bull_rmat.index
        bear_dates = daily_bear_rmat.index
        if bull_dates[0] > bear_dates[0]:
            drop_index = list(bear_dates).index(bull_dates[0])
            daily_bear_rmat = daily_bear_rmat.drop(bear_dates[:drop_index])
        elif bull_dates[0] < bear_dates[0]:
            drop_index = list(bull_dates).index(bear_dates[0])
            daily_bull_rmat = daily_bull_rmat.drop(bull_dates[:drop_index])
        self.daily_bull_rmat = daily_bull_rmat
        self.daily_bear_rmat = daily_bear_rmat
        #################################################################
        #self.weekly_bull_rmat = weekly_data(self.daily_bull_rmat)

    def bull_weekly_return(self):
        trial = returns_vector(self.daily_bull_rmat, np.array([1/len(self.bull_tickers) for i in self.bull_tickers]))
        final = weekly_data(trial)
        return final

    def bull_optimal_weights(self):
        return max_sharpe_ratio(self.daily_bull_rmat)["x"]


    def weekly_optimization(self):
        ##################### CHECK IF EVERYTHING IS BEING DONE RIGHT!
        ## must match up regime predictions with the returns dfs
        all_weights_bull = []
        all_weights_bear = []
        regime_df = self.regime_predict.all_predictions_df
        probs_df = self.regime_predict.all_prob
        dates = self.bull_weekly_return().index
        ######################### Need to adjust here!!!!
        drop_index = list(regime_df.index).index(dates[1])
        if drop_index == 0:
            regime_df = regime_df.drop(regime_df.index[:drop_index])
        else:
            regime_df = regime_df.drop(regime_df.index[:drop_index])
            probs_df = probs_df.drop(probs_df.index[:(drop_index-1)])

        daily_returns = []
        for i in range(1,len(dates)):
            if i == len(dates) - 1:
                temp_probs = probs_df.loc[dates[i]] #array of probability of low, med, high vol
                #Need to get lagged data to find this week's optimal weights#
                temp_bull_portfolio_lag = self.daily_bull_rmat.loc[dates[i-1]:]
                temp_bear_portfolio_lag = self.daily_bear_rmat.loc[dates[i-1]:]

                temp_bull_portfolio = self.daily_bull_rmat.loc[dates[i]:]
                temp_bear_portfolio = self.daily_bear_rmat.loc[dates[i]:]
            else:
                temp_probs = probs_df.loc[dates[i]] #array of probability of low, med, high vol
                temp_bull_portfolio_lag = self.daily_bull_rmat.loc[dates[i-1]:dates[i]].drop(dates[i], axis=0)
                temp_bear_portfolio_lag = self.daily_bear_rmat.loc[dates[i-1]:dates[i]].drop(dates[i], axis=0)

                temp_bull_portfolio = self.daily_bull_rmat.loc[dates[i]:dates[i+1]].drop(dates[i+1], axis=0)
                temp_bear_portfolio = self.daily_bear_rmat.loc[dates[i]:dates[i+1]].drop(dates[i+1], axis=0)

            num_assets = len(self.bull_tickers)
            temp_bull_optweights = np.array(num_assets*[1./num_assets,])
            temp_bear_optweights = max_sharpe_ratio(temp_bear_portfolio_lag)["x"] #change back to temp_bear_portfolio_lag

            all_weights_bull = all_weights_bull + [temp_bull_optweights.tolist()]
            all_weights_bear = all_weights_bear + [temp_bear_optweights.tolist()]

            bull_returns_vec = returns_vector(temp_bull_portfolio, temp_bull_optweights) * temp_probs[0]
            bear_returns_vec = returns_vector(temp_bear_portfolio, temp_bear_optweights) * temp_probs[2]# changed back to bear
            med_returns_vec = returns_vector(temp_bear_portfolio, temp_bear_optweights) * temp_probs[1] # changed back to bear

            weekly_returns_vec = (bull_returns_vec.add(bear_returns_vec)).add(med_returns_vec)
            daily_returns+= list(weekly_returns_vec)

        print(len(self.daily_bull_rmat.index[5:]))
        print(len(daily_returns))
        daily_rmat = pd.DataFrame({"Date":self.daily_bull_rmat.index[len(self.daily_bull_rmat.index)-(len(daily_returns)):], "Return":daily_returns})
        daily_rmat = daily_rmat.set_index("Date")

        bull_weights_df = pd.DataFrame(np.array(all_weights_bull))
        bull_weights_df.index = dates[1:]
        bear_weights_df = pd.DataFrame(np.array(all_weights_bear))
        bear_weights_df.index = dates[1:]
        bull_weights_df.to_csv("bull_weights.csv")
        bear_weights_df.to_csv("bear_weights.csv")
        return daily_rmat


##### REAL TEST ######## Run all code below to test
# fred_s = ["DCOILBRENTEU" ,"BAMLH0A0HYM2", "GOLDAMGBD228NLBM","DAAA","RIFSPPFAAD01NB","BAMLHE00EHYIOAS", "DEXCHUS", "DEXUSEU", "T10Y3M", "BAMLEMFSFCRPITRIV"]
# trial_vol = volClass.Vol_Data("2007-12-21", "2020-06-14", fred_strings = fred_s)
# trial_regime_predict = rfClass.Regime_Predict(trial_vol)
#
# returns_matrix(bear_tickers,start_date="2010-12-20", end_date="2020-06-07")
#
# trial_port = Portfolio(start_date="2008-01-01", end_date="2020-06-14",bull_tickers = ["PNQI", "SPY", "XLK", "SPXL", "XLY", "XLF","SHY"], bear_tickers= ["PNQI", "SPY", "XLK", "SPXL", "XLY", "XLF","SHY"] ,regime_predict=trial_regime_predict)
# trial_port.daily_bull_rmat.to_csv("bull_portfolio_returns.csv")
# opt_daily  = trial_port.weekly_optimization()
#
# annual_vol = np.sqrt(opt_daily.var()) * np.sqrt(252)
# annual_return = np.mean(opt_daily.mean())*252
#
# annual_return
# annual_vol
# ((opt_daily+1).cumprod()-1).plot()
#
#
#
#
# opt_daily.to_csv("Optimization_returns.csv")
# trial_port.daily_bear_rmat.to_csv("bear_portfolio_returns.csv")
# trial_port.daily_bull_rmat.to_csv("bull_portfolio_returns.csv")
# trial_port.regime_predict.all_prob.to_csv("regime_probs.csv")
#
#
# bear_tickers= ["PNQI", "SPY", "XLK", "SPXL", "TQQQ", "XLY", "XLF","SHY"]
