import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import scipy.optimize as sco

def returns_matrix(tickers, start_date, end_date):
    #NEED TO CHANGE FOR FACT THAT TICKERS HAVE VERY DIFFERENT DATA THAT THEY GO BACK TO
    '''
    Creates a daily returs dataframe where each column is a different stock
    '''
    rmat = []
    for tick in tickers:
        try:
            temp_df = pdr.DataReader(tick, "yahoo", start_date, end_date)
        except:
            print(str(tick) + " Data doesn't go back that far.")
            continue

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
    return -p_ret/p_sd

def max_sharpe_ratio(rmat):
    '''
    Uses optimization techniques to output the optimal weight vector for max sharpe ratio
    rmat: returns matrix where each column is a different stock's returns
    Output: A dictionary of different values. The x key contains the weights
    '''
    num_assets = len(rmat.columns)
    args = (rmat)
    constraints = constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) #make sure weights sum to 1
    bound = (0,1)
    bounds = tuple(bound for asset in range(num_assets))
    initial_guess = np.array(num_assets*[1./num_assets,]) #first guess is equal weight
    result = sco.minimize(neg_sharpe_ratio, initial_guess , args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
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
    bear_tickers = ["GDX", "VXX", "SHY", "XLU", "VHT", "UUP", "PNQI"]):
        self.bull_tickers = bull_tickers
        self.bear_tickers = bear_tickers
        self.start_date = start_date
        self.end_date = end_date
        self.daily_bull_rmat = returns_matrix(self.bull_tickers, self.start_date, self.end_date)
        self.daily_bear_rmat = returns_matrix(self.bear_tickers,self.start_date, self.end_date)

    def bull_weekly_return(self):
        trial = returns_vector(self.daily_bull_rmat, np.array([1/len(self.bull_tickers) for i in self.bull_tickers]))
        final = weekly_data(trial)
        return final

    def bull_optimal_weights(self):
        return max_sharpe_ratio(self.daily_bull_rmat)






trial_port = Portfolio(start_date="2019-01-01")
weights = trial_port.bull_optimal_weights()["x"]
rmat = trial_port.daily_bull_rmat
neg_sharpe_ratio(weights, rmat)
portfolio_return(rmat, weights)*252
np.sqrt(portfolio_variance(rmat, weights))*np.sqrt(252)
weights
#portfolio_variance(rmat, np.array([1/len(trial_port.bull_tickers) for i in trial_port.bull_tickers]))

### trying functions

##################### This indexing is important and how I will create backtest by isolating each week ##################
week_starts = weekly_r.index
rmat.loc[week_starts[0]:week_starts[1]].drop(week_starts[1], axis = 0)
############################################################################################################


rmeans=np.array(rmat.mean(axis = 0))
weights = np.array([1/len(trial_port.bull_tickers) for i in trial_port.bull_tickers])

weights.shape
rmeans.shape
len(rmat.iloc[0:,0])
num_assets=len(rmat.columns)
w = np.array(num_assets*[1./num_assets,])
portfolio_return(rmat, w)
neg_sharpe_ratio(rmat, w)
portfolio_variance(rmat,w)
