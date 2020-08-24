import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import scipy.optimize as sco
import os

regime_probs=pd.read_csv("regime_probs.csv")
regime_probs = regime_probs.set_index("Week")
bear_port_returns=pd.read_csv("bear_portfolio_returns.csv")
bear_port_returns = bear_port_returns.set_index("Date")
bull_portfolio_returns = pd.read_csv("bull_portfolio_returns.csv")
bull_port_returns=bull_portfolio_returns.set_index("Date")

((bull_port_returns+1).cumprod()-1).plot()

((bear_port_returns+1).cumprod()-1).plot()

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
    result = sco.minimize(neg_sharpe_ratio, initial_guess , args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints) #just need to change first paramter
    return result

SPY_data = bull_port_returns["SPY"]
for i in bear_port_returns.columns:
    print(SPY_data.corr(bear_port_returns[i]))


bear_weights=max_sharpe_ratio(bear_port_returns)["x"]
bear_weights
portfolio_return(bear_port_returns, bear_weights)*252
np.sqrt(portfolio_variance(bear_port_returns, bear_weights))*np.sqrt(252)

bull_weights = max_sharpe_ratio(bull_port_returns)["x"]
portfolio_return(bull_port_returns, bull_weights)*252
np.sqrt(portfolio_variance(bull_port_returns, bull_weights))*np.sqrt(252)

all_bull_weights = pd.read_csv("bull_weights.csv")
all_bull_weights["Week"] = pd.to_datetime(all_bull_weights["Week"])
all_bull_weights = all_bull_weights.set_index("Week").dropna()
all_bull_weights.columns = bull_port_returns.columns
all_bull_weights.loc[pd.to_datetime("2019-11-01"):pd.to_datetime("2020-01-10")]

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

opt_returns = pd.read_csv("Optimization_returns.csv")
opt_returns["Date"] = pd.to_datetime(opt_returns["Date"])
opt_returns=opt_returns.set_index("Date")
opt_returns=opt_returns.dropna()
((opt_returns+1).cumprod()-1).plot()

or_weekly=weekly_data(opt_returns["Return"])
or_weekly.tail(20)


SPY_data.index = pd.to_datetime(SPY_data.index)

spy_weekly = weekly_data(SPY_data)
((spy_weekly["Returns"]+1).cumprod()-1).plot()

annual_vol = np.sqrt(SPY_data.var()) * np.sqrt(252)
annual_return = np.mean(SPY_data.mean())*252
annual_vol
annual_return


corona_virus = or_weekly.loc[pd.to_datetime("2018-10-01"):pd.to_datetime("2019-01-10")]["Returns"]
spy_weekly_corona = spy_weekly.loc[pd.to_datetime("2018-10-01"):pd.to_datetime("2019-01-10")]["Returns"]
np.sqrt(spy_weekly_corona.var())*np.sqrt(52)
np.mean(spy_weekly_corona)*52

np.sqrt(corona_virus.var())*np.sqrt(52)
np.mean(corona_virus)*52
###############################################################################################


def returns_matrix(tickers, start_date, end_date):
    #NEED TO CHANGE FOR FACT THAT TICKERS HAVE VERY DIFFERENT DATA THAT THEY GO BACK TO#
    '''
    Creates a daily returns dataframe where each column is a different stock
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

rmat = returns_matrix(tickers = ["SPY","FXI", "VXX", "GLD", "USO","BTAL", "EWS","EWH"], start_date="2015-01-01", end_date="2020-07-06")
rmat.corr()
np.mean(rmat["SPY"])*(252)
weekly_data(rmat["EWS"])
np.corrcoef(ews, ewh)[0][1]
import matplotlib.pyplot as plt
ews = rmat["EWS"]
ewh = rmat["EWH"]
spy = rmat["SPY"]
corrs = []
for i in range(252,len(rmat["EWS"])):
    temp_ews = ews[i-252:i]
    temp_ewh = ewh[i-252:i]
    corrs.append(np.corrcoef(temp_ews,temp_ewh)[0][1])


corr_df = pd.DataFrame({"Date":rmat.index[252:], "Corr":corrs})
corr_df = corr_df.set_index("Date")
corr_df.plot()
plt.title("Yearly Moving Correlation Hong Kong/Singapore")
plt.ylabel("Correlation")
plt.show()
np.std(corrs)
