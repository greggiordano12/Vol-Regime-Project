import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import scipy.optimize as sco

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

annual_return
annual_vol
