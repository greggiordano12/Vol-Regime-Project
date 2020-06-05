import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import scipy.optimize as sco

class Portfolio:
    def __init__(self, tickers, start_date, end_date = None):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def returns_matrix(self):
        '''
        Creates a returs matrix where each row is a different stock's set of returns
        '''
        rmat = []
        for tick in self.tickers:
            try:
                temp_df = pdr.DataReader(tick, "yahoo", self.start_date, self.end_date)
            except:
                print(str(tick) + " Data doesn't go back that far.")
                continue

            temp_returns = np.log(temp_df.Close) - np.log(temp_df.Close.shift(1))
            temp_returns = list(temp_returns[1:])
            rmat = rmat + [temp_returns]
        rmat = np.array(rmat)
        dates = temp_df.index
        rdf=pd.DataFrame({self.tickers[i]:rmat[i] for i in range(len(self.tickers))})
        rdf.index = dates[1:]
        return rdf

    ##### Incorporate Jupyter Code ##########

    def portfolio_performance(wt, returns_mat):
        return portfolio_variance(wt,returns_mat), portfolio_return(wt,returns_mat)

    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        return -(p_ret - risk_free_rate) / p_var

    def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix, risk_free_rate)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = (-2,2) # change to negative becasue this is bounds of weights
        bounds = tuple(bound for asset in range(num_assets))
        result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                            method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    ##############################################################################


trial_port = Portfolio(["SPY", "PNQI", "XLK", "SPXL"], start_date="2010-01-01")
rmat=trial_port.returns_matrix()
