# pip install fredapi
# you should be good to run it thru my api_key
from fredapi import Fred
import pandas as pd
fred = Fred(api_key = '81fb59583aa03d2ce139e065d694c299')

# input variables - looking for weekly
# stock market volume
# GDP Weekly
# unemployment
# interest rates
# bank data (repo rates)
# housing market data
# commodity data


data = fred.get_series_latest_release('RPONTSYD')
data.df = pd.DataFrame(data)
data.df.columns = ['Overnight Repo']
print(data.df.head())
print(data.df.columns)

# Do we use seasonally adjusted data or not?
# initial unemployment claims seasonally adjusted
icsa = fred.get_series("ICSA", observation_start = "2000-01-01")
type(icsa)
icsa.df = pd.DataFrame(icsa)
print(icsa.df.head())


# High yield index option adjusted spread
highYieldOptionSpread = pd.DataFrame(fred.get_series("BAMLH0A0HYM2", observation_start = "2000-01-03"))
print(highYieldOptionSpread.head())

# Gold fixing Price
goldFixingPrice = pd.DataFrame(fred.get_series("GOLDAMGBD228NLBM", observation_start = "2000-01-04"))
print(goldFixingPrice.head())

# Corporate bond yields AAA


# Overnight AA financial commercial paper interest rate
overnightPaperInterestRate = pd.DataFrame(fred.get_series("RIFSPPFAAD01NB", observation_start = "2000-01-01"))
print(overnightPaperInterestRate.head())


fred.search('BAMLH0A0HYM2')
