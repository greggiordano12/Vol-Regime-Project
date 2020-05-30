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


# initial unemployment claims seasonally adjusted
icsa = pd.DataFrame(fred.get_series("ICSA", observation_start = "2000-01-01"))
print(icsa.head())

# Crude oil prices
crudeOilEuro = pd.DataFrame(fred.get_series("DCOILBRENTEU", observation_start = "2000-01-01"))
print(crudeOilEuro.head())

# High yield index option adjusted spread
highYieldOptionSpread = pd.DataFrame(fred.get_series("BAMLH0A0HYM2", observation_start = "2000-01-01"))
print(highYieldOptionSpread.head())

# Gold fixing Price
goldFixingPrice = pd.DataFrame(fred.get_series("GOLDAMGBD228NLBM", observation_start = "2000-01-01"))
print(goldFixingPrice.head())

# Corporate bond yields AAA
corporateBondYield = pd.DataFrame(fred.get_series("DAAA", observation_start = "2000-01-01"))
print(corporateBondYield.head())

# Overnight AA financial commercial paper interest rate
overnightPaperInterestRate = pd.DataFrame(fred.get_series("RIFSPPFAAD01NB", observation_start = "2000-01-01"))
print(overnightPaperInterestRate.head())

# Euro high yield index-option adjusted
euroHighYieldOptionSpread = pd.DataFrame(fred.get_series("BAMLHE00EHYIOAS", observation_start = "2000-01-01"))
print(euroHighYieldOptionSpread.head())


fred.search('volume')
