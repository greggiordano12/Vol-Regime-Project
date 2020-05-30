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
crudeOilEuro.shape
ce_index = crudeOilEuro.index

# High yield index option adjusted spread
highYieldOptionSpread = pd.DataFrame(fred.get_series("BAMLH0A0HYM2", observation_start = "2000-01-01"))
hy_index = highYieldOptionSpread.index

highYieldOptionSpread.shape

ce_index
drop_indicies = hy_index.difference(ce_index)
drop_indicies
highYieldOptionSpread.drop(drop_indicies)

# Gold fixing Price
goldFixingPrice = pd.DataFrame(fred.get_series("GOLDAMGBD228NLBM", observation_start = "2000-01-01"))
print(goldFixingPrice.head())
goldFixingPrice.shape
# Corporate bond yields AAA
corporateBondYield = pd.DataFrame(fred.get_series("DAAA", observation_start = "2000-01-01"))
print(corporateBondYield.head())
corporateBondYield.shape

# Overnight AA financial commercial paper interest rate
overnightPaperInterestRate = pd.DataFrame(fred.get_series("RIFSPPFAAD01NB", observation_start = "2000-01-01"))
print(overnightPaperInterestRate.head())
overnightPaperInterestRate.shape

# Euro high yield index-option adjusted
euroHighYieldOptionSpread = pd.DataFrame(fred.get_series("BAMLHE00EHYIOAS", observation_start = "2000-01-01"))
euroHighYieldOptionSpread.columns = ["BAMLHE00EHYIOAS"]
euroHighYieldOptionSpread.head()
euroHighYieldOptionSpread.shape


fred.search('volume')


pd.concat([highYieldOptionSpread, euroHighYieldOptionSpread], axis=1)
