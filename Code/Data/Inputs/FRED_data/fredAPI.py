# pip install fredapi
# you should be good to run it thru my api_key
from fredapi import Fred
import pandas as pd
import pandas_datareader.data as pdr
fred = Fred(api_key = '81fb59583aa03d2ce139e065d694c299')



stocks = ["SPY", "AAPL", "MSFT"]
input_df = pd.DataFrame({})
for id in stocks:
    temp_df = pdr.DataReader(id, "yahoo", start = "2018-01-01")
    pd.concat(input_df, pd.DataFrame(temp_df.Volume))
ptype(temp_df.Volume)
temp_df.head()

# input variables - looking for weekly
# stock market volume
# GDP Weekly
# unemployment
# interest rates
# bank data (repo rates)
# housing market data
# commodity data
# China / US Daily FX rates
# Helpful to get data from financial sector

# data we have and can use
# initial unemployment claims seasonally adjusted - icsa - weekly
# continued claims - ccsa - weekly
# Crude oil prices - DCOILBRENTEU - daily
# High yield index option adjusted spread - BAMLH0A0HYM2 - daily
# Gold fixing Price - GOLDAMGBD228NLBM - daily
# Corporate bond yields AAA - DAAA - daily
# Overnight AA financial commercial paper interest rate - RIFSPPFAAD01NB - daily
# Euro high yield index-option adjusted - BAMLHE00EHYIOAS - daily
# China/US FX rates - DEXCHUS - Daily
# US/EURO EXchange rates - DEXUSEU - Daily
# 10 year - 3 month interest rate - T10Y3M - Daily
# private sector financial EM data - BAMLEMFSFCRPITRIV - DAILY
# Trade weighted dollar to foreign goods and currencies - DTWEXAFEGS - DAILY, ONLY GOES BACK TO 2015


df = fred.get_series_info("BAMLHE00EHYIOAS")
df["frequency_short"]
fred.search('housing')
print(df)
