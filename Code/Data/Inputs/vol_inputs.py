import pandas as pd
import pandas_datareader as pdr
import numpy as np
import os
from Data.Inputs import volClass
from Data.Inputs.FRED_data import convertFred
from pandas_datareader import wb
from sklearn.datasets import load_iris
from sklearn import preprocessing



trial_vol = volClass.Vol_Data("2000-01-01")
trial_vol.weekly_vix()

######## FRED DATA IMPORTS ####################
option_spreads_US_csv = pd.read_csv("Data/Inputs/FRED_data/high_yield_index_option_adjusted_spread.csv")
euro_high_yield_csv = pd.read_csv("Data/Inputs/FRED_data/euro_high_yield.csv")
corporate_bond_csv = pd.read_csv("Data/Inputs/FRED_data/corporate_bond_yields.csv")
gold_fix_US_csv = pd.read_csv("Data/Inputs/FRED_data/gold_fixing_price_US.csv")
brent_europe_csv = pd.read_csv("Data/Inputs/FRED_data/brent_europe.csv")

####### Convert FRED Data ######################
option_spreads_US = convertFred.FredConvert(option_spreads_US_csv).weekly_data()
option_spreads_US
convertFred.ConvertFred()

############### CONSIDER important exchange rates like chinese yuan USD, yen USD, EURO USD, etc. ###################
