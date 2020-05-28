import pandas as pd
import pandas_datareader as pdr
import numpy as np
import os
from Data.Inputs import volClass
from pandas_datareader import wb

os.getcwd()

trial_vol = volClass.Vol_Data("2000-01-01")
option_spreads = pd.read_csv("Data/Inputs/FRED_data/high_yield_index_option_adjusted_spread.csv")
option_spreads
trial_vol.weekly_spy_volume()

############### CONSIDER important exchange rates like chinese yuan USD, yen USD, EURO USD, etc. ###################
