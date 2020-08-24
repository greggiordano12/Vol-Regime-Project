import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import scipy.optimize as sco
from Code.Data.Inputs import volClass
from Code.Data import rfClass
from Code.Trading_Strats import portfolioclass
import scipy.stats as st

class portfolioAnalytics:
    def __init__(self, rvec):
        self.rvec = rvec

    def VaR(self,conf_level = .95):
        vol = np.std(self.rvec)
        final = -(st.norm.ppf(.95))*vol*np.sqrt(252)
        return final





-st.norm.ppf(.95)
