import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
import datetime
import calendar
from fredapi import Fred

class Vol_Data:

    def __init__(self, start_date, end_date=None, fred_strings = None):
        self.start_date = start_date
        self.end_date = end_date
        #create attributes for rest of class
        self.vix_df = pdr.DataReader("^VIX", "yahoo", start_date, end_date)
        self.dates = self.vix_df.index
        self.vix_close = self.vix_df.Close
        self.vix_close_arr = np.array(self.vix_close)
        self.spy_df = pdr.DataReader("SPY", "yahoo", start_date, end_date)
        self.spy_volume = self.spy_df.Volume
        self.spy_volume_arr = np.array(self.spy_volume)
        self.fred_strings = fred_strings


    def line_plot(self):
        #Provides a line chart of the vix time-series w 20 and 30 hor lines
        hor_30 = [30 for i in self.vix_close]
        hor_20 = [20 for i in self.vix_close]
        plt.plot(self.vix_close)
        plt.plot(self.vix_df.index, hor_30)
        plt.plot(self.vix_df.index, hor_20)
        plt.xlabel("Year")
        plt.ylabel("Closing Price")
        plt.title("Time Series of VIX Close")
        plt.show()

    def hist_plot(self):
        #Provides a histogram of all closing prices
        plt.hist(self.vix_close_arr, bins = 40)
        plt.xlabel("Closing Price")
        plt.ylabel("Number of Days")
        plt.title("Distribution of VIX Closing Price")
        plt.show()

    def gen_regime_data(self):
        #Use this method to populate low_vol_dates, med_vol_dates, high_vol_dates and the vol_regime_df
        dates = self.vix_df.index
        vol_regime = []
        for i in range(len(self.vix_close)):
            if(self.vix_df.Close[i]<20):
                vol_regime.append(1)
            elif(self.vix_df.Close[i]>30):
                vol_regime.append(3)
            else:
                vol_regime.append(2)
        vol_regime_df = pd.DataFrame({"Vol Regime":vol_regime})
        vol_regime_df.index = dates
        return vol_regime_df


    def weekly_stats(self, data_df):
        #outputs weekly_data list that is the average weekly data from the given data array. Also outputs
        #week_start_dates that is a list of the first day of each week for the weekly data (used to identify start of weeks)
        # data: meant to be an array defined in constructor that matches self.dates in index

        weekly_data = []
        week_start_dates = []
        dates = self.dates
        data_dates = data_df.index
        drop_indicies = data_dates.difference(dates)
        data_df = data_df.drop(drop_indicies)
        #### check to make sure market data exceeds fred data
        new_dates = dates.difference(data_df.index)
        self.dates = self.dates.drop(new_dates)
        dates = self.dates
        ###################################################
        data = data_df.iloc[0:,0]
        i = 0
        while i<len(dates):
            temp_week = []
            temp_week_day = []
            temp_week_day.append(dates[i].weekday()) # adding start of the week day
            week_start_dates.append(dates[i]) # adding the start date to dates list
            temp_week.append(data[i]) # add the closing price of the start of the week
            i+=1 # increment up
            if i < len(dates):
                while (dates[i].weekday() not in temp_week_day) and (dates[i].weekday() != 0):
                    # This while loop gathers data for current week. Adjusts for weeks that have Mondays off.
                    temp_week.append(data[i])
                    temp_week_day.append(dates[i].weekday())
                    i+=1
                    if i >= len(dates):
                        break
            temp_avg_data = sum(temp_week)/len(temp_week)
            weekly_data.append(temp_avg_data)

        return weekly_data, week_start_dates

    def weekly_vix(self):
        #leverages weekly_stats to produce average weekly volatiltiy regimes. >30 is 3, <20 is 1, <30 and >20 is 2
        #outputs a dataframe that is indexed with the date of the start of the week and the data is the 1,2,3s representing
        #the weekly vol regime
        avg_vol_data, week_start_dates = self.weekly_stats(pd.DataFrame(self.vix_close))
        weekly_vol_regimes = []
        for temp_avg_vol in avg_vol_data:
            if temp_avg_vol <20:
                weekly_vol_regimes.append(1)
            elif temp_avg_vol > 30:
                weekly_vol_regimes.append(3)
            else:
                weekly_vol_regimes.append(2)

        week_dic = {"Date":week_start_dates}
        weekly_vol_regimes = pd.DataFrame({"Week":week_start_dates,"Weekly_Vol":weekly_vol_regimes})
        weekly_vol_regimes = weekly_vol_regimes.set_index("Week")
        return weekly_vol_regimes

    def weekly_spy_volume(self):
        #outputs dataframe of weekly spy volume averages with index as start of week
        avg_volume_data, week_start_dates = self.weekly_stats(pd.DataFrame(self.spy_volume))
        week_dic = {"Date":week_start_dates}
        weekly_volume_avg = pd.DataFrame({"Week":week_start_dates,"Weekly_Volume":avg_volume_data})
        weekly_volume_avg = weekly_volume_avg.set_index("Week")
        return weekly_volume_avg

    def weekly_fred_data(self):
        fred = Fred(api_key = '81fb59583aa03d2ce139e065d694c299')
        fred_ids = self.fred_strings
        input_df = self.weekly_spy_volume()
        for id in fred_ids:
            temp_df =  pd.DataFrame(fred.get_series(id, observation_start = self.start_date))
            for i in range(len(temp_df.iloc[0:,0])):
                if np.isnan(temp_df.iloc[0:,0][i]):
                    try:
                        temp_df.iloc[0:,0][i] = temp_df.iloc[0:,0][i-1]
                    except:
                        temp_df.iloc[0:,0][i] = temp_df.iloc[0:,0][i+1]
            avg_temp_data, temp_start_dates = self.weekly_stats(temp_df)
            temp_weekly_df = pd.DataFrame({"Week":temp_start_dates, id:avg_temp_data})
            temp_weekly_df = temp_weekly_df.set_index("Week")
            input_df = pd.concat([input_df, temp_weekly_df], axis = 1)
        return input_df



fred_s = ["DCOILBRENTEU","BAMLH0A0HYM2", "GOLDAMGBD228NLBM","DAAA","RIFSPPFAAD01NB","BAMLHE00EHYIOAS"]
trial_vol = Vol_Data("2000-01-01", fred_strings = fred_s)
trial_vol.weekly_fred_data()


trial_vol.spy_volume
trial_vol.weekly_vix() #weekly_vix should be the target data set for when we run our tests.
