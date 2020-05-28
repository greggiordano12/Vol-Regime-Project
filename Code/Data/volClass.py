import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
import datetime
import calendar


class Vol_Data:

    def __init__(self, start_date, end_date=None, ):
        self.start_date = start_date
        self.end_date = end_date
        #create attributes for rest of class
        self.vix_df = pdr.DataReader("^VIX", "yahoo", start_date, end_date)
        dates = self.vix_df.index
        #this loop reconfigures data frame to always start on a monday (beginning of the week)
        dropped_dates = []
        for d in dates:
            if d.weekday() == 0:
                break
            dropped_dates.append(d)
        self.vix_df = self.vix_df.drop(dropped_dates)
        self.vix_close = self.vix_df.Close
        self.vix_close_arr = np.array(self.vix_close)


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


    def weekly_vol(self, number_of_days = 5):
        #I need to adjust this function for the weeks that do not start on Mondays!
        weekly_vol_regimes = []
        week_start_dates = []
        dates = self.vix_df.index
        i = 0
        while i<len(dates):
            temp_week = []
            temp_week_day = []
            if dates[i].weekday() == 0:
                temp_week_day.append(dates[i].weekday())
                week_start_dates.append(dates[i])
                temp_week.append(self.vix_close_arr[i])
                i+=1
                if i < len(dates):
                    while dates[i].weekday() not in temp_week_day :
                        temp_week.append(self.vix_close_arr[i])
                        temp_week_day.append(dates[i].weekday())
                        i+=1
                        if i >= len(dates):
                            break
                temp_avg_vol = sum(temp_week)/len(temp_week)

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







    #run gen_regime_data


trial_vol = Vol_Data("2007-01-02", "2020-05-25")
trial_vol.weekly_vol()



findDay(vol_reg[1])

lst = [1,2,3,4]
