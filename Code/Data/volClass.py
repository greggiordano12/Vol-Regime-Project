import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt

class Vol_Data:

    def __init__(self, start_date, end_date=None, ):
        self.start_date = start_date
        self.end_date = end_date
        #create attributes for rest of class
        self.vix_df = pdr.DataReader("^VIX", "yahoo", start_date, end_date)
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


    def regime_trailing_avg(self, number_of_days = 5):
        all_averages = list(self.vix_close_arr[:number_of_days])
        for i in range(number_of_days, len(self.vix_close_arr)):
            temp_avg = sum(self.vix_close_arr[i-number_of_days:i])/number_of_days
            if temp_avg < 20:
                all_averages.append(1)
            elif temp_avg > 30:
                all_averages.append(3)
            else:
                all_averages.append(2)

        trailing_df = pd.DataFrame({"Vol Regime " + str(number_of_days) + " Day Avg":all_averages[number_of_days:]})
        trailing_df.index = self.vix_df.index[number_of_days:]

        return trailing_df



    #run gen_regime_data


Greg = Vol_Data("2007-01-01", "2020-05-25")
