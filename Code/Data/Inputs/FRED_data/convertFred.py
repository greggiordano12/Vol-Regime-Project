import pandas as pd
import pandas_datareader as pdr
import numpy as np

def updated_fred(data):
    updated_df = data
    updated_df["DATE"] = [pd.to_datetime(d) for d in updated_df["DATE"]]
    updated_df=updated_df.set_index("DATE")
    updated_df=updated_df.apply(pd.to_numeric, errors = 'coerce')
    values = updated_df.iloc[0:,0]
    for i in range(len(values)):
        if values[i] != values[i]:
            values[i] = values[i-1]
    updated_df.iloc[0:,0] = values

    return updated_df
class FredConvert:
    def __init__(self, data):
        self.data = updated_fred(data)
        self.dates = self.data.index

    def weekly_data(self):
        #outputs weekly_data list that is the average weekly data from the given data array. Also outputs
        #week_start_dates that is a list of the first day of each week for the weekly data (used to identify start of weeks)
        # data: meant to be an array defined in constructor that matches self.dates in index
        data = np.array(self.data.iloc[0:,0])
        weekly_data = []
        week_start_dates = []
        dates = self.dates
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
        final_df = pd.DataFrame({"Date":week_start_dates, "Fred Data":weekly_data})
        final_df = final_df.set_index("Date")

        return final_df

# option_spreads = pd.read_csv("Data/Inputs/FRED_data/high_yield_index_option_adjusted_spread.csv")
#
# trial_fred = FredConvert(option_spreads).weekly_data()
# trial_fred
