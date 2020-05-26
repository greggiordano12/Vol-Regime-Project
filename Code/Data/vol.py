import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt


vix_df = pdr.DataReader("^VIX","yahoo", '2007-01-01')
vix_close = vix_df.Close
vix_close_arr = np.array(vix_close)

# Visualize the VIX time series with lines at $30 and $20 to separate high and low volatilty regimes. Visually appears that above 30 is high vol, between 20
# and 30 is med vol, and below 20 is low vol
hor_30 = [30 for i in vix_close]
hor_20 = [20 for i in vix_close]
plt.plot(vix_close)
plt.plot(vix_df.index, hor_30)
plt.plot(vix_df.index, hor_20)
plt.xlabel("Year")
plt.ylabel("Closing Price")
plt.title("Time Series of VIX Close 2007-Present")
plt.show()

# Histogram to further visualize the distribution
plt.hist(vix_close_arr, bins = 40)
plt.xlabel("Closing Price")
plt.ylabel("Number of Days")
plt.title("Distribution of VIX Closing Price")
plt.show()



# We first can separate the vix data into the 3 regimes: low vol, med vol high vol
# Below 20 is low vol (use number 1 to represent that)
# 20 - 30 is med vol (use number 2 to represent)
# Above 30 is high vol (use number 3 to represent)


# Assigning the dates to a regime and creating regime dataset
dates = vix_df.index
low_vol_dates, med_vol_dates, high_vol_dates = [], [], []
vol_regime = []
count = 0
for i in range(len(vix_close)):
    count +=1
    if(vix_df.Close[i] < 20):
        low_vol_dates.append(dates[i])
        vol_regime.append(1)
    elif(vix_df.Close[i] > 30):
        high_vol_dates.append(dates[i])
        vol_regime.append(3)
    else:
        med_vol_dates.append(dates[i])
        vol_regime.append(2)


# Create volatility regime distribution
plt.bar(["Low Vol", "Med Vol", "High Vol"], [len(low_vol_dates), len(med_vol_dates), len(high_vol_dates)])
plt.xlabel("Volatility Regime")
plt.ylabel("Number of Days")
plt.title("Volatility Regime Distribution")


#Create 5 day trailing average for vix data
all_averages = list(vix_close_arr[:5])
for i in range(5,len(vix_close_arr)):
    temp_avg = sum(vix_close_arr[i-5:i])/5
    all_averages.append(temp_avg)

hor_30 = [30 for i in vix_close]
hor_20 = [20 for i in vix_close]
plt.plot(vix_df.index,all_averages)
plt.plot(vix_df.index, hor_30)
plt.plot(vix_df.index, hor_20)
plt.xlabel("Year")
plt.ylabel("5 day average")
plt.title("Time Series of VIX 5-day average 2007-Present")
plt.show()

#Create data set with trailing averages that we can use to test with random forest
low_vol_dates, med_vol_dates, high_vol_dates = [], [], []
vol_regime = []
for i in range(len(all_averages)):
    if(all_averages[i] < 20):
        low_vol_dates.append(dates[i])
        vol_regime.append(1)
    elif(all_averages[i] > 30):
        high_vol_dates.append(dates[i])
        vol_regime.append(3)
    else:
        med_vol_dates.append(dates[i])
        vol_regime.append(2)

final_data = pd.DataFrame({"Vol Regime": vol_regime})
final_data.index = dates
final_data
