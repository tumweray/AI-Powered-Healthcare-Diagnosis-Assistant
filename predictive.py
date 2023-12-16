import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#automatic prediction
from pmdarima.arima import auto_arima 
from math import sqrt
# importing a package for Dickey fuller's formular from stat module
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


data = pd.read_csv(r"C:\Users\PAVILION\Downloads\AirPassengers.csv")
print(data.info())
print(data.isnull())
print(data.describe())
print(data.head())


# plt.ylabel('Passengers')
# plt.xlabel('Month')

# checking the data types if they are the ones required for operations is in the right format .
print(data.info())

# data is changed  using type casting
# converting month column to an index
data["Month"] = pd.to_datetime(data["Month"], format="%Y-%m")
print(data.head())

# converting month column to an index
# data should always be manually indexed because some models dont know how to deal with automatically indexed data.
data.index = data["Month"]
del data["Month"]
print(data.head())

sns.lineplot(data)
# sns.lineplot(data =data , x= 'Mouth', y='#Passengers')
# (#Passengers) hash means numner of passengers
# plt.show()
"""
# Testing for stationality(A stationary process has the property that the mean, 
variance and autocorrelation structure do not change over time.)
"""
# the manner in which data is changing is constant so stationary.
# stationality show data that has a predictable pattern or consistent pattern
# Dickey fuller's formula is used to test stationality

# methodologies of forecasting
# 1. Arima(Autogressive moving average)
# 2. Salima(Seasonal autogressive moving average)


rolling_mean = data.rolling(8).mean()
rolling_std = data.rolling(8).std()

# Dispalying the orignal curve
# plt.plot(data,color= 'blue', label ='Orignal data')
# plt.show()

# plotting the mean
plt.plot(rolling_mean, color="red", label="Rolling mean passenger number")
# plt.show()

# plotting the std
plt.plot(rolling_std, color="purple", label="Rolling std passenger number")
plt.title(label="Passenger rolling mean and standard deviation")

# loc function automatically sets the colors.
plt.legend(loc="best")

# Dickey fuller's formula is used to test stationality
adft = adfuller(data, autolag="AIC")
output_data = pd.DataFrame(
    {
        "Values": [
            adft[0],
            adft[1],
            adft[2],
            adft[3],
            adft[4]["1%"],
            adft[4]["5%"],
            adft[4]["10%"],
        ],
        "Metric": [
            "Test statistics",
            "P-value",
            "Number of lags used",
            "Number of observations used",
            "Critical values(1%)",
            "Critical values(5%)",
            "Critical values(10%)",
        ],
    }
)
print(output_data)
"""
if the P-value > 5%(0.05) , the critical value of the data isn't stationary ,
if the P-value < 5%(0.05) ,the  the critical value of the data is stationary.
"""
"""
Autocorrelation is the measure of how correlated time series
 data is at a given time with past values.
"""
"""
this means that if our data has a strong correlation 
we can assume that there will be a strong likelyhood that if
our passenger are high today,they will also be high tomorrow.
"""
auto_correlation_1 = data['#Passengers'].autocorr(lag = 1)
auto_correlation_3 = data['#Passengers'].autocorr(lag = 3)
auto_correlation_6 = data['#Passengers'].autocorr(lag = 6)
print('One  month lag:',auto_correlation_1)
print('Three month lag:',auto_correlation_3)
print('Six month lag:',auto_correlation_6)
"""
In short and long term we have high correlation.
"""
#Decomposition
decompose = seasonal_decompose(data["#Passengers"],model="additive", period= 7)
decompose.plot()
# Resid - rise and fall each year(residue)

#Forecasting
"""
Forecasting basing on the current and past values using ARIMA(Autogressive moving average)
 in terms of linear combination of the past values.
Split the data into two training and testing data 
"""
#creating another data frame.
#index combines the two columns together.
#Train Data
data['Date'] = data.index
train = data[data['Date'] < pd.to_datetime('1960-08', format='%Y-%m')]
del train['Date']
train['train'] = train['#Passengers']
del train['#Passengers']

#Test data
test = data[data['Date'] >= pd.to_datetime('1960-08',format='%Y-%m')]
del test['Date']
test['test'] = test['#Passengers']
del test['#Passengers']

plt.plot(train,color ='green')
plt.plot(test, color= 'red')
plt.title('Train & Test split for passenger data')
plt.ylabel('Passenger Number')
plt.xlabel('Year-Month')
sns.set()
plt.show()
#generating predictions with auto arima
model = auto_arima(train, trace=True, error_action='ignore',suppress_warnings=True)
model.fit(train)

forecast = model.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast,index=test.index,columns=['Predictions'])
plt.plot(train,color ='green')
plt.plot(test, color= 'red')
plt.plot(forecast,color='black')
plt.title("Arima's best predictions")
plt.ylabel('Passenger Number')
plt.xlabel('Year-Month')
plt.show()
print(forecast)

# calculate the root mean squared error(shows you how accurate or how far off the prediction is)