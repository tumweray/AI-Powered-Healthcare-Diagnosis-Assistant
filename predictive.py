import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np



data =pd.read_csv(r"C:\Users\PAVILION\Downloads\AirPassengers.csv")
print(data.info())
print(data.isnull())
print(data.describe())
print(data.head())


#plt.ylabel('Passengers')
#plt.xlabel('Month')

#checking the data types if they are the ones required for operations is in the right format .
print(data.info())

#data is changed  using type casting 
#converting month column to an index 
data['Month'] =pd.to_datetime(data['Month'],format='%Y-%m')
print(data.head())

#converting month column to an index 
# data should always be manually indexed because some models dont know how to deal with automatically indexed data.
data.index = data['Month']
del data['Month']
print(data.head())

sns.lineplot(data )
# sns.lineplot(data =data , x= 'Mouth', y='#Passengers')
# (#Passengers) hash means numner of passengers
plt.show()

#Testing for stationality(A stationary process has the property that the mean, variance and autocorrelation structure do not change over time.)
# the manner in which data is changing is constant so stationary.
#stationality show data that has a predictable pattern or consistent pattern
# dickey fuller's formula is used to test stationality 

