# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import pmdarima as pmd


# Read Data CSV file
file = pd.read_csv('Demands.csv')
# Set time column as index for our loads instead of numbers of each row
file.set_index('time', inplace=True)
file.index = pd.to_datetime(file.index)
# Remove any NaNs in our data
file.dropna(inplace=True)
# Print Data Table
print(file.head())



# Perform decomposition for our data (First parameter: our loads, Second Parameter: type of seasonal data "Additive" or "multiplicative", Third parameter: period of our data)
seasonality = seasonal_decompose(file['load'], model='additive', period=24)
# Plotting the components of our data
seasonality.plot()
plt.show()


# Dickey Fuller Test for stationarity
StationarityTest = adfuller(file.load, autolag='AIC')
print(StationarityTest[1])


# Estimating our model parameter
arima_model = pmd.auto_arima(file['load'], start_P=1, start_q=1, test='adf', m=24, seasonal=True, trace=True)



# Building Model using orders obtained from above
sarima_model = SARIMAX(file['load'], order=(2,0,1), seasonal_order=(0,0,2,24))
predicted_loads = sarima_model.fit().predict()


# Plotting Curve predicted against actual loads
plt.plot(predicted_loads, label='predicted')
plt.plot(file['load'], label='actual')
plt.legend()
plt.show()
