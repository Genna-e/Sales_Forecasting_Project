
# Sales Forecasting System using ARIMA
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Load Dataset
data = pd.read_csv('Ecommerce_Sales_Prediction_Dataset.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Aggregate daily sales
daily_sales = data.groupby('Date')['Units_Sold'].sum()

# Train-test split
train_size = int(len(daily_sales) * 0.8)
train, test = daily_sales[:train_size], daily_sales[train_size:]

# Fit ARIMA model
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# RMSE
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"RMSE: {rmse}")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(train, label="Train Data")
plt.plot(test, label="Test Data", color="orange")
plt.plot(test.index, forecast, label="ARIMA Forecast", color="green", linestyle="--")
plt.title("Sales Forecasting with ARIMA")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.grid(True)
plt.show()
