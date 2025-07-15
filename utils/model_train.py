import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

def get_data(ticker):
    """Get stock data from Yahoo Finance"""
    stock_data = yf.download(ticker, start='2020-01-01')
    return stock_data[['Close']]  # This is a DataFrame with one column

def stationary_check(close_price):
    """Check if the time series is stationary using ADF test"""
    adf_test = adfuller(close_price.dropna())
    p_value = round(adf_test[1], 3)
    return p_value

def get_rolling_mean(close_price):
    """Calculate 7-day rolling mean of closing prices"""
    # Ensure we are working with a Series
    if isinstance(close_price, pd.DataFrame):
        close_price = close_price['Close']
    rolling_price = close_price.rolling(window=7).mean().dropna()
    # Only convert to DataFrame if it's NOT already a DataFrame
    if not isinstance(rolling_price, pd.DataFrame):
        rolling_price = rolling_price.to_frame(name='Close')
    print("rolling_price type before return:", type(rolling_price))
    print("rolling_price columns before numeric conversion:", rolling_price.columns)
    return rolling_price

def get_differencing_order(close_price):
    """Determine optimal differencing order for stationarity"""
    p_value = stationary_check(close_price)
    d = 0
    while p_value > 0.05 and d < 2:
            d += 1
            close_price = close_price.diff().dropna()
            p_value = stationary_check(close_price)
    return d

def fit_model(data, differencing_order):
    """Fit ARIMA model and generate forecasts
    
    Args:
        data: Time series data
        differencing_order: Order of differencing (d)
    
    Returns:
        Array of 30-day forecast values
    """
    model = ARIMA(data, order=(30, differencing_order, 30))
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=30)
    return forecast.predicted_mean

def evaluate_model(original_price, differencing_order):
    """Evaluate ARIMA model using RMSE on test data"""
    train_data = original_price[:-30]
    test_data = original_price[-30:]
    predictions = fit_model(train_data, differencing_order)
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    return round(rmse, 2)

def scaling(close_price):
    """Scale the data using StandardScaler"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(np.array(close_price).reshape(-1, 1))
    return scaled_data, scaler

def get_forecast(scaled_data, original_price_df, differencing_order):
    model = ARIMA(scaled_data, order=(5, differencing_order, 1))
    model_fit = model.fit()
    forecast_steps = 30
    forecast = model_fit.get_forecast(steps=forecast_steps)
    predictions = forecast.predicted_mean

    last_date = original_price_df.index[-1]
    forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps, freq='D')
    forecast_df = pd.DataFrame(predictions, index=forecast_index, columns=['Close'])
    return forecast_df

def inverse_scaling(scaler, scaled_data):
    """
    Inverse transform scaled data back to original scale.
    """
    if isinstance(scaled_data, (pd.Series, pd.DataFrame)):
        scaled_data = scaled_data.values
    return scaler.inverse_transform(np.array(scaled_data).reshape(-1, 1))

