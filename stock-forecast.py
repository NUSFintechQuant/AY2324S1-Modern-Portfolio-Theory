import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

def test_stationarity(timeseries):
    """
    Perform Augmented Dickey-Fuller test to assess the stationarity of a time series.
    
    Parameters:
    timeseries (pd.Series): The time series to be tested.
    
    Returns:
    float: The p-value of the test. A p-value below a threshold (e.g., 0.05) suggests stationarity.
    """
    dftest = adfuller(timeseries, autolag='AIC')
    return dftest[1]

def plot_acf_pacf(timeseries, lags=50):
    """
    Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) for a given time series.
    
    Parameters:
    timeseries (pd.Series): The time series for which to plot the ACF and PACF.
    lags (int): Number of lags to include in the plots.
    
    Returns:
    None
    """
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(121)
    plot_acf(timeseries, ax=ax1, lags=lags)
    ax2 = plt.subplot(122)
    plot_pacf(timeseries, ax=ax2, lags=lags)
    plt.show()

def best_arima_params(data):
    """
    Determine the best parameters for an ARIMA model using a grid search approach.
    
    Parameters:
    data (pd.Series): The time series data to fit the ARIMA model.
    
    Returns:
    tuple: The best (p, d, q) parameters based on the Akaike Information Criterion (AIC).
    """
    p_values = range(0, 5)
    d_values = range(0, 3)
    q_values = range(0, 5)
    best_score, best_order = float('inf'), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(data, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_score:
                        best_score, best_order = results.aic, (p, d, q)
                except:
                    continue
    return best_order

def forecast_stock_prices(data, periods=5, plot=False):
    """
    Forecast future stock prices using an ARIMA model and optionally plot the forecast.
    
    Parameters:
    data (pd.Series): The time series data on which the ARIMA model will be trained.
    periods (int): The number of future periods to forecast.
    plot (bool): Whether to plot the forecast against the historical data.
    
    Returns:
    pd.Series: The forecasted values for the given number of periods.
    """
    # Check if data is stationary and difference if necessary
    if test_stationarity(data) > 0.05:
        data_diff = data.diff().dropna()
    else:
        data_diff = data

    order = best_arima_params(data_diff)
    
    if order is not None:
        model = ARIMA(data_diff, order=order)
        results = model.fit()

        # Uncomment this part for the predictions throughout the entire time series
        # forecast = results.get_prediction(end=1500)
        # forecast_series = forecast.predicted_mean
        # forecast_index = pd.date_range(start=data.index[0], periods=len(forecast_series), freq='D')
        # last_value = data.iloc[0] 
        # forecast_series = np.cumsum(forecast_series)  # Cumulative sum of differences
        # forecast_series += last_value

        # Uncomment this part for the predictions for the next 60 days
        forecast = results.get_forecast(steps=periods)
        forecast_series = forecast.predicted_mean
        forecast_index = pd.date_range(start=data.index[-1], periods=len(forecast_series), freq='D')

        last_value = data.iloc[-1]  # Last known value before forecasting
        forecast_series = np.cumsum(forecast_series)  # Cumulative sum of differences
        forecast_series += last_value  # Add the last known value to the cumulative differences

        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(data, label='Historical Prices')
            plt.plot(forecast_index, forecast_series, label='Forecasted Prices', color='red')
            plt.legend()
            plt.show()
        return forecast_series
    else:
        print("Could not find suitable ARIMA parameters")
        return None

if __name__ == "__main__":
    ticker = "AAPL"
    # Download stock data
    data = yf.download(ticker, start="2020-01-01", end="2023-12-18")['Close']
    data.index = pd.to_datetime(data.index)  # Ensure index is datetime

    # Forecast future stock prices for the specified stock
    forecasted_prices = forecast_stock_prices(data, periods=60, plot=True)
    print(f"Forecasted Prices for {ticker}: {forecasted_prices}")
