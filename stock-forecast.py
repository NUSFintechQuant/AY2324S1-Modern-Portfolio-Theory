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
    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)
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

def forecast_stock_prices(data, periods=5):
    """
    Forecast future stock prices using an ARIMA model.
    
    Parameters:
    data (pd.Series): The time series data on which the ARIMA model will be trained.
    periods (int): The number of future periods to forecast.
    
    Returns:
    pd.Series: The forecasted values for the given number of periods.
    """
    # Check if data is stationary and difference if necessary
    if test_stationarity(data) > 0.05:
        data = data.diff().dropna()

    # Plot ACF and PACF for parameter identification
    plot_acf_pacf(data)

    order = best_arima_params(data)
    
    if order is not None:
        model = ARIMA(data, order=order)
        results = model.fit()
        forecast = results.forecast(steps=periods)
        return forecast
    else:
        print("Could not find suitable ARIMA parameters")
        return None

if __name__ == "__main__":
    ticker = "AAPL"
    # Download stock data
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
    data.index = pd.to_datetime(data.index)  # Ensure index is datetime
    stock_prices = data['Close']
    stock_prices.index = pd.DatetimeIndex(data.index).to_period('D')

    # Forecast future stock prices for the specified stock
    forecasted_prices = forecast_stock_prices(stock_prices, periods=5)
    print(f"Forecasted Prices for {ticker}: {forecasted_prices}")
