import requests
import pandas as pd
import json

#Documentation: https://www.alphavantage.co/documentation/#

#sample ticker inputs
ticker_symbols = ["AAPL", "GOOG", "MSFT", "TSLA", "BA"]
bonds_maturity = ["10year", "2year"]
commodities_symbols = ["Brent", "NATURAL_GAS", "COPPER"]
crypto_symbols = ["BTC", "ETH"]

#This function scrapes data for EQUITIES (stocks and ETFs from AlphaVantage)
#Example input parameter: ["QQQ", "SPY", "AAPL", "TSLA", "BA"]
def scrape_equities(tickers):
    data_list = []
    for symb in tickers:
        output = "compact" #compact returns only the latest 100 data points; full returns the full-length time series of 20+ years of historical data. 
        #The "compact" option is recommended if you would like to reduce the data size of each API call.
        func = "TIME_SERIES_DAILY" 

        outputSize = "outputsize=" + output
        function = "function=" + func
        symbol = "symbol=" + symb
        
        url = 'https://www.alphavantage.co/query?' + outputSize + "&" + function + "&" + symbol + "&" + "apikey=ZXVCKXV4Q9WHK3US"
        r = requests.get(url)
        data = r.json()
        data_list.append(data)
    return data_list
    #Sample return data: https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey=demo

#This function scrapes data of the US Treasury Yield (BONDS)
#Example input parameter: ["10year", "2year"]
def scrape_bonds(maturity):
    data_list = []
    for time in maturity:
        func = "TREASURY_YIELD"
        interval = "daily" #accepts daily, weekly, monthly

        function = "function=" + func
        interval1 = "interval=" + interval
        mature = "maturity=" + time #accepts 3month, 2year, 5year, 7year, 10year, and 30year

        url = "https://www.alphavantage.co/query?" + function + "&" + interval1 + "&" + mature + "&" + "apikey=ZXVCKXV4Q9WHK3US"
        r = requests.get(url)
        data = r.json()
        data_list.append(data)
    return data_list
    #Sample return data: https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=monthly&maturity=10year&apikey=demo

#This function scrapes data of COMMODITIES
#Example input parameter: ["Brent", "NATURAL_GAS", "COPPER"]
def scrape_commodities(commodity):
    data_list = []
    for symb in commodity:
        interval = "daily"

        function = "function=" + symb
        interval1 = "interval=" + interval

        url = "https://www.alphavantage.co/query?" + function + "&" + interval1 + "&" + "apikey=ZXVCKXV4Q9WHK3US"
        r = requests.get(url)
        data = r.json()
        data_list.append(data)
    return data_list
    #sample return data: https://www.alphavantage.co/query?function=BRENT&interval=monthly&apikey=demo

def scrape_crypto(crypto):
    data_list = []
    for symb in crypto:
        func = "DIGITAL_CURRENCY_DAILY"
        market = "USD"

        function = "function=" + func
        symbol = "symbol=" + symb
        market_line = "market=" + market
        
        url = 'https://www.alphavantage.co/query?' + function + "&" + symbol + "&" + market_line + "&" + "apikey=ZXVCKXV4Q9WHK3US"
        r = requests.get(url)
        data = r.json()
        data_list.append(data)
    return data_list

        
#sample function calls, only call one at a time
equities_data = scrape_equities(ticker_symbols)
#bonds_data = scrape_bonds(bonds_maturity)
#commodities_data = scrape_commodities(commodities_symbols)
#crypto_data = scrape_crypto(crypto_symbols)

outputList = []

for json_data in  equities_data:
    symbols = json_data["Meta Data"]["2. Symbol"]
    close_values = [entry["4. close"] for entry in json_data["Time Series (Daily)"].values()]
    output = {
        "symbols": symbols,
        "prices": close_values
    }
    outputList.append(output)

#make sure to run this file before running backtest.py
df = pd.DataFrame({obj['symbols']: [float(price) for price in obj['prices']] for obj in outputList})
df_percentage_change = df.pct_change()
df_percentage_change.to_json('object.json', orient='columns')
print(df_percentage_change)
