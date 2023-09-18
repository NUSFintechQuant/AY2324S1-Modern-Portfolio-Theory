import requests
#Documentation: https://www.alphavantage.co/documentation/#

#sample parameters inputs
ticker_symbols = ["QQQ", "SPY", "AAPL", "TSLA", "BA"]
bonds_maturity = ["10year", "2year"]
commodities_symbols = ["Brent", "NATURAL_GAS", "COPPER"]

#This function scrapes data for EQUITIES (stocks and ETFs from AlphaVantage)
#Example input parameter: ["QQQ", "SPY", "AAPL", "TSLA", "BA"]
def scrape_equities(tickers):
    data_list = []
    for symb in tickers:
        output = "full" #compact returns only the latest 100 data points; full returns the full-length time series of 20+ years of historical data. 
        #The "compact" option is recommended if you would like to reduce the data size of each API call.
        func = "TIME_SERIES_DAILY" 

        outputSize = "outputsize=" + output
        function = "function=" + func
        symbol = "symbol=" + symb
        
        url = 'https://www.alphavantage.co/query?' + outputSize + "&" + function + "&" + symbol + "&" + "apikey=FU9AQUPLZDYW1NAH"
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

        url = "https://www.alphavantage.co/query?" + function + "&" + interval1 + "&" + mature + "&" + "apikey=FU9AQUPLZDYW1NAH"
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

        url = "https://www.alphavantage.co/query?" + function + "&" + interval1 + "&" + "apikey=FU9AQUPLZDYW1NAH"
        r = requests.get(url)
        data = r.json()
        data_list.append(data)
    return data_list
    #sample return data: https://www.alphavantage.co/query?function=BRENT&interval=monthly&apikey=demo
        
#sample function calls
equities_data = scrape_equities(ticker_symbols)
bonds_data = scrape_bonds(bonds_maturity)
commodities_data = scrape_commodities(commodities_symbols)

for data in bonds_data: #, equities_data, commodities_data:
    print(data)
