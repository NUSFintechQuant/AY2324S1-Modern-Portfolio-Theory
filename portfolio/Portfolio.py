from PortfolioConfig import PortfolioConfig
import pandas as pd
import pathos.multiprocessing as pmp

class LackOfDataException(Exception):
    def __init__(self):
        msg = "Failure to find dataset, ensure it has been loaded with WDQS and stored in ./data."
        super.__init__(msg)

"""
This class represents a portfolio.
"""

class Portfolio:
    
    """
    Constructor for portfolio.
    
    @param config Portfolio configuration file representing holdings.
    @param n_cores Defaults to half of available cores on cpu, determines cores available for speeding up df generation.
    """
    def __init__(self, config : PortfolioConfig, n_cores = pmp.cpu_count()//2) -> None:
        self.config = config
        self.evaluated = False
        self.results = None
        self.n_cores = n_cores

    """
    Generates the pandas DataFrame representing the holdings in the portfolio with respect
    to their respective weights.

    @return Portfolio dataframe.
    """
    def generate_df(self) -> pd.DataFrame:
        if self.evaluated: # cache
            return self.results
        name_weight = list(self.config.holdings.items())
        pool = pmp.ProcessingPool(nodes = self.n_cores)
        base_asset = name_weight[0]
        base = self.generate_columns_needed(base_asset[0])
        with pool:
            results = pool.amap(
                lambda x : self.generate_columns_needed(x[0]), name_weight[1:])
            for res in results.get():
                df_curr, _ = res
                base = pd.merge(base, res, how = "inner", on = "date")
        self.results = base
        self.evaluated = True
        return self.results
    
    """
    Generates needed weights for the backtesting engine.
    
    @return Weights for the backtesting engine.
    """
    def generate_weights(self) -> pd.DataFrame:
        ret_df = {}
        for asset, weight in self.config.holdings.items():
            ret_df[asset] = [weight]
        return pd.DataFrame(ret_df)
    
    """
    Helper function to extract columns needed and reformat it for backtester.
    
    @param asset The name of the asset (should be saved as {asset}.csv).
    @param weight The weights to adjust the assets by.
    
    @return Date-price in column format.
    """
    def generate_columns_needed(self, asset : str) -> pd.DataFrame:
        #NOTE: Relative import used here, data needs to be pulled via wrds and stored in data dir.
        try:
            if not self.config.download:
                if asset not in self.config.downloaded:
                    df = pd.read_csv("../data/{}.csv".format(asset))
                else:
                    df = self.config.downloaded[asset]
            else:
                df = pd.read_csv("../data/{}.csv".format(asset))
        except:
            raise LackOfDataException()
        date_series = pd.to_datetime(df["datadate"]) # Convert to datetime for consistency.
        price = df["prccd"]
        ret_df = pd.DataFrame(
            {"date" : date_series, asset : price})
        return ret_df
    
    """
    Generates portfolio returns.
    
    @return The portfolio performance as a single variable.
    """
    def generate_consolidated(self) -> pd.DataFrame:
        if self.evaluated:
            df = self.results
        else:
            results = self.generate_df()
        price_df = df.loc[:, df.columns != 'date']
        price_df["consolidated"] = price_df.sum(axis = 1)
        ret_df = pd.DataFrame({"date" : df["date"], 
                               self.config.name : price_df["consolidated"]})
        return ret_df
    
    """
    Backtests the portfolio.
    
    @param backtester The backtest engine.
    @param start The start of testing period
    @param end The end of testing period
    @return Performance of the portfolio.
    """
    def backtest(self, engine, start='1900-01-01', end='2050-01-01') -> dict:
        engine.set_weights(self.generate_weights())
        df = self.generate_df()
        price_df = df.copy()
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df = price_df[(price_df['date'] >= start) & (price_df['date'] <= end)]
        price_df.set_index('date', inplace=True)
        # convert price df to % change price df
        percentage_change_price_df = price_df.pct_change()
        # print(percentage_change_price_df)
        engine.load_data(percentage_change_price_df)
        results = engine.run()
        metrics = ['Sharpe Ratio', 'Max Drawdown', 'PnL', 'Beta']
        for metric in metrics:
            print("{}: {}".format(metric, results[metric]))
        return results
    
"""
Default portfolios for backtesting, allocation given as per released NFS paper.
Note that keys in the holding dictionary corresponds to .csv name in ./data.
"""

"""
All-Weather Portfolio:

    US Large Cap (SnP500) : 25%
    Long Term US Bonds (TLT) : 40%
    Intermediate Term US Bonds (IEI) : 15%
    Gold (GLD) : 7.5%
    Commodities (DBC) : 7.5%
    
"""
ALL_WEATHER = Portfolio(PortfolioConfig("all_weather", {
    "snp500" : 0.3,
    "us_bonds" : 0.4,
    "inter_us_bonds" : 0.15,
    "gold" : 0.075,
    "commodities" : 0.075,
}, download=True))

"""
Bernstein Portfolio:

    US Large Cap (SnP500) : 25%
    US Small Cap (VB) : 25%
    Foreign Large Cap (VEA) : 25%
    US Bonds (TLT): 25%
    
"""
BERNSTEIN = Portfolio(PortfolioConfig("bernstein", {
    "us_bonds" : 0.25,
    "snp500" : 0.25,
    "foreign_large_cap" : 0.25,
    "us_small_cap" : 0.25,
}, download=True))

"""
MVO Portfolio:
    US Bonds : 3.9%
    US Large Cap (SnP500) : 96.0%
"""
MVO = Portfolio(PortfolioConfig("mvo", {
    "us_bonds" : 0.03915074751969351,
    "snp500": 0.9608492524803065,
}, download=True))

"""
SnP500 Index Portfolio:

    US Large Cap (SnP500) : 100%
    
"""
SNP500 = Portfolio(PortfolioConfig("snp500", {
    "snp500" : 1,
}, download=True))
