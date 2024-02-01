import pandas as pd
import math, os
import wrds
import refinitiv.data

"""
This class represents a portfolio configuration, the portfolio configuration is defined by
the holdings dictionary, containing assets and weights.

Example of holdings dictionary:
    {
        "APPL" : 0.5,
        "GOOG" : 0.5
    }
"""

class InvalidWeightException(Exception):
    def __init__(self, asset):
        super().__init__("You have an invalid ratio for {}".format(asset))

class InvalidRatioException(Exception):

    def __init__(self, ratio):
        super().__init__("The sum of your current portfolio weights is {}! Check your ratios again.".format(ratio))
        
class DataPullException(Exception):
    def __init__(self):
        super().__init__("Problem with retrieving data, check your ticker symbol against WRDS data dictionary!")

class PortfolioConfig:
    
    """
    @param name The name of the portfolio.
    @param holdings The dictionary of holdings, consisting of ticker and weights.
    @param download True if we have all the data in ./data, if some of our tickers are not in ./data, we pull from WRDS.
    """
    def __init__(self, name : str, holdings : dict, download=False):
        self.name = name
        self.holdings = holdings
        self.download = download
        self.validate_holdings_ratio()
        self.path = "../data/"
        if not self.download:
            self.downloaded = {}
            self.get_data()
    
    """
    Ensure portfolio weights are valid.
    """
    def validate_holdings_ratio(self) -> None:
        sum_ = 0.0
        for asset, ratios in self.holdings.items():
            if ratios > 1 or ratios < 0:
                raise InvalidWeightException(asset)
            sum_ += float(ratios)
        if not math.isclose(sum_, 1.0):
            raise InvalidRatioException(sum_)
            
    """
    Gets data needed from WRDS.
    """
    def get_data(self) -> None:
        conn = wrds.Connection()
        for asset in self.holdings.keys():
            path = os.path.join(self.path, "{}.csv".format(asset))
            if os.path.isfile(path):
                continue
            try:
                asset_data = conn.raw_sql("""
                select datadate, conm, tic, prccd
                from comp_na_daily_all.secd
                where tic = '{}'
                """.format(asset))
            except:
                raise DataPullException()
            if self.download:
                asset_data.to_csv(path)
            else: # We don't download to ./data, but keep pulled data in memory.
                self.downloaded[asset] = asset_data
                
    
