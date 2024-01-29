import pandas as pd
import math

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

class PortfolioConfig:

    def __init__(self, name : str, holdings : dict):
        self.name = name
        self.holdings = holdings
        self.validate_holdings_ratio()
    
    """
    Ensure portfolio weights are valid.
    """
    def validate_holdings_ratio(self):
        sum_ = 0.0
        for asset, ratios in self.holdings.items():
            if ratios > 1 or ratios < 0:
                raise InvalidWeightException(asset)
            sum_ += float(ratios)
        if not math.isclose(sum_, 1.0):
            raise InvalidRatioException(sum_)
