import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
from typing import Optional, Union

class Backtest:
    def __init__(self, initial_weights: Optional[pd.DataFrame] = None) -> None:
        self.initial_weights = initial_weights
        self.historical_returns = None
        self.results = {}
    
    def set_weights(self, initial_weights):
        self.initial_weights = initial_weights

    def load_data(self, data: Optional[pd.DataFrame] = None, source: Optional[str] = 'database') -> None:
        """
        Load historical data.

        Sample data.
        data = pd.DataFrame({
            'AAPL': [0.01, -0.005, 0.015, -0.01],
            'GOOGL': [0.005, 0.015, -0.01, 0.005],
            'MSFT': [-0.01, 0.005, 0.015, -0.005]
        })

        Parameters:
        - data: User-provided data. If not provided, fetch from source.
        - source: Either 'database' or 'scraper'. Determines where to fetch data if not provided by user.
        """
        if data is not None:
            self.historical_returns = data
        else:
            assets = self.initial_weights.columns.tolist()
            if source == 'scraper':
                pass
                #     self.daily_returns = scrape_data(assets)  # Assuming scrape_data can take a list of assets
            elif source == 'database':
                pass
            #     self.daily_returns = fetch_data(assets)

    def scrape_data(assets):
        NotImplementedError("Function not implemented yet")
        #write the scrappers here such that it returns in the form of Annex A in main class

    def fetch_data(assets):
        NotImplementedError("Function not implemented yet")

    def _calculate_portfolio_returns(self) -> pd.Series:
        """
        Calculate daily portfolio returns based on provided or fetched data and initial weights.
        """
        weights = self.initial_weights.values[0]
        portfolio_returns = self.historical_returns.dot(weights)
        return portfolio_returns

    def _calculate_sharpe(self, risk_free_rate_annual: Optional[float] = 0.03) -> float:
        """
        Calculates Sharpe assuming a risk-free rate of 3%.
        """
        portfolio_returns = self._calculate_portfolio_returns()

        # discount risk-free rate (assuming 252 trading days a year)
        # 1.03 = (1 + x)^{252}
        risk_free_rate_daily = math.exp(math.log(risk_free_rate_annual + 1) / 252) - 1

        mean_return_daily = portfolio_returns.mean()
        return_std_daily = portfolio_returns.std()

        sharpe_ratio_daily = (mean_return_daily - risk_free_rate_daily) / return_std_daily
        sharpe_ratio_annual = sharpe_ratio_daily * math.sqrt(252)

        return sharpe_ratio_annual

    def _calculate_max_drawdown(self) -> float:
        portfolio_returns = self._calculate_portfolio_returns()

        # Calculate the cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak

        max_drawdown = drawdown.min()
        return max_drawdown

    def _calculate_pnl(self) -> float:
        portfolio_returns = self._calculate_portfolio_returns()

        # Assuming initial investment is 1 for simplicity
        initial_value = 1
        final_value = initial_value * (1 + portfolio_returns).prod()

        pnl = final_value - initial_value
        return pnl

    def _calculate_beta(self) -> float:
        """
        Calculates beta assuming 'S&P500' is a column in your data for the S&P 500 returns
        """
        if 'S&P500' not in self.historical_returns.columns:
            return np.nan

        market_returns = self.historical_returns['S&P500']
        portfolio_returns = self._calculate_portfolio_returns()
        covariance_matrix = np.cov(portfolio_returns, market_returns)
        beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
        return beta

    def _plot_portfolio_performance(self, cumulative_returns: pd.Series) -> None:
        """
        Plot cumulative portfolio returns over time.
        """
        sns.set(style="ticks", palette="muted", context="notebook")

        # Plotting
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=cumulative_returns*100, linewidth=2.5, color="royalblue")

        # Setting plot title and labels
        plt.title("")
        plt.xlabel("")
        plt.ylabel("")

        sns.despine()
        plt.tight_layout()
        plt.show()

    def run(self) -> pd.DataFrame:
        """
        Calculate and store various portfolio metrics. Also visualises graph of portfolio returns.
        """
        portfolio_returns = self._calculate_portfolio_returns()
        
        self.results['Portfolio Returns'] = portfolio_returns.tolist()
        self.results['Cumulative Returns'] = (1 + portfolio_returns).cumprod()
        self.results['Sharpe Ratio'] = self._calculate_sharpe()
        self.results['Max Drawdown'] = self._calculate_max_drawdown()
        self.results['PnL'] = self._calculate_pnl()
        self.results['Beta'] = self._calculate_beta()
        
        self._plot_portfolio_performance(self.results['Cumulative Returns'])

        return self.results