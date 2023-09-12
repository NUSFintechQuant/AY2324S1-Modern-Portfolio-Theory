import numpy as np
import matplotlib.pyplot as plt
import random


def generate_random_portfolio_weights(num_assets):
    """
    Generate random portfolio weights for a given number of assets.

    Parameters:
        num_assets (int): The number of assets in the portfolio.

    Returns:
        weights (np.ndarray): An array of random portfolio weights that sum to 1.
    """
    weights = np.random.rand(num_assets)
    weights /= weights.sum()
    return weights


def generate_dirichlet_portfolios(num_assets):
    """
    Generate random portfolios using a Dirichlet distribution.

    Parameters:
        num_assets (int): The number of assets in the portfolio.

    Returns:
        s (np.ndarray): An array of random portfolio weights that sum to 1.
    """
    s = np.random.dirichlet(num_assets * [1.0], size=1)
    return s[0]


def calculate_portfolio_return(weights, returns):
    """
    Calculate the expected portfolio return based on given weights and asset returns.

    Parameters:
        weights (np.ndarray): Portfolio weights for each asset.
        returns (np.ndarray): Expected returns for each asset.

    Returns:
        portfolio_return (float): Expected portfolio return.
    """
    portfolio_return = np.sum(weights * returns)
    return portfolio_return


def calculate_portfolio_covariance(weights, cov_matrix):
    """
    Calculate the portfolio covariance based on given weights and the covariance matrix.

    Parameters:
        weights (np.ndarray): Portfolio weights for each asset.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.

    Returns:
        portfolio_covariance (float): Portfolio covariance.
    """
    portfolio_covariance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return portfolio_covariance


def calculate_portfolio_volatility(weights, cov_matrix, annualized=False):
    """
    Calculate the portfolio volatility (risk) based on given weights and the covariance matrix.

    Parameters:
        weights (np.ndarray): Portfolio weights for each asset.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
        annualized (bool): Whether to annualize the volatility (default is False).

    Returns:
        portfolio_volatility (float): Portfolio volatility (risk).
    """
    if annualized:
        portfolio_volatility = np.sqrt(calculate_portfolio_covariance(weights, cov_matrix) * 252) # annualized to 252 trading days
    else:
        portfolio_volatility = np.sqrt(calculate_portfolio_covariance(weights, cov_matrix))
   
    return portfolio_volatility


def generate_and_plot_random_portfolios(returns, cov_matrix, num_portfolios=10000, num_assets=None):
    """
    Generate and plot random portfolios on the efficient frontier.

    Parameters:
        returns (np.ndarray): Array of expected returns for each asset.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
        num_portfolios (int): Number of random portfolios to generate (default is 10000).
        num_assets (int): Number of assets to consider in each portfolio (default is None).

    Returns:
        None
    """
    portfolio_returns = []
    portfolio_volatilities = []

    # If num_assets is not specified, use all assets
    if num_assets is None:
        num_assets = len(returns)

    for _ in range(num_portfolios):
        
        # Create a vector of weights for the entire portfolio
        full_weights = np.zeros(len(returns)) 

        # Generate random weights that sum to 1 for the specified number of assets
        weights = generate_dirichlet_portfolios(num_assets)

        # Randomly select the positions (assets) for the held assets
        asset_positions = random.sample(range(len(returns)), num_assets)

        # Set the weights for the held assets
        full_weights[asset_positions] = weights

        # Normalize the weights so they sum to 1
        full_weights /= full_weights.sum()

        # Calculate the portfolio return and volatility
        portfolio_return = calculate_portfolio_return(full_weights, returns)
        portfolio_volatility = calculate_portfolio_volatility(full_weights, cov_matrix, annualized=True)

        portfolio_returns.append(portfolio_return)
        portfolio_volatilities.append(portfolio_volatility)

    # Convert the lists to NumPy arrays
    portfolio_returns = np.array(portfolio_returns)
    portfolio_volatilities = np.array(portfolio_volatilities)

    # Plot the efficient frontier
    plt.figure(figsize=(10, 6))
    plt.scatter(
        portfolio_volatilities,
        portfolio_returns,
        c=portfolio_returns / portfolio_volatilities,
        marker="o",
    )
    plt.title("Efficient Frontier")
    plt.xlabel("Volatility (Risk)")
    plt.ylabel("Return")
    plt.colorbar(label="Sharpe Ratio")
    plt.grid(True)
    plt.show()


# Run the code to generate and plot random portfolios
if __name__ == "__main__":
    returns = np.array([0.1, 0.15, 0.12, 0.08])
    cov_matrix = np.array(
        [
            [0.04, 0.02, 0.015, 0.01],
            [0.02, 0.03, 0.02, 0.015],
            [0.015, 0.02, 0.03, 0.02],
            [0.01, 0.015, 0.02, 0.04],
        ]
    )

    # Generate and plot random portfolios with default number of portfolios (10,000) and all assets in each portfolio
    generate_and_plot_random_portfolios(returns, cov_matrix)

    # You can vary the number of portfolios and the number of assets (chosen randomly) to hold in the portfolio
    # Generate and plot random portfolios with 100 portfolios and 2 assets in each portfolio
    generate_and_plot_random_portfolios(returns, cov_matrix, num_portfolios=100, num_assets=2)
