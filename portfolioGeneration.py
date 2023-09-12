import numpy as np
import matplotlib.pyplot as plt


def generate_random_portfolio_weights(num_assets):

    weights = np.random.rand(num_assets)
    weights /= weights.sum()
    return weights


def generate_dirichlet_portfolios(num_assets):
    s = np.random.dirichlet(num_assets * [1.0], size=1)
    return s[0]


def calculate_portfolio_return(weights, returns):
    portfolio_return = np.sum(weights * returns)
    return portfolio_return


def calculate_portfolio_volatility(weights, cov_matrix):
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_volatility


def generate_and_plot_random_portfolios(returns, cov_matrix, num_portfolios=10000):
    portfolio_returns = []
    portfolio_volatilities = []

    for _ in range(num_portfolios):
        weights = generate_random_portfolio_weights(len(returns))
        portfolio_return = calculate_portfolio_return(weights, returns)
        portfolio_volatility = calculate_portfolio_volatility(weights, cov_matrix)

        portfolio_returns.append(portfolio_return)
        portfolio_volatilities.append(portfolio_volatility)

    # Convert the lists to NumPy arrays
    portfolio_returns = np.array(portfolio_returns)
    portfolio_volatilities = np.array(portfolio_volatilities)

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


# Example usage:
if __name__ == "__main__":
    # Example data (replace with your own)
    returns = np.array([0.1, 0.15, 0.12, 0.08])
    cov_matrix = np.array(
        [
            [0.04, 0.02, 0.015, 0.01],
            [0.02, 0.03, 0.02, 0.015],
            [0.015, 0.02, 0.03, 0.02],
            [0.01, 0.015, 0.02, 0.04],
        ]
    )

    generate_and_plot_random_portfolios(returns, cov_matrix)
    generate_and_plot_random_portfolios(returns, cov_matrix, 10)
    generate_and_plot_random_portfolios(returns, cov_matrix, 100)
