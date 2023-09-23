import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import yfinance as yf
import datetime


# CONSTANTS
VARIANCE_THRESHOLD = 0.95  # Adjust this threshold as needed
TICKERS = [
    "AAPL",
    "GOOGL",
    "META",
    "TSLA",
    "AMZN",
    "MSFT",
    "JPM",
    "JNJ",
    "V",
    "PG",
    "UNH",
    "HD",
    "MA",
    "NVDA",
    "DIS",
    "PYPL",
    "BAC",
    "CMCSA",
    "XOM",
    "AVGO",
    "QQQ",
    "WMT",
    "ADBE",
    "CRM",
    "PFE",
    "NFLX",
    "KO",
    "INTC",
    "TMO",
    "CSCO",
    "VZ",
    "ABT",
    "PEP",
    "ABBV",
    "MRK",
    "NKE",
    "CVX",
    "ACN",
    "MCD",
    "T",
    "COST",
    "TXN",
    "MDT",
    "NEE",
    "LLY",
    "DHR",
]


def get_expected_returns(data):
    """
    Calculate expected returns from asset price data.
    Normalize stock returns data by subtracting the mean and 
    dividing by standard diviation. 
    Normalisation is required by PCA.

    Parameters:
        data (pd.DataFrame): DataFrame containing asset price data.

    Returns:
        pd.DataFrame: DataFrame containing expected returns.
    """
    simple_returns = pd.DataFrame(
        data=np.zeros(shape=(len(data.index), data.shape[1])),
        columns=data.columns.values,
        index=data.index,
    )
    normalised_returns = simple_returns
    simple_returns = data.pct_change().dropna()
    normalised_returns = (simple_returns - simple_returns.mean()) / simple_returns.std()
    return normalised_returns


def get_data(tickers):
    """
    Retrieve normalised asset price data from Yahoo Finance.

    Parameters:
        tickers (list): List of asset tickers to retrieve data for.

    Returns:
        pd.DataFrame: DataFrame containing normalised asset price data.
    """
    data = yf.download(
        tickers=tickers,
        start=datetime.datetime(2018, 8, 10),
        end=datetime.datetime(2023, 8, 3),
    )[["Close"]]

    data = get_expected_returns(data)
    return data


def calculate_covariance_matrix(data):
    """
    Calculate the covariance matrix of asset returns.

    Parameters:
        data (pd.DataFrame): DataFrame containing asset returns.

    Returns:
        np.ndarray: Covariance matrix.
    """
    return np.cov(data, rowvar=False)


def plot_PCA_variance(
    pca, num_assets, cumulative_variance, var_threshold=VARIANCE_THRESHOLD
):
    """
    Plot the PCA cumulative explained variance.

    Parameters:
        pca (PCA): PCA object.
        num_assets (int): Number of assets.
        cumulative_variance (np.ndarray): Array of cumulative explained variances.
        var_threshold (float): Variance threshold for plotting.

    Returns:
        None
    """
    # Plot the PCA spectrum
    if pca is not None:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the cumulative explained variance
        ax.plot(
            np.arange(0, num_assets), cumulative_variance, marker="o", linestyle="-"
        )
        ax.set_xlabel("Number of Principal Components")
        ax.set_ylabel("Cumulative Variance")
        ax.set_title("PCA Spectrum")
        plt.xticks(np.arange(0, num_assets, 1.0))

        # Add a horizontal line for the threshold
        ax.axhline(
            y=VARIANCE_THRESHOLD,
            color="red",
            linestyle="--",
            label=f"{var_threshold * 100}% Threshold",
        )
        ax.legend()

        plt.grid()
        plt.show()


def plot_percentage_variance(pca):
    """
    Plot the percentage of variance explained by each principal component.

    Parameters:
        pca (PCA): PCA object.

    Returns:
        None
    """
    if pca is not None:
        bar_width = 0.9
        n_assets = pca.components_.shape[1]
        x_indx = np.arange(n_assets)
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 4)

        # Eigenvalues are measured as percentage of explained variance.
        rects = ax.bar(
            x_indx, pca.explained_variance_ratio_[:n_assets], bar_width, color="orange"
        )
        ax.set_xticks(x_indx + bar_width)
        ax.set_xticklabels(list(range(n_assets)), rotation=45)
        ax.set_title("Percent variance explained")
        ax.legend((rects[0],), ("Percent variance explained by principal components",))
        plt.show()


def generate_eigenportfolios(cov_matrix, plot_cumulative_pca=False, plot_pct_var=False):
    """
    Generate eigenportfolios from the covariance matrix.

    Parameters:
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
        plot_cumulative_pca (bool): Whether to plot cumulative PCA spectrum.
        plot_pct_var (bool): Whether to plot percentage of variance explained.

    Returns:
        np.ndarray: Array of eigenportfolios.
    """
    pca = PCA().fit(cov_matrix)

    eigenvectors = pca.components_  # THIS IS ALREADY SORTED IN DECREASING ORDER

    for i in range(len(eigenvectors)):
        eigenvectors[i] = (
            eigenvectors[i] / eigenvectors[i].sum()
        )  # Normalise weights so they sum to 1

    # Calculate the cumulative sum of the explained variance ratio
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumulative_variance >= VARIANCE_THRESHOLD) + 1

    print(
        f"{VARIANCE_THRESHOLD * 100}% variance is explained by {num_components} principal components."
    )

    if plot_cumulative_pca:
        plot_PCA_variance(pca, pca.components_.shape[1], cumulative_variance)

    if plot_pct_var:
        plot_percentage_variance(pca)

    for i in range(num_components):
        print(f"Eigenportfolio { i } weights are: \n{ eigenvectors[i] }")
        print(
            f"Eigenportfolio { i } variance is: \n{ pca.explained_variance_ratio_[i] }"
        )
        print("-------------------------------------------------")

    return eigenvectors[:num_components]


def plot_eigenportfolio_weights(eigenportfolios, index, tickers):
    """
    Plot the weights of a specific eigenportfolio.

    Parameters:
        eigenportfolios (np.ndarray): Array of eigenportfolios.
        index (int): Index of the eigenportfolio to plot.
        tickers (list): List of asset tickers.

    Returns:
        None
    """
    assert index < len(
        eigenportfolios
    ), "Index must be less than the number of eigenportfolios"

    # Plot the weights of the index-th eigen-portfolio
    eigen_prtf1 = pd.DataFrame(
        data={"weights": eigenportfolios[index].squeeze() * 100}, index=tickers
    )
    eigen_prtf1.sort_values(by=["weights"], ascending=False, inplace=True)
    print("Sum of weights of first eigen-portfolio: %.2f" % np.sum(eigen_prtf1))
    eigen_prtf1.plot(
        title=f"Eigenportfolio { index } weights",
        figsize=(12, 6),
        xticks=range(0, len(tickers), 1),
        rot=45,
        linewidth=3,
    )
    plt.show()


if __name__ == "__main__":
    data = get_data(TICKERS)

    cov_matrix = calculate_covariance_matrix(data)

    eigenportfolios = generate_eigenportfolios(
        cov_matrix, plot_cumulative_pca=True, plot_pct_var=True
    )

    # Plot the weights of the index-th eigenportfolio but it has to be less than the total number of eigenportfolios above
    plot_eigenportfolio_weights(
        eigenportfolios, index=0, tickers=TICKERS
    )
    plot_eigenportfolio_weights(
        eigenportfolios, index=1, tickers=TICKERS
    )
