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
N_ASSETS = len(TICKERS)

# np.random.seed(0)
# num_samples = 50           # Number of portfolios
# num_assets = 20           # Number of assets in each portfolio
# data = np.random.rand(num_samples, num_assets) * 100


def clean_up_data(data):
    data.pct_change().dropna(inplace=True)
    return data


def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def get_data(tickers):
    data = yf.download(
        tickers=tickers,
        start=datetime.datetime(2018, 8, 10),
        end=datetime.datetime(2023, 8, 3),
    )[["Close"]]
    data = clean_up_data(data)
    data = normalize_data(data)
    return data


def calculate_covariance_matrix(data):
    return np.cov(data, rowvar=False)


def plot_PCA_spectrum(
    pca, num_assets, cumulative_variance, var_threshold=VARIANCE_THRESHOLD
):
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
    if pca is not None:
        bar_width = 0.9
        x_indx = np.arange(N_ASSETS)
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 4)

        # Eigenvalues are measured as percentage of explained variance.
        rects = ax.bar(
            x_indx, pca.explained_variance_ratio_[:N_ASSETS], bar_width, color="orange"
        )
        ax.set_xticks(x_indx + bar_width)
        ax.set_xticklabels(list(range(N_ASSETS)), rotation=45)
        ax.set_title("Percent variance explained")
        ax.legend((rects[0],), ("Percent variance explained by principal components",))
        plt.show()


def generate_eigenportfolios(cov_matrix, plot_cumulative_pca=False, plot_pct_var=False):
    pca = PCA().fit(cov_matrix)

    eigenvectors = pca.components_  # THIS IS ALREADY SORTED IN DECREASING ORDER

    for i in range(len(eigenvectors)):
        eigenvectors[i] = (
            eigenvectors[i] / eigenvectors[i].sum()
        )  # Normalize weights so they sum to 1

    # Calculate the cumulative sum of the explained variance ratio
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumulative_variance >= VARIANCE_THRESHOLD) + 1

    print(
        f"{VARIANCE_THRESHOLD * 100}% variance is explained by {num_components} principal components."
    )

    if plot_cumulative_pca:
        plot_PCA_spectrum(pca, pca.components_.shape[1], cumulative_variance)

    if plot_pct_var:
        plot_percentage_variance(pca)

    for i in range(num_components):
        print(f"Eigenportfolio { i + 1 } weights are: \n{ eigenvectors[i] }")
        print(
            f"Eigenportfolio { i + 1 } variance is: \n{ pca.explained_variance_ratio_[i] }"
        )
        print("-------------------------------------------------")

    return eigenvectors[:num_components]


if __name__ == "__main__":
    data = get_data(TICKERS)

    cov_matrix = calculate_covariance_matrix(data)

    eigenportfolios = generate_eigenportfolios(cov_matrix, plot_cumulative_pca=False, plot_pct_var=False)

    
