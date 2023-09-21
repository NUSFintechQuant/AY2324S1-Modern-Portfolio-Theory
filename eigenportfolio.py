import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# CONSTANTS
VARIANCE_THRESHOLD = 0.30  # Adjust this threshold as needed
np.random.seed(0)
num_samples = 50           # Number of portfolios
num_assets = 20           # Number of assets in each portfolio 


# Creating a random dataset with 5 assets and 100 samples
data = np.random.rand(num_samples, num_assets) * 100

def clean_up_data(data):
    # We will do this next time
    return data

def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

normalized_data = normalize_data(data)

def calculate_covariance_matrix(data):
    return np.cov(data, rowvar=False)

cov_matrix = calculate_covariance_matrix(normalized_data)

def plot_PCA_spectrum(pca, num_assets, cumulative_variance, var_threshold=VARIANCE_THRESHOLD):
    # Plot the PCA spectrum
    if pca is not None:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the cumulative explained variance
        ax.plot(np.arange(0, num_assets), cumulative_variance, marker='o', linestyle='-')
        ax.set_xlabel('Number of Principal Components')
        ax.set_ylabel('Cumulative Variance')
        ax.set_title('PCA Spectrum')
        plt.xticks(np.arange(0, num_assets, 1.0))

        # Add a horizontal line for the threshold
        ax.axhline(y=VARIANCE_THRESHOLD, color='red', linestyle='--', label=f'{var_threshold * 100}% Threshold')
        ax.legend()
        
        plt.grid()
        plt.show()


def generate_eigenportfolios(cov_matrix, plot=False):
    pca = PCA().fit(cov_matrix)

    eigenvectors = pca.components_   # THIS IS ALREADY SORTED

    for i in range(len(eigenvectors)):
        eigenvectors[i] = eigenvectors[i] / eigenvectors[i].sum()  # Normalize weights so they sum to 1

    # Calculate the cumulative variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumulative_variance >= VARIANCE_THRESHOLD) + 1

    print(f"{VARIANCE_THRESHOLD * 100}% variance is explained by {num_components} principal components.")
    if plot:
        plot_PCA_spectrum(pca, pca.components_.shape[1], cumulative_variance)

    for i in range(num_components):
        print(f"Eigenportfolio {i+1} weights are: \n{eigenvectors[i]}")
        print(f"Eigenportfolio {i+1} variance is: \n{pca.explained_variance_ratio_[i]}")
        print("-------------------------------------------------")

    return eigenvectors[:num_components]

generate_eigenportfolios(cov_matrix, plot=False)


