"""
================================================================================
Python script to demonstrate t-SNE (t-Stochastic Neighbour Embedding )
================================================================================
This script demonstrates t-SNE on a synthetic dataset to visualise customer clusters
based on their financial behaviour

Requirements
--------------------------
 pip install matplotlib scikit-learn

Author : Nyanjui
Date   : 12 September 2026
"""
# ================================================================================
# SECTION 0: Import the required modules
# ================================================================================
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE

# ================================================================================
# SECTION 1: Generate the dataset
# ================================================================================
# Variables to be used in generating a synthetic dataset
n_samples = 300
n_features = 4       # Number of customer features e.g. age, income, gender, spending socre & savings
centers = 5          # Number of customer clusters
cluster_std = 1.5    # Standard deviation of the clusters

# Create the customer features
features = ['Age','Income', 'Spending Score', 'Savings']

# Use make_blobs to generate a synthetic dataset
X,y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std,
                 n_features=n_features, random_state=42)

# ================================================================================
# SECTION 2: t-SNE & visualisation
# ================================================================================
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.legend(scatter.legend_elements(),title="Classes")
plt.title('t-SNE Visualisation of Synthetic Customer Data')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.colorbar(scatter)
plt.show()