# clustering_comparison.py
"""
Clustering Algorithm Comparison using Different Machine Learning Approaches

This script compares KMeans, DBSCAN, and Agglomerative Clustering on the Mall Customers dataset.
It includes preprocessing, PCA for visualization, silhouette score, Calinski-Harabasz index, Davies-Bouldin index,
and cluster-wise analysis.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# -------------------------------
# Step 1: Load and Preprocess Data
# -------------------------------

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("clustering/Mall_Customers.csv")

# Drop unnecessary columns
df.drop('CustomerID', axis=1, inplace=True)

# Encode categorical features
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])  # Male=1, Female=0

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# -------------------------------
# Step 2: Define Clustering Models
# -------------------------------

models = {
    'KMeans': KMeans(n_clusters=5, random_state=42),
    'DBSCAN': DBSCAN(eps=0.8, min_samples=5),
    'Agglomerative': AgglomerativeClustering(n_clusters=5)
}

# -------------------------------
# Step 3: Evaluate & Visualize Clusters
# -------------------------------

for name, model in models.items():
    # Fit and predict clusters
    labels = model.fit_predict(X_scaled)

    print(f"\n{name} Evaluation Metrics:")
    try:
        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        print(f"Silhouette Score         : {sil:.3f}")
        print(f"Calinski-Harabasz Index  : {ch:.3f}")
        print(f"Davies-Bouldin Index     : {db:.3f}")
    except Exception as e:
        print("Could not compute one or more scores:", str(e))

    # Plot clusters using PCA
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=40)
    plt.title(f"{name} Clustering Visualization")
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.tight_layout()
    plt.savefig(f"outputs/{name.lower()}_clusters.png")
    plt.close()

# -------------------------------
# Step 4: Cluster-wise Summary
# -------------------------------

print("\nCluster-wise Summary:")
for name, model in models.items():
    labels = model.fit_predict(X_scaled)
    df_copy = df.copy()
    df_copy['Cluster'] = labels

    print(f"\n{name} Cluster Summary:")
    print(df_copy.groupby('Cluster').mean(numeric_only=True))
