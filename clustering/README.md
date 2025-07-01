# Clustering Algorithm Comparison using Different Machine Learning Approaches

This project compares the performance of three unsupervised machine learning clustering algorithms — **KMeans**, **DBSCAN**, and **Agglomerative Clustering** — on the **Mall Customers dataset**.

It includes:
- Preprocessing
- PCA for dimensionality reduction
- Multiple evaluation metrics (Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index)
- Cluster-wise summary analysis
- Visualizations

---
## Dataset
**Mall_Customers.csv**

The dataset contains:
- `CustomerID`
- `Gender`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1–100)`

---
## Technologies Used
- Python 3.11
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---
## How It Works

### 1. **Preprocessing**
- Encodes gender (`Male=1`, `Female=0`)
- Drops `CustomerID`
- Standardizes the dataset

### 2. **PCA Transformation**
- Reduces features to 2D for easier visualization

### 3. **Clustering Algorithms Compared**
- **KMeans** (n_clusters=5)
- **DBSCAN** (eps=0.8, min_samples=5)
- **Agglomerative Clustering** (n_clusters=5)

### 4. **Evaluation Metrics**
| Metric                   | KMeans | DBSCAN | Agglomerative |
|--------------------------|--------|--------|---------------|
| Silhouette Score         | 0.272  | 0.172  | **0.287 ✅**   |
| Calinski-Harabasz Index  | 62.13  | 28.75  | **64.47 ✅**   |
| Davies-Bouldin Index     | **1.181 ✅** | 2.736  | 1.220    |

> Based on metrics, **Agglomerative Clustering performed best**.

---

## 📊 Visualizations

Visualizations of the clusters using PCA:

| KMeans | DBSCAN | Agglomerative |
|--------|--------|---------------|
| ![](outputs/kmeans_clusters.png) | ![](outputs/dbscan_clusters.png) | ![](outputs/agglomerative_clusters.png) |

---

## 🔍 Business Insights

- Clusters reveal **young, high-income, high-spending customers** — ideal for targeted marketing.
- Other clusters represent **older or low-spending segments** — useful for retention strategy.
- Each algorithm highlights different segments — showing the value of evaluating multiple approaches.

---

## How to Run

1. Clone this repo or download the project.  
2. Ensure dependencies are installed:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

   3. Run the script:

      ```bash
      python clustering_comparison.py
      ```
---
##  Result

This project demonstrates how unsupervised learning can help businesses:

- Segment customers based on purchasing behavior  
- Identify high-value customer groups  
- Improve targeted marketing strategies  
