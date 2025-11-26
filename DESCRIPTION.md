# cluster_maker — Package Description

`cluster_maker` is a small educational Python package used for synthetic
data generation, preprocessing, clustering, evaluation, and plotting.
It is designed for teaching purposes in MA52109 (Programming for Data Science).

---

## 1. Data Generation

### `define_dataframe_structure()`
Creates a seed DataFrame describing the centres of clusters. Each row
represents a cluster and each column a feature.

### `simulate_data()`
Uses the seed DataFrame to simulate synthetic clustered points with
Gaussian noise. Produces a DataFrame with feature columns and a 
`true_cluster` column.

---

## 2. Preprocessing

### `select_features()`
Extracts user-specified columns from a DataFrame.

### `standardise_features()`
Applies standardisation (zero mean, unit variance) to numeric arrays.

---

## 3. Clustering Algorithms

### `kmeans()`
Custom implementation of k-means:
- centroid initialisation
- iterative assignment + update
- returns labels & centroids

### `sklearn_kmeans()`
Wrapper around scikit-learn’s KMeans.

---

## 4. Evaluation

### `compute_inertia()`
Measures within-cluster sum of squared distances.

### `silhouette_score_sklearn()`
Computes silhouette score using sklearn.

### `elbow_curve()`
Runs clustering for multiple values of *k* and returns inertias.

---

## 5. Plotting

### `plot_clusters_2d()`
Simple scatter plot of clustered data + centroids.

### `plot_elbow()`
Plots inertia against the number of clusters.

---

## 6. High-Level Interface

### `run_clustering()`
Full workflow:
1. Read CSV  
2. Select & standardise features  
3. Run chosen clustering algorithm  
4. Compute metrics  
5. Generate cluster plot and optional elbow plot  
6. Export labelled data (optional)

---

This package provides the basic structure needed for clustering tasks in the practical exam, including data simulation, algorithm execution, evaluation, and visualisation.

