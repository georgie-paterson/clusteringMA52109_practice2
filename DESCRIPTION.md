
# cluster_maker: A Lightweight Framework for Cluster Simulation and Analysis

`cluster_maker` is a small Python package designed to support teaching and assessment in data science and introductory machine-learning modules.  
It provides tools to:

- define cluster centres  
- simulate synthetic clustered data  
- preprocess and validate features  
- run clustering algorithms  
- evaluate clustering performance  
- visualise cluster assignments  
- export results  

The package is modular, easy to read, and structured to demonstrate a full clustering workflow from start to finish.

---

## 1. Overview of the Clustering Workflow

A typical run of `cluster_maker` follows these steps:

1. **Load a CSV dataset**
2. **Select appropriate numeric features**
3. **Standardise the data (optional)**
4. **Cluster the data using KMeans**
5. **Compute evaluation metrics (inertia, silhouette)**
6. **Produce diagnostic plots**
7. **Save clustered data to disk**

All of these steps are orchestrated by the high-level function  
`run_clustering()` inside the package.

The included demo script (`demo/cluster_analysis.py`) demonstrates this workflow end-to-end using an input CSV file.

---

## 2. Module-by-Module Breakdown

Below is a summary of the package’s main components and what each part does.

---

### 2.1 dataframe_builder

This module is responsible for defining cluster centres and generating synthetic data.

#### `define_dataframe_structure(column_specs)`  
Creates a dataframe where:

- each column corresponds to a feature  
- each row is a cluster centre  
- values come from the `"reps"` lists in `column_specs`

Used primarily for simulation or testing.

#### `simulate_data(seed_df, n_points, cluster_std, random_state)`  
Generates synthetic points around the cluster centres using Gaussian noise.  
Returns a dataframe with:

- the simulated coordinates  
- a `"true_cluster"` label  

---

### 2.2 preprocessing

Contains tools for preparing and validating data before clustering.

#### `select_features(df, feature_cols)`  
- Ensures the requested feature columns exist  
- Verifies they are numeric  
- Returns a clean dataframe of only the selected features  

Errors are raised clearly if columns are missing or incompatible.

#### `standardise_features(X)`  
- Scales each feature to zero mean and unit variance  
- Improves clustering performance, especially for KMeans  

---

### 2.3 algorithms

Implements clustering algorithms.

#### `kmeans(X, k, random_state)`  
A simple custom implementation of the KMeans algorithm.  
Returns:

- `labels`: cluster assignments  
- `centroids`: resulting cluster centres  

#### `sklearn_kmeans(X, k, random_state)`  
Wrapper around `sklearn.cluster.KMeans` providing the same output format.

---

### 2.4 evaluation

Provides metrics and diagnostics that help assess clustering effectiveness.

#### `compute_inertia(X, labels, centroids)`  
Returns the total within-cluster sum of squared distances  
(lower is better).

#### `silhouette_score_sklearn(X, labels)`  
Computes the silhouette coefficient when possible.

#### `elbow_curve(X, k_values, random_state, use_sklearn)`  
Calculates inertia for a range of *k* values, supporting elbow-curve analysis.

---

### 2.5 plotting_clustered

Generates visualisations for interpreting clustering results.

#### `plot_clusters_2d(X, labels, centroids)`  
Produces a 2D cluster scatter plot:
- points coloured by cluster  
- centroids marked separately  
- returns a Matplotlib figure  

#### `plot_elbow(k_values, inertias)`  
Plots the elbow curve to help choose an appropriate *k*.

---

### 2.6 data_exporter

Contains lightweight tools for saving results.

#### `export_to_csv(df, output_path, delimiter, include_index)`  
Writes the dataframe (including cluster labels) to a CSV file.

---

### 2.7 interface (optional)  
In some versions, this module wraps the high-level API  
but is not required for basic functionality.

---

## 3. The Demo Script

The demo file:
demo/cluster_analysis.py
provides a complete example of the package in action.

### What the demo does:

1. Loads an input CSV file from the command line  
2. Displays summary information about the dataset  
3. Selects the first two numeric columns  
4. Validates them using `select_features`  
5. Runs `run_clustering()` with:
   - KMeans  
   - k=3  
   - standardisation enabled  
   - elbow curve computation  
6. Saves:
   - `clustered_data.csv`  
   - `cluster_plot.png`  
   - `elbow_plot.png`  
   into a folder named `demo_output/`

### How to run the demo:

```bash
python -m demo.cluster_analysis path/to/data.csv
