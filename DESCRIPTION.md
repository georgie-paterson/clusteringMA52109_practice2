# DESCRIPTION

`cluster_maker` is a modular Python package designed to guide users through the entire workflow of clustering analysis — from creating or loading data, all the way to visualising results. 

The design emphasises clarity and transparency: each stage of the workflow lives in its own module, and together they form a complete, logical pipeline for clustering tasks. Whether users want to understand how K-means works step-by-step or simply run a high-level clustering routine, the package supports both approaches.

---

## dataframe_builder.py

### Purpose  
This module is all about **creating structured data to work with**. It allows users to define cluster centres and generate synthetic datasets — a helpful feature for demonstrations or when you need controlled data for testing.

### Functions  
- **define_dataframe_structure(column_specs)**  
  Builds a “seed” DataFrame where each row corresponds to a cluster and each column corresponds to a feature.  
  Helpful input checks ensure that all feature definitions match in length and format.

- **simulate_data(seed_df, n_points, cluster_std, random_state)**  
  Generates realistic-looking data points around each cluster centre by adding Gaussian noise.  
  It returns a DataFrame that includes both the feature values and a `true_cluster` label for evaluation.

---

## data_analyser.py

### Purpose  
A small but useful module that provides **quick statistical insights** into your dataset before clustering.

### Functions  
- **calculate_descriptive_statistics(data)**  
  Returns common descriptive statistics, helping users get a feel for the scale and spread of their data.

- **calculate_correlation(data)**  
  Computes a correlation matrix across numeric columns — great for spotting relationships between features.

---

## data_exporter.py

### Purpose  
Once data is processed and analysed, this module helps users **save their results cleanly**.

### Functions  
- **export_to_csv(data, filename, delimiter, include_index)**  
  Saves a DataFrame to a CSV file in a safe, controlled way.

- **export_formatted(data, file, include_index)**  
  Writes a neat, human-readable table to a text file — useful for reports or simple inspection.

---

## preprocessing.py

### Purpose  
Before applying clustering algorithms, data needs to be cleaned and prepared. This module handles that step.

### Functions  
- **select_features(data, feature_cols)**  
  Ensures that user-specified feature columns exist and are numeric.  
  If something is wrong (missing column or non-numeric data), the function raises clear, helpful errors.

- **standardise_features(X)**  
  Uses scikit-learn’s `StandardScaler` to put all features on the same scale — a key step for algorithms like K-means.

---

## algorithms.py

### Purpose  
This module provides both a **manual implementation of K-means** (to teach the mechanics) and a **scikit-learn wrapper** for a more robust, real-world solution.

### Core Components  
- **init_centroids(X, k)** — randomly choose starting centroids  
- **assign_clusters(X, centroids)** — assign each point to the nearest centre  
- **update_centroids(X, labels, k)** — recalculate centroids based on assigned points  
- **kmeans(X, k)** — puts the above steps together into a complete K-means routine  
- **sklearn_kmeans(X, k)** — a wrapper around scikit-learn’s implementation  
  (useful when you want reliable results without implementing everything manually)

This module is great for learning: users can compare the behaviour of the manual and sklearn versions.

---

## evaluation.py

### Purpose  
These functions help users **judge cluster quality**, offering essential metrics for analysis.

### Functions  
- **compute_inertia(X, labels, centroids)**  
  Calculates within-cluster variance (the K-means objective function).

- **silhouette_score_sklearn(X, labels)**  
  Computes the silhouette score, giving a sense of cluster separation.

- **elbow_curve(X, k_values, use_sklearn)**  
  Calculates inertia values for a range of k values — useful for determining an appropriate number of clusters.

---

## plotting_clustered.py

### Purpose  
Visualisation is a key part of clustering analysis, and this module focuses entirely on producing clear, informative plots.

### Functions  
- **plot_clusters_2d(X, labels, centroids, title)**  
  Shows a 2D scatter plot of clustered data, with colour-coded labels and optional centroid markers.  
  Perfect for quickly understanding how the algorithm performed.

- **plot_elbow(k_values, inertias, title)**  
  Draws the elbow curve to help users visually assess the best number of clusters.

All plots follow sensible defaults and come with axis labels, titles, and clean layouts.

---

## interface.py

### Purpose  
This is the **heart of the package** — a high-level function that takes care of the entire workflow from start to finish.

### Function  
- **run_clustering(input_path, feature_cols, algorithm, k, standardise, output_path, random_state, compute_elbow)**  
  Handles the full process:
  1. Load data  
  2. Select useful features  
  3. Standardise (optional but recommended)  
  4. Run the chosen clustering algorithm  
  5. Compute inertia and silhouette score  
  6. Produce both a cluster plot and an optional elbow plot  
  7. Save the labelled dataset if requested  

It returns a dictionary containing everything a user might want: labelled data, metrics, centroids, and figures.

This function makes the package accessible for less experienced users while still allowing deeper exploration of the underlying components.

---

## __init__.py

### Purpose  
Collects and re-exports all public functions so they can be accessed directly from the top-level package. This makes the package feel clean and easy to use:

# Final Notes

`cluster_maker` integrates data generation, preprocessing, clustering algorithms, evaluation metrics, and visualisation tools into a coherent and well-structured framework. Its modular design ensures that each component of the workflow is easy to understand, maintain, and extend, making the package particularly suitable for teaching, practical exercises, and assessed debugging tasks.

The package supports both exploratory learning — for example, understanding the mechanics of K-means through the manual algorithm — and more applied workflows through the high-level `run_clustering` interface. Users can therefore engage with the package at the level most appropriate to their needs, whether they are studying core concepts or running complete clustering analyses.

Overall, `cluster_maker` aims to provide a clear, reliable, and educational environment for working with clustered data, enabling users to focus on analytical reasoning and practical problem-solving without unnecessary complexity.
