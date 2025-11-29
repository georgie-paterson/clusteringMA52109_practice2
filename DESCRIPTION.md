Description of the cluster_maker package:

algorithms.py: Provides the K-means clustering algorithm to use on our data. This implements the scikit learn algorithms and creates our clusters and iterate the algorithms to continue the k means process.

data_analyser.py: Computes descriptive statistics for each numerical column in our data. It also computes the correlation matrix between numeric columns within the dataframe.

data_exporter.py: Either exports the dataframe to a csv or text file

evaluation.py: Computes the sum of squares distances (residuals) amongst clusters, this is called the inertia. This also computes the silohuette score and an "elbow score" for inertia values in the k means algorithm.

dataframe_builder.py: the function define_dataframe_structure builds a seed DataFrame where each row is a cluster and each column is a feature. A key element within dataframe_builder is the column_specs variable, which is a list of dictionarys like {"name": "x", "reps": [0.0, 1.0, 2.0]} where reps length defines the number of clusters created. Throughout the module, the code validates input shapes and types and raises errors such as ValueError and TypeError. The function simulate_data simulates n points around the centres in the input seed_df by adding Gaussian noise (std = cluster_std) and returns a dataframe with the original features plus a true_cluster column. This then validates the n points and cluster std.

interface.py: is a high level orchestrator that runs the full clustering workflow start to finish, and uses all the functions from the other modules.

plotting_clustered.py: pretty self explanatory, this module takes the in dataframe, centroids, labels and titles and plots the k means iterative plots of the clusters. With the centroids marked as a cross and different clusters coloured differently.

preprocessing: the function select_features selects a subset of columns and ensures they're numeric. It pulls in a data frame and a list of columns that you want in the dataframe and returns this subset dataframe.