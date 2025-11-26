###
## cluster_maker: demo for cluster analysis
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np

# --- 1. PATH FIX: Automatically adds the project root to sys.path ---
# This resolves 'from cluster_maker import...' without needing PYTHONPATH=.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
# -----------------------------------------------------------------

# New imports needed for self-contained data generation
from cluster_maker import run_clustering, select_features
from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data

OUTPUT_DIR = "demo_output"
# Define a temporary path for the synthetic input data
TEMP_INPUT_PATH = os.path.join(OUTPUT_DIR, "temp_data_input.csv")


def main() -> None:
    print("=== cluster_maker demo: clustering analysis (Self-Contained) ===\n")
    print("-" * 60)

    # --- 2. DATA FIX: Generate and save input data internally ---
    print("Generating synthetic data...")
    seed_specs = [
        {"name": "FeatureA", "reps": [0.0, 5.0, -5.0]},
        {"name": "FeatureB", "reps": [0.0, 5.0, -5.0]},
    ]
    seed_df = define_dataframe_structure(seed_specs)
    # Generate 300 points in 3 clusters
    df = simulate_data(seed_df, n_points=300, cluster_std=0.8, random_state=42)

    # Save data to a temporary file (as run_clustering requires a path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(TEMP_INPUT_PATH, index=False)
    
    input_path = TEMP_INPUT_PATH
    print(f"Synthetic data saved successfully to: {input_path}")
    print("-" * 60)
    # -----------------------------------------------------------

    # Setup features based on the generated column names
    feature_cols = ["FeatureA", "FeatureB"]

    # No need for command-line argument checks now.
    
    # Inspection after generation
    print("Simulated data loaded and ready for analysis.")
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Validate feature columns using the package function
    try:
        # We use the DataFrame we generated, but run_clustering uses the CSV saved above.
        _ = select_features(df, feature_cols)
    except Exception as exc:
        print(f"\nFATAL ERROR validating features with select_features:\n{exc}")
        sys.exit(1)

    # Run the orchestrator
    print("\nRunning clustering with run_clustering(...)")
    result = run_clustering(
        input_path=input_path, # Now points to the generated CSV
        feature_cols=feature_cols,
        algorithm="kmeans",
        k=3,
        standardise=True,
        output_path=os.path.join(OUTPUT_DIR, "clustered_data.csv"),
        random_state=42,
        compute_elbow=True,
    )

    print("\nClustering completed.")
    print("Metrics:")
    for key, value in result["metrics"].items():
        print(f"  {key}: {value}")
    print("-" * 60)

    # Save plots
    cluster_plot_path = os.path.join(OUTPUT_DIR, "cluster_plot.png")
    elbow_plot_path = os.path.join(OUTPUT_DIR, "elbow_plot.png")

    print(f"Saving 2D cluster plot to:\n  {cluster_plot_path}")
    result["fig_cluster"].savefig(cluster_plot_path, dpi=150)

    if result["fig_elbow"] is not None:
        print(f"Saving elbow plot to:\n  {elbow_plot_path}")
        result["fig_elbow"].savefig(elbow_plot_path, dpi=150)
    else:
        print("No elbow plot was generated (fig_elbow is None).")

    print("\nClustered data saved to:")
    print(f"  - {os.path.join(OUTPUT_DIR, 'clustered_data.csv')}")
    print("Plots saved to:")
    print(f"  - {cluster_plot_path}")
    if result["fig_elbow"] is not None:
        print(f"  - {elbow_plot_path}")

    print("\n=== End of demo ===")


if __name__ == "__main__":
    # Call main() without passing sys.argv, as arguments are no longer needed
    main()