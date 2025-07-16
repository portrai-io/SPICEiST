import numpy as np
import pandas as pd
import scanpy as sc
from torch.utils.data import DataLoader, random_split
import torch
from spiceist import process_tile  # Assuming the package is installed or in PYTHONPATH
import os
from pyarrow import parquet
from spiceist import create_tiles

# Define paths (replace with actual paths)
path = '/path/to/data'  # Example path
xenium_prime = 'dataset_name'  # Example dataset name

# Load transcript data
df_transcript = parquet.read_table(os.path.join(path, xenium_prime, 'transcripts.parquet'))
df_transcript = df_transcript.to_pandas()

# Split into tiles
df_tiles = create_tiles(df_transcript, n_grid=4)

# Load cell feature matrix
adata_cell_1 = sc.read_10x_mtx(os.path.join(path, xenium_prime, 'cell_feature_matrix'))
sc.pp.filter_cells(adata_cell_1, min_counts=10)
adata_cell_1.layers["counts"] = adata_cell_1.X.copy()
sc.pp.normalize_total(adata_cell_1, target_sum=100)
sc.pp.log1p(adata_cell_1)

# Define hyperparameters
alpha_list = [0.1, 0.5, 1.0]  # Example alphas
resol_list = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8]  # Resolutions
mpp = 1.0  # Microns per pixel

# For demonstration, assume data is loaded and proceed with one tile example
# The following is based on the provided running code, integrated into process_tile usage

# Note: process_tile expects specific column names; adapt DataFrame columns if necessary
# For example, rename 'x_location' to 'x_global_px', 'feature_name' to 'target', etc.

# Example loop
results = []
for alpha in alpha_list:
    for i, df_trans in enumerate(df_tiles):
        # Adapt column names to match process_tile expectations
        if 'x_location' in df_trans.columns:
            df_trans = df_trans.rename(columns={'x_location': 'x_global_px', 'y_location': 'y_global_px',
                                                'cell_id': 'cell_ID', 'feature_name': 'target'})
        # Now call process_tile
        df_metrics = process_tile(df_trans, adata_cell_1, alpha, i, resol_list, output_dir='./results_prime', mpp=1.0)
        results.append(df_metrics)

# Combine results
combined_results = pd.concat(results, ignore_index=True)
combined_results.to_csv('./results_prime/metrics.csv', index=False)
print('Processing complete.')

# The provided code snippet is integrated into process_tile; if custom adaptations are needed, modify train.py accordingly.