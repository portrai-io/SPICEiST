# SPICEiST

Graph autoencoder framework that integrates subcellular transcript distribution patterns with cell-level gene expression profiles for enhanced cell clustering in imaging-based ST.

## Installation

### Using Conda

You can create a Conda environment and install the package from GitHub:

```bash
conda create -n spiceist python=3.9
conda activate spiceist
pip install git+https://github.com/portrai-io/SPICEiST.git
```

Alternatively, clone the repository and install locally:

```bash
git clone https://github.com/portrai-io/SPICEiST.git
cd SPICEiST
pip install -e .
```

### Requirements

The package depends on the following libraries (listed in `requirements.txt`):

- numpy
- pandas
- scanpy
- torch
- torch_geometric
- scipy
- scikit-learn
- geopandas
- libpysal
- networkx
- scib
- pyarrow

You can install them using:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) License - see the [LICENSE](LICENSE.md) file for details.

## Usage Example

Below is an example script (`main.py`) that demonstrates how to load data, split transcripts into tiles, and process them using the `spiceist` package.

```python
import os
import numpy as np
import pandas as pd
import scanpy as sc
from pyarrow import parquet
from torch.utils.data import DataLoader, random_split
import torch
from spiceist import process_tile, create_tiles  # Import from the installed package

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

# Load cell metadata
cell_meta = pd.read_csv(os.path.join(path, xenium_prime, 'cells.csv.gz'), compression='gzip')

# Define hyperparameters
alpha_list = [0.1, 0.5, 1.0]  # Example alphas
resol_list = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8]  # Resolutions

# Example loop
results = []
for alpha in alpha_list:
    for i, df_trans in enumerate(df_tiles):
        # Adapt column names to match process_tile expectations
        if 'x_location' in df_trans.columns:
            df_trans = df_trans.rename(columns={'x_location': 'x_global_px', 'y_location': 'y_global_px',
                                                'cell_id': 'cell_ID', 'feature_name': 'target'})
        # Now call process_tile
        df_metrics = process_tile(df_trans, adata_cell_1, alpha, i, resol_list, output_dir='./results_prime', cell_meta=cell_meta)
        results.append(df_metrics)

# Combine results
combined_results = pd.concat(results, ignore_index=True)
combined_results.to_csv('./results_prime/metrics.csv', index=False)
print('Processing complete.')
```