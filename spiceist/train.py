import os
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.decomposition import PCA
from scipy.sparse import coo_matrix
from .models import GraphAutoencoder
from .utils import gaussian_smoothing_per_cell, build_edge_index_knn
from .metrics import compute_clustering_metrics, compute_assort
import torch.optim as optim


def train_gae(graphs, alpha, output_dir, tile_id, batch_size=32, lr=1e-3, num_epochs=50, patience=3, hid_ch=128, lat_dim=64, dropout_p=0.2):
    """
    Train Graph Autoencoder on list of graphs, with early stopping, and extract embeddings.

    Args:
        graphs (list): List of PyG Data objects.
        alpha (float): Weight for global loss.
        output_dir (str): Directory to save model.
        tile_id (int): Tile identifier for model naming.
        ... (hyperparameters)

    Returns:
        np.ndarray: Extracted embeddings (n_graphs, lat_dim).
    """
    n_total = len(graphs)
    n_train = int(0.8 * n_total)
    train_graphs, eval_graphs = random_split(graphs, [n_train, n_total - n_train])
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_graphs, batch_size=batch_size, shuffle=False)
    in_ch = graphs[0].x.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphAutoencoder(in_ch, hid_ch=hid_ch, lat_dim=lat_dim, dropout_p=dropout_p).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    epochs_without_improve = 0
    print(f"Starting training for tile {tile_id} with alpha {alpha}")
    for epoch in range(1, num_epochs+1):
        model.train()
        total_train = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            x_hat, z_graph = model(batch)
            loss_recon = F.mse_loss(x_hat, batch.x)
            y_true = torch.stack([g.y for g in batch.to_data_list()]).to(device)
            loss_global = F.mse_loss(z_graph, y_true)
            loss = loss_recon + alpha * loss_global
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_train += loss.item() * batch.num_graphs
        train_loss = total_train / len(train_loader.dataset)
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                batch = batch.to(device)
                x_hat, z_graph = model(batch)
                loss_recon = F.mse_loss(x_hat, batch.x)
                y_true = torch.stack([g.y for g in batch.to_data_list()]).to(device)
                loss_global = F.mse_loss(z_graph, y_true)
                loss = loss_recon + alpha * loss_global
                total_val += loss.item() * batch.num_graphs
        val_loss = total_val / len(eval_loader.dataset)
        print(f"Epoch {epoch:02d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improve = 0
            torch.save(model.state_dict(), f"{output_dir}/best_model_{tile_id}_{alpha}.pth")
        else:
            epochs_without_improve += 1
        if epochs_without_improve >= patience:
            print(f"Validation loss hasn’t improved in {patience} epochs, stopping early.")
            break
    model.load_state_dict(torch.load(f"{output_dir}/best_model_{tile_id}_{alpha}.pth"))
    model.eval()
    all_embs = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            _, zg = model(batch)
            all_embs.append(zg.cpu())
    X_emb = torch.cat(all_embs, dim=0).numpy()
    print(f"Training complete for tile {tile_id}")
    return X_emb


def process_tile(df_trans, adata_cell_1, alpha, tile_id, resol_list, output_dir='./results_cosmx_luad', mpp=1.0, 
                 batch_size=32, lr=1e-3, num_epochs=50, patience=3, hid_ch=128, lat_dim=64, dropout_p=0.2, sigma=1.5, knn_k=4, 
                 n_neighbors=16, cell_id_col='cell_ID', save_output=True, cell_meta=None):
    """
    Process a single tile: prepare data, train GAE, extract embeddings, perform clustering, compute metrics, save results.

    Args:
        df_trans (pd.DataFrame): Transcript data with coordinates and features.
        adata_cell_1 (sc.AnnData): Cell-level AnnData object.
        alpha (float): Weight for global loss in training.
        tile_id (int): Identifier for the tile.
        resol_list (list): List of resolutions for Louvain clustering.
        output_dir (str, optional): Directory to save outputs. Defaults to './reuslts_xenium'.
        mpp (float, optional): Microns per pixel for bin sizing. Defaults to 1.0.
        batch_size (int, optional): Batch size for DataLoaders. Defaults to 32.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        num_epochs (int, optional): Maximum training epochs. Defaults to 50.
        patience (int, optional): Patience for early stopping. Defaults to 3.
        hid_ch (int, optional): Hidden channels in GCN. Defaults to 128.
        lat_dim (int, optional): Latent dimension. Defaults to 64.
        dropout_p (float, optional): Dropout probability. Defaults to 0.2.
        sigma (float, optional): Sigma for Gaussian smoothing. Defaults to 1.5.
        knn_k (int, optional): K for KNN edges. Defaults to 4.
        n_neighbors (int, optional): Neighbors for clustering. Defaults to 16.
        cell_id_col (str, optional): Column name for cell IDs in df_trans. Defaults to 'cell_ID'.
        save_output (bool, optional): Whether to save model and AnnData files. Defaults to True.
        cell_meta (pd.DataFrame, optional): Metadata DataFrame for cell coordinates. Defaults to None.

    Returns:
        pd.DataFrame: Metrics for clustering performance.
    """
    print(f"Processing Tile Number: {tile_id} with alpha: {alpha}")
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Copy the AnnData to avoid modifying the original
    adata_cell = adata_cell_1.copy()
    print(f"Starting bin calculation for tile {tile_id}")
    # Calculate number of bins based on coordinate ranges and mpp
    x_range = df_trans['x_global_px'].max() - df_trans['x_global_px'].min()
    y_range = df_trans['y_global_px'].max() - df_trans['y_global_px'].min()
    x_n_bin = int(x_range * mpp // 2)
    y_n_bin = int(y_range * mpp // 2)
    print(f"Starting grouping and normalization for tile {tile_id}")
    # Filter transcripts: assigned to cells and not negative probes
    df_assigned = df_trans[(df_trans[cell_id_col]!=0) & ~df_trans['target'].str.contains('NegPrb')]
    # Extract coordinates as numpy arrays
    x_coord = df_assigned['x_global_px'].values
    y_coord = df_assigned['y_global_px'].values
    # Create division arrays for binning
    x_div_arr = np.linspace(np.min(x_coord), np.max(x_coord), num=x_n_bin, endpoint=False)[1:]
    y_div_arr = np.linspace(np.min(y_coord), np.max(y_coord), num=y_n_bin, endpoint=False)[1:]
    # Assign bin indices to each transcript
    df_assigned['array_col'] = np.searchsorted(x_div_arr, x_coord, side='right')
    df_assigned['array_row'] = np.searchsorted(y_div_arr, y_coord, side='right')
    # Group and count transcripts per cell/grid/feature
    grp = (df_assigned.groupby([cell_id_col,'array_col','array_row','target'], sort=False).size().reset_index(name='count'))
    # Normalize counts to fractions per cell
    cell_tot = grp.groupby(cell_id_col)['count'].transform('sum')
    grp['frac'] = grp['count'] / cell_tot
    # Factorize grid positions and features to integers
    grid_idx, grid_codes = pd.factorize(list(zip(grp.array_col, grp.array_row)), sort=True)
    feat_idx, feat_codes = pd.factorize(grp['target'], sort=True)
    # Build sparse COO matrix from fractions
    M_coo = coo_matrix((grp.frac.values, (grid_idx, feat_idx)), shape=(len(grid_codes), len(feat_codes)))
    # Convert to CSR for efficiency
    grid_tx_count_sparse = M_coo.tocsr()
    # Unpack grid codes
    array_cols, array_rows = zip(*grid_codes)
    # Create grid metadata DataFrame
    idx = [f"{c}_{r}" for c, r in zip(array_cols, array_rows)]
    grid_metadata = pd.DataFrame({'array_col': array_cols, 'array_row': array_rows}, index=idx)
    # Get unique cell_ID per grid, removing duplicates
    df_cells = (df_assigned[['array_col','array_row',cell_id_col]].drop_duplicates().drop_duplicates(subset=['array_col','array_row'], keep=False))
    # Reset index for merging
    gm = grid_metadata.reset_index(drop=True)
    # Merge grid metadata with cell assignments
    merged = gm.merge(df_cells, on=['array_col','array_row'], how='left')
    # Subset to cells present in adata
    merged_sub = merged[merged[cell_id_col].isin(adata_cell.obs_names.tolist())]
    # Subset sparse matrix to selected grids
    grid_tx_mtx = grid_tx_count_sparse[merged_sub.index,:]
    # Map positions to cells
    cell_index_map = merged_sub.groupby(cell_id_col).indices
    present_cells = list(cell_index_map.keys())
    adata_cell_sub = adata_cell[present_cells, :].copy()
    cell_ids = adata_cell_sub.obs_names.tolist()
    adata_cell_sub_ = adata_cell_sub.copy()
    # Prepare coordinates and groups
    coords = merged_sub[['array_col','array_row']].values.astype(int)
    groups = [cell_index_map[c] for c in cell_ids]
    # Convert sparse matrix to dense for smoothing
    feats_all = grid_tx_mtx.toarray()
    labs_all = merged_sub[cell_id_col].values
    coords_np = coords
    print(f"Applying Gaussian smoothing for tile {tile_id}")
    # Apply Gaussian smoothing per cell
    sm_feats = gaussian_smoothing_per_cell(feats_all, coords_np, labs_all, sigma=sigma)
    # L2 normalize smoothed features
    norms = np.linalg.norm(sm_feats, axis=1, keepdims=True).clip(min=1e-6)
    sm_feats /= norms
    print(f"Scaling and PCA for gene expression in tile {tile_id}")
    # Scale gene expression data
    sc.pp.scale(adata_cell_sub_, max_value=10)
    # Compute PCA on gene expression
    pca_gene = PCA(n_components=64, random_state=0)
    X_gene = pca_gene.fit_transform(adata_cell_sub_.X)
    adata_cell_sub.obsm['X_pca'] = X_gene
    print(f"Building graphs for tile {tile_id}")
    # Build PyG Data objects for each cell
    graphs = []
    for ci, pi in enumerate(groups):
        x_i = torch.from_numpy(sm_feats[pi]).float()
        edge_idx = build_edge_index_knn(coords_np[pi], k=knn_k)
        y_vec = torch.from_numpy(X_gene[ci]).float()
        data = Data(x=x_i, edge_index=edge_idx, y=y_vec)
        graphs.append(data)
    X_emb = train_gae(graphs, alpha, output_dir, tile_id, batch_size, lr, num_epochs, patience, hid_ch, lat_dim, dropout_p, save_output=save_output)
    adata_cell_sub.obsm["X_emb"] = X_emb
    print("Final joint shape:", X_emb.shape)
    print(f"Starting clustering and metrics computation for tile {tile_id}")
    # Clustering and metrics computation
    metrics_list = []
    for label, embed_key in [("concat", "X_emb"), ("gex", "X_pca")]:
        adata = adata_cell_sub.copy()
        sc.pp.neighbors(adata, use_rep=embed_key, n_neighbors=n_neighbors)
        sc.tl.umap(adata)
        for resol in resol_list:
            sc.tl.louvain(adata, resolution=resol, key_added='louvain')
            if len(adata.obs['louvain'].unique()) == 1:
                print(f"Skipping resolution {resol} due to single cluster")
                continue
            if save_output:
                adata.write_h5ad(f'{output_dir}/adata_cell_{embed_key}_{tile_id}_{alpha}_{resol}.h5ad')
            clust_metrics = compute_clustering_metrics(adata, embed_key, 'louvain')
            assort = compute_assort(adata, cell_meta=cell_meta)
            all_metrics = {**clust_metrics, 'Assort': assort}
            all_metrics['Type'] = label
            all_metrics['Resolution'] = resol
            all_metrics['Tile_No'] = tile_id
            all_metrics['Alpha'] = alpha
            metrics_list.append(all_metrics)
    # Cleanup
    del graphs
    if 'model' in locals():
        del model
    if 'train_loader' in locals():
        del train_loader
    if 'eval_loader' in locals():
        del eval_loader
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    print(f"Processing complete for tile {tile_id}")
    # Prepare metrics DataFrame
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics = pd.melt(df_metrics, id_vars=['Tile_No', 'Alpha', 'Type', 'Resolution'], var_name='Index', value_name='Value')
    return df_metrics 