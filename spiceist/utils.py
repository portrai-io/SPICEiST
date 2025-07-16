import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from libpysal.weights import KNN
import networkx as nx
import torch

def to_dense_if_sparse(mat):
    return mat.toarray() if hasattr(mat, "toarray") else mat

def gaussian_smoothing_per_cell(features, coords, cell_labels, sigma=1.5):
    smoothed = np.zeros_like(features)
    for lab in np.unique(cell_labels):
        idx = np.where(cell_labels == lab)[0]
        feats_c  = to_dense_if_sparse(features[idx])
        coords_c = coords[idx]
        tree = cKDTree(coords_c)
        for j, pt in enumerate(coords_c):
            nbrs = tree.query_ball_point(pt, r=3*sigma)
            d   = np.linalg.norm(coords_c[nbrs] - pt, axis=1)
            w   = np.exp(- (d**2) / (2*sigma**2))
            w  /= (w.sum()+1e-12)
            smoothed[idx[j]] = (feats_c[nbrs] * w[:,None]).sum(axis=0)
    return smoothed

def build_edge_index_knn(coords, k=4):
    tree = cKDTree(coords)
    n_pts = coords.shape[0]
    edges = []
    for i, pt in enumerate(coords):
        _, idxs = tree.query(pt, k=min(k+1, n_pts))
        for j in np.atleast_1d(idxs):
            if j!=i:
                edges.append([i,j])
    if edges:
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        return torch.empty((2,0), dtype=torch.long)

def calculate_assortativity(X: np.ndarray, Y: np.ndarray, k: int = 16) -> float:
    if X.shape[0] != Y.shape[0]:
        raise ValueError("Input arrays X and Y must have the same number of rows.")
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("Input array X must have a shape of (N, 2).")
    gdf = gpd.GeoDataFrame(
        {'label': Y},
        geometry=gpd.points_from_xy(X[:, 0], X[:, 1])
    )
    w = KNN.from_dataframe(gdf, k=k)
    graph = w.to_networkx()
    labels_dict = gdf['label'].to_dict()
    nx.set_node_attributes(graph, labels_dict, 'label')
    coefficient = nx.attribute_assortativity_coefficient(graph, 'label')
    return coefficient 

def create_tiles(df_transcript, n_grid=4):
    """
    Split transcript DataFrame into tiles based on a grid.

    Args:
        df_transcript (pd.DataFrame): DataFrame with 'x_location', 'y_location', 'cell_id'.
        n_grid (int, optional): Number of grid divisions. Defaults to 4.

    Returns:
        list: List of DataFrames, each for a tile.
    """
    x_min, y_min = df_transcript[['x_location','y_location']].min(axis=0)
    x_max, y_max = df_transcript[['x_location','y_location']].max(axis=0)
    y0, x0 = int(y_min), int(x_min)
    y1, x1 = int(y_max), int(x_max)
    x_grid = np.linspace(x0, x1, n_grid + 1, dtype=int)
    y_grid = np.linspace(y0, y1, n_grid + 1, dtype=int)
    coords = df_transcript[['x_location','y_location']].to_numpy()
    df_tiles = []
    for i in range(n_grid):
        for j in range(n_grid):
            xi0, xi1 = x_grid[j], x_grid[j + 1]
            yi0, yi1 = y_grid[i], y_grid[i + 1]
            in_tile_mask = ((coords[:, 0] >= xi0) & (coords[:, 0] < xi1) & (coords[:, 1] >= yi0) & (coords[:, 1] < yi1))
            coord_tile = np.where(in_tile_mask)[0]
            if len(coord_tile) > 0:
                df_tile = df_transcript.loc[coord_tile]
                if (df_tile['cell_id']!='UNASSIGNED').sum() > 0:
                    df_tiles.append(df_tile)
            else:
                print(i, j)
    return df_tiles 