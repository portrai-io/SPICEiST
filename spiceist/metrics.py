import scib
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
import numpy as np
from .utils import calculate_assortativity

def compute_clustering_metrics(adata, embed_key, cluster_key):
    asw = scib.me.silhouette(adata, label_key=cluster_key, embed=embed_key)
    chi = calinski_harabasz_score(adata.obsm[embed_key], adata.obs[cluster_key])
    dbi = davies_bouldin_score(adata.obsm[embed_key], adata.obs[cluster_key])
    return {'ASW': asw, 'CHI': chi, 'DBI': dbi}

def compute_assort(adata, coord_keys=['x_centroid', 'y_centroid'], cluster_key='louvain', k=16, cell_meta=None, id_col='cell_id', coord_cols=['x_centroid', 'y_centroid'], set_obs=True):
    if cell_meta is not None:
        df_coord = cell_meta.set_index(id_col).loc[adata.obs_names.tolist(), coord_cols]
        coords = df_coord.values.astype(float)
        if set_obs:
            adata.obs[['array_col', 'array_row']] = coords
    else:
        coords = adata.obs[coord_keys].values.astype(float)
    labels = adata.obs[cluster_key].astype(str).values
    return calculate_assortativity(coords, labels, k=k)