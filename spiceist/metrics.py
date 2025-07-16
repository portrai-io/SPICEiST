import scib
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
import numpy as np
from .utils import calculate_assortativity

def compute_clustering_metrics(adata, embed_key, cluster_key):
    asw = scib.me.silhouette(adata, label_key=cluster_key, embed=embed_key)
    chi = calinski_harabasz_score(adata.obsm[embed_key], adata.obs[cluster_key])
    dbi = davies_bouldin_score(adata.obsm[embed_key], adata.obs[cluster_key])
    return {'ASW': asw, 'CHI': chi, 'DBI': dbi}

def compute_assort(adata, coord_keys=['CenterX_global_px', 'CenterY_global_px'], cluster_key='louvain', scale=0.12, k=16):
    coords = adata.obs[coord_keys].values.astype(float) * scale
    labels = adata.obs[cluster_key].astype(str).values
    return calculate_assortativity(coords, labels, k=k) 