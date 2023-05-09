#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
"""


# %% Libraries
import os
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import kmeans_plusplus as sklearn_kmeans_plusplus
from typing import Final


# %% Paths
# Default subdirectory that would contain the configurations for initialising centroids
CONFIG_DIR: Final[str] = 'data/init-configs'
# Default subdirectory that would contain the initialised centroids
SAVE_DIR: Final[str] = 'data/init'


# %% Initialise random centroids from the dataset
def random(X: DataFrame, n_clusters: int, random_state: int) -> DataFrame:
    assert X.drop_duplicates().shape[0] >= n_clusters > 0
    centroids = X.drop_duplicates().sample(n=n_clusters, random_state=random_state).reset_index(drop=True)
    centroids.index.name = 'cluster'
    return centroids


# %% Initialise centroids using k-means++
def kmeans_plusplus(X: DataFrame, n_clusters: int, random_state: int) -> DataFrame:
    assert X.drop_duplicates().shape[0] >= n_clusters > 0
    centroids, indices = sklearn_kmeans_plusplus(
        X=X.to_numpy(), n_clusters=n_clusters, random_state=random_state
        )
    assert (centroids == X.iloc[indices].to_numpy()).all(axis=None)
    centroids = DataFrame(centroids, columns=X.columns)
    centroids.index.name = 'cluster'
    return centroids


# %% Export centroids
def export(centroids: DataFrame, dataset_name: str, random_state: int, init_method: str):
    n_clusters = len(centroids)
    save_dir = os.path.join(SAVE_DIR, dataset_name, f'k={n_clusters}', f'{init_method}')
    filename = f'{dataset_name}.k{n_clusters}.r{random_state}.{init_method}.csv'
    csvpath = os.path.join(save_dir, filename)
    print(f"Exporting centroids to '{csvpath}'")
    os.makedirs(save_dir, exist_ok=True)
    centroids.to_csv(csvpath, mode='x')


# %% Load centroids
def load(csvpath: str) -> DataFrame:
    print(f"Loading centroids from '{csvpath}'")
    centroids = pd.read_csv(csvpath, index_col=0)
    centroids.columns.name = 'attribute'
    print(f"  shape: {centroids.shape}")
    return centroids


# %% END OF FILE
