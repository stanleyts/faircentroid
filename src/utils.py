#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
"""


# %% Libraries
import glob
import numpy as np
import os
import pandas as pd
import re
import zipfile
from collections.abc import Sequence
from pandas import DataFrame, Index, MultiIndex, Series
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


# %% Compute objects' distances from all centroids
def compute_distances(X: DataFrame, centroids: DataFrame) -> DataFrame:
    assert centroids.columns.equals(X.columns)
    distances = cdist(X, centroids, metric='sqeuclidean')
    return DataFrame(distances, index=X.index, columns=centroids.index)


# %% Get objects' losses
def get_object_losses(distances: DataFrame, c: Series) -> Series:
    assert distances.index.equals(c.index)
    assert set(c.unique()) == set(distances.columns)
    idx, cols = pd.factorize(c)
    object_losses = distances.reindex(cols, axis='columns').to_numpy()[np.arange(len(c)), idx]
    return Series(object_losses, index=c.index, name='object loss')


# %% Compute objects' losses
def compute_object_losses(X: DataFrame, c: Series, centroids: DataFrame) -> Series:
    distances = compute_distances(X=X, centroids=centroids)
    object_losses = get_object_losses(distances=distances, c=c)
    return object_losses


# %% Compute overall groups' losses
def compute_overall_group_losses(object_losses: Series, s: Series, sensitive_groups: Sequence[str] | None = None) -> Series:
    assert object_losses.index.equals(s.index)
    overall_group_losses = object_losses.groupby(s, sort=False).mean()
    overall_group_losses.name = 'overall group loss'
    if sensitive_groups is None:
        return overall_group_losses
    assert set(s.unique()) == set(sensitive_groups)
    index = Index(sensitive_groups, name=s.name)
    return overall_group_losses.reindex(index, fill_value=0)


# %% Compute groups' statistics
def compute_group_stats(object_losses: Series, s: Series, c: Series, sensitive_groups: Sequence[str], n_clusters: int) -> DataFrame:
    assert object_losses.index.equals(s.index)
    assert c.index.equals(s.index)
    assert set(s.unique()) == set(sensitive_groups)
    assert set(c.unique()) == set(range(n_clusters))
    group_stats = object_losses.groupby([c, s], sort=False).aggregate(['sum', 'count'])
    index = MultiIndex.from_product([range(n_clusters), sensitive_groups], names=group_stats.index.names)
    return group_stats.reindex(index, fill_value=0)


# %% Get groups' losses
def get_group_losses(group_stats: DataFrame) -> Series:
    group_losses = group_stats['sum'].divide(group_stats['count']).fillna(0)
    group_losses.name = 'group loss'
    return group_losses


# %% Compute groups' losses in each cluster
def compute_group_losses(object_losses: Series, s: Series, c: Series, sensitive_groups: Sequence[str] | None = None, n_clusters: int | None = None) -> Series:
    assert object_losses.index.equals(s.index)
    assert c.index.equals(s.index)
    group_losses = object_losses.groupby([c, s], sort=False).mean()
    group_losses.name = 'group loss'
    if sensitive_groups is None or n_clusters is None:
        return group_losses
    assert set(s.unique()) == set(sensitive_groups)
    assert set(c.unique()) == set(range(n_clusters))
    index = MultiIndex.from_product([range(n_clusters), sensitive_groups], names=group_losses.index.names)
    return group_losses.reindex(index, fill_value=0)


# %% Get worst-off group in each cluster
def get_worstoff(group_losses: Series) -> dict[int, str]:
    worstoff = group_losses.groupby(level='cluster', sort=False).idxmax()
    return dict(worstoff.values)


# %% Quantify centroid shift
def compute_centroid_shift(centroids: DataFrame, new_centroids: DataFrame) -> float:
    assert new_centroids.index.equals(centroids.index)
    assert new_centroids.columns.equals(centroids.columns)
    return np.linalg.norm(new_centroids.subtract(centroids), ord='fro')


# %% Standardise a Series
def standardise(ser: Series) -> Series:
    values = ser.values.reshape(-1, 1)
    standardised = StandardScaler().fit_transform(values)
    return Series(standardised.reshape(1, -1)[0], index=ser.index)


# %% Get last iteration of algorithm
def get_last_iteration(save_dir: str, compress: bool) -> int:
    last = dict()
    for output in ['clusters', 'centroids']:
        dirpath = os.path.join(save_dir, output)
        if compress is True:
            with zipfile.ZipFile(f'{dirpath}.zip') as zf:
                paths = zf.namelist()
        else:
            paths = glob.glob(os.path.join(dirpath, '*.csv'))
        for path in paths:
            filename = path.split('/')[-1]
            match = re.fullmatch(r"^(\d+)\.csv$", filename)
            if match is None:
                raise ValueError(f"File name in '{path}' does not match the pattern '^(\d+)\.csv$'")
            i = int(match.group(1))
            if output not in last or last[output] < i:
                last[output] = i
    if last['clusters'] != last['centroids']:
        raise ValueError(f"Last iteration differs for clusters ({last['clusters']}) and centroids ({last['centroids']})")
    return last['clusters']


# %% Check if archived data is same as the original
def check_archives(save_dir: str):
    last = get_last_iteration(save_dir=save_dir, compress=False)
    last_zip = get_last_iteration(save_dir=save_dir, compress=True)
    if last_zip != last:
        raise ValueError(f"Last iteration differs for original ({last}) and compressed ({last_zip})")
    for output in ['clusters', 'centroids']:
        dirpath = os.path.join(save_dir, output)
        with zipfile.ZipFile(f'{dirpath}.zip') as zf:
            for i in range(1, last+1):
                data = pd.read_csv(os.path.join(dirpath, f'{i}.csv'), index_col=0)
                data_zip = pd.read_csv(zf.open(f'{i}.csv'), index_col=0)
                if not data_zip.equals(data):
                    raise ValueError(f"{output} differ at '{i}.csv'")


# %% END OF FILE
