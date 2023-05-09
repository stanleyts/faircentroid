#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
"""


# %% Libraries
import os
import pandas as pd
import shutil
import zipfile
from pandas import DataFrame, Series
from sklearn.cluster import _kmeans
from tabulate import tabulate
from typing import Final

from .. import utils
from ..experiments import SAVE_DIR


# %% Parameters
MAX_ITER: Final[int] = 200    # maximum number of iterations
TOL: Final[float] = 1e-4      # relative tolerance to declare convergence


# %% Compute objective
def compute_objective(object_losses: Series) -> float:
    return object_losses.sum()


# %% Estimate cluster assignment
def estimate_assignment(distances: DataFrame) -> Series:
    new_c = distances.idxmin(axis='columns')
    assert set(new_c.unique()) == set(distances.columns.values), "Some clusters are empty"
    new_c.name = distances.columns.name
    return new_c


# %% Estimate cluster centroids
def estimate_centroids(X: DataFrame, c: Series) -> DataFrame:
    assert c.index.equals(X.index)
    return X.groupby(c).mean()


# %% Run the algorithm
def run(n_clusters: int, X: DataFrame, init_centroids: DataFrame, max_iter: int = MAX_ITER, tol: float = TOL, export: bool = False, dataset_name: str | None = None, random_state: int | None = None, init_method: str | None = None, compress: bool = False) -> tuple[Series, DataFrame]:
    assert n_clusters > 0
    assert set(init_centroids.index) == set(range(n_clusters))
    assert init_centroids.columns.equals(X.columns)
    assert not init_centroids.duplicated().any()
    assert max_iter >= 1
    assert tol > 0
    assert dataset_name is not None if export is True else True
    assert random_state is not None if export is True else True
    assert init_method is not None if export is True else True

    print("Configuration:")
    print(tabulate(
        {'algorithm': ['Lloyd'],
         'n_clusters': [n_clusters],
         'max_iter': [max_iter],
         'tol': [tol]
         },
        headers='keys',
        tablefmt='rounded_outline',
        stralign='right'
        ))

    if export is True:
        save_dir = os.path.join(SAVE_DIR, dataset_name, f'k={n_clusters}', init_method, f'r={random_state}', 'lloyd')
        os.makedirs(save_dir, exist_ok=False)
        print(f"Exporting clusters and centroids to '{save_dir}'")
        c_dir = os.path.join(save_dir, 'clusters')
        centroids_dir = os.path.join(save_dir, 'centroids')
        os.makedirs(c_dir, exist_ok=False)
        os.makedirs(centroids_dir, exist_ok=False)

    # set tolerance dependent on the dataset (as done in sklearn's kmeans)
    tol_ = _kmeans._tolerance(X, tol)

    # initialisation
    print("Initialising centroids")
    centroids = init_centroids
    if export is True:
        centroids.to_csv(os.path.join(centroids_dir, '1.csv'))

    print("Running algorithm")
    distances = utils.compute_distances(X=X, centroids=centroids)
    c = estimate_assignment(distances=distances)
    if export is True:
        c.to_csv(os.path.join(c_dir, '1.csv'))

    # display stats
    print("╭──────┬───────────────┬───────────────┬────────────────╮")
    print("│ iter │     objective │ reassignments │ centroid shift │")
    print("├──────┼───────────────┼───────────────┼────────────────┤")
    object_losses = utils.get_object_losses(distances=distances, c=c)
    objective = compute_objective(object_losses=object_losses)
    print(f"│ {1:4} │ {objective:13.5f} │ {len(c):>13} │ {'-':>14} │")

    converged = False
    for i in range(2, max_iter+1):
        # estimate cluster centroids and cluster assignment
        new_centroids = estimate_centroids(X=X, c=c)
        if export is True:
            new_centroids.to_csv(os.path.join(centroids_dir, f'{i}.csv'))
        distances = utils.compute_distances(X=X, centroids=new_centroids)
        new_c = estimate_assignment(distances=distances)
        if export is True:
            new_c.to_csv(os.path.join(c_dir, f'{i}.csv'))

        n_reassignments = sum(new_c.eq(c) == False)
        centroid_shift = utils.compute_centroid_shift(centroids=centroids, new_centroids=new_centroids)

        c = new_c
        centroids = new_centroids

        # display stats
        object_losses = utils.get_object_losses(distances=distances, c=c)
        objective = compute_objective(object_losses=object_losses)
        print(f"│ {i:4} │ {objective:13.5f} │ {n_reassignments:>13} │ {centroid_shift:>14.5e} │")

        # stop if cluster assignment has not changed
        if n_reassignments == 0:
            converged = True
            print("╰──────┴───────────────┴───────────────┴────────────────╯")
            print(f"Converged at iteration {i}: strict convergence")
            break
        # stop if centroids have not changed enough
        if centroid_shift <= tol_:
            converged = True
            print("╰──────┴───────────────┴───────────────┴────────────────╯")
            print(f"Converged at iteration {i}: centroid shift {centroid_shift} within tolerance {tol_}")
            break

    if not converged:
        print("╰──────┴───────────────┴───────────────┴────────────────╯")
        print(f"Terminated at iteration {i}: maximum number of iterations reached")

    if export is True and compress is True:
        _ = shutil.make_archive(base_name=c_dir, format='zip', root_dir=c_dir)
        _ = shutil.make_archive(base_name=centroids_dir, format='zip', root_dir=centroids_dir)
        utils.check_archives(save_dir=save_dir)
        shutil.rmtree(c_dir, ignore_errors=True)
        shutil.rmtree(centroids_dir, ignore_errors=True)

    return c, centroids


# %% Load algorithm outputs
def load(n_clusters: int, dataset_name: str, random_state: int, init_method: str, i: int = -1, compress: bool = False) -> tuple[Series, DataFrame]:
    assert n_clusters > 0
    assert i == -1 or i > 0
    save_dir = os.path.join(SAVE_DIR, dataset_name, f'k={n_clusters}', init_method, f'r={random_state}', 'lloyd')
    if i == -1:
        i = utils.get_last_iteration(save_dir=save_dir, compress=compress)
    c_dir = os.path.join(save_dir, 'clusters')
    centroids_dir = os.path.join(save_dir, 'centroids')
    if compress is True:
        print(f"Loading clusters from '{c_dir}.zip/{i}.csv'")
        with zipfile.ZipFile(f'{c_dir}.zip') as zf:
            c = pd.read_csv(zf.open(f'{i}.csv'), index_col=0).squeeze()
        print(f"Loading centroids from '{centroids_dir}.zip/{i}.csv'")
        with zipfile.ZipFile(f'{centroids_dir}.zip') as zf:
            centroids = pd.read_csv(zf.open(f'{i}.csv'), index_col=0)
    else:
        c_path = os.path.join(c_dir, f'{i}.csv')
        print(f"Loading clusters from '{c_path}'")
        c = pd.read_csv(c_path, index_col=0).squeeze()
        centroids_path = os.path.join(centroids_dir, f'{i}.csv')
        print(f"Loading centroids from '{centroids_path}'")
        centroids = pd.read_csv(centroids_path, index_col=0)
    assert isinstance(c, Series)
    return c, centroids


# %% END OF FILE
