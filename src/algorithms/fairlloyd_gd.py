#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
"""


# %% Libraries
import numpy as np
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
ETA: Final[float] = 1e-3      # learning rate
PHI: Final[float] = 1000      # multiplier for amplifying loss
PATIENCE: Final[int] = 10     # maximum number of iterations with no reduction in centroid shift


# %% Compute objective
def compute_objective(object_losses: Series, s: Series) -> float:
    assert object_losses.index.equals(s.index)
    return object_losses.groupby(s, sort=False).mean().max()


# %% Estimate cluster assignment
def estimate_assignment(distances: DataFrame) -> Series:
    new_c = distances.idxmin(axis='columns')
    assert set(new_c.unique()) == set(distances.columns.values), "Some clusters are empty"
    new_c.name = distances.columns.name
    return new_c


# %% Compute derivative of approximate objective
def _compute_approx_objective_derivative(centroids: DataFrame, group_sums: DataFrame, group_sizes: Series, overall_group_sizes: Series, overall_group_losses: Series, n_clusters: int, phi: float) -> DataFrame:
    # exp(phi * mean_{x \in S}(loss(x))
    # subtract maximum to prevent overflow due to np.exp
    w = np.exp(phi * (overall_group_losses - max(overall_group_losses)))

    # \sum_{x \in C \cap S}(x)  -  \mu_C * |C \cap S|
    difference = group_sums.subtract(centroids.multiply(group_sizes, axis='index', level=centroids.index.name))
    # -2 * 1/|S| * (\sum_{x \in C \cap S}(x)  -  \mu_C * |C \cap S|)
    overall_group_losses_derivative = -2 * difference.divide(overall_group_sizes, axis='index', level=overall_group_sizes.index.name)

    # derivative
    weighted_sum = overall_group_losses_derivative.multiply(w, axis='index', level=w.index.name).groupby(level=centroids.index.name, sort=False).sum()
    derivative = weighted_sum.divide(sum(w))
    assert set(derivative.index) == set(range(n_clusters))
    assert derivative.notna().all(axis=None)
    return derivative.reindex(range(n_clusters))


# %% Estimate cluster centroids
def estimate_centroids(X: DataFrame, s: Series, c: Series, centroids: DataFrame, n_clusters: int, tol_: float, eta: float, phi: float, patience: int) -> DataFrame:
    assert s.index.equals(X.index)
    assert c.index.equals(X.index)
    assert n_clusters > 0
    assert set(centroids.index) == set(range(n_clusters))
    assert set(c.unique()) == set(range(n_clusters))
    assert centroids.columns.equals(X.columns)
    assert not centroids.duplicated().any()
    assert tol_ > 0
    assert eta > 0
    assert phi >= 1
    assert patience >= 1

    grouped = X.groupby([c, s], sort=False)
    # \sum_{x \in C \cap S}(x)
    group_sums = grouped.sum()
    # |C \cap S|
    group_sizes = grouped.size()
    # 1/|S|
    overall_group_sizes = group_sizes.groupby(level=s.name, sort=False).sum()

    n_bad_iter = 0
    min_centroid_shift = float('inf')
    while True:
        object_losses = utils.compute_object_losses(X=X, c=c, centroids=centroids)
        overall_group_losses = utils.compute_overall_group_losses(object_losses=object_losses, s=s)
        objective_derivative = _compute_approx_objective_derivative(centroids=centroids, group_sums=group_sums, group_sizes=group_sizes, overall_group_sizes=overall_group_sizes, overall_group_losses=overall_group_losses, n_clusters=n_clusters, phi=phi)
        new_centroids = centroids.subtract(eta * objective_derivative)
        centroid_shift = utils.compute_centroid_shift(centroids=centroids, new_centroids=new_centroids)
        if centroid_shift <= tol_:
            break
        if centroid_shift < min_centroid_shift:
            min_centroid_shift = centroid_shift
            n_bad_iter = 0
        else:
            n_bad_iter += 1
            if n_bad_iter >= patience:
                break
        centroids = new_centroids
    return centroids


# %% Run the algorithm
def run(n_clusters: int, X: DataFrame, s: Series, init_centroids: DataFrame, max_iter: int = MAX_ITER, tol: float = TOL, eta: float = ETA, phi: float = PHI, patience: int = PATIENCE, export: bool = False, dataset_name: str | None = None, random_state: int | None = None, init_method: str | None = None, compress: bool = False) -> tuple[Series, DataFrame]:
    assert n_clusters > 0
    assert s.index.equals(X.index)
    assert set(init_centroids.index) == set(range(n_clusters))
    assert init_centroids.columns.equals(X.columns)
    assert not init_centroids.duplicated().any()
    assert max_iter >= 1
    assert tol > 0
    assert eta > 0
    assert phi >= 1
    assert patience >= 1
    assert dataset_name is not None if export is True else True
    assert random_state is not None if export is True else True
    assert init_method is not None if export is True else True

    print("Configuration:")
    print(tabulate(
        {'algorithm': ['Fair-Lloyd (Gradient Descent)'],
         'n_clusters': [n_clusters],
         'max_iter': [max_iter],
         'tol': [tol],
         'eta': [eta],
         'phi': [phi],
         'patience': [patience]
         },
        headers='keys',
        tablefmt='rounded_outline',
        stralign='right'
        ))

    if export is True:
        save_dir = os.path.join(SAVE_DIR, dataset_name, f'k={n_clusters}', init_method, f'r={random_state}', 'fairlloyd_gd')
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
    print("╭──────┬───────────┬───────────────┬────────────────╮")
    print("│ iter │ objective │ reassignments │ centroid shift │")
    print("├──────┼───────────┼───────────────┼────────────────┤")
    object_losses = utils.get_object_losses(distances=distances, c=c)
    objective = compute_objective(object_losses=object_losses, s=s)
    print(f"│ {1:4} │ {objective:9.5f} │ {len(c):>13} │ {'-':>14} │")

    converged = False
    for i in range(2, max_iter+1):
        # estimate cluster centroids and cluster assignment
        new_centroids = estimate_centroids(X=X, s=s, c=c, centroids=centroids, n_clusters=n_clusters, tol_=tol_, eta=eta, phi=phi, patience=patience)
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
        objective = compute_objective(object_losses=object_losses, s=s)
        print(f"│ {i:4} │ {objective:9.5f} │ {n_reassignments:>13} │ {centroid_shift:>14.5e} │")

        # stop if cluster assignment has not changed
        if n_reassignments == 0:
            converged = True
            print("╰──────┴───────────┴───────────────┴────────────────╯")
            print(f"Converged at iteration {i}: strict convergence")
            break
        # stop if centroids have not changed enough
        if centroid_shift <= tol_:
            converged = True
            print("╰──────┴───────────┴───────────────┴────────────────╯")
            print(f"Converged at iteration {i}: centroid shift {centroid_shift} within tolerance {tol_}")
            break

    if not converged:
        print("╰──────┴───────────┴───────────────┴────────────────╯")
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
    save_dir = os.path.join(SAVE_DIR, dataset_name, f'k={n_clusters}', init_method, f'r={random_state}', 'fairlloyd_gd')
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


# %% Compute derivative of exact objective
def _compute_exact_objective_derivative(centroids: DataFrame, group_sums: DataFrame, group_sizes: Series, overall_group_sizes: Series, overall_group_losses: Series, n_clusters: int) -> DataFrame:
    # identify overall worst-off group
    worstoff = overall_group_losses.idxmax()
    # \sum_{x \in C \cap worstoff}(x)  -  \mu_C * |C \cap worstoff|
    difference = group_sums.subtract(centroids.multiply(group_sizes, axis='index', level=centroids.index.name)).xs(worstoff, axis='index', level=overall_group_sizes.index.name)
    # -2 * 1/|worstoff| * (\sum_{x \in C \cap worstoff}(x)  -  \mu_C * |C \cap worstoff|)
    derivative = difference.multiply(-2 / overall_group_sizes.loc[worstoff])
    assert set(derivative.index) == set(range(n_clusters))
    assert derivative.notna().all(axis=None)
    return derivative.reindex(range(n_clusters))


# %% END OF FILE
