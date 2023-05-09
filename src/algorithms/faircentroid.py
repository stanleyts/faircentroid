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
from collections.abc import Sequence
from pandas import DataFrame, Series
from sklearn.cluster import _kmeans
from tabulate import tabulate
from typing import Final

from .. import utils
from ..experiments import SAVE_DIR


# %% Parameters
LAMBDA_: Final[float] = .5    # trade-off between fairness (1) and utility (0)
MAX_ITER: Final[int] = 200    # maximum number of iterations
TOL: Final[float] = 1e-4      # relative tolerance to declare convergence
ETA: Final[float] = 1e-3      # learning rate
PHI: Final[float] = 1000      # multiplier for amplifying loss
PATIENCE: Final[int] = 10     # maximum number of iterations with no reduction in centroid shift


# %% Compute utility
def compute_utility(group_stats: DataFrame) -> float:
    assert (group_stats >= 0).all(axis=None)
    return group_stats['sum'].sum()


# %% Compute fairness
def compute_fairness(group_stats: DataFrame) -> float:
    assert (group_stats >= 0).all(axis=None)
    group_losses = group_stats['sum'].divide(group_stats['count'])
    return group_losses.groupby(level='cluster', sort=False).max().sum()


# %% Compute objective
def compute_objective(group_stats: DataFrame, lambda_: float) -> float:
    assert 0 <= lambda_ <= 1
    objective = 0
    if lambda_ < 1:
        utility_coeff = (1-lambda_) / group_stats['count'].sum()
        objective += utility_coeff * compute_utility(group_stats=group_stats)
    if lambda_ > 0:
        fairness_coeff = lambda_ / len(group_stats.index.unique(level='cluster'))
        objective += fairness_coeff * compute_fairness(group_stats=group_stats)
    return objective


# %% Estimate object's best reassignment
def _get_best_reassignment(S: str, C: int, d: Series, group_stats: DataFrame, group_losses: Series, worstoff: dict[int, str], objective: float, lambda_: float) -> tuple[int, DataFrame, float]:
    # test reassignment only if one of the following three cases hold
    # case 2: `S` is the worst-off group in `C` and `d[C]` > `group_losses[C, S]`
    if S == worstoff[C] and d[C] > group_losses[C, S]:
        candidates = [(new_C, d_new_C)
                      for new_C, d_new_C in d.items()
                      if new_C != C
                      ]
    else:
        candidates = list()
        for new_C, d_new_C in d.items():
            if new_C == C:
                continue
            # case 1: `d[new_C]` < `d[C]`
            elif d_new_C < d[C]:
                candidates.append((new_C, d_new_C))
            # case 3: `S` is the worst-off group in `new_C` and `d[new_C]` < `group_losses[new_C, S]`
            elif S == worstoff[new_C] and d_new_C < group_losses[new_C, S]:
                candidates.append((new_C, d_new_C))

    if len(candidates) == 0:
        return C, group_stats, objective

    best_C = C
    best_group_stats = group_stats
    best_objective = objective

    # subtract object's current loss and decrease count for cluster `C`
    group_stats_sub = group_stats.copy()
    sum_ = group_stats_sub.at[(C, S), 'sum'] - d[C]
    assert sum_ >= -1e-12
    group_stats_sub.at[(C, S), 'sum'] = max(0, sum_)
    group_stats_sub.at[(C, S), 'count'] -= 1
    assert (group_stats_sub.loc[(C, S)] >= 0).all(axis=None)

    for new_C, d_new_C in candidates:
        # add object's new loss and increase count for cluster `new_C`
        new_group_stats = group_stats_sub.copy()
        new_group_stats.at[(new_C, S), 'sum'] += d_new_C
        new_group_stats.at[(new_C, S), 'count'] += 1
        new_objective = compute_objective(group_stats=new_group_stats, lambda_=lambda_)
        if new_objective < best_objective:
            best_C = new_C
            best_group_stats = new_group_stats
            best_objective = new_objective

    return best_C, best_group_stats, best_objective


# %% Estimate cluster assignment
def estimate_assignment(distances: DataFrame, s: Series, c: Series, sensitive_groups: Sequence[str], n_clusters: int, lambda_: float) -> Series:
    assert distances.index.equals(s.index)
    assert c.index.equals(s.index)
    assert len(sensitive_groups) > 0
    assert set(s.unique()) == set(sensitive_groups)
    assert n_clusters > 0
    assert set(distances.columns) == set(range(n_clusters))
    assert set(c.unique()) == set(range(n_clusters))
    assert 0 <= lambda_ <= 1

    new_c = c.copy()
    object_losses = utils.get_object_losses(distances=distances, c=new_c)
    group_stats = utils.compute_group_stats(object_losses=object_losses, s=s, c=new_c, sensitive_groups=sensitive_groups, n_clusters=n_clusters)
    group_losses = utils.get_group_losses(group_stats=group_stats)
    worstoff = utils.get_worstoff(group_losses=group_losses)
    objective = compute_objective(group_stats=group_stats, lambda_=lambda_)

    # reorder objects in `distances` so that the ones with least maximum disadvantage are seen first
    # this way, the objects with higher maximum disadvantages would likely remain in their nearest clusters
    reordered_distances = distances.reindex(index=distances.max(axis='columns').sort_values().index)

    # reassign objects to clusters such that the objective decreases
    for x, d in reordered_distances.iterrows():
        new_C, group_stats, objective = _get_best_reassignment(S=s[x], C=new_c[x], d=d, group_stats=group_stats, group_losses=group_losses, worstoff=worstoff, objective=objective, lambda_=lambda_)
        if new_C != new_c[x]:
            assert group_stats.loc[new_c[x], 'count'].sum() > 0, f"Cluster {new_c[x]} is empty"
            new_c[x] = new_C
#            assert (abs(utils.compute_group_stats(object_losses=utils.get_object_losses(distances=distances, c=new_c), s=s, c=new_c, sensitive_groups=sensitive_groups, n_clusters=n_clusters) - group_stats) < 1e-6).all(axis=None)
            group_losses = utils.get_group_losses(group_stats=group_stats)
            worstoff = utils.get_worstoff(group_losses=group_losses)

    assert set(new_c.unique()) == set(distances.columns.values), "Some clusters are empty"
    return new_c


# %% Compute derivative of utility
def _compute_utility_derivative(centroids: DataFrame, group_sums: DataFrame, group_sizes: Series, n_clusters: int) -> DataFrame:
    # \sum_{x \in C}(x)
    cluster_sums = group_sums.groupby(level=centroids.index.name, sort=False).sum()
    # |C|
    cluster_sizes = group_sizes.groupby(level=centroids.index.name, sort=False).sum()
    # \sum_{x \in C}(x)  -  \mu_C * |C|
    difference = cluster_sums.subtract(centroids.multiply(cluster_sizes, axis='index', level=centroids.index.name))
    # -2 * (\sum_{x \in C}(x)  -  \mu_C * |C|)
    derivative = -2 * difference
    assert set(derivative.index) == set(range(n_clusters))
    assert derivative.notna().all(axis=None)
    return derivative.reindex(range(n_clusters))


# %% Compute derivative of approximate fairness
def _compute_approx_fairness_derivative(centroids: DataFrame, group_sums: DataFrame, group_sizes: Series, group_losses: Series, n_clusters: int, phi: float) -> DataFrame:
    # exp(phi * loss(C, S))
    # subtract maximum in each cluster to prevent overflow due to np.exp
    max_per_cluster = group_losses.groupby(level=centroids.index.name, sort=False).max()
    w = np.exp(phi * group_losses.subtract(max_per_cluster, level=centroids.index.name))

    # mean_{x \in C \cap S}(x)  -  \mu_C
    group_means = group_sums.divide(group_sizes, axis='index', level=centroids.index.name)
    difference = group_means.subtract(centroids, axis='index', level=centroids.index.name)
    # -2 * (mean_{x \in C \cap S}(x)  -  \mu_C)
    group_losses_derivative = -2 * difference

    # derivative
    weighted_sum = group_losses_derivative.multiply(w, axis='index').groupby(level=centroids.index.name, sort=False).sum()
    weight_sum = w.groupby(level=centroids.index.name, sort=False).sum()
    derivative = weighted_sum.divide(weight_sum, axis='index')
    assert set(derivative.index) == set(range(n_clusters))
    assert derivative.notna().all(axis=None)
    return derivative.reindex(range(n_clusters))


# %% Compute derivative of approximate objective
def _compute_approx_objective_derivative(centroids: DataFrame, group_sums: DataFrame, group_sizes: Series, group_losses: Series, n_clusters: int, lambda_: float, phi: float) -> DataFrame:
    derivative = 0
    if lambda_ < 1:
        utility_coeff = (1-lambda_) / group_sizes.sum()
        derivative += utility_coeff * _compute_utility_derivative(centroids=centroids, group_sums=group_sums, group_sizes=group_sizes, n_clusters=n_clusters)
    if lambda_ > 0:
        fairness_coeff = lambda_ / n_clusters
        derivative += fairness_coeff * _compute_approx_fairness_derivative(centroids=centroids, group_sums=group_sums, group_sizes=group_sizes, group_losses=group_losses, n_clusters=n_clusters, phi=phi)
    return derivative


# %% Estimate cluster centroids
def estimate_centroids(X: DataFrame, s: Series, c: Series, centroids: DataFrame, n_clusters: int, lambda_: float, tol_: float, eta: float, phi: float, patience: int) -> DataFrame:
    assert s.index.equals(X.index)
    assert c.index.equals(X.index)
    assert n_clusters > 0
    assert set(centroids.index) == set(range(n_clusters))
    assert set(c.unique()) == set(range(n_clusters))
    assert centroids.columns.equals(X.columns)
    assert not centroids.duplicated().any()
    assert 0 <= lambda_ <= 1
    assert tol_ > 0
    assert eta > 0
    assert phi >= 1
    assert patience >= 1

    grouped = X.groupby([c, s], sort=False)
    # \sum_{x \in C \cap S}(x)
    group_sums = grouped.sum()
    # |C \cap S|
    group_sizes = grouped.size()

    n_bad_iter = 0
    min_centroid_shift = float('inf')
    while True:
        object_losses = utils.compute_object_losses(X=X, c=c, centroids=centroids)
        group_losses = utils.compute_group_losses(object_losses=object_losses, s=s, c=c)
        objective_derivative = _compute_approx_objective_derivative(centroids=centroids, group_sums=group_sums, group_sizes=group_sizes, group_losses=group_losses, n_clusters=n_clusters, lambda_=lambda_, phi=phi)
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
def run(n_clusters: int, X: DataFrame, s: Series, init_centroids: DataFrame, sensitive_groups: Sequence[str], lambda_: float = LAMBDA_, max_iter: int = MAX_ITER, tol: float = TOL, eta: float = ETA, phi: float = PHI, patience: int = PATIENCE, export: bool = False, dataset_name: str | None = None, random_state: int | None = None, init_method: str | None = None, compress: bool = False) -> tuple[Series, DataFrame]:
    assert n_clusters > 0
    assert s.index.equals(X.index)
    assert set(init_centroids.index) == set(range(n_clusters))
    assert init_centroids.columns.equals(X.columns)
    assert not init_centroids.duplicated().any()
    assert len(sensitive_groups) > 0
    assert set(s.unique()) == set(sensitive_groups)
    assert 0 <= lambda_ <= 1
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
        {'algorithm': ['Fair-Centroid'],
         'n_clusters': [n_clusters],
         'max_iter': [max_iter],
         'tol': [tol],
         'eta': [eta],
         'phi': [phi],
         'patience': [patience],
         'lambda_': [lambda_]
         },
        headers='keys',
        tablefmt='rounded_outline',
        stralign='right'
        ))

    if export is True:
        save_dir = os.path.join(SAVE_DIR, dataset_name, f'k={n_clusters}', init_method, f'r={random_state}', 'faircentroid', f'l={lambda_}')
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
    c = distances.idxmin(axis='columns')
    assert set(c.unique()) == set(range(n_clusters)), "Some clusters are empty"
    c.name = distances.columns.name
    if export is True:
        c.to_csv(os.path.join(c_dir, '1.csv'))

    # display stats
    print("╭──────┬───────────────┬───────────┬───────────┬───────────────┬────────────────╮")
    print("│ iter │       utility |  fairness | objective │ reassignments │ centroid shift │")
    print("├──────┼───────────────┼───────────┼───────────┼───────────────┼────────────────┤")
    object_losses = utils.get_object_losses(distances=distances, c=c)
    group_stats = utils.compute_group_stats(object_losses=object_losses, s=s, c=c, sensitive_groups=sensitive_groups, n_clusters=n_clusters)
    utility = compute_utility(group_stats=group_stats)
    fairness = compute_fairness(group_stats=group_stats)
    objective = compute_objective(group_stats=group_stats, lambda_=lambda_)
    print(f"│ {1:4} │ {utility:13.5f} │ {fairness:9.5f} │ {objective:9.5f} │ {len(c):>13} │ {'-':>14} │")

    converged = False
    for i in range(2, max_iter+1):
        # estimate cluster centroids and cluster assignment
        new_centroids = estimate_centroids(X=X, s=s, c=c, centroids=centroids, n_clusters=n_clusters, lambda_=lambda_, tol_=tol_, eta=eta, phi=phi, patience=patience)
        if export is True:
            new_centroids.to_csv(os.path.join(centroids_dir, f'{i}.csv'))
        distances = utils.compute_distances(X=X, centroids=new_centroids)
        new_c = estimate_assignment(distances=distances, s=s, c=c, sensitive_groups=sensitive_groups, n_clusters=n_clusters, lambda_=lambda_)
        if export is True:
            new_c.to_csv(os.path.join(c_dir, f'{i}.csv'))

        n_reassignments = sum(new_c.eq(c) == False)
        centroid_shift = utils.compute_centroid_shift(centroids=centroids, new_centroids=new_centroids)

        c = new_c
        centroids = new_centroids

        # display stats
        object_losses = utils.get_object_losses(distances=distances, c=c)
        group_stats = utils.compute_group_stats(object_losses=object_losses, s=s, c=c, sensitive_groups=sensitive_groups, n_clusters=n_clusters)
        utility = compute_utility(group_stats=group_stats)
        fairness = compute_fairness(group_stats=group_stats)
        objective = compute_objective(group_stats=group_stats, lambda_=lambda_)
        print(f"│ {i:4} │ {utility:13.5f} │ {fairness:9.5f} │ {objective:9.5f} │ {n_reassignments:>13} │ {centroid_shift:>14.5e} │")

        # stop if cluster assignment has not changed
        if n_reassignments == 0:
            converged = True
            print("╰──────┴───────────────┴───────────┴───────────┴───────────────┴────────────────╯")
            print(f"Converged at iteration {i}: strict convergence")
            break
        # stop if centroids have not changed enough
        if centroid_shift <= tol_:
            converged = True
            print("╰──────┴───────────────┴───────────┴───────────┴───────────────┴────────────────╯")
            print(f"Converged at iteration {i}: centroid shift {centroid_shift} within tolerance {tol_}")
            break

    if not converged:
        print("╰──────┴───────────────┴───────────┴───────────┴───────────────┴────────────────╯")
        print(f"Terminated at iteration {i}: maximum number of iterations reached")

    if export is True and compress is True:
        _ = shutil.make_archive(base_name=c_dir, format='zip', root_dir=c_dir)
        _ = shutil.make_archive(base_name=centroids_dir, format='zip', root_dir=centroids_dir)
        utils.check_archives(save_dir=save_dir)
        shutil.rmtree(c_dir, ignore_errors=True)
        shutil.rmtree(centroids_dir, ignore_errors=True)

    return c, centroids


# %% Load algorithm outputs
def load(n_clusters: int, lambda_: float, dataset_name: str, random_state: int, init_method: str, i: int = -1, compress: bool = False) -> tuple[Series, DataFrame]:
    assert n_clusters > 0
    assert i == -1 or i > 0
    assert 0 <= lambda_ <= 1
    save_dir = os.path.join(SAVE_DIR, dataset_name, f'k={n_clusters}', init_method, f'r={random_state}', 'faircentroid', f'l={lambda_}')
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


# %% Compute derivative of exact fairness
def _compute_exact_fairness_derivative(centroids: DataFrame, group_sums: DataFrame, group_sizes: Series, group_losses: Series, n_clusters: int) -> DataFrame:
    # identify worst-off group in each cluster
    worstoff = list(utils.get_worstoff(group_losses=group_losses).items())
    # mean_{x \in C \cap worstoff(C)}(x)  -  \mu_C
    group_means = group_sums.divide(group_sizes, axis='index', level=centroids.index.name).loc[worstoff]
    sensitive_attr = [name for name in group_means.index.names if name != centroids.index.name][0]
    group_means = group_means.droplevel(level=sensitive_attr, axis='index')
    difference = group_means.subtract(centroids, axis='index')
    # -2 * (mean_{x \in C \cap worstoff(C)}(x)  -  \mu_C)
    derivative = -2 * difference
    assert set(derivative.index) == set(range(n_clusters))
    assert derivative.notna().all(axis=None)
    return derivative.reindex(range(n_clusters))


# %% Compute derivative of exact objective
def _compute_exact_objective_derivative(centroids: DataFrame, group_sums: DataFrame, group_sizes: Series, group_losses: Series, n_clusters: int, lambda_: float) -> DataFrame:
    derivative = 0
    if lambda_ < 1:
        utility_coeff = (1-lambda_) / group_sizes.sum()
        derivative += utility_coeff * _compute_utility_derivative(centroids=centroids, group_sums=group_sums, group_sizes=group_sizes, n_clusters=n_clusters)
    if lambda_ > 0:
        fairness_coeff = lambda_ / n_clusters
        derivative += fairness_coeff * _compute_exact_fairness_derivative(centroids=centroids, group_sums=group_sums, group_sizes=group_sizes, group_losses=group_losses, n_clusters=n_clusters)
    return derivative


# %% END OF FILE
