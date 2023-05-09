#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
"""


# %% Libraries
import pandas as pd
from collections.abc import Sequence
from pandas import DataFrame, Series
from tabulate import tabulate

from src.algorithms import lloyd, fairlloyd_gd, faircentroid
from src.datasets import Dataset
from src import metrics


# %% Run algorithms
def run_algorithms(n_clusters: int, dataset: Dataset, init_centroids: DataFrame, lambda_: float, random_state: int, init_method: str, compress: bool) -> Series:
    dataset_name = dataset['name']
    sensitive_groups = dataset['sensitive_groups']
    X = dataset['X']
    s = dataset['s']

    try:
        # run Lloyd's algorithm
        print("Running Lloyd's algorithm")
        c_lloyd, centroids_lloyd = lloyd.run(n_clusters=n_clusters, X=X.copy(), init_centroids=init_centroids.copy(), export=True, dataset_name=dataset_name, random_state=random_state, init_method=init_method, compress=compress)
    except FileExistsError:
        print("Export directory already exists; the experiment may have previously been run")
        # load outputs
        c_lloyd, centroids_lloyd = lloyd.load(n_clusters=n_clusters, dataset_name=dataset_name, random_state=random_state, init_method=init_method, compress=compress)
    print()
    # evaluate
    scores_lloyd = metrics.evaluate(X=X.copy(), s=s.copy(), c=c_lloyd, centroids=centroids_lloyd)
    print()

    try:
        # run Fair-Lloyd algorithm
        print("Running Fair-Lloyd algorithm")
        c_fairlloyd, centroids_fairlloyd = fairlloyd_gd.run(n_clusters=n_clusters, X=X.copy(), s=s.copy(), init_centroids=init_centroids.copy(), export=True, dataset_name=dataset_name, random_state=random_state, init_method=init_method, compress=compress)
    except FileExistsError:
        print("Export directory already exists; the experiment may have previously been run")
        # load outputs
        c_fairlloyd, centroids_fairlloyd = fairlloyd_gd.load(n_clusters=n_clusters, dataset_name=dataset_name, random_state=random_state, init_method=init_method, compress=compress)
    print()
    # evaluate
    scores_fairlloyd = metrics.evaluate(X=X.copy(), s=s.copy(), c=c_fairlloyd, centroids=centroids_fairlloyd)
    print()

    try:
        # run Fair-Centroid algorithm
        print("Running Fair-Centroid algorithm")
        c_faircentroid, centroids_faircentroid = faircentroid.run(n_clusters=n_clusters, X=X.copy(), s=s.copy(), init_centroids=init_centroids.copy(), sensitive_groups=sensitive_groups, lambda_=lambda_, export=True, dataset_name=dataset_name, random_state=random_state, init_method=init_method, compress=compress)
    except FileExistsError:
        print("Export directory already exists; the experiment may have previously been run")
        # load outputs
        c_faircentroid, centroids_faircentroid = faircentroid.load(n_clusters=n_clusters, lambda_=lambda_, dataset_name=dataset_name, random_state=random_state, init_method=init_method, compress=compress)
    print()
    # evaluate
    scores_faircentroid = metrics.evaluate(X=X.copy(), s=s.copy(), c=c_faircentroid, centroids=centroids_faircentroid)
    print()

    # compare all algorithms
    print("Algorithm comparison")
    scores = pd.concat({'Lloyd': scores_lloyd, 'Fair-Lloyd': scores_fairlloyd, f'Fair-Centroid (λ={lambda_})': scores_faircentroid}, names=['algorithm'])
    parts = tabulate(scores.unstack(level='algorithm'), headers='keys', tablefmt='rounded_outline').split('\n')
    sep = parts[2]
    print("\n".join([*parts[:4], sep, *parts[4:7], sep, *parts[7:]]))
    print()
    scores = pd.concat({(n_clusters, init_method, random_state): scores}, names=['n_clusters', 'init_method', 'random_state'])
    scores = scores.reorder_levels(['n_clusters', 'init_method', 'metric', 'random_state', 'algorithm'])
    scores = scores.reindex(scores.index.unique('metric'), level='metric')

    return scores


# %% Evaluate algorithms
def eval_algorithms(n_clusters: int, dataset: Dataset, lambda_: float, random_state: int, init_method: str, compress: bool) -> Series:
    dataset_name = dataset['name']
    X = dataset['X']
    s = dataset['s']

    # evaluate Lloyd's algorithm
    c_lloyd, centroids_lloyd = lloyd.load(n_clusters=n_clusters, dataset_name=dataset_name, random_state=random_state, init_method=init_method, compress=compress)
    scores_lloyd = metrics.evaluate(X=X.copy(), s=s.copy(), c=c_lloyd, centroids=centroids_lloyd, display=False)
    # evaluate Fair-Lloyd algorithm
    c_fairlloyd, centroids_fairlloyd = fairlloyd_gd.load(n_clusters=n_clusters, dataset_name=dataset_name, random_state=random_state, init_method=init_method, compress=compress)
    scores_fairlloyd = metrics.evaluate(X=X.copy(), s=s.copy(), c=c_fairlloyd, centroids=centroids_fairlloyd, display=False)
    # evaluate Fair-Centroid algorithm
    c_faircentroid, centroids_faircentroid = faircentroid.load(n_clusters=n_clusters, lambda_=lambda_, dataset_name=dataset_name, random_state=random_state, init_method=init_method, compress=compress)
    scores_faircentroid = metrics.evaluate(X=X.copy(), s=s.copy(), c=c_faircentroid, centroids=centroids_faircentroid, display=False)
    # concatenate all scores
    scores = pd.concat({'Lloyd': scores_lloyd, 'Fair-Lloyd': scores_fairlloyd, f'Fair-Centroid (λ={lambda_})': scores_faircentroid}, names=['algorithm'])
    scores = pd.concat({(n_clusters, init_method, random_state): scores}, names=['n_clusters', 'init_method', 'random_state'])
    scores = scores.reorder_levels(['n_clusters', 'init_method', 'metric', 'random_state', 'algorithm'])
    scores = scores.reindex(scores.index.unique('metric'), level='metric')

    return scores


# %% Run Fair-Centroid with different lambdas
def run_lambdas(n_clusters: int, dataset: Dataset, init_centroids: DataFrame, lambdas: Sequence[float], random_state: int, init_method: str, compress: bool) -> Series:
    dataset_name = dataset['name']
    sensitive_groups = dataset['sensitive_groups']
    X = dataset['X']
    s = dataset['s']

    scores = dict()
    for lambda_ in lambdas:
        try:
            # run Fair-Centroid algorithm
            print("Running Fair-Centroid algorithm")
            c, centroids = faircentroid.run(n_clusters=n_clusters, X=X.copy(), s=s.copy(), init_centroids=init_centroids.copy(), sensitive_groups=sensitive_groups, lambda_=lambda_, export=True, dataset_name=dataset_name, random_state=random_state, init_method=init_method, compress=compress)
        except FileExistsError:
            print("Export directory already exists; the experiment may have previously been run")
            # load outputs
            c, centroids = faircentroid.load(n_clusters=n_clusters, lambda_=lambda_, dataset_name=dataset_name, random_state=random_state, init_method=init_method, compress=compress)
        print()
        # evaluate
        scores[f'λ={lambda_}'] = metrics.evaluate(X=X.copy(), s=s.copy(), c=c, centroids=centroids)
        print()

    # compare across lambdas
    print("λ comparison")
    scores = pd.concat(scores, names=['lambda_'])
    parts = tabulate(scores.unstack(level='lambda_'), headers='keys', tablefmt='rounded_outline').split('\n')
    sep = parts[2]
    print("\n".join([*parts[:4], sep, *parts[4:7], sep, *parts[7:]]))
    print()
    scores.index = scores.index.map(lambda x: (float(x[0][2:]), x[1]))
    scores = pd.concat({(n_clusters, init_method, random_state, 'Fair-Centroid'): scores}, names=['n_clusters', 'init_method', 'random_state', 'algorithm'])
    scores = scores.reorder_levels(['n_clusters', 'init_method', 'metric', 'random_state', 'algorithm', 'lambda_'])
    scores = scores.reindex(scores.index.unique('metric'), level='metric')

    return scores


# %% Evaluate Fair-Centroid with different lambdas
def eval_lambdas(n_clusters: int, dataset: Dataset, lambdas: Sequence[float], random_state: int, init_method: str, compress: bool) -> tuple[Series, list[tuple[int, float]]]:
    dataset_name = dataset['name']
    X = dataset['X']
    s = dataset['s']

    scores = dict()
    bad = list()
    # evaluate Lloyd's algorithm
    c_lloyd, centroids_lloyd = lloyd.load(n_clusters=n_clusters, dataset_name=dataset_name, random_state=random_state, init_method=init_method, compress=compress)
    scores[('Lloyd', 0)] = metrics.evaluate(X=X.copy(), s=s.copy(), c=c_lloyd, centroids=centroids_lloyd, display=False)
    # evaluate Fair-Centroid algorithm for different lambdas
    for lambda_ in lambdas:
        try:
            c_faircentroid, centroids_faircentroid = faircentroid.load(n_clusters=n_clusters, lambda_=lambda_, dataset_name=dataset_name, random_state=random_state, init_method=init_method, compress=compress)
            scores[('Fair-Centroid', lambda_)] = metrics.evaluate(X=X.copy(), s=s.copy(), c=c_faircentroid, centroids=centroids_faircentroid, display=False)
        except FileNotFoundError as e:
            print(f"FileNotFoundError for r={random_state}, l={lambda_}")
            bad.append((random_state, lambda_))
    # concatenate all scores
    scores = pd.concat(scores, names=['algorithm', 'lambda_'])
    scores = pd.concat({(n_clusters, init_method, random_state): scores}, names=['n_clusters', 'init_method', 'random_state'])
    scores = scores.reorder_levels(['n_clusters', 'init_method', 'metric', 'random_state', 'algorithm', 'lambda_'])
    scores = scores.reindex(scores.index.unique('metric'), level='metric')

    return scores, bad


# %% END OF FILE
