#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
"""


# %% Libraries
from pandas import DataFrame, Series
from tabulate import tabulate

from . import utils


# %% K-means objective
def kmeans_objective(object_losses: Series) -> float:
    return object_losses.mean()


# %% Fair k-means objective
def fair_kmeans_objective(object_losses: Series, s: Series) -> float:
    assert object_losses.index.equals(s.index)
    return object_losses.groupby(s).mean().max()


# %% Fair centroid objective
def fair_centroid_objective(object_losses: Series, s: Series, c: Series) -> float:
    assert object_losses.index.equals(s.index)
    assert object_losses.index.equals(c.index)
    group_losses = object_losses.groupby([c, s]).mean()
    worst_losses = group_losses.groupby(level=c.name).max()
    return worst_losses.mean()


# %% Disparity in overall losses
def overall_disparity(object_losses: Series, s: Series, c: Series | None = None, standardise: bool = False) -> float:
    assert object_losses.index.equals(s.index)
    if standardise is True:
        assert c is not None
        assert object_losses.index.equals(c.index)
        object_losses = object_losses.groupby(c).transform(utils.standardise)
    return object_losses.groupby(s).mean().var(ddof=0)


# %% Disparity in losses averaged over clusters
def average_cluster_disparity(object_losses: Series, s: Series, c: Series, standardise: bool = True) -> float:
    assert object_losses.index.equals(s.index)
    assert object_losses.index.equals(c.index)
    if standardise is True:
        object_losses = object_losses.groupby(c).transform(utils.standardise)
    group_losses = object_losses.groupby([c, s]).mean()
    return group_losses.groupby(level=c.name).var(ddof=0).mean()


# %% Evaluate against all metrics
def evaluate(X: DataFrame, s: Series, c: Series, centroids: DataFrame, display: bool = True) -> Series:
    print("Evaluating against all metrics")
    object_losses = utils.compute_object_losses(X=X, c=c, centroids=centroids)
    scores = Series({
        'average cluster disparity': average_cluster_disparity(object_losses=object_losses, s=s, c=c, standardise=True),
        'k-means objective': kmeans_objective(object_losses=object_losses),
        'fair centroid objective': fair_centroid_objective(object_losses=object_losses, s=s, c=c),
        'fair k-means objective': fair_kmeans_objective(object_losses=object_losses, s=s),
#        'overall disparity (not standardised)': overall_disparity(object_losses=object_losses, s=s, standardise=False),
#        'overall disparity': overall_disparity(object_losses=object_losses, s=s, c=c, standardise=True),
#        'average cluster disparity (not standardised)': average_cluster_disparity(object_losses=object_losses, s=s, c=c, standardise=False),
        }, name='score')
    scores.index.name = 'metric'
    if display is True:
        # parts = tabulate(scores.to_frame(), tablefmt='rounded_outline').split('\n')
        # sep = parts[0].replace("╭", "├").replace("┬", "┼").replace("╮", "┤")
        # print("\n".join([*parts[:2], sep, *parts[2:5], sep, *parts[5:]]))
        print(tabulate(scores.to_frame(), tablefmt='rounded_outline'))
    return scores


# %% END OF FILE
