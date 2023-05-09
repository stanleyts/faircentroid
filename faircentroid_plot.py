#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
"""


# %% Libraries
import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime

from src import experiments, utils
from src.datasets import adult, creditcard
from src.algorithms import faircentroid


# %% Set rcParams
plt.rcParams['lines.linewidth'] = .5
plt.rcParams['lines.markersize'] = 1
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 5
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.edgecolor'] = 'grey'
plt.rcParams['axes.linewidth'] = .3
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.formatter.limits'] = [-2, 6]
plt.rcParams['xtick.major.size'] = plt.rcParams['ytick.major.size'] = 2
plt.rcParams['xtick.major.pad'] = plt.rcParams['ytick.major.pad'] = 1
plt.rcParams['xtick.color'] = plt.rcParams['ytick.color'] = 'grey'
plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = 'small'
plt.rcParams['grid.color'] = 'gainsboro'
plt.rcParams['grid.linewidth'] = .3
plt.rcParams['grid.alpha'] = 1
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.handlelength'] = 1.0
plt.rcParams['figure.labelsize'] = 'medium'
MM = 1/25.4
plt.rcParams['figure.figsize'] = (75*MM, 70*MM)
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = .01
plt.rcParams['savefig.transparent'] = True


# %% Parse arguments
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot trends in groups' representativities for Fair-Centroid.",
        argument_default=argparse.SUPPRESS
        )
    parser.add_argument('--dataset_name', type=str, default='Adult_race', help="dataset name (default: %(default)s)", dest='DATASET_NAME')
    parser.add_argument('--n_clusters', type=int, default=3, help="number of clusters to form (default: %(default)s)", dest='N_CLUSTERS')
    parser.add_argument('--init_method', type=str, default='kmeans_plusplus', help="method for initialising centroids (default: %(default)s)", dest='INIT_METHOD')
    parser.add_argument('--random_state', type=int, default=46, help="random state (default: %(default)s)", dest='RANDOM_STATE')
    parser.add_argument('--lambda_', type=float, default=.5, help="trade-off between fairness and utility (default: %(default)s)", dest='LAMBDA_')
    return parser.parse_args()


# %% main
if __name__ == '__main__':
    args = _parse_args()

    print(f"SCRIPT STARTED @{datetime.today().isoformat()}")
    print()

    analysis_dir = os.path.join(experiments.SAVE_DIR, args.DATASET_NAME, 'analysis')

    # load dataset
    if args.DATASET_NAME == 'Adult_sex':
        dataset = adult.load(yamlpath=adult.SEX)
    elif args.DATASET_NAME == 'Adult_race':
        dataset = adult.load(yamlpath=adult.RACE)
    elif args.DATASET_NAME == 'CreditCard_SEX':
        dataset = creditcard.load(yamlpath=creditcard.SEX)
    print()
    dataset_name = dataset['name']
    sensitive_groups = dataset['sensitive_groups']
    X = dataset['X']
    s = dataset['s']

    # evaluate clusters
    group_losses = dict()
    i = 1
    while True:
        try:
            c, centroids = faircentroid.load(n_clusters=args.N_CLUSTERS, lambda_=args.LAMBDA_, dataset_name=args.DATASET_NAME, random_state=args.RANDOM_STATE, init_method=args.INIT_METHOD, i=i, compress=True)
            object_losses = utils.compute_object_losses(X=X, c=c, centroids=centroids)
            group_losses[i] = utils.compute_group_losses(object_losses=object_losses, s=s, c=c, sensitive_groups=sensitive_groups, n_clusters=args.N_CLUSTERS)
            i += 1
        except KeyError:
            i -= 1
            break
    group_losses = pd.concat(group_losses, names=['iteration'])
    group_losses = group_losses.reorder_levels(['cluster', 'iteration', s.name]).sort_index()
    group_losses = group_losses.unstack(level=s.name)
    mm = 1/25.4
    fig, ax = plt.subplots(nrows=args.N_CLUSTERS)
    for j in range(args.N_CLUSTERS):
        _ = group_losses.loc[j].plot.line(style='.-', ylabel=None, ax=ax[j])
        if j != args.N_CLUSTERS - 1:
            _ = ax[j].xaxis.set_ticklabels([])
        _ = ax[j].set_xlabel('')
        ax[j].legend().set_visible(False)
    _ = fig.supxlabel('iteration', y=.05)
    _ = fig.supylabel('group representativity', x=.05)
    filename = f"{args.DATASET_NAME}.k{args.N_CLUSTERS}.r{args.RANDOM_STATE}.pdf"
    filepath = os.path.join(analysis_dir, filename)
    if os.path.isfile(filepath):
        raise FileExistsError
    fig.savefig(filepath)
    print()

    print(f"SCRIPT ENDED @{datetime.today().isoformat()}")


# %% END OF FILE
