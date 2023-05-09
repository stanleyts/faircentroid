#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
"""


# %% Libraries
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
from datetime import datetime

from src import experiments


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
MM = 1/25.4
plt.rcParams['figure.figsize'] = (29*MM, 35*MM)
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = .01
plt.rcParams['savefig.transparent'] = True


# %% Parse arguments
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot effect of number of clusters.",
        argument_default=argparse.SUPPRESS
        )
    parser.add_argument('DATASET_NAME', type=str, help="dataset name")
    parser.add_argument('--init_method', type=str, default='kmeans_plusplus', help="method for initialising centroids (default: %(default)s)", dest='INIT_METHOD')
    return parser.parse_args()


# %% main
if __name__ == '__main__':
    args = _parse_args()

    print(f"SCRIPT STARTED @{datetime.today().isoformat()}")
    print()

    analysis_dir = os.path.join(experiments.SAVE_DIR, args.DATASET_NAME, 'analysis')

    # read aggregated scores
    aggregated_scores_path = os.path.join(analysis_dir, 'n_clusters_eval_agg_scores.csv')
    print(f"Loading aggregated scores from '{aggregated_scores_path}'")
    aggregated_scores = pd.read_csv(aggregated_scores_path, index_col=[0, 1, 2, 3])
    aggregated_scores = aggregated_scores.xs(args.INIT_METHOD, axis='index', level='init_method')

    # generate plots
    print(f"Generating plots and exporting to '{analysis_dir}'")
    algorithms = aggregated_scores.index.unique('algorithm')
    metric_dict = {
        'k-means objective': r'$k$-means objective',
        'fair k-means objective': r'fair $k$-means objective',
        }
    for metric in aggregated_scores.index.unique('metric'):
        data = aggregated_scores.xs(metric, axis='index', level='metric')['mean']
        data = data.unstack(level='algorithm')[algorithms]
        data.columns = data.columns.str.replace(r'^Fair-Centroid.*$', 'Fair-Centroid', regex=True)
        _ = plt.figure()
        ylabel = metric_dict.get(metric, metric) if args.DATASET_NAME == 'Adult_sex' else None
        ax = data.plot.line(style='.-', xlabel=r'$k$', ylabel=ylabel)
        _ = ax.set_xlim(left=0)
        _ = ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.legend(title=None)
        if args.DATASET_NAME != 'Adult_sex':
            ax.legend().set_visible(False)
        filename = f"n_clusters vs {metric} - {args.DATASET_NAME}.pdf"
        filepath = os.path.join(analysis_dir, filename)
        if os.path.isfile(filepath):
            raise FileExistsError
        ax.figure.savefig(filepath)
    print()

    # get elbow for Lloyd's algorithm
    data = aggregated_scores.xs(('k-means objective', 'Lloyd'), axis='index', level=('metric', 'algorithm'))['mean']
    data = data.sort_index()
    assert set(data.index.unique()) == set(range(data.index.min(), data.index.max()+1))
    slope_change = data.diff(periods=1).diff(periods=-1)
    k_elbow = slope_change.idxmin()
    print(f"Elbow at k={k_elbow}")
    print()

    print(f"SCRIPT ENDED @{datetime.today().isoformat()}")


# %% END OF FILE
