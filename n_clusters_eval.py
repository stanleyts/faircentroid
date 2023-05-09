#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
"""


# %% Libraries
import argparse
import glob
import os
import pandas as pd
import re
from datetime import datetime
from importlib import import_module

from src import analysis, experiments


# %% Parse arguments
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate effect of number of clusters.",
        argument_default=argparse.SUPPRESS
        )
    parser.add_argument('DATASET_MODULE', type=str, help="name of the dataset module")
    parser.add_argument('DATASET_CONFIG', type=str, help="path of the YAML file containing the dataset's configuration")
    parser.add_argument('--init_method', type=str, default='kmeans_plusplus', help="method for initialising centroids (default: %(default)s)", dest='INIT_METHOD')
    parser.add_argument('--lambda_', type=float, default=.5, help="trade-off between fairness and utility (default: %(default)s)", dest='LAMBDA_')
    args = parser.parse_args()
    assert 0 <= args.LAMBDA_ <= 1
    return args


# %% main
if __name__ == '__main__':
    args = _parse_args()

    print(f"SCRIPT STARTED @{datetime.today().isoformat()}")
    print()

    # load dataset
    dataset = import_module(f"src.datasets.{args.DATASET_MODULE}").load(yamlpath=args.DATASET_CONFIG)
    print()
    dataset_name = dataset['name']

    # retrieve the relevant output directories
    print(f"Retrieving relevant output directories in '{experiments.SAVE_DIR}'")
    save_dirs = glob.glob(
        os.path.join(
            experiments.SAVE_DIR,
            dataset_name,
            'k=*',
            args.INIT_METHOD,
            'r=*'
            )
        )
    print(f"  {len(save_dirs)} retrieved")
    print()

    # evaluate
    scores = list()
    bad = list()
    for save_dir in save_dirs:
        match = re.fullmatch(fr"^{experiments.SAVE_DIR}/{dataset_name}/k=(\d+)/{args.INIT_METHOD}/r=(\d+)$", save_dir)
        if match is None:
            raise ValueError(f"'{save_dir}' does not match the pattern '^{experiments.SAVE_DIR}/{dataset_name}/k=(\d+)/{args.INIT_METHOD}/r=(\d+)$'")
        n_clusters, random_state = match.groups()
        n_clusters = int(n_clusters)
        random_state = int(random_state)
        try:
            scores.append(analysis.eval_algorithms(n_clusters=n_clusters, dataset=dataset, lambda_=args.LAMBDA_, random_state=random_state, init_method=args.INIT_METHOD, compress=True))
        except FileNotFoundError:
            print(f"FileNotFoundError for k={n_clusters}, r={random_state}")
            bad.append((n_clusters, random_state))
    print(f"Failed to read for {len(bad)} outputs")
    scores = pd.concat(scores)
    scores = scores.sort_index(level=('n_clusters', 'random_state'), sort_remaining=False)
    scores = scores.reindex(scores.index.unique('metric'), level='metric')
    print()
    # export scores
    analysis_dir = os.path.join(experiments.SAVE_DIR, dataset_name, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    scores_path = os.path.join(analysis_dir, 'n_clusters_eval_scores.csv')
    print(f"Exporting scores to '{scores_path}'")
    scores.to_csv(scores_path, mode='x')

    # aggregate over random states
    print("Aggregating over random states")
    grouped = scores.groupby(level=['n_clusters', 'init_method', 'metric', 'algorithm'], sort=False)
    aggregated_scores = grouped.aggregate(mean='mean', std='std', n_random_states='count')
    if aggregated_scores['n_random_states'].nunique() > 1:
        print("Number of random states differ")
        print(aggregated_scores['n_random_states'].groupby(level='n_clusters').min())
    aggregated_scores = aggregated_scores.drop('n_random_states', axis='columns')
    # export aggregated scores
    aggregated_scores_path = os.path.join(analysis_dir, 'n_clusters_eval_agg_scores.csv')
    print(f"Exporting aggregated scores to '{aggregated_scores_path}'")
    aggregated_scores.to_csv(aggregated_scores_path, mode='x')
    print()

    print(f"SCRIPT ENDED @{datetime.today().isoformat()}")


# %% END OF FILE
