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
        description="Evaluate effect of λ.",
        argument_default=argparse.SUPPRESS
        )
    parser.add_argument('DATASET_MODULE', type=str, help="name of the dataset module")
    parser.add_argument('DATASET_CONFIG', type=str, help="path of the YAML file containing the dataset's configuration")
    parser.add_argument('--init_method', type=str, default='kmeans_plusplus', help="method for initialising centroids (default: %(default)s)", dest='INIT_METHOD')
    parser.add_argument('--n_clusters', type=int, help="number of clusters to form", dest='N_CLUSTERS')
    args = parser.parse_args()
    assert args.N_CLUSTERS > 0
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
            f'k={args.N_CLUSTERS}',
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
        match = re.fullmatch(fr"^{experiments.SAVE_DIR}/{dataset_name}/k={args.N_CLUSTERS}/{args.INIT_METHOD}/r=(\d+)$", save_dir)
        if match is None:
            raise ValueError(f"'{save_dir}' does not match the pattern '^{experiments.SAVE_DIR}/{dataset_name}/k={args.N_CLUSTERS}/{args.INIT_METHOD}/r=(\d+)$'")
        random_state, = match.groups()
        random_state = int(random_state)
        faircentroid_dirs = glob.glob(
            os.path.join(save_dir, 'faircentroid', 'l=*')
            )
        lambdas = list()
        for faircentroid_dir in faircentroid_dirs:
            match = re.fullmatch(fr"^{experiments.SAVE_DIR}/{dataset_name}/k={args.N_CLUSTERS}/{args.INIT_METHOD}/r={random_state}/faircentroid/l=(\d\.\d+)$", faircentroid_dir)
            if match is None:
                raise ValueError(f"'{faircentroid_dir}' does not match the pattern '^{experiments.SAVE_DIR}/{dataset_name}/k={args.N_CLUSTERS}/{args.INIT_METHOD}/r={random_state}/faircentroid/l=(\d\.\d+)$'")
            lambda_, = match.groups()
            lambda_ = float(lambda_)
            lambdas.append(lambda_)
        r_scores, r_bad = analysis.eval_lambdas(n_clusters=args.N_CLUSTERS, dataset=dataset, lambdas=lambdas, random_state=random_state, init_method=args.INIT_METHOD, compress=True)
        scores.append(r_scores)
        bad.extend(r_bad)
    print(f"Failed to read for {len(bad)} outputs")
    scores = pd.concat(scores, names=['random_state'])
    scores = scores.sort_index(level='random_state', sort_remaining=False)
    scores = scores.reindex(scores.index.unique('lambda_').sort_values(), level='lambda_')
    scores = scores.reindex(scores.index.unique('metric'), level='metric')
    print()
    # export scores
    analysis_dir = os.path.join(experiments.SAVE_DIR, dataset_name, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    scores_path = os.path.join(analysis_dir, 'lambda_eval_scores.csv')
    print(f"Exporting scores to '{scores_path}'")
    scores.to_csv(scores_path, mode='x')

    # aggregate over random states
    print("Aggregating over random states")
    grouped = scores.groupby(level=['n_clusters', 'init_method', 'metric', 'algorithm', 'lambda_'], sort=False)
    aggregated_scores = grouped.aggregate(mean='mean', std='std', n_random_states='count')
    if aggregated_scores['n_random_states'].nunique() > 1:
        print("Number of random states differ")
        print(aggregated_scores['n_random_states'].groupby(level='lambda_').min())
    aggregated_scores = aggregated_scores.drop('n_random_states', axis='columns')
    # export aggregated scores
    aggregated_scores_path = os.path.join(analysis_dir, 'lambda_eval_agg_scores.csv')
    print(f"Exporting aggregated scores to '{aggregated_scores_path}'")
    aggregated_scores.to_csv(aggregated_scores_path, mode='x')
    print()

    print(f"SCRIPT ENDED @{datetime.today().isoformat()}")


# %% END OF FILE
