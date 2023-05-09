#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
"""


# %% Libraries
import argparse
import glob
import os
import re
from datetime import datetime

from src import analysis, experiments, init


# %% Parse arguments
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment with different λ.",
        argument_default=argparse.SUPPRESS
        )
    parser.add_argument('DATASET_NAME', type=str, help="dataset name")
    parser.add_argument('RANDOM_STATE', type=int, help="random state")
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

    # retrieve the relevant YAML files
    print(f"Retrieving relevant YAML files in '{experiments.CONFIG_DIR}'")
    config_paths = glob.glob(
        os.path.join(
            experiments.CONFIG_DIR,
            args.DATASET_NAME,
            f'k={args.N_CLUSTERS}',
            args.INIT_METHOD,
            f'r={args.RANDOM_STATE}',
            f'{args.DATASET_NAME}.k{args.N_CLUSTERS}.r{args.RANDOM_STATE}.l*.{args.INIT_METHOD}.yaml'
            )
        )
    print(f"  {len(config_paths)} retrieved")
    print()

    config_path_dict = dict()
    for yamlpath in config_paths:
        filename = yamlpath.split('/')[-1]
        match = re.fullmatch(fr"^{args.DATASET_NAME}\.k{args.N_CLUSTERS}\.r{args.RANDOM_STATE}\.l(\d\.\d+)\.{args.INIT_METHOD}\.yaml$", filename)
        if match is None:
            raise ValueError(f"'{filename}' does not match the pattern '^{args.DATASET_NAME}\.k{args.N_CLUSTERS}\.r{args.RANDOM_STATE}\.l(\d\.\d+)\.{args.INIT_METHOD}\.yaml$'")
        lambda_, = match.groups()
        lambda_ = float(lambda_)
        config_path_dict[lambda_] = yamlpath

    dataset = None
    init_centroids = None
    scores = dict()
    for lambda_, yamlpath in config_path_dict.items():
        # load experiment configuration
        experiment = experiments.load(yamlpath=yamlpath)
        print()
        if dataset is None:
            # load dataset
            dataset_module = experiment['dataset_module']
            dataset_config = experiment['dataset_config']
            dataset = dataset_module.load(yamlpath=dataset_config)
            print()
            if dataset['name'] != args.DATASET_NAME:
                raise ValueError
        elif experiment['dataset_module'] != dataset_module:
            raise ValueError(f"'{yamlpath}' has a different `dataset_module`")
        elif experiment['dataset_config'] != dataset_config:
            raise ValueError(f"'{yamlpath}' has a different `dataset_config`")
        if experiment['n_clusters'] != args.N_CLUSTERS:
            raise ValueError(f"'{yamlpath}' has a different `n_clusters`")
        if experiment['random_state'] != args.RANDOM_STATE:
            raise ValueError(f"'{yamlpath}' has a different `random_state`")
        if experiment['init_method'] != args.INIT_METHOD:
            raise ValueError(f"'{yamlpath}' has a different `init_method`")
        if init_centroids is None:
            # load initial centroids
            init_path = experiment['init_path']
            init_centroids = init.load(csvpath=init_path)
            print()
        elif experiment['init_path'] != init_path:
            raise ValueError(f"'{yamlpath}' has a different `init_path`")
        if experiment['lambda_'] != lambda_:
            raise ValueError(f"'{yamlpath}' has mismatch in name and contents")

    lambdas = sorted(list(config_path_dict))
    _ = analysis.run_lambdas(n_clusters=args.N_CLUSTERS, dataset=dataset, init_centroids=init_centroids, lambdas=lambdas, random_state=args.RANDOM_STATE, init_method=args.INIT_METHOD, compress=True)

    print(f"SCRIPT ENDED @{datetime.today().isoformat()}")


# %% END OF FILE
