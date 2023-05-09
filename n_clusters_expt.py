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
        description="Experiment with number of clusters.",
        argument_default=argparse.SUPPRESS
        )
    parser.add_argument('DATASET_NAME', type=str, help="dataset name")
    parser.add_argument('RANDOM_STATE', type=int, help="random state")
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

    # retrieve the relevant YAML files
    print(f"Retrieving relevant YAML files in '{experiments.CONFIG_DIR}'")
    config_paths = glob.glob(
        os.path.join(
            experiments.CONFIG_DIR,
            args.DATASET_NAME,
            'k=*',
            args.INIT_METHOD,
            f'r={args.RANDOM_STATE}',
            f'{args.DATASET_NAME}.k*.r{args.RANDOM_STATE}.l{args.LAMBDA_}.{args.INIT_METHOD}.yaml'
            ),
        )
    print(f"  {len(config_paths)} retrieved")
    print()

    config_path_dict = dict()
    for yamlpath in config_paths:
        filename = yamlpath.split('/')[-1]
        match = re.fullmatch(fr"^{args.DATASET_NAME}\.k(\d+)\.r{args.RANDOM_STATE}\.l{args.LAMBDA_}\.{args.INIT_METHOD}\.yaml$", filename)
        if match is None:
            raise ValueError(f"'{filename}' does not match the pattern '^{args.DATASET_NAME}\.k(\d+)\.r{args.RANDOM_STATE}\.l{args.LAMBDA_}\.{args.INIT_METHOD}\.yaml$'")
        n_clusters, = match.groups()
        n_clusters = int(n_clusters)
        config_path_dict[n_clusters] = yamlpath

    dataset = None
    for n_clusters in sorted(config_path_dict):
        # load experiment configuration
        yamlpath = config_path_dict[n_clusters]
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
        if experiment['n_clusters'] != n_clusters:
            raise ValueError(f"'{yamlpath}' has mismatch in name and contents")
        if experiment['random_state'] != args.RANDOM_STATE:
            raise ValueError(f"'{yamlpath}' has a different `random_state`")
        if experiment['init_method'] != args.INIT_METHOD:
            raise ValueError(f"'{yamlpath}' has a different `init_method`")
        init_path = experiment['init_path']
        if experiment['lambda_'] != args.LAMBDA_:
            raise ValueError(f"'{yamlpath}' has a different `lambda_`")

        # load initial centroids
        init_centroids = init.load(csvpath=init_path)
        print()

        _ = analysis.run_algorithms(n_clusters=n_clusters, dataset=dataset, init_centroids=init_centroids, lambda_=args.LAMBDA_, random_state=args.RANDOM_STATE, init_method=args.INIT_METHOD, compress=True)

    print(f"SCRIPT ENDED @{datetime.today().isoformat()}")


# %% END OF FILE
