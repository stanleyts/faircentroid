#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
"""


# %% Libraries
import argparse
import os
from datetime import datetime
from ruamel import yaml

from src import init


# %% Parse arguments
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate configurations for initialising centroids.",
        argument_default=argparse.SUPPRESS
        )
    parser.add_argument('DATASET_MODULE', type=str, help="name of the dataset module")
    parser.add_argument('DATASET_CONFIG', type=str, help="path of the YAML file containing the dataset's configuration")
    parser.add_argument('--min_clusters', type=int, default=3, help="minimum number of clusters (default: %(default)s)", dest='MIN_CLUSTERS')
    parser.add_argument('--max_clusters', type=int, default=12, help="maximum number of clusters (default: %(default)s)", dest='MAX_CLUSTERS')
    parser.add_argument('--n_random_states', type=int, default=100, help="number of random states (default: %(default)s)", dest='N_RANDOM_STATES')
    parser.add_argument('--init_method', type=str, default='kmeans_plusplus', help="method for initialising centroids (default: %(default)s)", dest='INIT_METHOD')
    args = parser.parse_args()
    assert 1 <= args.MIN_CLUSTERS <= args.MAX_CLUSTERS
    assert args.N_RANDOM_STATES >= 1
    return args


# %% main
if __name__ == '__main__':
    args = _parse_args()

    print(f"SCRIPT STARTED @{datetime.today().isoformat()}")
    print()

    dataset_name = args.DATASET_CONFIG.split('/')[-2]
    base_dir = os.path.join(init.CONFIG_DIR, dataset_name)
    print(f"Generating configurations for initialising centroids in '{base_dir}'")
    for n_clusters in range(args.MIN_CLUSTERS, args.MAX_CLUSTERS+1):
        save_dir = os.path.join(base_dir, f'k={n_clusters}', f'{args.INIT_METHOD}')
        os.makedirs(save_dir, exist_ok=False)
        for random_state in range(args.N_RANDOM_STATES):
            config = {
                'dataset_module': args.DATASET_MODULE,
                'dataset_config': args.DATASET_CONFIG,
                'n_clusters': n_clusters,
                'random_state': random_state,
                'init_method': args.INIT_METHOD
                }
            filename = f'{dataset_name}.k{n_clusters}.r{random_state}.{args.INIT_METHOD}.yaml'
            with open(os.path.join(save_dir, filename), 'x') as f:
                yaml.round_trip_dump(config, f)
    print()

    print(f"SCRIPT ENDED @{datetime.today().isoformat()}")


# %% END OF FILE
