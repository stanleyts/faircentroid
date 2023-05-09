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
from ruamel import yaml

from src import experiments, init


# %% Parse arguments
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate configurations for experiments.",
        argument_default=argparse.SUPPRESS
        )
    parser.add_argument('DATASET_MODULE', type=str, help="name of the dataset module")
    parser.add_argument('DATASET_CONFIG', type=str, help="path of the YAML file containing the dataset's configuration")
    parser.add_argument('--n_clusters', type=int, default=None, help="number of clusters to form (default: %(default)s)", dest="N_CLUSTERS")
    parser.add_argument('--lambdas', type=float, nargs='+', default=[.5], help="trade-off between fairness and utility (default: %(default)s)", dest='LAMBDAS')
    args = parser.parse_args()
    assert all(0 <= lambda_ <= 1 for lambda_ in args.LAMBDAS)
    if args.N_CLUSTERS is None:
        args.N_CLUSTERS = '*'
    return args


# %% main
if __name__ == '__main__':
    args = _parse_args()

    print(f"SCRIPT STARTED @{datetime.today().isoformat()}")
    print()

    dataset_name = args.DATASET_CONFIG.split('/')[-2]
    init_dir = os.path.join(init.SAVE_DIR, dataset_name)

    # retrieve the relevant CSV files
    print(f"Retrieving relevant CSV files in '{init_dir}'")
    init_paths = glob.glob(
        os.path.join(
            init_dir,
            f'k={args.N_CLUSTERS}',
            '*',
            f'{dataset_name}.k{args.N_CLUSTERS}.r*.*.csv'
            )
        )
    print(f"  {len(init_paths)} retrieved")
    print()

    base_dir = os.path.join(experiments.CONFIG_DIR, dataset_name)
    print(f"Generating experiment configurations in '{base_dir}'")
    for path in init_paths:
        init_filename = path.split('/')[-1]
        match = re.fullmatch(fr"^{dataset_name}\.k(\d+)\.r(\d+)\.(\w+)\.csv$", init_filename)
        if match is None:
            raise ValueError(f"'{init_filename}' does not match the pattern '^{dataset_name}\.k(\d+)\.r(\d+)\.(\w+)\.csv$'")
        n_clusters, random_state, init_method = match.groups()
        n_clusters = int(n_clusters)
        random_state = int(random_state)
        save_dir = os.path.join(base_dir, f'k={n_clusters}', f'{init_method}', f'r={random_state}')
        os.makedirs(save_dir, exist_ok=True)
        for lambda_ in set(args.LAMBDAS):
            config = {
                'dataset_module': args.DATASET_MODULE,
                'dataset_config': args.DATASET_CONFIG,
                'n_clusters': n_clusters,
                'random_state': random_state,
                'init_method': init_method,
                'init_path': path,
                'lambda_': lambda_
                }
            filename = f'{dataset_name}.k{n_clusters}.r{random_state}.l{lambda_}.{init_method}.yaml'
            with open(os.path.join(save_dir, filename), 'x') as f:
                yaml.round_trip_dump(config, f)
    print()

    print(f"SCRIPT ENDED @{datetime.today().isoformat()}")


# %% END OF FILE
