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
from importlib import import_module
from ruamel import yaml

from src import init


# %% Parse arguments
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialise centroids.",
        argument_default=argparse.SUPPRESS
        )
    parser.add_argument('DATASET_NAME', type=str, help="dataset name")
    return parser.parse_args()


# %% main
if __name__ == '__main__':
    args = _parse_args()

    print(f"SCRIPT STARTED @{datetime.today().isoformat()}")
    print()

    # retrieve the relevant YAML files
    print(f"Retrieving relevant YAML files in '{init.CONFIG_DIR}'")
    config_paths = sorted(glob.glob(
        os.path.join(
            init.CONFIG_DIR,
            args.DATASET_NAME,
            'k=*',
            '*',
            f'{args.DATASET_NAME}.k*.r*.*.yaml'
            )
        ))
    print(f"  {len(config_paths)} retrieved")
    print()

    dataset = None
    for yamlpath in config_paths:
        filename = yamlpath.split('/')[-1]
        match = re.fullmatch(fr"^{args.DATASET_NAME}\.k(\d+)\.r(\d+)\.(\w+)\.yaml$", filename)
        if match is None:
            raise ValueError(f"'{filename}' does not match the pattern '^{args.DATASET_NAME}\.k(\d+)\.r(\d+)\.(\w+)\.yaml$'")
        n_clusters, random_state, init_method = match.groups()
        n_clusters = int(n_clusters)
        assert n_clusters > 0
        random_state = int(random_state)
        init_method = getattr(import_module("src.init"), init_method)

        # load configuration
        print(f"Loading configuration from '{yamlpath}'")
        with open(yamlpath) as f:
            config = yaml.round_trip_load(f)
        if dataset is None:
            # load dataset
            dataset_module = import_module(f"src.datasets.{config['dataset_module']}")
            dataset_config = config['dataset_config']
            dataset = dataset_module.load(yamlpath=dataset_config)
            print()
            if dataset['name'] != args.DATASET_NAME:
                raise ValueError(f"'{yamlpath}' has mismatch in path")
            X = dataset['X']
        elif config['dataset_config'] != dataset_config:
            raise ValueError(f"'{yamlpath}' has a different `dataset_config`")
        if config['n_clusters'] != n_clusters:
            raise ValueError(f"'{yamlpath}' has mismatch in name and contents")
        if config['random_state'] != random_state:
            raise ValueError(f"'{yamlpath}' has mismatch in name and contents")
        if config['init_method'] != init_method.__name__:
            raise ValueError(f"'{yamlpath}' has mismatch in name and contents")

        # initialise centroids
        print(f"Initialising centroids using {init_method.__name__}")
        centroids = init_method(X=X.copy(), n_clusters=n_clusters, random_state=random_state)

        # export centroids
        init.export(
            centroids=centroids,
            dataset_name=args.DATASET_NAME,
            random_state=random_state,
            init_method=init_method.__name__
            )
        print()

    print(f"SCRIPT ENDED @{datetime.today().isoformat()}")


# %% END OF FILE
