#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
"""


# %% Libraries
from importlib import import_module
from ruamel import yaml
from types import ModuleType
from typing import Final, TypedDict


# %% Paths
# Default subdirectory that would contain the experiment configurations
CONFIG_DIR: Final[str] = 'data/expt-configs'
# Default subdirectory that would contain the clusters and centroids
SAVE_DIR: Final[str] = 'data/expts'


# %% Experiment type hint
class Experiment(TypedDict):
    dataset_module: ModuleType
    dataset_config: str
    n_clusters: int
    random_state: int
    init_method: str
    init_path: str
    lambda_: float


# %% Load an experiment configuration
def load(yamlpath: str) -> Experiment:
    experiment = dict()
    # load YAML file
    print(f"Loading experiment configuration from '{yamlpath}'")
    with open(yamlpath) as f:
        config = yaml.round_trip_load(f)
    experiment['dataset_module'] = import_module(f"src.datasets.{config['dataset_module']}")
    experiment['dataset_config'] = config['dataset_config']
    if config['n_clusters'] <= 0:
        raise ValueError("`n_clusters` must be greater than 0")
    experiment['n_clusters'] = config['n_clusters']
    experiment['random_state'] = config['random_state']
    experiment['init_method'] = config['init_method']
    experiment['init_path'] = config['init_path']
    if not 0 <= config['lambda_'] <= 1:
        raise ValueError("`lambda_` must be between 0 and 1")
    experiment['lambda_'] = config['lambda_']
    return experiment


# %% END OF FILE
