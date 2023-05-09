#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
"""


# %% Libraries
import pandas as pd
from ruamel import yaml
from tabulate import tabulate
from typing import Final

from . import Dataset


# %% Dataset configuration path
SEX: Final[str] = 'data/datasets/CreditCard_SEX/creditcard.yaml'


# %% Categorical attributes
CATEGORICAL: list[str] = [
    'SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2',
    'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'DEFAULT'
    ]


# %% Load a processed Credit Card Clients dataset
def load(yamlpath: str) -> Dataset:
    def _to_dict(items):
        return {
            k: list(v) if isinstance(v, list) else v
            for k, v in items
            }

    dataset = dict()
    # load YAML file
    print(f"Loading dataset configuration from '{yamlpath}'")
    with open(yamlpath) as f:
        config = yaml.round_trip_load(f)
    dataset['name'] = config.pop('name')
    nonsensitive = list(config.pop('nonsensitive'))
    sensitive_info = _to_dict(items=config.pop('sensitive').items())
    if len(sensitive_info) > 1:
        raise RuntimeError(f"More than one sensitive attribute"
                           f" found: {list(sensitive_info)}")
    sensitive, dataset['sensitive_groups'] = list(sensitive_info.items())[0]
    _ = config.pop('predictor')    # we don't need the predictor
    # load dataset
    print(f"Loading processed Credit Card Clients dataset ({dataset['name']})"
          f" from '{config['filepath_or_buffer']}'")
    dtype = {attr: str for attr in CATEGORICAL}
    data = pd.read_csv(**config, dtype=dtype)
    data.columns.name = 'attribute'
    print(f"  shape: {data.shape}")
    dataset['X'] = data.filter(nonsensitive, axis='columns')
    dataset['s'] = data.filter([sensitive], axis='columns').squeeze()
    print(tabulate(
        {'dataset': [dataset['name']],
         'samples': [len(dataset['X'])],
         'dimensionality': [len(dataset['X'].columns)],
         'sensitive attribute': [dataset['s'].name],
         'sensitive groups': [len(dataset['sensitive_groups'])]
         },
        headers='keys',
        tablefmt='rounded_outline',
        stralign='right'
        ))
    return dataset


# %% END OF FILE
