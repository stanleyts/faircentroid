#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
"""


# %% Libraries
from pandas import DataFrame, Series
from typing import TypedDict


# %% Dataset type hint
class Dataset(TypedDict):
    name: str
    sensitive_groups: list[str]
    X: DataFrame
    s: Series


# %% END OF FILE
