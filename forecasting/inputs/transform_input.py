#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
#


def transform_input(data, transformation, parameter):
    if transformation is None:
        assert parameter is None
        ans = data
    elif transformation == 'lag':
        assert type(parameter) == pd.DateOffset
        ans = data.set_index(data.index + parameter).reindex(data.index, method = 'bfill')
    elif transformation == 'smoothing':
        assert type(parameter) == float
        ans = data.ewm(alpha = parameter, adjust = False, axis = 0).mean()
    elif transformation == 'difference':
        assert type(parameter) == pd.DateOffset
        ans = data - data.set_index(data.index + parameter).reindex(data.index, method = 'bfill')
    elif transformation == 'minimum':
        assert type(parameter) == str
        ans = data.rolling(parameter).min()
    elif transformation == 'maximum':
        assert type(parameter) == str
        ans = data.rolling(parameter).max()
    else:
        raise ValueError
    return ans
