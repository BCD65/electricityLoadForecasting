#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
#


def transform_input(data, transformation, parameter):
    if transformation == '':
        assert parameter == ''
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
        assert type(parameter) == pd.DateOffset
        ans = data.rolling('{0}H'.format(parameter.hours)).min()
    elif transformation == 'maximum':
        assert type(parameter) == pd.DateOffset
        ans = data.rolling('{0}H'.format(parameter.hours)).max()
    else:
        raise ValueError
    return ans
