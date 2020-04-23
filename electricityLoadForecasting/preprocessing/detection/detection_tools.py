#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy    as np
import pandas as pd
import os
import pickle
from termcolor import colored
#
import electricityLoadForecasting.tools as tools



def compute_local_trimmed_mean(serie, ind, ind_error):
    # Use integer indices instead of dates for speed
    low_ratio    = 0.2 # threshold to find corrupted data with trimmed_mean
    nb_weeks     = 5
    window_ind   = np.arange(ind - 7*24*min(nb_weeks,
                                            (ind//(7*24)),
                                            ),
                             ind + 7*24*min(nb_weeks,
                                            (serie.shape[0] - ind)//(7*24),
                                            ),
                             7*24,
                             )
    valid_ind   = [jj
                   for jj in window_ind
                   if  jj not in ind_error
                   ]
    bad_window        = (len(valid_ind) <= 2)
    if bad_window:
        return bad_window, None, None
    loc_serie         = serie[valid_ind]
    trimmed_loc_serie = sorted(loc_serie)[1:-1]
    trimmed_mean      = np.mean(trimmed_loc_serie)
    too_low           = serie[ind] < (low_ratio * trimmed_mean)
    #too_high       = serie[ind] > (high_ratio * trimmed_mean) # Activate to discard sites with unusual large values
    return bad_window, too_low, trimmed_mean



def detection_errors(df_load, prefix = None):
    """
    Identify instant with zeros, Nan or negative values then with trimmed means
    Returns dictionaries with (key, value) as (name_of_site, instants with corrupted data)
    """
    print('detection_errors  - ', end = '')
    fname = os.path.join(prefix, 'dikt_errors.pkl')
    try:
        with open(fname, 'rb') as f:
            dikt_errors = pickle.load(f)
        print('Loaded dikt_errors')
    except Exception as e:
        print('\n{0}'.format(colored(e,'red')))
        print('dikt_errors not loaded')
        dikt_errors = {
                       'bad_win' : {},
                       'neg'     : {},
                       'null'    : {}, 
                       'too_low' : {},
                       'zero'    : {}, 
                       }
        ###############
        print('### Detect null/zero/neg values')
        for ii, site in enumerate(df_load.columns):
            print('{0:5} / {1:5} - '.format(ii, df_load.shape[1]), end = '')
            dates_null_values = df_load.index[np.where(pd.isnull(df_load[site]))[0]]
            dates_zero_values = df_load.index[np.where(df_load[site] == 0)[0]]
            dates_neg_values  = df_load.index[np.where(df_load[site] <0)[0]]
            ind_null_values   = np.where(pd.isnull(df_load[site]))[0]
            ind_zero_values   = np.where(df_load[site] == 0)[0]
            ind_neg_values    = np.where(df_load[site] <0)[0]
            nb_detections = dates_null_values.shape[0] + dates_zero_values.shape[0] + dates_neg_values.shape[0]
            print('{0:25} -> {1:5} detection{2}'.format(str(site), nb_detections, 's'*(nb_detections>1)))
            if dates_null_values.shape[0] > 0:
                dikt_errors['null'][site] = list(zip(ind_null_values, dates_null_values))
            if dates_zero_values.shape[0] > 0:
                dikt_errors['zero'][site] = list(zip(ind_zero_values, dates_zero_values))
            if dates_neg_values.shape[0]  > 0:    
                dikt_errors['neg'][site]  = list(zip(ind_neg_values, dates_neg_values))
        ###############
        print('### Detect errors with trimmed_mean')
        for ii, site in enumerate(df_load.columns):
            print('{0:5} / {1:5} - '.format(ii, df_load.shape[1]), end = '')
            dates_errors  = [dd
                             for error_type in dikt_errors
                             for ii, dd in dikt_errors[error_type].get(site, [])
                             ]
            ind_errors  = [ii
                           for error_type in dikt_errors
                           for ii, dd in dikt_errors[error_type].get(site, [])
                           ]
            detections_bad_window = []
            detections_too_low    = []
            for ind, date in enumerate(df_load.index):
                if date in dates_errors: 
                    continue
                bad_window, too_low, tmp_trimmed_mean = compute_local_trimmed_mean(df_load[site].values,
                                                                                   ind,
                                                                                   ind_errors,
                                                                                   )
                if bad_window:
                    detections_bad_window.append((ind, date))
                elif too_low:
                    detections_too_low.append((ind, date))
            nb_detections = len(detections_bad_window) + len(detections_too_low)
            print('{0:25} -> {1:5} detection{2}'.format(str(site), nb_detections, 's'*(nb_detections>1)))
            if len(detections_bad_window):
                dikt_errors['bad_win'][site] = detections_bad_window
            if len(detections_too_low):
                dikt_errors['too_low'][site] = detections_too_low
        ###############
        with open(fname, 'wb') as f:
            pickle.dump(dikt_errors, f)    
    print(*['\rlen(dikt_errors[{key}]) {fill}= {size}\n'.format(key = key, 
                                                                size = len(dikt_errors[key]),
                                                                fill = ' '*(7 - len(key)),
                                                                ) 
            for key in sorted(dikt_errors.keys())
            ],
          '\rdone\n{0}'.format('#'*tools.NB_SIGNS),
          )
    return dikt_errors

