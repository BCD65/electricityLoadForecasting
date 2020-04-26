
import pandas as pd

"""
Hyperparameters for GAM
"""

def set_gam(hprm):    
    
    hprm.update({
                 'gam.univariate_functions'   : {
                                                 ('daylight',        '',           '')                          : ('s', ''),
                                                 ('holidays',        '',           '')                          : ('',  ''),
                                                 ('holidays',        'lag',        pd.DateOffset(hours = 24))   : ('',  ''),
                                                 ('holidays',        'lag',      - pd.DateOffset(hours = 24))   : ('',  ''),
                                                 ('hour',            '',           '')                          : ('s', ''), 
                                                 ('nebulosity',      '',           '')                          : ('s', ''),
                                                 ('target',          'lag',        pd.DateOffset(hours = 24))   : ('s', ''),
                                                 ('temperature',     '',           '')                          : ('s', ''), 
                                                 ('temperature',     'lag',        pd.DateOffset(hours = 24))   : ('s', ''),
                                                 ('temperature',     'smoothing',  0.99)                        : ('s', ''),
                                                 ('temperature',     'minimum',    pd.DateOffset(hours = 24))   : ('s', ''),
                                                 ('temperature',     'maximum',    pd.DateOffset(hours = 24))   : ('s', ''),
                                                 ('timestamp',       '',           '')                          : ('',  ''), 
                                                 ('week_day_binary', '',           '')                          : ('',  ''), 
                                                 #('week_hour',       '',           '')                          : ('s', 'k = 7, bs = "cc"'), 
                                                 ('xmas',            '',           '')                          : ('',  ''),
                                                 ('year_day',        '',           '')                          : ('s', ''),
                                                 },
                 'gam.bivariate_functions'    : {
                                                 #('hour',      'week_day')  : ('ti', 'bs = c("cc", "tp"), k = c(24, 7)'),
                                                 #('hour',      'year_day')  : ('ti', 'bs = c("cc", "tp")'), 
                                                 #('week_day',  'year_day')  : ('ti', 'bs = c("tp", "tp")'), 
                                                 #('targetlag', 'week_day')  : ('ti', 'bs = c("tp", "tp")'), 
                                                 },
                   })
    return hprm
