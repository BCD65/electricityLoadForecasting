
import pandas as pd

"""
Hyperparameters for GAM
"""

def set_gam(hprm):    
    
    hprm.update({
                 'gam.univariate_functions'   : {
                                                 ('daylight',        '',           '')                          : ('',  ''),
                                                 ('holidays',        '',           '')                          : ('',  ''),
                                                 ('holidays',        'lag',        pd.DateOffset(hours = 24))   : ('',  ''),
                                                 ('holidays',        'lag',      - pd.DateOffset(hours = 24))   : ('',  ''),
                                                 #('hour',            '',           '')                          : ('s', ''), 
                                                 ('nebulosity',      '',           '')                          : ('s', ''),
                                                 ('target',          'lag',        pd.DateOffset(hours = 24))   : ('s', ''),
                                                 ('temperature',     '',           '')                          : ('s', ''), 
                                                 ('temperature',     'lag',        pd.DateOffset(hours = 24))   : ('s', ''),
                                                 ('temperature',     'smoothing',  0.99)                        : ('s', ''),
                                                 ('temperature',     'minimum',    pd.DateOffset(hours = 24))   : ('s', ''),
                                                 ('temperature',     'maximum',    pd.DateOffset(hours = 24))   : ('s', ''),
                                                 ('timestamp',       '',           '')                          : ('',  ''), 
                                                 ('week_day',        'binary',     '')                          : ('',  ''), 
                                                 #('week_hour',       '',           '')                          : ('s', 'k = 7, bs = "cc"'), 
                                                 ('xmas',            '',           '')                          : ('',  ''),
                                                 ('year_day',        '',           '')                          : ('s', ''),
                                                 },
                 'gam.bivariate_functions'    : {
                                                  # (('hour',            '',           ''),
                                                  #  ('week_day',        'binary',     ''))                         : ('by',), 
                                                 
                                                  # (('hour',            '',           ''),
                                                  #  ('year_day',        '',           ''))                         : ('ti', 'bs = c("tp", "tp")'), 
                                                 
                                                  (('year_day',        '',           ''),
                                                  ('week_day',        'binary',     ''))                         : ('by',), 
                                                 
                                                 (('target',          'lag',        pd.DateOffset(hours = 24)),
                                                  ('week_day',        'binary',     ''))                         : ('by',), 
                                                 },
                   })
    return hprm
