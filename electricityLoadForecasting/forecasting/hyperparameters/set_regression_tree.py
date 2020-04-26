
import pandas as pd

"""
Parameters for the svr model
"""


def set_regression_tree(hprm):
    
    assert hprm['learning.model'] == 'regression_tree'
    hprm.update({'regression_tree.inputs' : {
                                                 ('daylight',      '',           ''),
                                                 ('holidays',      '',           ''),
                                                 ('holidays',      'lag',        pd.DateOffset(hours = 24)),
                                                 ('holidays',      'lag',      - pd.DateOffset(hours = 24)),
                                                 #('hour',          '',           ''),
                                                 ('nebulosity',    '',           ''),
                                                 ('target',        'lag',        pd.DateOffset(hours = 24)),
                                                 ('temperature',   '',           ''), 
                                                 ('temperature',   'lag',        pd.DateOffset(hours = 24)),
                                                 #('temperature',   'smoothing',  0.99),
                                                 ('temperature',   'minimum',    pd.DateOffset(hours = 24)),
                                                 ('temperature',   'maximum',    pd.DateOffset(hours = 24)),
                                                 #('temperature',   'difference', pd.DateOffset(hours = 24)),
                                                 ('timestamp',     '',           ''), 
                                                 ('week_hour',     '',           ''), 
                                                 #('wind_speed',    '',           ''), 
                                                 ('xmas',          '',           ''), 
                                                 ('year_day',      '',           ''), 
                                                 },
                  })
            
    return hprm