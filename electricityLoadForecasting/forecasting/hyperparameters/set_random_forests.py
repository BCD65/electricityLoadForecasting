
import pandas as pd

"""
Parameters for the random forests
"""


def set_random_forests(hprm):
    
    assert hprm['learning.model'] == 'random_forests'
    hprm.update({'random_forests.inputs' : {
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
                 'random_forests.n_estimators' : 100,
                 })
            
    return hprm

