

import pandas as pd


def set_mars(hprm):
    
    assert hprm['learning.model'] == 'mars'
    hprm.update({'mars.inputs' : {
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
                 'mars.verbose' : 1,
                 'mars.thresh'  : 0.00001,
                 })
    
    return hprm

