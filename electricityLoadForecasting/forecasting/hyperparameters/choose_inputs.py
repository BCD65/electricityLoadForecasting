


import pandas as pd


##############################
#        Select inputs       #
##############################

def choose_inputs(hprm):
    hprm.update({
                 'inputs.nb_sites_per_site'   : 1,
                 'inputs.nb_weather_per_site' : 2,
                 'inputs.selection' : {
                                       ('daylight',      '',           ''),
                                       ('holidays',      '',           ''),
                                       ('holidays',      'lag',        pd.DateOffset(hours = 24)),
                                       ('holidays',      'lag',      - pd.DateOffset(hours = 24)),
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
                                       ('xmas',          '',           ''), 
                                       ('year_day',      '',           ''), 
                                       },
                 
                 ### Fixed values
                 'inputs.cyclic' : {
                                    'hour'        : 24,
                                    'week_hour'   : 168,
                                    'year_day'    : 365,
                                    }

                  
                  })

    return hprm