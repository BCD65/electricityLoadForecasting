


import pandas as pd


##############################
#        Select inputs       #
##############################

def choose_inputs(hprm):
    hprm.update({
                 'inputs.nb_sites_per_site'   : 1,
                 'inputs.nb_weather_per_site' : 2,
                 'inputs.selection' : {
                                       ('target',        'lag',        pd.DateOffset(hours = 24)),
                                       ('temperature',   '',           ''), 
                                       ('temperature',   'smoothing',  0.99),
                                       ('temperature',   'minimum',    '1D'),
                                       ('temperature',   'maximum',    '1D'),
                                       ('temperature',   'difference', pd.DateOffset(hours = 24)),
                                       ('timestamp',     '',           ''), 
                                       ('week_hour',     '',           ''), 
                                       },
                 
                 ### Fixed values
                 'inputs.cyclic' : {
                                    'target'      : False,
                                    'temperature' : False,
                                    'timestamp'   : False,
                                    'week_hour'   : 168,
                                    }

                  
                  })

    return hprm