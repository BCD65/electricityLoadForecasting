#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import default_config
#import copy as cp
import pandas as pd


##############################
#        Select inputs       #
##############################

def choose_inputs(hprm):
    hprm.update({
                 'inputs.nb_sites_per_site'   : 1,
                 'inputs.nb_weather_per_site' : 2,
                 'inputs.selection' : {
                                       ('temperature',   None,         None), 
                                       ('week_hour',     None,         None), 
                                       ('timestamp',     None,         None), 
                                       ('target',        'lag',        pd.DateOffset(hours = 24)),
                                       ('temperature',   'smoothing',  0.99),
                                       ('temperature',   'minimum',    '1D'),
                                       ('temperature',   'maximum',    '1D'),
                                       ('temperature',   'difference', pd.DateOffset(hours = 24)),
                                       }

                  
                  })

    return hprm