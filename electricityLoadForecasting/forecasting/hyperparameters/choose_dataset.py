
import pandas as pd

"""
Select consumption sites and weather stations
"""

def choose_dataset(hprm):
   
    hprm.update({
                 # Choose the database
                 'database' : 'RTE.quick_test',
                 })
    assert hprm['database'] in [
                                'eCO2mix.France',
                                'eCO2mix.administrative_regions',
                                'RTE.substations',
                                'RTE.quick_test',
                                ]
                                        
    hprm.update({
                 # Choose the electricity load data
                 'sites.zone'                : 'all', 
                 'sites.aggregation'         : None,
                 'sites.trash'               : [],# Eliminate additional sites from the dataset
                 })
                 
    hprm.update({
                 # Choose the weather data
                 'weather.zone'              : 'all',
                 'weather.source'            : 'observed', 
                 'weather.aggregation'       : {('eCO2mix.France',                 None) : 'mean',
                                                ('eCO2mix.administrative_regions', None) :  None, 
                                                ('RTE.substations',                None) :  None, 
                                                ('RTE.substations',               'sum') : 'wmean', 
                                                }.get((hprm['database'], hprm['sites.aggregation'])),
                 'weather.extra_latitude'    : {'eCO2mix.France'                 : 5,
                                                'eCO2mix.administrative_regions' : 0.1, 
                                                }.get(hprm['database'], 0.1),
                 'weather.extra_longitude'   : {'eCO2mix.France'                 : 8,
                                                'eCO2mix.administrative_regions' : 0.1, 
                                                }.get(hprm['database'], 0.1),
                 
                 # Select the training and the test set
                 'training_set.form'         : 'continuous',
                 'training_set.first_sample' : pd.to_datetime('2013-01-07 00:00', format = '%Y/%m/%d %H:%M').tz_localize('UTC'),
                 'training_set.length'       : pd.DateOffset(years = 3),
                 'validation_set.length'     : pd.DateOffset(years = 1),
                 })

    return hprm
