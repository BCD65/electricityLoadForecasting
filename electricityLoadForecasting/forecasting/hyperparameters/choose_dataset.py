
import pandas as pd

"""
Select consumption sites and weather stations
"""

def choose_dataset(hprm):
   
    hprm.update({
                 # Choose the database
                 'database' : 'eCO2mix.administrative_regions',
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
    assert hprm['sites.aggregation'] in [
                                         None,
                                         'sum',
                                         ]
                 
    hprm.update({
                 # Choose the weather data
                 'weather.zone'              : 'all',
                 'weather.source'            : 'observed', 
                 'weather.aggregation'       : {('eCO2mix.France',                 None)     : 'mean',
                                                ('eCO2mix.administrative_regions', None)     :  None, 
                                                ('RTE.substations',                None)     :  None, 
                                                ('RTE.substations',               'sum')     : 'national_weighted_mean', 
                                                ('RTE.substations',               'administrative_regions') : 'regional_weighted_mean', 
                                                ('RTE.substations',               'RTE_regions')            : None 
                                                }.get((hprm['database'], hprm['sites.aggregation'])),
                 'weather.extra_latitude'    : {('eCO2mix.France',                 None)     : 5,
                                                ('eCO2mix.administrative_regions', None)     : 0.1, 
                                                ('RTE.substations',                None)     : 0.1, 
                                                ('RTE.substations',               'sum')     : 5,
                                                ('RTE.substations',               'administrative_regions') : 5,
                                                ('RTE.substations',               'RTE_regions')            : 5,
                                                }.get((hprm['database'], hprm['sites.aggregation']), 0.1),
                 'weather.extra_longitude'   : {('eCO2mix.France',                 None)     : 8,
                                                ('eCO2mix.administrative_regions', None)     : 0.1, 
                                                ('RTE.substations' ,               None)     : 0.1, 
                                                ('RTE.substations',               'sum')     : 8,
                                                ('RTE.substations',               'administrative_regions') : 8, 
                                                ('RTE.substations',               'RTE_regions')            : 8, 
                                                }.get((hprm['database'], hprm['sites.aggregation']), 0.1),
                 
                 # Select the training and the test set
                 'training_set.form'         : 'continuous',
                 'training_set.first_sample' : pd.to_datetime('2013-01-07 00:00', format = '%Y/%m/%d %H:%M').tz_localize('UTC'),
                 'training_set.length'       : pd.DateOffset(years = 3),
                 'validation_set.length'     : pd.DateOffset(years = 1),
                 })

    return hprm
