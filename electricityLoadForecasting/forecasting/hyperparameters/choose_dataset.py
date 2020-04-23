
import pandas as pd

####################################################################
#        Select consumption sites and weather stations             #
####################################################################

def choose_dataset(hprm):
   
    hprm.update({
                 # Choose the database
                 'database' : 'eCO2mix.France',
                 })
    assert hprm['database'] in [
                                'eCO2mix.France',
                                'eCO2mix.administrative_regions',
                                'RTE.substations',
                                'RTE.quick_test',
                                ]
                                        
    hprm.update({
                 # Choose the load data
                 'sites.zone'                : 'all', 
                 'sites.aggregation'         : None,
                 'sites.trash'               : [],    # Eliminate additional sites from the dataset
                 
                 # Choose the weather data
                 'weather.zone'              : 'all', 
                 'weather.source'            : 'observed', 
                 'weather.aggregation'       : {'eCO2mix.France'                 : 'mean',
                                                'eCO2mix.administrative_regions' : None, 
                                                }[hprm['database']],
                 'weather.extra_latitude'    : {'eCO2mix.France'                 : 5,
                                                'eCO2mix.administrative_regions' : 0.1, 
                                                }[hprm['database']],
                 'weather.extra_longitude'   : {'eCO2mix.France'                 : 8,
                                                'eCO2mix.administrative_regions' : 0.1, 
                                                }[hprm['database']],
                 
                 # Select the training and the test set
                 'training_set.form'         : 'continuous', #'two_sets', # Continuous/Discontinuous training sets
                 'training_set.first_sample' : pd.to_datetime('2013-01-07 00:00', format = '%Y/%m/%d %H:%M'),
                 'training_set.length'       : pd.DateOffset(years = 3),
                 'validation_set.length'     : pd.DateOffset(years = 1),
                 })

    return hprm


                            # Possible Zones
                            
                            # '4Paris'              4 substations in Paris
                            # 'Paris'               A square around Paris
                            # 'zone_Normandie'      A list of substations in Normandie/Paris
                            # 'full'                All the substations
                            # 'e2m'                 eCO2mix time series
                            #  misc.reg_admin[0]    An administrative region
                            #  misc.reg_rte[0]      A RTE region
                            #  misc.stations[0]     A district
                            
                            # districts             The aggregated loads in the districts
                            # admReg                The aggregated loads in the administrative regions
                            # rteReg                The aggregated loads in the RTE regions
                            # nat                   The aggregated loads in the whole country
                            
                            # Possible values for stations_id
                            # all, mean, wmean (weighted mean according to RTE weights)