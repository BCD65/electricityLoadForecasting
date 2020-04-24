
import os

###############################   Inputs   ####################################

try:
    # Possible to customize but keep the structure for interoperability
    from .personal_folders import inputs, extras
    assert extras # Only used with RTE database
except ModuleNotFoundError:
    inputs = os.path.join(os.path.expanduser("~"),
                          'electricityLoadForecastingInputs',
                          )

def path_database(dataset):
    assert dataset in {
                       'eCO2mix.administrative_regions',
                       'eCO2mix.France',
                       'RTE.substations',
                       'RTE.quick_test',
                       }
    
    if dataset == 'RTE.substations':
        path_database = os.path.join(inputs, 
                                     'RTE', 
                                     'transformed_data',
                                     )
    elif dataset == 'RTE.quick_test':
        path_database = os.path.join(inputs, 
                                     'RTE', 
                                     'transformed_data', 
                                     'quick_test',
                                     )
    elif dataset == 'eCO2mix.administrative_regions':
        path_database =  os.path.join(inputs,
                                      'eCO2mix',
                                      'transformed_data',
                                      'administrative_regions',
                                      )
    elif dataset == 'eCO2mix.France':
        path_database =  os.path.join(inputs,
                                      'eCO2mix',
                                      'transformed_data',
                                      'France',
                                      )
    return path_database

##############################   Outputs   ####################################

outputs = os.path.join(os.path.expanduser("~"),
                       'electricityLoadForecastingOutputs',
                       )

###############################################################################





