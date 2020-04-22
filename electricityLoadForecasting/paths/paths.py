
import os

###############################   Inputs   ####################################

try:
    # Possible to customize but keep the structure for interoperability
    from .personal_folders import inputs
except ModuleNotFoundError:
    inputs = os.path.join(os.path.expanduser("~"),
                          'electricityLoadForecastingInputs',
                          )

extras   = os.path.join(inputs, 
                        'additional_data',
                        )

def path_database(dataset):
    assert dataset in {
                       'Eco2mix.administrative_regions',
                       'Eco2mix.France',
                       'RTE.full',
                       'RTE.quick_test',
                       }
    
    if dataset == 'RTE.full':
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
    elif dataset == 'Eco2mix.administrative_regions':
        path_database =  os.path.join(inputs,
                                    'Eco2mix',
                                    'transformed_data',
                                    'administrative_regions',
                                    )
    elif dataset == 'Eco2mix.France':
        path_database =  os.path.join(inputs,
                                    'Eco2mix',
                                    'transformed_data',
                                    'France',
                                    )
    return path_database

##############################   Outputs   ####################################

outputs = os.path.join(os.path.expanduser("~"),
                       'electricityLoadForecastingOutputs',
                       )

###############################################################################





