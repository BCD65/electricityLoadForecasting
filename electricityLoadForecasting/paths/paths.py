
import os

###############################   Inputs   ####################################

try:
    # Possible to customize but keep the structure for interoperability
    from .personal_folders import folder_inputs
except ModuleNotFoundError:
    folder_inputs = os.path.join(os.path.expanduser("~"),
                                 'electricityLoadForecastingInputs',
                                 )

path_extras   = os.path.join(folder_inputs, 
                             'additional_data',
                             )

def path_dataset(dataset):
    assert dataset in {
                       'Eco2mix.administrative_regions',
                       'Eco2mix.France',
                       'RTE.full',
                       'RTE.quick_test',
                       }
    
    if dataset == 'RTE.full':
        path_dataset = os.path.join(folder_inputs, 
                                   'RTE', 
                                   'transformed_data',
                                   )
    elif dataset == 'RTE.quick_test':
        path_dataset = os.path.join(folder_inputs, 
                                   'RTE', 
                                   'transformed_data', 
                                   'quick_test',
                                   )
    elif dataset == 'Eco2mix.administrative_regions':
        path_dataset =  os.path.join(folder_inputs,
                                    'Eco2mix',
                                    'transformed_data',
                                    'administrative_regions',
                                    )
    elif dataset == 'Eco2mix.France':
        path_dataset =  os.path.join(folder_inputs,
                                    'Eco2mix',
                                    'transformed_data',
                                    'France',
                                    )
    return path_dataset

##############################   Outputs   ####################################

path_outputs = os.path.join(os.path.expanduser("~"),
                            'electricityLoadForecastingOutputs',
                            )

###############################################################################





