
import os
import socket
#


### Paths ###

try:
    # Possible to customize but keep the structure for interoperability
    from .personal_folders import root_folder
except ModuleNotFoundError:
    root_folder = os.path.expanduser("~")
    
###############################################################################

def path_inputs(database):
    assert database in {
                       'Eco2mix.administrative_regions',
                       'Eco2mix.France',
                       'RTE.full',
                       'RTE.quick_test',
                       }
    
    if database == 'RTE.full':
        path_inputs = os.path.join(root_folder, 
                                   'RTE', 
                                   'transformed_data',
                                   )
    elif database == 'RTE.quick_test':
        path_inputs = os.path.join(root_folder, 
                                   'RTE', 
                                   'transformed_data', 
                                   'quick_test',
                                   )
    elif database == 'Eco2mix.administrative_regions':
        path_inputs =  os.path.join(root_folder,
                                    'Eco2mix',
                                    'transformed_data',
                                    'administrative_regions',
                                    )
    elif database == 'Eco2mix.France':
        path_inputs =  os.path.join(root_folder,
                                    'Eco2mix',
                                    'transformed_data',
                                    'France',
                                    )
    return path_inputs

###############################################################################

extras   = os.path.join(root_folder, 
                        'additional_data',
                        )
outputs = os.path.join(os.path.expanduser("~"),
                       'forecasting_outputs',
                       )

###############################################################################





