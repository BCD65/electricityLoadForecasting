
import os
import socket
#


### Paths ###

if socket.gethostname() == 'MB6.local':
    root_folder = '/Volumes/_TData'
else:
    root_folder = os.path.expanduser("~")
    
###############################################################################

def path_inputs(database):
    assert database in {
                       'Eco2mix_administrative_regions',
                       'Eco2mix_France',
                       'RTE',
                       'RTE_quick_test',
                       }
    
    if database == 'RTE':
        path_inputs = os.path.join(root_folder, 
                                   'RTE', 
                                   'transformed_data',
                                   )
    elif database == 'RTE_quick_test':
        path_inputs = os.path.join(root_folder, 
                                   'RTE', 
                                   'transformed_data', 
                                   'quick_test',
                                   )
    elif database == 'Eco2mix_administrative_regions':
        path_inputs =  os.path.join(root_folder,
                                    'Eco2mix',
                                    'transformed_data',
                                    'administrative_regions',
                                    )
    elif database == 'Eco2mix_France':
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





