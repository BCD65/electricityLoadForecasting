
import os
import socket
#


### Paths ###
if socket.gethostname() == 'MB6.local':
    root_folder = '/Volumes/_TData'
else:
    root_folder = os.path.expanduser("~")
###############################################################################

dataset     = 'Eco2mix_administrative_regions'

assert dataset in {
                   'Eco2mix_administrative_regions',
                   'Eco2mix_France',
                   'RTE',
                   'RTE_quick_test',
                   }

if dataset == 'RTE':
    path_inputs = os.path.join(root_folder, 
                               'RTE', 
                               'transformed_data',
                               )
    path_extras = os.path.join(root_folder, 
                               'additional_data',
                               )
elif dataset == 'RTE_quick_test':
    path_inputs = os.path.join(root_folder, 
                               'RTE', 
                               'transformed_data', 
                               'quick_test',
                               )
    path_extras = os.path.join(root_folder, 
                               'additional_data',
                               )
elif dataset == 'Eco2mix_administrative_regions':
    path_inputs =  os.path.join(root_folder,
                                'Eco2mix',
                                'transformed_data',
                                'administrative_regions',
                                )
elif dataset == 'Eco2mix_France':
    path_inputs =  os.path.join(root_folder,
                                'Eco2mix',
                                'transformed_data',
                                'France',
                                )


###




