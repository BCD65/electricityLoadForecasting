
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
    inputs = os.path.join(root_folder, 
                          'RTE', 
                          'transformed_data',
                          )
elif dataset == 'RTE_quick_test':
    inputs = os.path.join(root_folder, 
                          'RTE', 
                          'transformed_data', 
                          'quick_test',
                          )
elif dataset == 'Eco2mix_administrative_regions':
    inputs =  os.path.join(root_folder,
                           'Eco2mix',
                           'transformed_data',
                           'administrative_regions',
                           )
elif dataset == 'Eco2mix_France':
    inputs =  os.path.join(root_folder,
                           'Eco2mix',
                           'transformed_data',
                           'France',
                           )


extras   = os.path.join(root_folder, 
                        'additional_data',
                        )
outputs = os.path.join(os.path.expanduser("~"),
                       'forecasting_outputs',
                       )
###




