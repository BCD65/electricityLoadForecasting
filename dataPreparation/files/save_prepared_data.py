
import os
#
import electricityLoadForecasting.src.global_var as global_var


def save_sites(
               df_filtered_load,
               coordinates_sites,
               prefix = None,
               ):
    print('save_sites - ', end = '')
    dikt_variables = {
                      'df_sites'               : df_filtered_load,
                      'df_coordinates_sites'   : coordinates_sites,
                      }  
    os.makedirs(prefix, exist_ok = True)
    for ii, (key, value) in enumerate(dikt_variables.items()):
        value.to_csv(os.path.join(prefix, key+'.csv'),
                     index_label = value.columns.name,
                     )
    print('done\n{0}'.format('#'*global_var.NB_SIGNS))


def save_weather(
                 df_weather,
                 coordinates_weather,
                 prefix = None,
                 ):
    print('save_weather - ', end = '')
    dikt_variables = {
                      'df_weather'             : df_weather,
                      'df_coordinates_weather' : coordinates_weather,
                      }  
    os.makedirs(prefix, exist_ok = True)
    for ii, (key, value) in enumerate(dikt_variables.items()):
        value.to_csv(os.path.join(prefix, key+'.csv'),
                     index_label = value.columns.name,
                     )
    print('done\n{0}'.format('#'*global_var.NB_SIGNS))

