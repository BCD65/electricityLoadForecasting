
import numpy as np
import os
from termcolor import colored
import shutil
#
import electricityLoadForecasting.tools as tools
from electricityLoadForecasting.dataPreparation import correction, detection, files
import electricityLoadForecasting.dataPreparation.RTE as RTE


"""
Run to prepare the raw data (main function is at the end)
It detects corrupted data
It proposes modifications of the corrupted data
It saves everything in the end
This script takes about 24 hours and has not been optimized but only has to be run once.
"""
   
###############################################################################

def main_sites(): 

    # Load
    df_load     = RTE.process.load_raw_data(prefix = os.path.join(RTE.etc.dikt_folders['tmp_data'],
                                                                  'quick_test'*RTE.config.quick_test,
                                                                  ))
    dikt_errors = detection.detection_errors(df_load,
                                             prefix = os.path.join(RTE.etc.dikt_folders['tmp_data'],
                                                                   'quick_test'*RTE.config.quick_test,
                                                                   ))
    df_corrected_load, trash_sites = correction.correct_with_regression(df_load, 
                                                                        dikt_errors,
                                                                        prefix      = os.path.join(RTE.etc.dikt_folders['tmp_data'],
                                                                                                   'quick_test'*RTE.config.quick_test,
                                                                                                   ),
                                                                        prefix_plot = os.path.join(RTE.etc.dikt_folders['plots'],
                                                                                                   'quick_test'*RTE.config.quick_test,
                                                                                                   ),
                                                                        bool_plot_corrections = RTE.config.bool_plot_corrections, 
                                                                        bool_plot_trash       = RTE.config.bool_plot_trash, 
                                                                        )
    df_filtered_load, coordinates_sites = RTE.process.get_coordinates_sites(df_corrected_load)
    print(colored('{0} sites have been dropped'.format(len(trash_sites)), 'red'))
    
    # Checks
    tools.check_dates(df_filtered_load.index)
    assert (df_filtered_load.columns == coordinates_sites.index).all()
    
    # Save
    files.save_sites(
                     df_filtered_load,
                     coordinates_sites,
                     prefix = os.path.join(RTE.etc.dikt_folders['transformed_data'], 
                                           'quick_test'*RTE.config.quick_test, 
                                           )
                     )
    return df_filtered_load, coordinates_sites

                     
###############################################################################

def main_weather():
    
    # Load
    panel_RTE           = RTE.process.load_panel_RTE()
    df_weather          = RTE.process.load_raw_meteo(prefix = os.path.join(RTE.etc.dikt_folders['tmp_data'],
                                                                       'quick_test'*RTE.config.quick_test,
                                                                       ),
                                                 prefix_plot = os.path.join(RTE.etc.dikt_folders['plots'],
                                                                             'quick_test'*RTE.config.quick_test,
                                                                             ),
                                                 subset_weather = panel_RTE,
                                                 )
    coordinates_weather = RTE.process.get_coordinates_weather(panel_RTE)
    
    # Checks
    tools.check_dates(df_weather.index)
    assert (df_weather.columns.levels[0]   == coordinates_weather.index).all()
    assert np.all(df_weather.columns.levels[0] == panel_RTE)

    # Save
    files.save_weather(
                       df_weather,
                       coordinates_weather,
                       prefix = os.path.join(RTE.etc.dikt_folders['transformed_data'], 
                                             'quick_test'*RTE.config.quick_test, 
                                             )
                       )
    return df_weather, coordinates_weather
    

###############################################################################

if __name__ == '__main__':
    df_filtered_load, coordinates_sites = main_sites()
    df_weather, coordinates_weather     = main_weather()
    input('Hit any key (but ctrl+c) to delete temp variables and finish')
    shutil.rmtree(os.path.join(RTE.etc.dikt_folders['tmp_data'],
                               'quick_test'*RTE.config.quick_test,
                               ), 
                  ignore_errors = True,
                  )
    if os.path.isdir(RTE.etc.dikt_folders['tmp_data']) and not bool(os.listdir(RTE.etc.dikt_folders['tmp_data'])):
        shutil.rmtree(RTE.etc.dikt_folders['tmp_data'], 
                      ignore_errors = True,
                      )
        