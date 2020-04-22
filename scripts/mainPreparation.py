#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from termcolor import colored
#
import electricityLoadForecasting.tools as tools
from electricityLoadForecasting.dataPreparation import correction, detection, files
import electricityLoadForecasting.dataPreparation.Eco2mix as Eco2mix


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
    df_load     = Eco2mix.process.load_raw_load_data(prefix = os.path.join(Eco2mix.etc.dikt_folders['tmp_data'], 
                                                                           Eco2mix.config.LEVEL,
                                                                           ))
    dikt_errors = detection.detection_errors(df_load,
                                             prefix = os.path.join(Eco2mix.etc.dikt_folders['tmp_data'], 
                                                                   Eco2mix.config.LEVEL,
                                                                   ))
    df_filtered_load, trash_sites = correction.correct_with_regression(df_load, 
                                                                       dikt_errors, 
                                                                       prefix       = os.path.join(Eco2mix.etc.dikt_folders['tmp_data'], 
                                                                                                   Eco2mix.config.LEVEL,
                                                                                                   ),
                                                                       prefix_plot = os.path.join(Eco2mix.etc.dikt_folders['plots'],
                                                                                                  Eco2mix.config.LEVEL,
                                                                                                  ),
                                                                       bool_plot_corrections = Eco2mix.config.bool_plot_corrections, 
                                                                       bool_plot_trash       = Eco2mix.config.bool_plot_trash
                                                                       )
    print(colored('{0} sites have been dropped'.format(len(trash_sites)), 'red'))
    coordinates_sites = Eco2mix.process.get_coordinates_sites()
    
    # Checks
    tools.check_dates(df_filtered_load.index)
    assert (df_filtered_load.columns == coordinates_sites.index).all()
    
    # Save
    files.save_sites(
                     df_filtered_load,
                     coordinates_sites,
                     prefix = os.path.join(Eco2mix.etc.dikt_folders['transformed_data'], 
                                           Eco2mix.config.LEVEL,
                                           )
                     )
    return df_filtered_load, coordinates_sites


###############################################################################

def main_weather():

    # Load
    df_weather, coordinates_weather, trash_weather = Eco2mix.process.load_raw_weather_data(prefix      = os.path.join(Eco2mix.etc.dikt_folders['tmp_data'],
                                                                                                                      Eco2mix.config.LEVEL,
                                                                                                                      ),
                                                                                           prefix_plot = os.path.join(Eco2mix.etc.dikt_folders['plots'],
                                                                                                                      Eco2mix.config.LEVEL,
                                                                                                                      ),
                                                                                           )
    print(colored('{0} weather stations have been dropped'.format(len(trash_weather)), 'red'))

    # Checks
    tools.check_dates(df_weather.index)
    assert (df_weather.columns.levels[0]   == coordinates_weather.index).all()
    
    # Save
    files.save_weather(
                       df_weather,
                       coordinates_weather,
                       prefix = os.path.join(Eco2mix.etc.dikt_folders['transformed_data'], 
                                             Eco2mix.config.LEVEL,
                                             )
                       )
    return df_weather, coordinates_weather
                         

###############################################################################

if __name__ == '__main__':
    df_filtered_load, coordinates_sites = main_sites()
    df_weather, coordinates_weather     = main_weather()
    input('Hit any key (but ctrl+c) to delete temp variables and finish')
    shutil.rmtree(os.path.join(Eco2mix.etc.dikt_folders['tmp_data'],
                               Eco2mix.config.LEVEL,
                               ), 
                  ignore_errors = True,
                  )
    if os.path.isdir(Eco2mix.etc.dikt_folders['tmp_data']) and not bool(os.listdir(Eco2mix.etc.dikt_folders['tmp_data'])):
        shutil.rmtree(Eco2mix.etc.dikt_folders['tmp_data'], 
                      ignore_errors = True,
                      )
        