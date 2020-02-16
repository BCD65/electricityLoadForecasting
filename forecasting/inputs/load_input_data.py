#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
#
import electricityLoadForecasting.paths as paths


def load_coordinates_sites():
    coordinates_sites = pd.read_csv(os.path.join(paths.inputs,
                                                 'df_coordinates_sites.csv',
                                                 ), 
                                    index_col = [0],
                                    #header    = [0],
                                    )
    return coordinates_sites


def load_coordinates_weather():
    coordinates_weather = pd.read_csv(os.path.join(paths.inputs,
                                                   'df_coordinates_weather.csv',
                                                   ), 
                                      index_col = [0],
                                      #header    = [0],
                                      )
    return coordinates_weather


def load_target():
    df_sites = pd.read_csv(os.path.join(paths.inputs,
                                        'df_sites.csv',
                                        ), 
                           index_col = [0],
                           #header    = [0],
                           )
    df_sites.index = pd.to_datetime(df_sites.index).tz_localize('UTC') 
    return df_sites


def load_weather():
    df_weather = pd.read_csv(os.path.join(paths.inputs,
                                          'df_weather.csv',
                                          ), 
                             index_col = [0],
                             header    = [0,1,2],
                             )
    df_weather.index = pd.to_datetime(df_weather.index).tz_localize('UTC') 
    return df_weather
    



###############################################################################

#profile
def load_input_data():
    
    data = {
            'df_sites'               : load_target(),
            'df_weather'             : load_weather(),
            'df_coordinates_sites'   : load_coordinates_sites(),
            'df_coordinates_weather' : load_coordinates_weather(),
            }
    
    common_dates       = sorted(set(data['df_sites'].index).intersection(data['df_weather'].index))
    data['df_sites']   = data['df_sites']  .loc[common_dates]
    data['df_weather'] = data['df_weather'].loc[common_dates]
    
    assert (data['df_sites'].index          == data['df_weather'].index).all()
    assert (data['df_sites'].columns        == sorted(data['df_sites'].columns)).all()
    assert list(data['df_weather'].columns) == sorted(data['df_weather'].columns)

    return data
    
if __name__ == '__main__':
    data = load_input_data()