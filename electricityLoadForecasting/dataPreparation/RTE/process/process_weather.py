
import numpy as np
import pandas as pd
import os
from termcolor import colored
#
from electricityLoadForecasting import tools, etc
import electricityLoadForecasting.dataPreparation.plot_tools as plot_tools
import electricityLoadForecasting.dataPreparation.RTE        as RTE

###############################################################################

def load_panel_RTE():
    print('get_panel - ', end = '')
    with open(os.path.join(RTE.etc.dikt_files['weather.panel']), 'rb') as f_panel:
        panel_RTE = sorted(map(tools.format_weather_station_name, 
                               np.load(f_panel), 
                               ))
    if RTE.config.quick_test:
        panel_RTE = panel_RTE[:RTE.config.ncols_weather_quick_test]
        print(colored('panel has been shortened', 'green'))
    print('done - len(panel_RTE) = {0}\n{1}'.format(len(panel_RTE),
                                                   '#'*etc.global_var.NB_SIGNS),
                                                   )
    return panel_RTE


def load_raw_meteo(prefix = None, prefix_plot = None, subset_weather = slice(None)):  
    print('load_raw_weather_data - ', end = '')
    fname_weather = os.path.join(prefix, 'df_weather.csv')
    try:
        df_weather = pd.read_csv(fname_weather, 
                                 index_col = [0],
                                 header    = [0,1,2],
                                 )
        df_weather.index = pd.to_datetime(df_weather.index)       
        print('Loaded df_weather')
    except Exception as e:
        print('\n{0}'.format(colored(e,'red')))
        print('df_weather not loaded')
        descript = pd.read_csv(RTE.etc.dikt_files['weather.description'], 
                               sep=';',
                               )
        code_to_station      = descript[[RTE.etc.MeteoFrance_weather_id, RTE.etc.MeteoFrance_weather_name]].set_index(RTE.etc.MeteoFrance_weather_id).to_dict()[RTE.etc.MeteoFrance_weather_name]#
        tmp_forecast_weather = pd.read_csv(RTE.etc.dikt_files['weather.forecast'],
                                           sep = ',', 
                                           na_values=['.', '-99.90'], 
                                           )
        tmp_observed_weather = pd.read_csv(RTE.etc.dikt_files['weather.observed'],
                                           sep = ',', 
                                           na_values=['.', '-99.90'], 
                                           )
        
        tmp_forecast_weather[etc.user_source] = etc.user_source_forecast
        tmp_observed_weather[etc.user_source] = etc.user_source_observed
        
        dikt_weather_transcoding = {RTE.etc.MeteoFrance_temperature  : etc.user_value, 
                                    RTE.etc.MeteoFrance_nebulosity   : etc.user_value,
                                    RTE.etc.MeteoFrance_dt_UTC       : etc.user_dt_UTC,
                                    RTE.etc.MeteoFrance_weather_name : etc.user_weather_name,
                                    }
        
        tmp_forecast_temperature = tmp_forecast_weather[[RTE.etc.MeteoFrance_dt_UTC, RTE.etc.MeteoFrance_weather_id, etc.user_source, RTE.etc.MeteoFrance_temperature ]].rename(dikt_weather_transcoding, axis = 1)
        tmp_forecast_nebulosity  = tmp_forecast_weather[[RTE.etc.MeteoFrance_dt_UTC, RTE.etc.MeteoFrance_weather_id, etc.user_source, RTE.etc.MeteoFrance_nebulosity]  ].rename(dikt_weather_transcoding, axis = 1)
        tmp_observed_temperature = tmp_observed_weather[[RTE.etc.MeteoFrance_dt_UTC, RTE.etc.MeteoFrance_weather_id, etc.user_source, RTE.etc.MeteoFrance_temperature ]].rename(dikt_weather_transcoding, axis = 1)
        tmp_observed_nebulosity  = tmp_observed_weather[[RTE.etc.MeteoFrance_dt_UTC, RTE.etc.MeteoFrance_weather_id, etc.user_source, RTE.etc.MeteoFrance_nebulosity  ]].rename(dikt_weather_transcoding, axis = 1)
        
        tmp_forecast_temperature[etc.user_physical_quantity] = etc.user_temperature
        tmp_forecast_nebulosity[etc.user_physical_quantity ] = etc.user_nebulosity
        tmp_observed_temperature[etc.user_physical_quantity] = etc.user_temperature
        tmp_observed_nebulosity[etc.user_physical_quantity ] = etc.user_nebulosity
        
        tmp_weather = pd.concat([tmp_forecast_temperature,
                                 tmp_forecast_nebulosity,
                                 tmp_observed_temperature,
                                 tmp_observed_nebulosity,
                                 ],
                                axis = 0,
                                )
        
        tmp_weather[etc.user_dt_UTC]       = pd.to_datetime(tmp_weather[etc.user_dt_UTC], format = '%Y-%m-%dT%H:%M:%SZ').dt.tz_localize('UTC')
        tmp_weather[etc.user_weather_name] = tmp_weather[RTE.etc.MeteoFrance_weather_id].apply(lambda x : tools.transcoding.format_weather_station_name(code_to_station.get(x, str(x))))
        
        df_weather = tmp_weather.pivot_table(index   = etc.user_dt_UTC, 
                                             columns = [etc.user_weather_name, etc.user_physical_quantity, etc.user_source],
                                             values  = etc.user_value,
                                             )
        # Cheat/Complete columns with NaNs with steps of 24 hours
        df_weather = df_weather.groupby(by = df_weather.index.map(lambda x : (x.hour, x.minute))).fillna(method='ffill')

        ### Filter data
        print('Filter nan df_weather  ', df_weather.shape,   ' -> ', end = '')
        df_weather          = df_weather.dropna(axis=1, how='all')
        df_weather          = df_weather[subset_weather]
        df_weather.columns  = df_weather.columns.remove_unused_levels()
        
        # Check dates
        tools.check_dates(df_weather.index)    
        
        # Save
        os.makedirs(os.path.dirname(fname_weather), exist_ok = True)
        df_weather.to_csv(fname_weather)        
    if RTE.config.bool_plot_meteo:
        plot_tools.plot_meteo(df_weather, 
                              os.path.join(prefix_plot,
                                           'meteo',
                                           ),
                              )
    print('done : df_weather.shape = {0}\n{1}'.format(df_weather.shape, '#'*etc.global_var.NB_SIGNS))
    return df_weather

        
###############################################################################    
    
def get_coordinates_weather(panel_RTE):
    print('get_coordinates_stations - ', end = '')
    coordinates_stations = pd.read_csv(RTE.etc.dikt_files['weather.coordinates'],
                                       sep = ';',
                                       )
    coordinates_stations = coordinates_stations.rename({RTE.etc.MeteoFrance_latitude  : etc.user_latitude,
                                                        RTE.etc.MeteoFrance_longitude : etc.user_longitude,
                                                        }, axis = 1)
    coordinates_stations[etc.user_weather_name] = coordinates_stations[RTE.etc.MeteoFrance_weather_name2].apply(tools.format_weather_station_name)
    coordinates_stations = coordinates_stations.set_index(etc.user_weather_name)[[etc.user_latitude, etc.user_longitude]]
    coordinates_stations = coordinates_stations.filter(items = panel_RTE, axis = 0)
    for k in panel_RTE:
        assert k in coordinates_stations.index
    print('done - Names of the {0} weather stations : \n{1}'.format(coordinates_stations.shape[0],
                                                                    list(coordinates_stations.index),
                                                                    ))
    return coordinates_stations

