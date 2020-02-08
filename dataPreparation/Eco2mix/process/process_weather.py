
import numpy as np
import pandas as pd
import os
import pickle
from termcolor import colored
import shutil
import urllib
import gzip
#
import electricityLoadForecasting.tools as tools
import electricityLoadForecasting.src.global_var as global_var
import electricityLoadForecasting.dataPreparation.plot_tools      as plot_tools
import electricityLoadForecasting.dataPreparation.Eco2mix.config  as config
import electricityLoadForecasting.dataPreparation.Eco2mix.src     as src




###############################################################################

def download_raw_weather_data(year = None, month = None):
    assert type(year)  == int and year > 2000, year
    assert type(month) == int and month in np.arange(1,13), month
    os.makedirs(src.dikt_folders['weather'], exist_ok = True)
    gzip_file_url = os.path.join(src.dikt_url['weather'],
                                 src.dikt_files['weather.file_year_month'].format(year = year, month = month),
                                 ) + '.csv.gz'
    gz_file_path  = os.path.join(src.dikt_folders['weather'],
                                 src.dikt_files['weather.file_year_month'].format(year = year, month = month),
                                 ) + '.csv.gz'
    urllib.request.urlretrieve(gzip_file_url,
                               gz_file_path,
                               )
    csv_file_path = os.path.join(src.dikt_folders['weather'],
                                 src.dikt_files['weather.file_year_month'].format(year = year, month = month),
                                 ) + '.csv'
    with gzip.open(gz_file_path, 'rb') as f_in:
        with open(csv_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def read_raw_weather_data(year = None, month = None):
    assert type(year)  == int and year > 2000, year
    assert type(month) == int and month in np.arange(1,13), month
    csv_file_path = os.path.join(src.dikt_folders['weather'],
                                 src.dikt_files['weather.file_year_month'].format(year = year, month = month),
                                 ) + '.csv'    
    df        = pd.read_csv(csv_file_path,
                            sep = ';',
                            na_values = ['mq'],
                            )
    df[src.user_dt_UTC] = pd.to_datetime(df[src.MeteoFrance_dt_UTC], format = '%Y%m%d%H%M%S')  
    df = df.rename({src.MeteoFrance_temperature  : src.user_temperature, 
                    src.MeteoFrance_nebulosity   : src.user_nebulosity,
                    src.MeteoFrance_wind_speed   : src.user_wind_speed,
                    src.MeteoFrance_weather_id   : src.user_weather_id,
                    }, 
                   axis = 1,
                   )
    df.drop_duplicates(subset  = [src.user_dt_UTC, src.user_weather_id],
                       inplace = True,
                       keep    = 'first',
                       )
    df = df.set_index([src.user_dt_UTC, src.user_weather_id])
    df = df[[src.user_temperature, src.user_nebulosity, src.user_wind_speed]]
    df = df.astype(float)
    df[src.user_temperature] -= 273.15
    df = df.unstack()
    df = df.swaplevel(0,1, axis = 1)
    df.columns.names = [src.user_weather_id, src.user_physical_quantity]
    return df


def correct_filter_weather(df_weather):
    # Some series start with a few Nan so correct them or drop them
    length_missing_data_beginning = (1 - pd.isnull(df_weather)).idxmax(axis = 0) - df_weather.index[0]
    dropped_columns               = df_weather.columns[length_missing_data_beginning > pd.to_timedelta(24, unit='h')].remove_unused_levels()#.levels[0]
    df_weather                    = df_weather.drop(columns = dropped_columns)
    trash_weather                 = list(dropped_columns.levels[0])
    df_weather.columns            = df_weather.columns.remove_unused_levels()
    # Drop the stations that do not have all the physical quantities
    for station in (df_weather.columns.levels[0]).copy():
        if df_weather[station].columns.shape[0] < df_weather.columns.levels[1].shape[0]:
            df_weather = df_weather.drop(columns = station, level = 0)
            trash_weather.append(station)
    # Backfill the rest
    df_weather.fillna(method = 'bfill', axis = 0, inplace = True)    
    assert df_weather.shape[1] % df_weather.columns.levels[1].shape[0] == 0, (df_weather.shape[1], df_weather.columns.levels[1].shape[0])
    return df_weather, sorted(set(trash_weather))
    


def load_raw_weather_data(prefix = None, prefix_plot = None):
    print('load_raw_weather_data - ', end = '')
    fname_weather = os.path.join(prefix, 'df_weather.csv')
    fname_trash   = os.path.join(prefix, 'trash_weather.pkl')
    weather_description = load_weather_description()
    try:
        df_weather = pd.read_csv(fname_weather, 
                                 index_col = [0],
                                 header    = [0,1,2],
                                 )
        df_weather.index = pd.to_datetime(df_weather.index)
        with open(fname_trash, 'rb') as f:
            trash_weather = pickle.load(f)        
        print('Loaded df_weather and trash_weather')
    except Exception as e:
        print('\n{0}'.format(colored(e,'red')))
        print('df_weather and trash_weather not loaded')
        dikt_weather = {}
        for year in config.YEARS_WEATHER:
            for month in range(1,13):
                print('\ryear = {0:2} - month = {1:2}'.format(year, month), end = '')
                try:
                    dikt_weather[year, month] = read_raw_weather_data(year = year, month = month)
                except FileNotFoundError:
                    download_raw_weather_data(year = year, month = month)
                    dikt_weather[year, month] = read_raw_weather_data(year = year, month = month)
        print()
        df_weather = pd.concat([dikt_weather[year,month] 
                                for year in config.YEARS_WEATHER
                                for month in range(1,13)
                                ], 
                               axis = 0,
                               )
        df_weather = df_weather.dropna(axis = 1, how = 'all')
        df_weather.columns = df_weather.columns.set_levels([weather_description[src.user_weather_name][e] 
                                                            for e in df_weather.columns.levels[0]
                                                            ],
                                                           level = 0,
                                                           ).set_names(src.user_weather_name, level=0)
        df_weather = df_weather.sort_index(axis = 1)
        df_weather = df_weather.reindex(pd.date_range(start = df_weather.index.min(),
                                                      end   = df_weather.index.max() +pd.to_timedelta(2, unit='h'),
                                                      freq  = '1H',
                                                      name  = df_weather.index.name, 
                                                      ))
        df_weather = df_weather.interpolate(method='linear') 
        df_weather, trash_weather   = correct_filter_weather(df_weather)
        df_weather[src.user_source] = src.user_source_observed 
        df_weather                  = df_weather.set_index(src.user_source, append = True).unstack(src.user_source)
        df_weather.columns          = df_weather.columns.remove_unused_levels()
        # Check dates
        tools.check_dates(df_weather.index)        
        # Save
        os.makedirs(os.path.dirname(fname_weather), exist_ok = True)
        df_weather.to_csv(fname_weather)
        with open(fname_trash, 'wb') as f:
            pickle.dump(trash_weather, f)
    coordinates_weather = weather_description.set_index(src.user_weather_name)[[src.user_latitude, src.user_longitude]]
    coordinates_weather.sort_index(axis = 0, inplace = True)
    common_names = sorted(set(coordinates_weather.index).intersection(df_weather.columns.levels[0]))
    df_weather          = df_weather.loc[:,common_names]
    coordinates_weather = coordinates_weather.loc[common_names]
    if config.bool_plot_meteo:
        plot_tools.plot_meteo(df_weather, 
                              os.path.join(prefix_plot,
                                           'meteo',
                                           ),
                              )
    print('done : df_weather.shape = {0}\n{1}'.format(df_weather.shape, '#'*global_var.NB_SIGNS))
    return df_weather, coordinates_weather, trash_weather

###############################################################################

def download_weather_description():
    os.makedirs(src.dikt_folders['weather'], exist_ok = True)
    csv_file_path = os.path.join(src.dikt_folders['weather'],
                                 src.dikt_files['weather.description'],
                                 ) + '.csv'
    csv_file_url  = os.path.join(src.dikt_url['weather.stations'],
                                 src.dikt_files['weather.description'],
                                 ) + '.csv'
    urllib.request.urlretrieve(csv_file_url,
                               csv_file_path,
                               )

def read_weather_description():
    csv_file_path = os.path.join(src.dikt_folders['weather'],
                                 src.dikt_files['weather.description'],
                                 ) + '.csv'
    df        = pd.read_csv(csv_file_path,
                            sep = ';',
                            )
    df = df.rename({src.MeteoFrance_weather_id2 : src.user_weather_id,
                    src.MeteoFrance_name        : src.user_weather_name,
                    src.MeteoFrance_latitude    : src.user_latitude,
                    src.MeteoFrance_longitude   : src.user_longitude,
                    },
                   axis = 1,
                   )
    df = df.set_index(src.user_weather_id)
    return df


def load_weather_description():
    try:
        weather_description = read_weather_description()
    except FileNotFoundError:
        download_weather_description()
        weather_description = read_weather_description()
    weather_description[src.user_weather_name] = weather_description[src.user_weather_name].apply(tools.format_weather_station_name)
    return weather_description

###############################################################################

