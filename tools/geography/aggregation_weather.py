
import os
import pandas as pd
#
import electricityLoadForecasting.config as config
import electricityLoadForecasting.tools  as transcoding



def aggregation_weather(df_weather, df_coordinates, aggregation_level):
    if aggregation_level is None:
        agg_df_weather          = df_weather
        agg_coordinates_weather = df_coordinates
    elif aggregation_level == 'mean':
        agg_df_weather          = pd.DataFrame(df_weather.mean(axis = 1), columns = ['mean'])
        agg_coordinates_weather = pd.DataFrame(df_coordinates.mean(axis = 0), index = ['mean'])
    else:
        df_weights                = pd.read_csv(os.path.join(config.path_extras, 'poids_stations_meteo.csv'), sep = ';', decimal=',')
        df_weights                = df_weights.fillna(0)
        df_weights['Nom station'] = df_weights['Nom station'].apply(transcoding.format_weather_station_name)
        df_weights                = df_weights.set_index('Nom station')
        df_weights                = df_weights.astype(float)
        df_weights                = df_weights.loc[df_weather.columns]
        df_weights                = df_weights/df_weights.sum(axis = 0)
        if   aggregation_level == 'national_weighted_mean':
            df_weights              = pd.DataFrame(df_weights['FRAN'], columns = ['weighted_mean'])
        elif aggregation_level == 'regional_weighted_mean':
            df_weights              = pd.DataFrame(df_weights[[e for e in df_weights.columns if e != 'FRAN']])
        else:
            raise ValueError
        agg_df_weather          = df_weather  @ df_weights
        agg_coordinates_weather = df_weights.T @ df_coordinates
    return agg_df_weather, agg_coordinates_weather
