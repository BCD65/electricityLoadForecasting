
import os
#
import electricityLoadForecasting.dataPreparation.RTE as RTE


dikt_folders = {
                #
                'sites'               : os.path.join(RTE.config.root_folder_raw, 'raw_data', 'load'), 
                'weather'             : os.path.join(RTE.config.root_folder_raw, 'raw_data', 'weather'),  
                'tmp_data'            : os.path.join(RTE.config.root_folder_transformed, 'transformed_data', 'tmp'),
                'transformed_data'    : os.path.join(RTE.config.root_folder_transformed, 'transformed_data'),
                'plots'               : os.path.join(RTE.config.root_folder_transformed, 'plots')
                #
                }

dikt_files = {
              #
              'sites.loads_13_15' : os.path.join(dikt_folders['sites'], 'extract_demihoraire_2013_2015.csv'), 
              'sites.loads_16'    : os.path.join(dikt_folders['sites'], 'extract_demihoraire_2016.csv'), 
              'sites.loads_17'    : os.path.join(dikt_folders['sites'], 'extract_demihoraire_2017.csv'), 
              'sites.coordinates' : os.path.join(dikt_folders['sites'], 'coordonnees_postes.csv'), 
              #
              'weather.panel'       : os.path.join(dikt_folders['weather'], 'Panel_RTE'), 
              'weather.observed'    : os.path.join(dikt_folders['weather'], 'meteo_real.csv'), 
              'weather.forecast'    : os.path.join(dikt_folders['weather'], 'meteo_prev.csv'), 
              'weather.description' : os.path.join(dikt_folders['weather'], 'descriptif_stations.csv'), 
              'weather.coordinates' : os.path.join(dikt_folders['weather'], 'coordonnees_station_MF 2_name_modified_TBU.csv'), 
              #
              }
