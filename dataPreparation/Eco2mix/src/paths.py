
import os
#
import electricityLoadForecasting.config  as config

dikt_folders = {
                'sites'               : os.path.join(config.root_folder, 'Eco2mix', 'raw_data', 'load'), 
                'weather'             : os.path.join(config.root_folder, 'Eco2mix', 'raw_data', 'weather'),  
                'tmp_data'            : os.path.join(config.root_folder, 'Eco2mix', 'transformed_data/tmp'),
                'transformed_data'    : os.path.join(config.root_folder, 'Eco2mix', 'transformed_data'),
                'plots'               : os.path.join(config.root_folder, 'Eco2mix', 'plots'),
                }
dikt_files = {
              #
              'sites.load_site_year' : 'eCO2mix_RTE_{site}_Annuel-Definitif_{year:d}', 
              'sites.coordinates'    : 'coordinates_sites.csv',
              #
              'weather.file_year_month' : 'synop.{year:d}{month:02d}',
              'weather.description'     : 'postesSynop',
              #
              }