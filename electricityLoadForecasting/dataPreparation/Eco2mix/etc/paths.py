
import os
#
from electricityLoadForecasting import paths

dikt_folders = {
                'sites'               : os.path.join(paths.inputs, 'Eco2mix', 'raw_data', 'load'), 
                'weather'             : os.path.join(paths.inputs, 'Eco2mix', 'raw_data', 'weather'),  
                'tmp_data'            : os.path.join(paths.inputs, 'Eco2mix', 'transformed_data/tmp'),
                'transformed_data'    : os.path.join(paths.inputs, 'Eco2mix', 'transformed_data'),
                'plots'               : os.path.join(paths.inputs, 'Eco2mix', 'plots'),
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