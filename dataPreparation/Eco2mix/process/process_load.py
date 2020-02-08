

import pandas as pd
import os
from termcolor import colored
import urllib
import zipfile
import unidecode
#
import electricityLoadForecasting.tools                           as tools
import electricityLoadForecasting.src.global_var                  as global_var
import electricityLoadForecasting.dataPreparation.Eco2mix.config  as config
import electricityLoadForecasting.dataPreparation.Eco2mix.src     as src

###############################################################################

def download_raw_load_data(site = None, year = None):
    assert type(year) == int and year >= 2013, year
    assert type(site) == str, site
    prefix = src.dikt_folders['sites']
    os.makedirs(prefix, exist_ok = True)
    zip_file_url = os.path.join(src.dikt_url['sites'], 
                                src.dikt_files['sites.load_site_year'].format(site = src.URL_TRANSCODING.get(site, site), 
                                                                              year = year,
                                                                              ).replace('__', '_'),
                                ) + '.zip'
    zip_file_path    = os.path.join(prefix,
                                    src.dikt_files['sites.load_site_year'].format(site = site,
                                                                                  year = year,
                                                                                  ).replace('__', '_'),
                                    ) + '.zip'
    urllib.request.urlretrieve(zip_file_url,
                               zip_file_path,
                               )
    with zipfile.ZipFile(zip_file_path, 'r') as zipObj:
        zipObj.extractall(prefix)


def read_raw_load_data(site = None, year = None):
    assert type(year) == int and year >= 2013, year
    xls_file_path    = os.path.join(src.dikt_folders['sites'],
                                    src.dikt_files['sites.load_site_year'].format(site = src.XLS_TRANSCODING.get(site, site),
                                                                                  year = year,
                                                                                  ).replace('__', '_'),
                                ) + '.xls'    
    df        = pd.read_csv(xls_file_path,
                            sep        = '\t',
                            encoding   = 'ISO-8859-1',
                            skipfooter = 2,
                            index_col  = False,
                            engine     = 'python',
                            #skiprows  = 1,
                            na_values = ['ND'],
                            )
    df[src.user_dt_UTC] = pd.to_datetime(df[src.Eco2mix_date] + ' ' + df[src.Eco2mix_time_UTC], format = '%Y/%m/%d %H:%M')  
    print(colored('It is assumed that the data from Eco2mix is in UTC - to be checked', 'red', 'on_cyan'))
    df = df.set_index(src.user_dt_UTC)[[src.Eco2mix_load]]
    df.columns.name = src.user_site_name
    df = df.rename({src.Eco2mix_load : unidecode.unidecode(site) if site else 'France'}, axis = 1)
    df = df.iloc[::4]
    return df


def load_raw_load_data(prefix = None):
    print('load_raw_data - ', end = '')
    fname = os.path.join(prefix, 'df_raw_load.csv')
    try:
        df_load = pd.read_csv(fname,
                              index_col = [0],
                              header    = [0],
                              )
        df_load.index = pd.to_datetime(df_load.index)
        print('Loaded df_load')
    except Exception as e:
        print('\n{0}'.format(colored(e,'red')))
        print('df_load not loaded')
        dikt_load = {}
        for year in config.YEARS_LOADS:
            for site in src.coordinates_sites[config.LEVEL].index:
                print('year = {year} - site = {site}'.format(site = site, year = year))
                try:
                    dikt_load[site, year] = read_raw_load_data(site = site, year = year)
                except FileNotFoundError:
                    download_raw_load_data(site = site, year = year)
                    dikt_load[site, year] = read_raw_load_data(site = site, year = year)
        df_load = pd.concat([pd.concat([dikt_load[site, year]
                                        for year in config.YEARS_LOADS
                                        ],
                                       axis = 0,
                                       )
                             for site in src.coordinates_sites[config.LEVEL].index
                             ],
                            axis = 1,
                            )
        # Cheat for the first row that is missing in Eco2mix
        df_load.fillna(method = 'bfill', axis = 0, inplace = True)
        # Check dates
        tools.check_dates(df_load.index)
        df_load = df_load.reindex(pd.date_range(start = df_load.index.min(),
                                                end   = df_load.index.max(),
                                                freq  = '1H',
                                                name  = df_load.index.name, 
                                                ))
        # Save
        os.makedirs(os.path.dirname(fname), exist_ok = True)
        df_load.to_csv(fname, index_label = df_load.columns.name)
    print('done - df_load.shape = {0}\n{1}'.format(df_load.shape,
                                                   '#'*global_var.NB_SIGNS,
                                                   ))
    return df_load

###############################################################################    
        
def get_coordinates_sites():
    print('get_coordinates_sites - ', end = '')
    coordinates_sites = src.coordinates_sites[config.LEVEL]
    coordinates_sites =  coordinates_sites.rename({site : unidecode.unidecode(site)
                                                   for site in coordinates_sites.index
                                                   },
                                                  axis = 0,
                                                  )
    print('done - len(coordinates_sites) = {0}\n{1}'.format(len(coordinates_sites),
                                                            '#'*global_var.NB_SIGNS),
                                                            )
    print('Names of the {0} sites : \n{1}'.format(len(coordinates_sites), list(coordinates_sites.index))) 
    return coordinates_sites
      
