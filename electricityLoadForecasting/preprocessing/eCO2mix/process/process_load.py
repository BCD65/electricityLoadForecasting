
import pandas as pd
import os
from termcolor import colored
import urllib
import zipfile
import unidecode
import pytz
#
import electricityLoadForecasting.tools as tools
from .. import etc, config

###############################################################################

def download_raw_load_data(site = None, year = None):
    assert type(year) == int and year >= 2013, year
    assert type(site) == str, site
    prefix = etc.dikt_folders['sites']
    os.makedirs(prefix, exist_ok = True)
    zip_file_url = os.path.join(etc.dikt_url['sites'], 
                                etc.dikt_files['sites.load_site_year'].format(site = etc.URL_TRANSCODING.get(site, site), 
                                                                                      year = year,
                                                                                      ).replace('__', '_'),
                                ) + '.zip'
    zip_file_path    = os.path.join(prefix,
                                    etc.dikt_files['sites.load_site_year'].format(site = site,
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
    xls_file_path    = os.path.join(etc.dikt_folders['sites'],
                                    etc.dikt_files['sites.load_site_year'].format(site = etc.XLS_TRANSCODING.get(site, site),
                                                                                          year = year,
                                                                                          ).replace('__', '_'),
                                    ) + '.xls'    
    df        = pd.read_csv(xls_file_path,
                            sep        = '\t',
                            encoding   = 'ISO-8859-1',
                            skipfooter = 2,
                            index_col  = False,
                            engine     = 'python',
                            na_values = ['ND'],
                            )
    # According to RTE, the data is in CET. Given the NonExistentTimeErrors, it must be CET all year long and never CEST.
    df[tools.transcoding.user_dt_UTC] = pd.to_datetime(df[etc.eCO2mix_date] + ' ' + df[etc.eCO2mix_time_CET],
                                                       format = '%Y/%m/%d %H:%M',
                                                       ).dt.tz_localize(tz = pytz.FixedOffset(60)).dt.tz_convert('UTC')
    df = df.set_index(tools.transcoding.user_dt_UTC)[[etc.eCO2mix_load]]
    df = df.rename({etc.eCO2mix_load : unidecode.unidecode(site) if site else 'France'}, axis = 1)
    df = df.iloc[::4]
    return df


def load_raw_load_data(prefix = None):
    print('load_raw_data - ', end = '')
    fname = os.path.join(prefix, 'df_raw_load.csv')
    try:
        df_load = pd.read_csv(fname,
                              index_col = [0],
                              )
        df_load.index = pd.to_datetime(df_load.index)
        print('Loaded df_load')
    except Exception as e:
        print('\n{0}'.format(colored(e,'red')))
        print('df_load not loaded')
        dikt_load = {}
        for year in config.YEARS_LOADS:
            for site in etc.coordinates_sites[config.LEVEL].index:
                print('\r' + 'year = {year} - site = {site}'.format(site = site, year = year) + ' '*10,
                      end = '',
                      )
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
                             for site in etc.coordinates_sites[config.LEVEL].index
                             ],
                            axis = 1,
                            )
        # Cheat for the first row that is missing in eCO2mix
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
        df_load.to_csv(fname)
    print('done - df_load.shape = {0}\n{1}'.format(df_load.shape,
                                                   '#'*tools.NB_SIGNS,
                                                   ))
    return df_load

###############################################################################    
        
def get_coordinates_sites():
    print('get_coordinates_sites - ', end = '')
    coordinates_sites = etc.coordinates_sites[config.LEVEL]
    coordinates_sites = coordinates_sites.rename({site : unidecode.unidecode(site)
                                                  for site in coordinates_sites.index
                                                  },
                                                 axis = 0,
                                                 )
    print('done - len(coordinates_sites) = {0}\n{1}'.format(len(coordinates_sites),
                                                            '#'*tools.NB_SIGNS),
                                                            )
    print('Names of the {0} sites : \n{1}'.format(len(coordinates_sites), list(coordinates_sites.index))) 
    return coordinates_sites
      
