
import pandas as pd
import os
import datetime as dt
from termcolor import colored
#
from electricityLoadForecasting import tools, etc
import electricityLoadForecasting.dataPreparation.RTE as RTE


###############################################################################


def load_raw_data(prefix = None):
    print('load_raw_data - ', end = '')
    fname = os.path.join(prefix, 'df_load.csv')
    try:
        df_load = pd.read_csv(fname,
                              index_col = [0],
                              #header    = [0],
                              )
        df_load.index = pd.to_datetime(df_load.index)
        print('Loaded df_load')
    except Exception as e:
        print('\n{0}'.format(colored(e,'red')))
        print('df_load not loaded')
        # Read and add fictive data for 31 december 2016, 2017
        load_db_13_15 = pd.read_csv(os.path.join(RTE.etc.dikt_folders['sites'],
                                                 RTE.etc.dikt_files['sites.loads_13_15'], 
                                                 ),
                                    usecols = range(RTE.config.ncols_load_quick_test) if RTE.config.quick_test else None, 
                                    )
        load_db_13_15[etc.user_dt_UTC] = pd.to_datetime(load_db_13_15[etc.user_dt_UTC])
        
        load_db_16 = pd.read_csv(os.path.join(RTE.etc.dikt_folders['sites'],
                                              RTE.etc.dikt_files['sites.loads_16'], 
                                              ),
                                 usecols = range(RTE.config.ncols_load_quick_test) if RTE.config.quick_test else None, 
                                 ).rename(columns = {RTE.etc.RTE_dt_UTC:etc.user_dt_UTC})
        load_db_16[etc.user_dt_UTC] = pd.to_datetime(load_db_16[etc.user_dt_UTC]).dt.tz_localize('UTC')
        load_db_16 = pd.concat([load_db_16.iloc[:-1], 
                                load_db_16.iloc[-49:-1].assign(date_UTC = lambda x : x.date_UTC + dt.timedelta(hours = 24)), 
                                ])
        load_db_17 = pd.read_csv(os.path.join(RTE.etc.dikt_folders['sites'],
                                              RTE.etc.dikt_files['sites.loads_17'], 
                                              ),
                                 usecols = range(RTE.config.ncols_load_quick_test) if RTE.config.quick_test else None, 
                                 ).rename(columns = {RTE.etc.RTE_dt_UTC:etc.user_dt_UTC})
        load_db_17[etc.user_dt_UTC] = pd.to_datetime(load_db_17[etc.user_dt_UTC]).dt.tz_localize('UTC')
        load_db_17 = pd.concat([load_db_17.iloc[:-1], 
                                load_db_17.iloc[-49:-1].assign(date_UTC = lambda x : x.date_UTC + dt.timedelta(hours = 24)), 
                                ])
        df_load = pd.concat([load_db_13_15, 
                             load_db_16, 
                             load_db_17, 
                             ], 
                            axis = 0,
                            ).set_index(etc.user_dt_UTC).iloc[::2]
        # Correct rows that are all Nan
        df_load.loc[pd.to_datetime('2015-10-31 21:00:00+00:00'):pd.to_datetime('2015-11-01 00:00:00+00:00')] = df_load.loc[pd.to_datetime('2015-10-31 21:00:00+00:00'):pd.to_datetime('2015-11-01 00:00:00+00:00')].interpolate(axis = 0)
        df_load.loc[pd.to_datetime('2016-10-31 21:00:00+00:00'):pd.to_datetime('2016-11-01 00:00:00+00:00')] = df_load.loc[pd.to_datetime('2016-10-31 21:00:00+00:00'):pd.to_datetime('2016-11-01 00:00:00+00:00')].interpolate(axis = 0)
        # Format column names
        df_load.columns       = list(map(lambda x : tools.format_site_name(x), df_load.columns))  
        #df_load.columns.names = [etc.user_site_name]  
        #df_load.columns = pd.MultiIndex.from_arrays([df_load.columns], names=[etc.user_site_name]) 
        # Discard ignored substations
        df_load         = df_load[[e for e in df_load.columns if e not in RTE.config.ignored_substations]]
        # Checks
        tools.check_dates(df_load.index)        
        df_load = df_load.reindex(pd.date_range(start = df_load.index.min(),
                                                end   = df_load.index.max(),
                                                freq  = '1H',
                                                name  = df_load.index.name, 
                                                ))        
        # Save
        os.makedirs(os.path.dirname(fname), exist_ok = True)
        df_load.to_csv(fname,
                       #index_label = df_load.columns.name,
                       )
    print('done : df_load.shape = {0}\n{1}'.format(df_load.shape, '#'*etc.global_var.NB_SIGNS))
    return df_load

###############################################################################          
        
def get_coordinates_sites(df_corrected_load):
    print('get_coordinates_sites - ', end = '')
    tmp_coordinates_substations = pd.read_csv(RTE.etc.dikt_files['sites.coordinates'])
    tmp_coordinates_substations = tmp_coordinates_substations.rename(columns={RTE.etc.RTE_site_name : etc.user_site_name, 
                                                                              RTE.etc.RTE_longitude : etc.user_longitude, 
                                                                              RTE.etc.RTE_latitude  : etc.user_latitude,
                                                                              })
    tmp_coordinates_substations[etc.user_site_name] = tmp_coordinates_substations[etc.user_site_name].apply(tools.format_site_name) 
    tmp_coordinates_substations = tmp_coordinates_substations.set_index(etc.user_site_name)[[etc.user_latitude, etc.user_longitude]]
    substations_names           = sorted(set(df_corrected_load.columns).intersection(tmp_coordinates_substations.index))
    coordinates_substations     = tmp_coordinates_substations.filter(items = substations_names, axis = 0)
    df_filtered_load            = df_corrected_load.filter(items = substations_names, axis = 1)
    print('Names of the {0} substations : \n{1}'.format(df_corrected_load.shape[1],
                                                        substations_names,
                                                        )) 
    print('done - df_corrected_load.shape = {0} - len(coordinates_substations) = {1}\n{2}'.format(df_corrected_load.shape,
                                                                                                  len(coordinates_substations),
                                                                                                  '#'*etc.global_var.NB_SIGNS),
                                                                                                  )
    return df_filtered_load, coordinates_substations
      