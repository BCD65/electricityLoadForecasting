
import numpy    as np
import pandas as pd
import os
import pickle
from termcolor import colored
from sklearn.ensemble import RandomForestRegressor
try:
    import spams # spams should be installed manually : conda install -c conda-forge python-spams
except:
    pass
try:
    import xgboost as xgb # xgboost should be installed manually : conda install -c conda-forge xgboost
except:
    pass
#
import electricityLoadForecasting.etc.global_var             as global_var
import electricityLoadForecasting.dataPreparation.plot_tools as plot_tools



def correct_with_regression(df_load, dikt_errors, prefix = None, prefix_plot = None, bool_plot_corrections = None, bool_plot_trash = None): 
    """
    Learn a predictor for each bad site
    to predict the irrelevant values 
    from the values of the other sites 
    that do not have irrelevant values
    """
    print('correct_with_regression - ', end = '')
    fname_load  = os.path.join(prefix, 'df_corrected_load.csv')
    fname_trash = os.path.join(prefix, 'trash_sites.pkl')
    try:
        df_corrected_load = pd.read_csv(fname_load,
                                        index_col = 0,
                                        #header    = [0],
                                        )
        df_corrected_load.index = pd.to_datetime(df_corrected_load.index)
        with open(fname_trash, 'rb') as f:
            trash_sites = pickle.load(f)
        print('Loaded df_corrected_load and trash_sites')
    except Exception as e:
        print('\n{0}'.format(colored(e,'red')))
        print('df_corrected_load not loaded')
        bad_sites  = sorted(set([site
                                 for k, v in dikt_errors.items()
                                 for site in v
                                 ]))
        df_corrected_load = df_load.copy()
        trash_sites       = []
        X                 = df_load[sorted(set(df_load.columns) - set(bad_sites))]
        assert not pd.isnull(X).sum().sum()
        for ii, site in enumerate(bad_sites):  
            print('\r{0:6} / {1:6} - '.format(ii, len(bad_sites)), end = '')
            y             = df_load[site]
            flags = {dd : error_type 
                     for error_type in dikt_errors
                     for ii, dd in dikt_errors[error_type].get(site, [])
                     }
            samples_unkown = [(ii, dd) 
                               for error_type in dikt_errors
                               for ii, dd in dikt_errors[error_type].get(site, [])
                               ]
            ind_unknown, dates_unknown = list(zip(*samples_unkown))
            ind_unknown   = sorted(ind_unknown)
            dates_unknown = sorted(dates_unknown)
            ind_known     = [ii for ii in range(y.shape[0]) if ii not in ind_unknown] # Indices corresponding to sane observations
            assert not pd.isnull(y.iloc[ind_known]).sum()
            if len(ind_known) == 0:
                trash_sites.append((site, 'dates_known empty'))
                df_corrected_load = df_corrected_load.drop(site, axis = 1)
                print('{0:6} ->  drop because dates known empty'.format(site))
                continue    
            shuffled_ind_known = ind_known.copy()
            np.random.shuffle(shuffled_ind_known)
            cut = int(0.9*len(shuffled_ind_known))
            # Divide the sane observations into a training and a test sets
            ind_train = sorted(shuffled_ind_known[:cut])
            ind_test  = sorted(shuffled_ind_known[cut:])
            # Train
            y_train   = y.iloc[ind_train]
            X_train   = X.iloc[ind_train]
            # Validation
            y_test    = y.iloc[ind_test]
            X_test    = X.iloc[ind_test]
            # Pred
            X_pred    = X.iloc[ind_unknown]
            # Normalization covariates
            X_mean  = X_train.mean(axis = 0)
            X_std   = X_train.std(axis = 0)
            X_train = (X_train - X_mean)/X_std
            X_test  = (X_test  - X_mean)/X_std
            X_pred  = (X_pred  - X_mean)/X_std
            # Normalization target
            y_mean  = y_train.mean(axis = 0)
            y_std   = y_train.std (axis = 0)
            y_train = (y_train - y_mean)/y_std
            assert np.allclose(X_train.sum(), 0)
            assert np.allclose(y_train.sum(), 0)
            regressor = 'rf' # 'rf' # 'xgb' # 'spams' 
            # Assess the quality of a predictor from the other sane sites
            # We de not have a criteria to decide which algorithms is the most 
            # appropriate and have used alternatively spams of random forests.                            
            if regressor == 'rf':
                model = RandomForestRegressor()
                model.fit(X_train, y_train)
                y_hat_train = model.predict(X_train)
                y_hat_test  = model.predict(X_test)
                y_hat_pred  = model.predict(X_pred)
            elif regressor == 'xgb':
                model = xgb.XGBRegressor()
                model.fit(X_train, y_train)
                y_hat_train = model.predict(X_train)
                y_hat_test  = model.predict(X_test)
                y_hat_pred  = model.predict(X_pred)
            elif regressor == 'spams':
                hprm = {'loss'              : 'square', 
                        'numThreads'        : -1,
                        'verbose'           : False,
                        'lambda1'           : 0.03*X_train.shape[0],
                        'lambda2'           : 0.1, # For elastic_net
                        'it0'               : 10, # iter between two dual gap computations
                        'max_it'            : int(1e4), # (optional, maximum number of iterations, 100 by default)
                        'L0'                : 0.1, # (optional, initial parameter L in fista, 0.1 by default, should be small enough)
                        'regul'             : 'l2',
                        'tol'               : 1e-4, 
                        'intercept'         : False, #(optional, do not regularize last row of W, false by default)
                        'compute_gram'      : True,
                        'return_optim_info' : True
                        }
                beta0                = np.zeros((X_train.shape[1], 1), 
                                                dtype = np.float64, 
                                                order = "F",
                                                )
                beta_cen, optim_info = spams.fistaFlat(np.asfortranarray(y_train, dtype = np.float64).reshape((-1, 1)), 
                                                       np.asfortranarray(X_train, dtype = np.float64), 
                                                       beta0, 
                                                       **hprm,
                                                       )
                beta                 = beta_cen[:,0]
                y_hat_train = X_train @ beta
                y_hat_test  = X_test  @ beta
                y_hat_pred  = X_pred  @ beta
            y_train     = y_train * y_std + y_mean
            y_hat_train = y_hat_train * y_std + y_mean
            y_hat_test  = y_hat_test  * y_std + y_mean
            y_hat_pred  = y_hat_pred  * y_std + y_mean
            rr_train = 1 - ((y_train - y_hat_train)**2).mean()/y_train.std()**2
            rr_test  = 1 - ((y_test  - y_hat_test )**2).mean()/y_test .std()**2
            if not (rr_train > 0.9 and rr_test > 0.5): # If the performances are not good enough on the training and the test sets, drop the site
                trash_sites.append((site, 
                              'rr_train = {rr_train:.2} - rr_test = {rr_test:.2}'.format(
                                                                                         rr_train = rr_train,
                                                                                         rr_test  = rr_test,
                                                                                         )))
                df_corrected_load = df_corrected_load.drop(site, axis = 1)
                print('{0:6} ->  drop because prediction not good enough - rr_train = {rr_train:.2} - rr_test = {rr_test:.2}'.format(site,
                                                                                                                                     rr_train = rr_train,
                                                                                                                                     rr_test  = rr_test,
                                                                                                                                     ))
                continue
            if bool_plot_corrections:
                plot_tools.plot_corrections(y, 
                                            dates_unknown, 
                                            y_hat_pred, 
                                            os.path.join(prefix_plot, 
                                                         'corrections',
                                                         ),
                                            regressor, 
                                            rr_test,
                                            flags,
                                            )
            print('{0:6} -> {1:5} values corrected - rr_train = {rr_train:.2} - rr_test = {rr_test:.2}'.format(site, 
                                                                                                               len(ind_unknown), 
                                                                                                               rr_train = rr_train,
                                                                                                               rr_test  = rr_test,
                                                                                                               ))
            df_corrected_load[site].iloc[ind_unknown] = y_hat_pred  
        df_corrected_load.to_csv(fname_load, 
                                 #index_label = df_load.columns.name,
                                 )
        with open(fname_trash, 'wb') as f:
            pickle.dump(trash_sites, f)
    if bool_plot_trash:    
        plot_tools.plot_trash(trash_sites, 
                              df_load,
                              os.path.join(prefix_plot,
                                           'trash_sites',
                                           ),
                              ) # Plot the sites that are discarded
    print('done - df_corrected_load.shape = {0} - len(trash_sites) = {1}\n{2}'.format(df_corrected_load.shape,
                                                                                      len(trash_sites),
                                                                                      '#'*global_var.NB_SIGNS),
                                                                                      )
    return df_corrected_load, trash_sites
            
  
