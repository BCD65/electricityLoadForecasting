

import numpy as np
import copy as cp

def bool_separated(hprm):
    variable_separation = (hprm.get('separation_var')  == ())
    independent_models  = (hprm.get('separation_post') == True)
    return variable_separation or independent_models


def separated(exp):
                       
    args = (
            exp.param, 
            exp.dikt, 
            exp.posts_names, 
            exp.data_train, 
            exp.target_train, 
            exp.dates_train,
            exp.data_test,  
            exp.target_test,  
            exp.dates_test, 
            )
    
    if exp.param.get('separation_post'):
        (exp.prediction_train, exp.prediction_test, 
         exp.model_pred,       exp.ans_kw,
         ) = optim_sep_post_owned_data(
                                       exp.param,
                                       exp.IMMACULATE_PARAM,
                                       exp.posts_names
                                       )
        exp.pre_treatment_adjust(adjust_predictions = False)
    elif exp.param.get('separation_var'):
        exp.prediction_train, exp.prediction_test, exp.model_pred, exp.ans_kw = optim_sep_var(
                                                                                              exp,
                                                                                              exp.param['separation_var'][0], 
                                                                                              exp.param['separation_var'][1],               
                                                                                              *args,
                                                                                              )
        exp.pre_treatment_adjust()
    return exp



def optim_sep_post_owned_data(
                              param, 
                              IMMACULATE_PRM,
                              posts_names,  
                              ):
    raise NotImplementedError
    """
    Function to compute the models for the different substations sequentially 
    """
    prediction_train, prediction_test, model_pred, ans_kw = [[] for _ in range(4)]
    # Reload fresh data once
    loaded_data = munging.get_data(param)
    with open(param['path_inputs'] + 'dates.npy', 'rb') as f_dates: 
        loaded_dates      = np.load(f_dates, allow_pickle=True)
    for pp, post in enumerate(posts_names):
        print(colored('{0} {1} / {2}'.format(post, str(pp), str(posts_names.shape[0])), 'green', 'on_blue'))
        local_param = {**cp.deepcopy(param), 
                       'posts_id'        : [post], 
                       'separation_post' : False,
                       'download'        : False, 
                       'upload'          : False,
                       'load_perf'       : False, 
                       'data_cat'        : cp.deepcopy(IMMACULATE_PRM['data_cat']),
                       'stations_id'     : cp.deepcopy(IMMACULATE_PRM['stations_id']),
                       }
        local_exp = main(local_prm, 
                         load_data    = cp.deepcopy(load_data), 
                         weather_data = cp.deepcopy(weather_data),
                         )
        prediction_train.append(local_exp.prediction_train)
        prediction_test .append(local_exp.prediction_test )
    
    prediction_train = np.concatenate(prediction_train, 
                                      axis = 1, 
                                      )
    prediction_test  = np.concatenate(prediction_test , 
                                      axis = 1, 
                                      )
    return prediction_train, prediction_test, model_pred, ans_kw


#%%

def optim_sep_var(exp, var, var_sep, 
                  param,
                  dikt, posts_names,
                  data_train, target_train, dates_train,  
                  data_test,  target_test,  dates_test,    
                  ):
    raise NotImplementedError
    """
    Function to compute the models sequentially for the different values of an input time series (eg different hours of the day)
    """
    prediction_train = np.zeros(target_train.shape)
    prediction_test  = np.zeros(target_test .shape)
    witness_train    = np.zeros(target_train.shape)
    witness_test     = np.zeros(target_test .shape)
    model_pred, ans_kw = [[] for k in range(2)]
    if type(var_sep) == int:
        intervals = [((k-0.5)/var_sep,(k+0.5)/var_sep) for k in range(var_sep+(1-param['qo_modulo'][var]))]
    else:
        raise ValueError
    for ii, interval in enumerate(intervals):
        # For loop over the different values of the input time series (eg the differnt hours)
        dikt_loc = cp.deepcopy(dikt)
        for k, v in dikt_loc.items():
            w = v.split('/')
            w[0] += '_' + str(round(interval[0],5)) + '_' + str(round(interval[1],5))   
            dikt_loc[k] = str_make.format_string('/'.join(w))
        ind_train = np.array([data_train[var][k] >= interval[0] and data_train[var][k] < interval[1] 
                              for k in range(data_train[var].shape[0])
                              ]).reshape(-1)
        ind_test  = np.array([data_test [var][k] >= interval[0] and data_test [var][k] < interval[1] 
                              for k in range(data_test [var].shape[0])
                              ]).reshape(-1)
        args = (
                {**cp.deepcopy(param), 
                 'load_perf'   : False, # So that prediction can be concatenated
                 'normalize_y' : '',
                 'separation_var' : (),
                 },
                dikt_loc, 
                posts_names, 
                {k:v[ind_train] 
                 for k, v in cp.deepcopy(data_train).items() 
                 if k!= var
                },
                target_train[ind_train], 
                dates_train  [ind_train],
                {k:v[ind_test] 
                 for k, v in cp.deepcopy(data_test).items() 
                 if k!= var
                }, 
                target_test[ind_test],  
                dates_test  [ind_test], 
                )                        
        ans = exp.optim_model(*args)
        prediction_train[ind_train] = ans[0]
        witness_train[ind_train] = 1
        prediction_test [ind_test]  = ans[1]
        witness_test [ind_test ] = 1
        model_pred      .append(None)
        ans_kw          .append(None)
    assert not (witness_train!=1).sum()
    assert not (witness_test !=1).sum()
    return prediction_train, prediction_test, model_pred, ans_kw
