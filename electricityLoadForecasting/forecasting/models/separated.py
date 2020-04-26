

import numpy as np
import copy as cp
from termcolor import colored
#
from electricityLoadForecasting             import paths
from electricityLoadForecasting.forecasting import experience, inputs



def bool_separated(hprm):
    variable_separation = bool(hprm.get('learning.model.separation.input'))
    independent_models  = (hprm.get('learning.model.separation.sites') == True)
    return variable_separation or independent_models


def separated(exp):
                       
    # args = (
    #         exp.hprm, 
    #         exp.dikt, 
    #         #exp.posts_names, 
    #         exp.data_train, 
    #         exp.target_train, 
    #         exp.dates_train,
    #         exp.data_validation,  
    #         exp.target_validation,  
    #         exp.dates_validation, 
    #         )
    
    if exp.hprm.get('learning.model.separation.sites'):
        (exp.prediction_train,
         exp.prediction_validation,
         exp.model_pred,
         exp.ans_kw,
         ) = learn_sites_independently(exp.hprm,
                                       #exp.IMMACULATE_hprm,
                                       #exp.posts_names
                                       )
    elif exp.hprm.get('learning.model.separation.input'):
        (exp.prediction_train,
         exp.prediction_validation,
         exp.model_pred,
         exp.ans_kw,
         )= learn_input_separated_models(exp,
                                         exp.hprm['learning.model.separation.input'][0], 
                                         exp.hprm['learning.model.separation.input'][1],               
                                         )
    return exp



def learn_sites_independently(hprm, 
                              #IMMACULATE_PRM,
                              #posts_names,  
                              ):
    #raise NotImplementedError
    """
    Function to compute the models for the different substations sequentially 
    """
    prediction_train, prediction_validation, model_pred, ans_kw = [[] for _ in range(4)]
    # Reload fresh data once
    fresh_data = inputs.load_input_data(paths.path_database(hprm['database']))
    # with open(hprm['path_inputs'] + 'dates.npy', 'rb') as f_dates: 
    #     loaded_dates      = np.load(f_dates, allow_pickle=True)
    for ii, site in enumerate(fresh_data['df_sites'].columns):
        print(colored('{0} {1} / {2}'.format(site,
                                             ii,
                                             len(fresh_data['df_sites'].columns),
                                             ),
                      'green', 'on_blue'))
        local_hprm = {**cp.deepcopy(hprm),
                      #'posts_id'                        : [site], 
                      'learning.model.separation.sites' : False,
                      #'download'        : False, 
                      #'upload'          : False,
                      #'load_perf'       : False, 
                      #'stations_id'     : cp.deepcopy(IMMACULATE_PRM['stations_id']),
                      }
        local_exp = experience.main(hprm = local_hprm, 
                                    data = cp.deepcopy(fresh_data), 
                                    )
        prediction_training  .append(local_exp.prediction_train)
        prediction_validation.append(local_exp.prediction_validation)
    
    prediction_training   = np.concatenate(prediction_training, 
                                           axis = 1, 
                                           )
    prediction_validation = np.concatenate(prediction_validation, 
                                           axis = 1, 
                                           )
    return prediction_training, prediction_validation#, model_pred, ans_kw


#%%

def learn_input_separated_models(exp,
                                 name_inpt,
                                 nb_submodels, 
                                 ):
    assert type(nb_submodels) == int
        
    prediction_training    = np.zeros(exp.target_training.shape)
    prediction_validation  = np.zeros(exp.target_validation.shape)
    witness_training       = np.zeros(exp.target_train.shape)
    witness_validation     = np.zeros(exp.target_validation.shape)
    
    inpt_training     = exp.inputs_training  [name_inpt]
    inpt_validation   = exp.inputs_validation[name_inpt]
    
    min_value = inpt_training.min()
    max_value = inpt_training.max()
    intervals = [(min_value + (max_value - min_value)*ii/nb_submodels,
                  min_value + (max_value - min_value)*(ii + 1)/nb_submodels,
                  )
                 for ii in range(nb_submodels)
                 ]

    # For loop over the different values of the input time series (eg the differnt hours)
    for ii, interval in enumerate(intervals):
        local_exp = cp.deepcopy(exp)
        local_exp.hprm['learning.model.separation.input'] = ()
        for k, v in local_exp.dikt_files.items():
            w = v.split(os.sep)
            w[0] += '_'.join([name_inpt,
                              str(round(interval[0],5)),
                              str(round(interval[1],5)),
                              ])
            local_exp.dikt_files[k] = os.sep.join(w)
            
        ind_training    = np.logical_and(np.logical_or(inpt_training >= interval[0],
                                                       ii == 0,
                                                       ),
                                         np.logical_or(inpt_training <  interval[1],
                                                       ii == len(intervals) - 1,
                                                       ),
                                         ).reshape(-1)
        ind_validation  = np.logical_and(np.logical_or(inpt_validation >= interval[0],
                                                       ii == 0,
                                                       ),
                                         np.logical_or(inpt_validation <  interval[1],
                                                       ii == len(intervals) - 1,
                                                       ),
                                         ).reshape(-1)
        local_exp.target_training   = exp.target_training.loc[ind_training]
        local_exp.inputs_training   = exp.inputs_training.loc[ind_training] 
        local_exp.target_validation = exp.target_validation.loc[ind_validation]
        local_exp.inputs_validation = exp.inputs_validation.loc[ind_validation]                     
        local_exp.learning_process()
        prediction_training  [ind_training]   = local_exp.prediction_training
        witness_training     [ind_training]   = 1
        prediction_validation[ind_validation] = local_exp.prediction_validation
        witness_validation   [ind_validation] = 1
        #model_pred      .append(None)
        #ans_kw          .append(None)
        
    assert not (witness_train!=1).sum()
    assert not (witness_validation !=1).sum()
    exp.prediction_training   = prediction_training
    exp.prediction_validation = prediction_validation






