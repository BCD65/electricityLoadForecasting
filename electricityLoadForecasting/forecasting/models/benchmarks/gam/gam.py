

import os
import re
import pandas as pd
import subprocess
#import warnings
try: # pip install rpy2==3.3.1
    #warnings.simplefilter(action="ignore", category=FutureWarning)
    from rpy2.robjects import pandas2ri, r
    #warnings.resetwarnings()
except ModuleNotFoundError:
    pass
#
from electricityLoadForecasting import paths


def fit_and_predict(inputs_training, Y_training, inputs_validation, hprm, assignments = {}):
    Y_hat_training   = pd.DataFrame(0, 
                                    index   = Y_training.index, 
                                    columns = Y_training.columns,
                                    )
    Y_hat_validation = pd.DataFrame(0, 
                                    index   = inputs_validation.index, 
                                    columns = Y_training.columns,
                                    )
        
    for ii, site_name in enumerate(Y_training.columns):
        print('\r{0}/{1}'.format(ii, Y_training.shape[1]), end = '\n')
        site_inputs_training = {(qty, *prm) : (inputs_training[(qty, *prm)]
                                               if (qty not in assignments)
                                               else
                                               inputs_training[(qty, *prm)][assignments[qty][site_name]]
                                               ) 
                                for (qty, *prm) in inputs_training.columns.droplevel(-1).drop_duplicates()
                                }
        site_inputs_validation = {(qty, *prm) : (inputs_validation[(qty, *prm)]
                                                 if (qty not in assignments)
                                                 else
                                                 inputs_validation[(qty, *prm)][assignments[qty][site_name]]
                                                 ) 
                                  for (qty, *prm) in inputs_validation.columns.droplevel(-1).drop_duplicates()
                                  }
        Y_hat_training[site_name], Y_hat_validation[site_name] = call_fitter(site_inputs_training,
                                                                             Y_training[site_name],
                                                                             site_inputs_validation, 
                                                                             hprm,
                                                                             )
    return Y_hat_training, Y_hat_validation


def call_fitter(inputs_training, y_training, inputs_validation, hprm):
    assert y_training.ndim == 1
    path_R_files = os.path.join(paths.outputs, 
                                'R_files/', 
                                )
    os.makedirs(path_R_files,
                exist_ok = True,
                )
    
    data_training   = {**{simplify_inpt_name(qty, transform, prm, inpt_nb = str(ii) if data.shape[1] > 1 else '') : data.iloc[:,ii].values
                          for (qty, transform, prm), data in inputs_training.items()
                          for ii in range(data.shape[1])
                          },
                       'target' : y_training.values,
                       }
    
    data_validation = {simplify_inpt_name(qty, transform, prm, inpt_nb = str(ii) if data.shape[1] > 1 else '') : data.iloc[:,ii].values
                       for (qty, transform, prm), data in inputs_validation.items()
                       for ii in range(data.shape[1])
                       }
    
    univariate_formula = {simplify_inpt_name(qty, transform, prm) : v
                          for (qty, transform, prm), v in hprm['gam.univariate_functions'].items()
                          }
    
    bivariate_formula  = {simplify_inpt_name(qty, transform, prm) : v
                          for (qty, transform, prm), v in hprm['gam.bivariate_functions'].items()
                          }
            
    ### Convert arrays
    pandas2ri.activate()
    df_train = pandas2ri.py2rpy(pd.DataFrame.from_dict(data_training))
    df_test  = pandas2ri.py2rpy(pd.DataFrame.from_dict(data_validation))
    pandas2ri.deactivate()
    
    ### Save converted files
    r.assign("data_train", df_train)
    r("save(data_train, file='{0}/temp_dat_for_r_train.gzip', compress=TRUE)".format(path_R_files))
    r.assign("data_test",  df_test)
    r("save(data_test,  file='{0}/temp_dat_for_r_test.gzip',  compress=TRUE)".format(path_R_files))
    
    string_formula = make_gam_formula(univariate_formula, 
                                      bivariate_formula,
                                      data_training.keys(), 
                                      hprm,
                                      )
    
    ### Launch the R script
    path2script = os.path.join(os.path.dirname(__file__),
                               'load_fit_predict_savePredictions.R',
                               )
    args        = [string_formula, path_R_files]
    cmd         = ['Rscript', path2script] + args
    ### Python will quote what must be quoted in subprocess.check_output

    print('launch Rscript')
    x = subprocess.check_output(cmd, universal_newlines=True)
    print(x)

    y_hat_training   = r['read.table']("{0}/predictions_from_r_train.gzip".format(path_R_files))
    y_hat_training   = pandas2ri.rpy2py(y_hat_training)
    y_hat_training   = y_hat_training.values
    
    y_hat_validation = r['read.table']("{0}/predictions_from_r_test.gzip".format(path_R_files))
    y_hat_validation = pandas2ri.rpy2py(y_hat_validation)
    y_hat_validation = y_hat_validation.values
    
    return y_hat_training, y_hat_validation


def simplify_inpt_name(qty, transform, prm, inpt_nb = ''):
    ans = re.sub('_+', '_', '_'.join([qty, 
                                      transform[:3] if bool(transform) else '',
                                      str(prm.hours if hasattr(prm, 'hours') else prm) if bool(prm) else '',
                                      inpt_nb,
                                      ])).strip('_')
    return ans


def make_gam_formula(univariate_functions, bivariate_functions, list_keys, hprm):

    ### Init
    formula = 'target~'
    
    ### Univariate part
    for key in list_keys:
        category = '_'.join([e for e in key.split('_') if e[0].isalpha()])
        if category in univariate_functions:
            formula += (formula[-1]!='~')*'+' 
            formula += univariate_functions[category][0] 
            formula += '(' 
            formula += key    
            if univariate_functions[category][1]:
                formula += ',' + univariate_functions[category][1]
            formula += ')'
                
    ### Bivariate part
    for key1 in list_keys:
        category1 = '_'.join([e for e in key1.split('_') if e[0].isalpha()])
        for key2 in list_keys:
            category2 = '_'.join(key2.split('_')[:2])
            if (category1, category2) in bivariate_functions:
                if len(bivariate_functions[category1,category2]) == 1:
                    assert bivariate_functions[category1,category2][0] == 'by'
                    formula += (formula[-1]!='~')*'+' 
                    formula += 's({0},by={1})'.format(key1,
                                                key2,
                                                )
                elif len(bivariate_functions[category1,category2]) == 2:
                    formula += (formula[-1]!='~')*'+' 
                    formula += '{0}({1},{2}{3})'.format(bivariate_functions[category1,category2][0],
                                                  key1,
                                                  key2,
                                                  (',' + bivariate_functions[category1,category2][1] 
                                                   if bivariate_functions[category1,category2][1] 
                                                   else 
                                                   ''
                                                   ), 
                                                  )
                else:
                    raise ValueError

    return formula




