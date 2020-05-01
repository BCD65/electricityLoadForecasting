

import os
import re
import numpy as np
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


def fit_and_predict(inputs_training,
                    Y_training,
                    inputs_validation,
                    hprm,
                    assignments = {},
                    ):
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
        
        site_selection = [(inpt, trsfm, prm, location)
                          for inpt, trsfm, prm, location in inputs_training.columns
                          if (   inpt not in assignments
                              or location in assignments[inpt][site_name].values
                              )
                          ]
                        
        site_inputs_training   = inputs_training  [site_selection]
        site_inputs_validation = inputs_validation[site_selection]
        
        site_inputs_training.columns   = site_inputs_training.columns.remove_unused_levels()
        site_inputs_validation.columns = site_inputs_validation.columns.remove_unused_levels()
        
        # {(qty, *prm) : (inputs_training[(qty, *prm)]
        #                                        if (qty not in assignments)
        #                                        else
        #                                        inputs_training[(qty, *prm)][assignments[qty][site_name]]
        #                                        ) 
        #                         for (qty, *prm) in inputs_training.columns.droplevel(-1).drop_duplicates()
        #                         }
        # site_inputs_validation = {(qty, *prm) : (inputs_validation[(qty, *prm)]
        #                                          if (qty not in assignments)
        #                                          else
        #                                          inputs_validation[(qty, *prm)][assignments[qty][site_name]]
        #                                          ) 
        #                           for (qty, *prm) in inputs_validation.columns.droplevel(-1).drop_duplicates()
        #                           }
        Y_hat_training[site_name], Y_hat_validation[site_name] = call_fitter(site_inputs_training,
                                                                             Y_training[site_name],
                                                                             site_inputs_validation, 
                                                                             hprm,
                                                                             )
    return Y_hat_training, Y_hat_validation


def call_fitter(site_inputs_training,
                y_training,
                site_inputs_validation,
                hprm,
                ):
    assert y_training.ndim == 1
    path_R_files = os.path.join(paths.outputs, 
                                'R_files/', 
                                )
    os.makedirs(path_R_files,
                exist_ok = True,
                )
    
    ### Data
    data_training   = {**{simplify_inpt_name(inpt, trsfm, prm, location) : site_inputs_training[inpt, trsfm, prm, location].values
                          for inpt, trsfm, prm, location in site_inputs_training
                          },
                       # **{simplify_inpt_name(qty, transform, prm, inpt_nb = str(ii) if data.shape[1] > 1 else '') : data.iloc[:,ii].values
                       #    for (qty, transform, prm), data in site_inputs_training.items()
                       #    for ii in range(data.shape[1])
                       #    },
                       'target' : y_training.values,
                       }
    
    data_validation = {simplify_inpt_name(inpt, trsfm, prm, location) : site_inputs_validation[inpt, trsfm, prm, location].values
                       for inpt, trsfm, prm, location in site_inputs_validation
                       }
                      # {simplify_inpt_name(qty, transform, prm, inpt_nb = str(ii) if data.shape[1] > 1 else '') : data.iloc[:,ii].values
                      #  for (qty, transform, prm), data in site_inputs_validation.items()
                      #  for ii in range(data.shape[1])
                      #  }
            
    # Convert arrays
    pandas2ri.activate()
    df_train = pandas2ri.py2rpy(pd.DataFrame.from_dict(data_training))
    df_test  = pandas2ri.py2rpy(pd.DataFrame.from_dict(data_validation))
    pandas2ri.deactivate()
    
    # Save converted files
    r.assign("data_train", df_train)
    r("save(data_train, file='{0}/temp_dat_for_r_train.gzip', compress=TRUE)".format(path_R_files))
    r.assign("data_test",  df_test)
    r("save(data_test,  file='{0}/temp_dat_for_r_test.gzip',  compress=TRUE)".format(path_R_files))
    
    nb_unique = {k : len(np.unique(v))
                 for k, v in site_inputs_training.items()
                 }
    
    string_formula = make_gam_formula(site_inputs_training.columns,
                                      nb_unique,
                                      hprm,
                                      )
    
    ### Launch the R script
    path2script = os.path.join(os.path.dirname(__file__),
                               'load_fit_predict_savePredictions.R',
                               )
    args        = [string_formula, path_R_files]
    cmd         = ['Rscript', path2script] + args
    # Python will quote what must be quoted in subprocess.check_output

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


def simplify_inpt_name(qty,
                       transform,
                       prm,
                       location = '',
                       ):
    ans = re.sub('_+', '_', '_'.join([qty, 
                                      (transform[:3]
                                       if
                                       bool(transform)
                                       else
                                       ''
                                       ),
                                      (str(prm.n*prm.hours
                                           if hasattr(prm, 'hours')
                                           else prm
                                           ).replace('-', 'm')
                                       if bool(prm)
                                       else
                                       ''
                                       ),
                                      str(location).replace('-','_'),
                                      ])).strip('_')
    return ans


def make_gam_formula(list_cols,
                     nb_unique,
                     hprm,
                     ):
        
    ### Formula
    univariate_formula = {simplify_inpt_name(*inpt) : v
                          for inpt, v in hprm['gam.univariate_functions'].items()
                          }
    
    bivariate_formula  = {(simplify_inpt_name(*inpt1),
                           simplify_inpt_name(*inpt2),
                           ) : v
                          for (inpt1, inpt2), v in hprm['gam.bivariate_functions'].items()
                          }

    ### Init
    formula = 'target~'
    
    ### Univariate part
    for col in list_cols:
        *inpt, location = col
        # category = '_'.join([e
        #                      for e in key.split('_')
        #                      if e[0].isalpha()
        #                      ])
        simple_cat = simplify_inpt_name(*inpt)
        if simple_cat in univariate_formula:
            formula += (formula[-1]!='~')*'+' 
            if nb_unique[col] > 1:
                formula += univariate_formula[simple_cat][0] 
            formula += '(' 
            formula += simplify_inpt_name(*col)    
            if univariate_formula[simple_cat][1]:
                formula += ',' + univariate_formula[simple_cat][1]
            formula += ')'
                
    ### Bivariate part
    for col1 in list_cols:
        
        # category1 = '_'.join([e
        #                       for e in key1.split('_')
        #                       if e[0].isalpha()
        #                       ])
        for col2 in list_cols:
            *inpt1, location1 = col1
            *inpt2, location2 = col2
            simple_cat1 = simplify_inpt_name(*inpt1)
            simple_cat2 = simplify_inpt_name(*inpt2)
            #category2 = '_'.join(key2.split('_')[:2])
            if (simple_cat1, simple_cat2) in bivariate_formula:
                if len(bivariate_formula[simple_cat1, simple_cat2]) == 1:
                    assert bivariate_formula[simple_cat1, simple_cat2][0] == 'by'
                    formula += (formula[-1]!='~')*'+' 
                    if nb_unique[col1] > 1:
                        formula += 's'
                    formula += '({0},by={1})'.format(simplify_inpt_name(*col1),
                                                     simplify_inpt_name(*col2),
                                                     )
                elif len(bivariate_formula[simple_cat1, simple_cat2]) == 2:
                    formula += (formula[-1]!='~')*'+' 
                    formula += '{0}({1},{2}{3})'.format(bivariate_formula[simple_cat1, simple_cat2][0],
                                                        simplify_inpt_name(*col1),
                                                        simplify_inpt_name(*col2),
                                                        (',' + bivariate_formula[simple_cat1, simple_cat2][1] 
                                                         if bivariate_formula[simple_cat1, simple_cat2][1] 
                                                         else 
                                                         ''
                                                         ), 
                                                        )
                else:
                    raise ValueError

    return formula




