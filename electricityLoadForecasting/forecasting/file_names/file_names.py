

"""
This script creates long names of files from the hyperparameters 
to save the variables and the results.
"""

#import misc
import numpy as np
import pandas as pd
import os
#
import electricityLoadForecasting.tools  as tools


def make_dikt_files(hprm, nb_sites = None, nb_weather = None, dt_training = None, dt_validation = None):

    dikt = {}
    
    #=========================================#
    ###         Dataset                     ###
    #=========================================#
    str_dataset = (hprm['database'].replace('.','_'),
                   'sites',
                   hprm['sites.zone'],
                   str(nb_sites),
                   hprm['sites.aggregation'] if bool(hprm['sites.aggregation']) else '',
                   'mask{0}'.format(hprm['inputs.nb_sites_per_site']),
                   'weather',
                   hprm['weather.source'],
                   hprm['weather.zone'],
                   str(nb_weather), 
                   hprm['weather.aggregation'] if bool(hprm['weather.aggregation']) else '',
                   'mask{0}'.format(hprm['inputs.nb_weather_per_site']),
                   )
    str_split_training   = (
                            'training',
                                     dt_training.min().strftime(format = '%Y%m%d'),
                                     dt_training.max().strftime(format = '%Y%m%d'),
                                     )
    str_split_validation = (
                             'validation',
                             dt_validation.min().strftime(format = '%Y%m%d'),
                             dt_validation.max().strftime(format = '%Y%m%d'),
                             )
        
    str_learning = (hprm['learning.model'],
                    'independent_learning' if not hprm['learning.model.separation.sites'] else '',
                    '_'.join(['separation', *hprm['learning.model.separation.input']]) if hprm['learning.model.separation.input'] else '',
                    )

      
    #=========================================#
    ###             Model                   ###
    #=========================================#  
    
    if hprm['learning.model'] not in {'afm'}:
    
        str_inputs = {}
        for variable in hprm['inputs.selection']:
            list_attr = []
            for e in variable:
                if bool(e):
                    if type(e) == pd.DateOffset:
                        list_attr.append('_'.join([str(v)+k for k, v in e.kwds.items()]))
                    else:
                        list_attr.append(str(e))
            str_inputs[variable] = list_attr
    
        dikt['experience.whole'] = (
                                    str_dataset,
                                    str_split_training,
                                    str_split_validation,
                                    str_learning,
                                    *list(str_inputs.values()),
                                    )
        # Specific to GAM
        if hprm['learning.model'] in {'gam'}:
            formula_uni = tuple([e
                                 for inpt, (func, prm) in hprm['gam.univariate_functions'].items()
                                 for e in [func, inpt, prm]
                                 ])
            formula_biv = tuple([e
                                 for (inpt1, inpt2), (func, prm) in hprm['gam.bivariate_functions'].items()
                                 for e in [func, inpt1, inpt2, prm]
                                 ])
            dikt['experience.whole'] += (formula_uni, formula_biv)
        
        
    #=========================================#
    ###               afm                   ###
    #=========================================#    
    elif hprm['learning.model'] in {'afm'}:
        ############
        # Features #
        ############        
        order_splines    = ('order' +str(hprm.get('afm.order_splines', 1)))*(hprm.get('order_splines', 1)!=1)
        natural_splines  = 'nat'*hprm.get('afm.natural_splines',  0)
        boundary_scaling = 'bds'*hprm.get('afm.boundary_scaling', 0)
        all_products     = 'apr'*(hprm.get('gp_pen',0) > 0 and hprm.get('gp_matrix','') != '') # When apr, x1tx1 containes all interactions  
        univariate_func  = hprm['afm.formula'].loc[hprm['afm.formula']['nb_intervals'].apply(lambda x : type(x) != tuple)].reset_index().astype(str)
        univariate_data  = [tuple(univariate_func.iloc[ii][['input', 'nb_intervals']])
                            for ii in range(univariate_func.shape[0])
                            ]
        univariate_model = [tuple(univariate_func.iloc[ii])
                            for ii in range(univariate_func.shape[0])
                            ]
        bivariate_func   = hprm['afm.formula'].loc[hprm['afm.formula']['nb_intervals'].apply(lambda x : (type(x) == tuple and len(x) == 2))].reset_index().astype(str)
        bivariate_data   = [tuple(bivariate_func.iloc[ii][['input', 'nb_intervals']])
                            for ii in range(bivariate_func.shape[0])
                            ]
        bivariate_model  = [tuple(bivariate_func.iloc[ii])
                            for ii in range(bivariate_func.shape[0])
                            ]
        str_inter    = (hprm['afm.features.bivariate.combine_function'].__name__ if bool(bivariate_data) else '')
        str_data     = [str_dataset,
                        str_learning,
                        (natural_splines,
                         order_splines,
                         boundary_scaling,
                         all_products,
                         ),
                        str_split_training,
                        ]
        dikt['features.training.univariate']   =  str_data + univariate_data
        dikt['features.validation.univariate'] =  str_data + univariate_data  + [str_split_validation]
        dikt['features.training.bivariate']    = (str_data + [(str_inter,)]   + bivariate_data) if str_inter else []
        dikt['features.validation.bivariate']  = (str_data + [(str_inter,)]   + bivariate_data + [str_split_validation]) if str_inter else []
        dikt['features.training.all']          =  str_data + univariate_data  + ([(str_inter,)] + bivariate_data if str_inter else [])
        dikt['features.validation.all']        =  str_data + univariate_data  + ([(str_inter,)] + bivariate_data if str_inter else []) + [str_split_validation]
        dikt['experience.model']               =  str_data + univariate_model + ([(str_inter,)] + bivariate_model if str_inter else []) + [str_split_validation]
        
        #############
        # Algorithm #
        #############
        gp_reg         = (('gp', str(hprm.get('gp_matrix')), str(hprm.get('gp_pen')))
                          if (hprm.get('gp_pen', 0)!=0 and hprm.get('gp_matrix','') != '')
                          else
                          ''
                          )

        # Stopping criteria
        if hprm['afm.algorithm'] == 'L-BFGS':
            stop = '_'.join(['stop', 
                             str(hprm['afm.algorithm.lbfgs.tol']),
                             str(hprm['afm.algorithm.lbfgs.pgtol']),
                             '{:.0e}'.format(hprm['afm.algorithm.lbfgs.maxfun']).replace('s+', ''),
                             '{:.0e}'.format(hprm['afm.algorithm.lbfgs.maxiter']).replace('+', ''),
                             ])
        elif hprm['afm.algorithm'] == 'FirstOrder':
            stop = '_'.join(['nws'  *(1-hprm['afm.algorithm.first_order.try_warmstart']),
                             'stop', 
                             '{:.0e}'.format(hprm['afm.algorithm.first_order.max_iter']).replace('+', ''),
                             'noes' *(1-hprm['afm.algorithm.first_order.early_stop_validation']),
                             'noesi'*(1-hprm['afm.algorithm.first_order.early_stop_ind_validation']),
                             str(hprm['afm.algorithm.first_order.tol']), 
                             ]) 
        else:
            raise ValueError('Incorrect algorithm : {0}'.format(hprm['afm.algorithm']))                  
            
        dikt['experience.whole'] = dikt['experience.model'] + [gp_reg,
                                                               (stop,),
                                                               ]
        

    for k in dikt: 
        if bool(dikt[k]) and type(dikt[k]) != str:
            dikt[k] = os.path.join(*['_'.join(sub_tuple)
                                     for sub_tuple in dikt[k]
                                     ])
        else:
            dikt[k] = ''             

          
    #=========================================#
    #            Format                       #
    #=========================================#
    # Format strings that are too long - it should no longer occur
    for key, ss in dikt.items():
        too_long = np.sum([len(ee) > 250 for ee in ss.split(os.sep)])
        if too_long:
            list_chunks = ss.split(os.sep)
            new_list_chunks = []
            for ee in list_chunks:
                for ii in range(len(ee)//250+1):
                    new_list_chunks.append(ee[250*ii:250*(ii+1)])
            dikt[key] = os.sep.join(new_list_chunks)
        dikt[key] = tools.format_file_names(dikt[key])
    
    for ss in [
               'experience.whole',
               ]:
        print("dikt['{0}']\n{1}".format(ss, dikt[ss].replace(os.sep, '/\n\t\t\t')))
    return dikt

