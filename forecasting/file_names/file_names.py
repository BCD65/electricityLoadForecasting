

"""
This script creates long names of files from the hyperhprmeters 
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
    str_dataset = (hprm['database'],
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

    #=========================================#
    ###         Model                       ###
    #=========================================#
    str_learning = (hprm['learning.model'],
                    'coupled_models' if not hprm['learning.independent_models'] else '',
                    'individual_designs' if hprm['learning.individual_designs'] else '',
                    )    
    
    #=========================================#
    ###             GAM                     ###
    #=========================================#    
    if hprm['learning.model'] not in {'afm'}:
        # dikt_gam_to_str(dikt_uni, dikt_bi, data_cat)    
        #=========================================#
        ###         Variables                   ###
        #=========================================#
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
    
        dikt['experience.whole'] = os.path.join(
                                                str_dataset,
                                                str_split_training,
                                                str_split_validation,
                                                str_learning,
                                                *list(str_inputs.values()),
                                                )
    
    
    #=========================================#
    ###       approx tf                     ###
    #=========================================#    
    elif hprm['learning.model'] in {'afm'}:
        boundary_scaling = 'bds'*hprm.get('afm.boundary_scaling', 0)
        natural_splines  = 'nat'*hprm.get('afm.natural_splines',  0)
        order_splines    = ('order' +str(hprm.get('afm.order_splines', 1)))*(hprm.get('order_splines', 1)!=1)
        univariate_func  = hprm['afm.formula'].loc[hprm['afm.formula']['nb_intervals'].apply(lambda x : type(x) != tuple)].astype(str)
        univariate_data  = [tuple(univariate_func[['input', 'nb_intervals']].iloc[ii])
                            for ii in range(univariate_func.shape[0])
                            ]
        univariate_model = [tuple(univariate_func.iloc[ii])
                            for ii in range(univariate_func.shape[0])
                            ]
        bivariate_func   = hprm['afm.formula'].loc[hprm['afm.formula']['nb_intervals'].apply(lambda x : (type(x) == tuple and len(x) == 2))].astype(str)
        bivariate_data   = [tuple(bivariate_func[['input', 'nb_intervals']].iloc[ii])
                            for ii in range(bivariate_func.shape[0])
                            ]
        bivariate_model  = [tuple(bivariate_func.iloc[ii])
                            for ii in range(bivariate_func.shape[0])
                            ]
        all_products = 'apr'*(hprm.get('gp_pen',0) > 0 and hprm.get('gp_matrix','') != '') # When apr, x1tx1 containes all interactions  
        str_inter    = (str(hprm['afm.bivariate.combine_function']) if bool(bivariate_data) else '')
        str_data     = [str_dataset,
                        str_learning, 
                        (natural_splines,
                         order_splines,
                         boundary_scaling,
                         all_products,
                         ),
                        str_split_training,
                        ]
        dikt['training.univariate_data']   = str_data + univariate_data
        dikt['validation.univariate_data'] = str_data + univariate_data + [str_split_validation]
        dikt['training.bivariate_data']    = str_data + [(str_inter,)] + bivariate_data if str_inter else []
        dikt['validation.bivariate_data']  = str_data + [(str_inter,)] + bivariate_data + [str_split_validation] if str_inter else []
        dikt['training.univariate_bivariate_data']   = str_data + univariate_data + ([(str_inter,)] + bivariate_data if str_inter else [])
        dikt['validation.univariate_bivariate_data'] = str_data + univariate_data + ([(str_inter,)] + bivariate_data if str_inter else []) + [str_split_validation]
        dikt['validation.model'] = str_data + univariate_model + ([(str_inter,)] + bivariate_model if str_inter else []) + [str_split_validation]
        
        ##########
        # Algorithm
        ##########
        no_warmstart   = 'nws'  *(1-hprm['afm.algorithm.try_warmstart'])
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
        else:
            stop = '_'.join(['stop', 
                             '{:.0e}'.format(hprm['afm.algorithm.max_iter']).replace('+', ''),
                             'noes' *(1-hprm['afm.algorithm.early_stop_test']),
                             'noesi'*(1-hprm['afm.algorithm.early_stop_ind_test']),
                             str(hprm['afm.algorithm.tol']), 
                             ])                   
        #var_model = var_12 if str_inter else var1
        #dikt['model_wo_hyperp']  = dikt['validation.univariate_bivariate_parts']
        dikt['experience.whole'] = dikt['validation.model'] + [(no_warmstart, stop), gp_reg]

        for k in dikt: 
            if bool(dikt[k]):
                dikt[k] = os.path.join(*['_'.join(sub_tuple)
                                         for sub_tuple in dikt[k]
                                         ])
            else:
                dikt[k] = ''
        
        print("dikt['validation.univariate_data']", '\n', 
              *[' '*10 + dikt['validation.univariate_data'].split(os.sep)[ii] + '\n' 
                for ii in range(len(dikt['validation.univariate_data'].split(os.sep)))
                ],
              '\n',
              )
        print("dikt['validation.bivariate_data']", '\n', 
              *[' '*10 + dikt['validation.bivariate_data'].split(os.sep)[ii] + '\n'
                for ii in range(len(dikt['validation.bivariate_data'].split(os.sep)))
                ],
              '\n',
              )
        print("dikt['validation.univariate_bivariate_data']", '\n', 
              *[' '*10 + dikt['validation.univariate_bivariate_data'].split(os.sep)[ii] + '\n'
                for ii in range(len(dikt['validation.univariate_bivariate_data'].split(os.sep)))
                ],
              '\n',
              )        
#            
#    #=========================================#
#    #         Final model                     #
#    #=========================================#
#    dikt['experience.modeling'] = os.path.join(
#                                               str_dataset,
#                                               str_learning,
#                                               *list(str_inputs.values()),
#                                               )
#    dikt['experience.training'] = os.path.join(
#                                               str_dataset,
#                                               str_split_training,
#                                               str_learning,
#                                               *list(str_inputs.values()),
#                                               )
#    dikt['experience.validation'] = os.path.join(
#                                                 str_dataset,
#                                                 str_split_validation,
#                                                 str_learning,
#                                                 *list(str_inputs.values()),
#                                                 )

            
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
        
    print(dikt['experience.whole'].replace(os.sep, '/\n\t')) 
    
    return dikt


###############################################################################


def dikt_gam_to_str(dikt_uni, dikt_bi, data_cat):
    # save the gam formula as a string
    s = []
    for cat,(spline, kw) in dikt_uni.items():
        s += [cat, 
              spline, 
              kw.replace('c(','' )
                .replace('(' ,'' )
                .replace(')' ,'' )
                .replace(' ' ,'' )
                .replace('"' ,'' )
                .replace('=' ,'_')
                .replace(',' ,'_'),
              ]
    for (cat1,cat2),modeling in dikt_bi.items():
        assert type(modeling) == tuple
        if len(modeling) == 1:
            s += [
                  cat1+'#'+cat2,
                  'by', 
                  ]
        elif len(modeling) == 2:
            (spline, kw) = modeling
            s += [cat1+'#'+cat2, 
                  spline, 
                  kw.replace('c(','' )
                    .replace('(' ,'' )
                    .replace(')' ,'' )
                    .replace(' ' ,'' )
                    .replace('"' ,'' )
                    .replace('=' ,'_')
                    .replace(',' ,'_'),
                  ]
        else:
            raise ValueError
    return '_'.join(s)  


