

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
    str_dataset = '_'.join([hprm['database'],
                            'sites',
                            hprm['sites.zone'],
                            str(nb_sites),
                            hprm['sites.aggregation'] if bool(hprm['sites.aggregation']) else '',
                            'weather',
                            hprm['weather.zone'],
                            hprm['weather.source'],
                            str(nb_weather), 
                            hprm['weather.aggregation'] if bool(hprm['weather.aggregation']) else '',
                            ])
    str_split_training   = '_'.join([
                                     'training',
                                     dt_training.min().strftime(format = '%Y%m%d'),
                                     dt_training.max().strftime(format = '%Y%m%d'),
                                     ])
    str_split_validation = '_'.join([
                                     'validation',
                                     dt_validation.min().strftime(format = '%Y%m%d'),
                                     dt_validation.max().strftime(format = '%Y%m%d'),
                                     ])

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
        str_inputs[variable] = '_'.join(list_attr)

    #=========================================#
    ###         Model                       ###
    #=========================================#
    str_learning = '_'.join([hprm['learning.method'],
                             'coupled_models' if not hprm['learning.independent_models'] else '',
                             'individual_designs' if hprm['learning.individual_designs'] else '',
                             ])    
    import ipdb; ipdb.set_trace()
    
    
    #=========================================#
    ###       approx tf                     ###
    #=========================================#    
    if hprm['learning.method'] in {'approx_tf'}:
        
        ##########
        # Design
        ##########
        data_train = join_strings([stations, ind_train]) 
        data_test  = join_strings([stations, ind_test ])
        data_both  = join_strings([stations, ind_train, ind_test]) 
        bnd_scal   = format_string('bds'*hprm.get('tf_boundary_scaling', 0))
        nat_spli   = format_string('nat'*hprm.get('natural_splines'    , 0))
        hrch       = format_string(('hrch_'+join_strings([key + '_' + format_string(str(value)) for key, value in hprm.get('tf_hrch',{}).items()]))*(len(hprm.get('tf_hrch', {}))>0)) 
        order_spli = format_string(('ord' +str(hprm.get('order_splines', 1)))*(hprm.get('order_splines', 1)!=1))
        
        ##########
        # Inputs of univariate functions
        ##########
        inst_variables          = [k 
                                   for k in hprm['selected_variables'] 
                                   if 'lag' not in k
                                   ]
        str_inst_variables      = format_string('_'.join(inst_variables))
        #### Lags
        lag_variables           = [k 
                                   for k in hprm['selected_variables'] 
                                   if ('lag' in k or 'dif' in k) and hprm['lag'][k]!=()
                                   ]
        str_lag                 = {}
        for k in lag_variables:
            L = []
            for i in range(len(hprm['lag'][k])):
                if len(hprm['lag'][k][i]) == 1:
                    L += [str(hprm['lag'][k][i][0])]
                else :
                    L += [str(np.min(hprm['lag'][k][i])) + '__' + str(np.max(hprm['lag'][k][i])) + ('_' + str(len(hprm['lag'][k][i])))*bool(len(hprm['lag'][k][i]) != np.max(hprm['lag'][k][i]) - np.min(hprm['lag'][k][i]) +1)]
            str_lag[k] = '_'.join(L)
        ### Smoothed
        smo_variables           = [k for k in hprm['selected_variables'] if ('smo' in k) and hprm['smo'][k]!=()]
        str_smo                 = {}
        for k in smo_variables:
            L = []
            for i in range(len(hprm['smo'][k])):
                L += [str(hprm['smo'][k][i])]
            str_smo[k] = '_'.join(L)    
        
        list_univariate = sorted(set([e for coef, list_keys in hprm['tf_config_coef'].items() for e in list_keys if '#' not in e]))
        list_bivariate  = sorted(set([e for coef, list_keys in hprm['tf_config_coef'].items() for e in list_keys if '#'     in e]))
                                                                            
        dikt_var_1 = {k: list(filter(lambda x : bool(x), 
                                     [ k if k not in hprm.get('lag',{}) and k not in hprm.get('smo',{}) else '',
                                      (k + join_strings([str(hprm['lag'].get(k)[i][0]) for i in range(len(hprm['lag'].get(k)))])) if k in hprm.get('lag',{}) else '',
                                      (k + join_strings([str(hprm['smo'].get(k)[i]   ) for i in range(len(hprm['smo'].get(k)))])) if k in hprm.get('smo',{}) else '',
                                      str(hprm['tf_masks'].get(k, '')), 
                                      str(hprm['approx_nb_itv'][k])
                                      ]))
                      for k in list_univariate
                      }                                                                           
        lst_var_1 = [join_strings([*dikt_var_1[k], 
                                   'orth'*hprm.get('tf_orthogonalize', 
                                                    {}, 
                                                    ).get(k, 
                                                          False, 
                                                          ), 
                                   ]) 
                     for k in dikt_var_1
                     ]
        var_1     = format_string(join_strings(lst_var_1)) if lst_var_1 else ''
        
        dikt_var_2 = {(e,f): list(filter(lambda x : bool(x), 
                                     [a+'#'+b
                                      for a in ['_'.join(list(filter(lambda x : bool(x), 
                                                           [ e if e not in hprm.get('lag',{}) and e not in hprm.get('smo',{}) else '', 
                                                            (e + join_strings([str(hprm['lag'].get(e)[i][0]) for i in range(len(hprm['lag'].get(e)))])) if e in hprm.get('lag',{}) else '', 
                                                            (e + join_strings([str(hprm['smo'].get(e)[i]   ) for i in range(len(hprm['smo'].get(e)))])) if e in hprm.get('smo',{}) else '', 
                                                            str(hprm['tf_masks'].get(e, '')),
                                                            str(hprm['approx_nb_itv'][e+'#'+f][0]),
                                                            ])))]
                                      for b in ['_'.join(list(filter(lambda x : bool(x), 
                                                           [ f if f not in hprm.get('lag',{}) and f not in hprm.get('smo',{}) else '', 
                                                            (f + join_strings([str(hprm['lag'].get(f)[i][0]) for i in range(len(hprm['lag'].get(f)))])) if f in hprm.get('lag',{}) else '', 
                                                            (f + join_strings([str(hprm['smo'].get(f)[i]   ) for i in range(len(hprm['smo'].get(f)))])) if f in hprm.get('smo',{}) else '', 
                                                            str(hprm['tf_masks'].get(f, '')),
                                                            str(hprm['approx_nb_itv'][e+'#'+f][1]),
                                                            ])))]
                                      ])) 
                     for e,f in list(map(lambda x : x.split('#'), list_bivariate))
                     }                                                                           
        lst_var_2 = [join_strings([*dikt_var_2[k], 
                                   'orth'*hprm.get('tf_orthogonalize', 
                                                    {}, 
                                                    ).get(tuple(sorted(k)), 
                                                          False, 
                                                          ), 
                                   ]) 
                     for k in dikt_var_2
                     ]
        var_2     = format_string(join_strings(sorted(lst_var_2))) if lst_var_2 else ''
        
        ##########
        # Inputs both univariate and bivariate functions
        ##########  
        var_12       = '_'.join([var_1, 'XX' if var_2 else '', var_2])
        all_products = 'apr'*(hprm.get('gp_pen',0) > 0 and hprm.get('gp_matrix','') != '') # When apr, x1tx1 containes all interactions  
        
        w_inter    = bool(list_bivariate)
        minprod    = format_string(  'prod'*(1-hprm['tf_prod0_min1']) \
                                   + 'min' *(  hprm['tf_prod0_min1']) 
                                   ) if w_inter else ''
        ##########
        # Primitives
        ##########
        str_tmp                        = format_string(minprod)
        str_inter                      = format_string(str_tmp if (str_tmp and w_inter) else '')
        del str_tmp

        str_data                       = format_string(join_strings([db, zone, method, separated, posts, 
                                                                     data_train, weight_comp, pre_t, separated2, 
                                                                     separated3, nat_spli, order_spli, bnd_scal, 
                                                                     hrch, all_products, 
                                                                     ]) 
                                                       + '/')

        dikt['primitives_train1']      = join_strings([str_data, var_1]) 
        dikt['primitives_test1']       = join_strings([str_data, var_1, ind_test])
        dikt['primitives_train2']      = join_strings([str_data, str_inter, var_2]) 
        dikt['primitives_test2']       = join_strings([str_data, str_inter, var_2, ind_test])
        dikt['primitives_train12']     = join_strings([str_data, str_inter, var_12]) 
        dikt['primitives_test12']      = join_strings([str_data, str_inter, var_12, ind_test])
        
        
        print("dikt['primitives_test1']", '\n', 
              *[' '*10 + dikt['primitives_test1'].split('/')[ii] + '\n' for ii in range(len(dikt['primitives_test1'].split('/')))],
              '\n',
              )
        print("dikt['primitives_test2']", '\n', 
              *[' '*10 + dikt['primitives_test2'].split('/')[ii] + '\n' for ii in range(len(dikt['primitives_test2'].split('/')))],
              '\n',
              )
        print("dikt['primitives_test12']", '\n', 
              *[' '*10 + dikt['primitives_test12'].split('/')[ii] + '\n' for ii in range(len(dikt['primitives_test12'].split('/')))],
              '\n',
              )

        ##########
        # Regression variables        
        ##########
        str_conf  = []
        for var in sorted(hprm['tf_config_coef']):
            new_list = [var]
            if var == 'A':
                if hprm['tf_pen'].get(var,'')!= '' and hprm['tf_alpha'].get(var,0) > 0:
                    new_list+=[
                               str(hprm['tf_A_max']),
                               *(hprm['tf_pen'][var], str(hprm['tf_alpha'][var])), 
                               ]
            else:
                list_cov = hprm['tf_config_coef'][var]
                if var == 'Cbm': 
                    var = 'Cm'
                for cov in list_cov:
                    cond = (     hprm['tf_pen'][var].get(cov,'') != '' 
                            and (   type(hprm['tf_alpha'][var].get(cov,0)) == tuple 
                                 or hprm['tf_alpha'][var].get(cov,0) > 0
                                 )
                            )
                    new_list += [
                                 cov, 
                                 *((hprm['tf_pen'][var][cov][:3], str(hprm['tf_alpha'][var][cov])) if cond else (str(0),)),
                                 ]
            str_conf += new_list + ['/']
        str_conf = format_string(join_strings(str_conf)) 

        
        ##########
        # Additional hprmeters
        ##########
        rkB        = format_string('rkB'  + '_'.join([k+'_'+str(v) 
                                                      for k, v in hprm['tf_rk_B'].items() 
                                                      if k in hprm['tf_config_coef'].get('Blr',{})
                                                      ])) if 'Blr' in hprm['tf_config_coef'] else ''        
        rkUV       = format_string('rkC'  + str(hprm['tf_rk_UV'])) if 'Cuv'  in hprm['tf_config_coef'] else ''  
        trunc_svd  = format_string('tsvd' + str(hprm['trunc_svd'])) if ('trunc_svd' in hprm and hprm['trunc_svd'] > 0) else ''
        
        ##########
        # Algorithm
        ##########
        no_warmstart = format_string('nws'*(1-hprm['tf_try_ws']))
        early_stop     = format_string('noes'*(1-hprm['tf_early_stop_test']))
        early_stop_ind = format_string('noesi'*(1-hprm['tf_early_stop_ind_test']))
        iniB         = bool({'Blr', 'Bsp'}        - set(hprm['tf_config_coef'])) and hprm.get('tf_init_B', 0)
        iniC         = bool({'Csp', 'Cuv', 'Cbm'} - set(hprm['tf_config_coef'])) and hprm.get('tf_init_C', 0)
        init         =   'init_'                                         \
                       + ('B' + '_fr'*hprm.get('freeze_B',    ''))*iniB \
                       + ('C' + hprm.get('tf_method_init_UV', ''))*iniC \
                       if (iniB + iniC >= 1) \
                       else ''
        sp_coef1     = format_string((hprm.get('tf_ch_basis1', 0)!=0)*('spb'  + str(hprm.get('tf_ch_basis1', 0)))) 
        sp_coef2     = format_string((hprm.get('tf_ch_basis2', 0)!=0)*('spuv' + str(hprm.get('tf_ch_basis2', 0))))
        gp_reg       = format_string(  'gp' 
                                     + '_' 
                                     + str(hprm.get('gp_matrix')) 
                                     + '_' 
                                     + str(hprm.get('gp_pen'))
                                     ) * (hprm.get('gp_pen', 0)!=0 and hprm.get('gp_matrix','') != '')

        # Stopping criteria
        if np.sum(['bfgs' in key for key, value in hprm['tf_config_coef'].items() if value]):
            stop = format_string(join_strings(['stop', 
                                               str(hprm['tf_tol_lbfgs']),
                                               str(hprm['tf_pgtol_lbfgs']),
                                               '{:.0e}'.format(hprm['tf_maxfun_lbfgs']).replace('s+', ''),
                                               '{:.0e}'.format(hprm['tf_maxiter_lbfgs']).replace('+', ''),
                                               ]))
        else:
            stop = format_string(join_strings(['stop', 
                                               '{:.0e}'.format(hprm['tf_max_iter']).replace('+', ''),
                                               early_stop,
                                               early_stop_ind, 
                                               str(hprm['tf_tol']), 
                                               ]))
        
        if w_inter:
            var_model = var_12
        else:                      
            var_model = var_1
        dikt['model_wo_hyperp']       =         join_strings([str_data, 
                                                              var_model,    
                                                              str_inter*w_inter,
                                                              ])
        dikt['model_pred']            =         dikt['model_wo_hyperp'] + '/' + str_conf     \
                                              + join_strings([no_warmstart, init,         rkB,       rkUV,      
                                                              trunc_svd,    sp_coef1,     sp_coef2,      
                                                              gp_reg,       stop,
                                                              ])
        
            
    #=========================================#
    #         Final model                     #
    #=========================================#
        
    dikt['exp'] = os.path.join(
                               str_dataset,
                               str_split_training,
                               str_split_validation,
                               str_learning,
                               *list(str_inputs.values()),
                               )
    
    for key, ss in dikt.items():
        too_long = np.sum([len(ee) > 250 for ee in ss.split('/')])
        if too_long:
            list_chunks = ss.split('/')
            new_list_chunks = []
            for ee in list_chunks:
                for ii in range(len(ee)//250+1):
                    new_list_chunks.append(ee[250*ii:250*(ii+1)])
            dikt[key] = '/'.join(new_list_chunks)
        dikt[key] = tools.format_file_names(dikt[key])
        
    print(dikt['exp'].replace('/', '/\n\t'))    
        
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


