
"""
This module is used to compute from the input data 
all the covariates based on spline features.
"""


import numpy  as np
import scipy  as sp
import pandas as pd
import os
import sys
from subprocess import CalledProcessError, TimeoutExpired
from termcolor import colored
#
import electricityLoadForecasting.tools  as tools
import electricityLoadForecasting.paths  as paths
from . import masks


path_data =  os.path.join(paths.outputs,
                          'Data',
                          ) 
save_data         = True
big_size_to_print = 1e8


#@profile
def compute_design(inputs, 
                   hprm, 
                   #path, replace with os.path.join(paths.outputs, 'Data')
                   dikt_file_names, 
                   mask_univariate       = None,
                   mask_bivariate        = None,
                   size1                 = None, 
                   size2                 = None, 
                   dikt_func             = None,       
                   dikt_func12           = None,  
                   db                    = None,
                   size_tensor_bivariate = None, 
                   ):
    # This is the main script where the functions for the univariate and the bivariate covariates are called
    # From the number of nodes defined in the parameters, 
    # We compute sequentially the nodes, the corresponding splines functions, 
    # and the transformation of the data by these functions.
    if   db == 'training':
        assert dikt_func   == None
        assert dikt_func12 == None
    elif db == 'validation':
        assert bool(dikt_func)
    else:
        raise ValueError('Incorret db')
    precompute = not (hprm['afm.algorithm'] == 'L-BFGS')        
    ### Univariate part
    if hprm['afm.features.sparse_x1']: # Format to store the univariate covariates
        print(colored('X1_SPARSE', 'green'))
    prefix_features_univariate = dikt_file_names['features.{}.univariate'.format(db)] # string used to save/load the covariates
    try:
        # We try to load from the local machine or from a distant machine (in the module primitives)
        if db == 'training':
            dikt_nodes = tools.batch_load(path_data, prefix_features_univariate, data_name = 'dikt_nodes',     data_type = 'np')
            dikt_func  = tools.batch_load(path_data, prefix_features_univariate, data_name = 'dikt_func',      data_type = 'np')
            size1      = tools.batch_load(path_data, prefix_features_univariate, data_name = 'size_univariate',data_type = 'np')
        X1 = tools.batch_load(path_data,
                              prefix_features_univariate,
                              data_name = 'X1_'+db,        
                              data_type = ('dict_sp'
                                           if hprm['afm.features.sparse_x1']
                                           else
                                           'dict_np'
                                           ),
                              )
    except Exception as e:
        print(e)
        print('Start X1 {0}'.format(db))
        if db == 'training':
            dikt_nodes   = make_dikt_nodes(inputs,
                                           hprm,
                                           hprm['afm.formula'].loc[hprm['afm.formula']['nb_intervals'].apply(lambda x : type(x) != tuple)],
                                           ) # Compute the nodes from the number of nodes for each covariate
            dikt_func    = make_dikt_func(dikt_nodes,
                                          hprm,
                                          ) # Compute the associated functions        
        X1, size1 = make_X1(inputs,
                            dikt_func,
                            hprm,
                            sparse_x1 = hprm['afm.features.sparse_x1'],
                            ) # Compute the covariates and the size of the corresponding vectors
        for key in X1.keys():
            if key in mask_univariate:
                mm = mask_univariate[key]
                if type(mm) == list and len(mm) == 0: 
                    # If no substation should have access to this covariate, delete it
                    # This can happen when only a region is considered : the weather stations in other regions will not be considered
                    del X1[key], size1[key]
                elif type(mm) == list and len(mm) > 0:
                    pass
                elif type(mm) == slice and mm == slice(None):
                    pass
                else:
                    raise ValueError
        print('finished X1')
        ### Save 
        list_files = [
                      (X1, 'X1_'+db , prefix_features_univariate, ('dict_sp'
                                                                   if hprm['afm.features.sparse_x1']
                                                                   else
                                                                   'dict_np'
                                                                   )),
                      ]
        if db == 'training':
            list_files += [
                           (dikt_nodes, 'dikt_nodes',      prefix_features_univariate, 'np'), 
                           (dikt_func,  'dikt_func',       prefix_features_univariate, 'np'), 
                           (size1,      'size_univariate', prefix_features_univariate, 'np'), 
                           ]
        save_list_files(list_files, path_data, hprm)
        ### FINISHED X1
    X = {'X1_{0}'.format(db) : X1}
    if db == 'training':
        X.update({
                  'dikt_nodes'       : dikt_nodes, 
                  'dikt_func'        : dikt_func, 
                  'mask_univariate'  : mask_univariate, 
                  'size_univariate'  : size1, 
                  })
    if precompute:
        # Compute XtX
        # l-bfgs and mfilter are two possible algorithms that do not require the precomputations of XtX
        # since it is done later for l-bfgs and it is not done for mfilter.
        try:
            X1tX1 = tools.batch_load(path_data,
                                     prefix = prefix_features_univariate,
                                     data_name = 'X1tX1_'+db,
                                     data_type = ('dict_sp' 
                                                  if hprm.get('afm.features.sparse_x1')
                                                  else
                                                  'dict_np'
                                                  ),
                                     )
        except Exception as e:
            print(e)
            print('Start XtX1 '+db)
            X1tX1 = precomputations_1(X1, mask_univariate, all_products = (hprm.get('gp_pen',0) > 0 and hprm['gp_matrix'] != ''))
            for key, value in {**X1tX1}.items():
                assert type(value) in {np.ndarray, sp.sparse.csr_matrix}
            if len (X1tX1) < 1e4:
                list_files = [(X1tX1, 
                               'X1tX1_'+db,
                               prefix_features_univariate,
                               ('dict_sp'
                                if hprm.get('afm.features.sparse_x1')
                                else
                                'dict_np'),
                               )]
                save_list_files(list_files, path_data, hprm)
                print('finished XtX1')
            else:
                print('X1tX1 too large to save : len (X1tX1) = {0}'.format(len(X1tX1)))
        for keys in list(X1tX1.keys()):
            assert type(X1tX1[keys]) in {np.ndarray, sp.sparse.csr_matrix}
            key1, key2 = keys
            if (key1 != key2) and (key2, key1) not in X1tX1:
                X1tX1.update({(key2, key1) : (sp.sparse.csr_matrix(X1tX1[keys].T)
                                              if type(X1tX1[keys]) == sp.sparse.csr_matrix
                                              else
                                              X1tX1[keys].T
                                              )})
        X.update({'X1tX1_{0}'.format(db) : X1tX1})
    ### Bivariate part
    if not hprm['afm.formula'].loc[hprm['afm.formula']['nb_intervals'].apply(lambda x : (type(x) == tuple and len(x) == 2))].empty: # ie if there are interactions
        prefix_features_bivariate = dikt_file_names['features.{}.bivariate'.format(db)] # string used to save/load the covariates
        prefix_features_all       = dikt_file_names['features.{}.all'.format(db)] # string used to save/load the covariates
        if hprm['afm.features.sparse_x2']:
            print(colored('X2_SPARSE', 'green'))
        try:
            # Try to load the covariates corresponding to the interactions
            # Important because they may demand many more computations
            if db == 'training':
                dikt_nodes12          = tools.batch_load(path_data, prefix_features_bivariate, data_name = 'dikt_nodes12',          data_type = 'np')
                dikt_func12           = tools.batch_load(path_data, prefix_features_bivariate, data_name = 'dikt_func12',           data_type = 'np')
                size2                 = tools.batch_load(path_data, prefix_features_bivariate, data_name = 'size_bivariate',        data_type = 'np')
                size_tensor_bivariate = tools.batch_load(path_data, prefix_features_bivariate, data_name = 'size_tensor_bivariate', data_type = 'np')
            X2 = tools.batch_load(path_data,
                                  prefix    = prefix_features_bivariate,
                                  data_name = 'X2_' + db,
                                  data_type = ('dict_sp'
                                               if hprm['afm.features.sparse_x2']
                                               else
                                               'dict_np'
                                               ),
                                  )
        except Exception as e:
            print(e)
            print('Start X2 {0}'.format(db))
            if db == 'training':
                # Compute the nodes for the univariate functions used to compute the interactions
                # They may be different from the nodes used for the univariate covariates
                dikt_nodes12  = make_dikt_nodes(inputs,
                                                hprm,
                                                pd.DataFrame([(coef, inpt1, nb_itv1, *reg)
                                                              for (coef, inpts), (nb_itvs, *reg) in hprm['afm.formula'].loc[hprm['afm.formula']['nb_intervals'].apply(lambda x : type(x) == tuple)].iterrows()
                                                              for inpt1, nb_itv1 in zip(inpts, nb_itvs)
                                                              ],
                                                             columns = hprm['afm.formula'].reset_index().columns,
                                                             ).set_index(hprm['afm.formula'].index.names),
                                                )
                dikt_func12   = make_dikt_func(dikt_nodes12,
                                               hprm,
                                               )
            else:
                assert dikt_func12
            print('start X12')
            # Compute the transformations of the data with univariate functions later used to compute the interactions
            X12, size12 = make_X1(inputs, 
                                  dikt_func12, 
                                  hprm, 
                                  sparse_x1 = False, # More convenient to have non sparse matrices to compute products
                                  )
            print('finished X12')
            print('start X2')
            # Compute the product for the interactions
            X2, size2, size_tensor_bivariate = make_X2(X12,
                                                       hprm,
                                                       )
            for h,k in X2.keys():
                if (h,k) in mask_bivariate:
                    mm = mask_bivariate.get((h,k),
                                            mask_bivariate.get(k,h),
                                            )
                    if type(mm) == list and len(mm) == 0:
                        # Discard interactions that are used by zero substations
                        del X2[k], size2[k], size_tensor_bivariate[k]
            print('finished X2') 
            list_files = [(X2, 'X2_'+db, prefix_features_bivariate, 'dict_sp' if hprm['afm.features.sparse_x2'] else 'dict_np'), 
                          ] if len (X2) < 1e4 else []
            if db == 'training':
                list_files += [ 
                               (dikt_nodes12,          'dikt_nodes12',          prefix_features_bivariate, 'np'), 
                               (dikt_func12,           'dikt_func12',           prefix_features_bivariate, 'np'), 
                               (size2,                 'size_bivariate',        prefix_features_bivariate, 'np'),
                               (size_tensor_bivariate, 'size_tensor_bivariate', prefix_features_bivariate, 'np'),
                               ]
            save_list_files(list_files, path_data, hprm)
            
        X.update({'X2_'+db : X2})
        if db == 'training':
            X.update({
                      'mask_bivariate'        : mask_bivariate,
                      'size_bivariate'        : size2,
                      'size_tensor_bivariate' : size_tensor_bivariate,
                      })
        if precompute:
            # Compute the empirical covariance between the univariate/bivariate covariates
            try:
                X1tX2 = tools.batch_load(path_data,
                                         prefix    = prefix_features_all,
                                         data_name = 'X1tX2_'+db,
                                         data_type = ('dict_sp'
                                                      if hprm.get('afm.features.sparse_x1') and hprm['afm.features.sparse_x2']
                                                      else
                                                      'dict_np'
                                                      ),
                                         )
                X2tX2 = tools.batch_load(path_data,
                                         prefix    = prefix_features_bivariate,
                                         data_name = 'X2tX2_'+db,
                                         data_type = ('dict_sp'
                                                      if hprm['afm.features.sparse_x2']
                                                      else 'dict_np'
                                                      ),
                                         )
            except Exception:
                X1tX2, X2tX2 = precomputations_2(X1, X2, mask_univariate, mask_bivariate, hprm, all_products = (hprm.get('gp_pen',0) > 0 and hprm['gp_matrix'] != ''))
                for key, value in {**X1tX2}.items():
                    assert type(value) in {np.ndarray, sp.sparse.csr_matrix}
                for key, value in {**X2tX2}.items():
                    assert type(value) in {np.ndarray, sp.sparse.csr_matrix}
                list_files = []
                if len (X1tX2) < 1e4:
                    list_files += [(X1tX2, 
                                    'X1tX2_'+db, 
                                    prefix_features_all, 
                                    ('dict_sp'
                                     if hprm.get('afm.features.sparse_x1') and hprm['afm.features.sparse_x2']
                                     else
                                     'dict_np'
                                     ),
                                    )]
                if len (X1tX2) < 1e4:    
                    list_files += [(X2tX2, 
                                    'X2tX2_'+db, 
                                    prefix_features_bivariate, 
                                    'dict_sp' if hprm['afm.features.sparse_x2'] else 'dict_np',
                                    )]
                save_list_files(list_files, path_data, hprm) 
                print('finished XtX2')
        if precompute:
            X2tX1 = {}
            for keys in list(X1tX2.keys()):
                key1, key2  = keys
                X2tX1.update({(key2, key1) : sp.sparse.csr_matrix(X1tX2[keys].T) 
                                             if type(X1tX2[keys]) == sp.sparse.csr_matrix 
                                             else 
                                             X1tX2[keys].T
                                             })
            for keys in list(X2tX2.keys()):
                assert type(X2tX2[keys]) == sp.sparse.csr_matrix
                key1, key2 = keys
                if key1!=key2:
                    X2tX2.update({(key2,key1) : X2tX2[keys].T})
                    if type(X2tX2[keys]) == sp.sparse.csr_matrix:
                        X2tX2[key2,key1] = sp.sparse.csr_matrix(X2tX2[key2,key1])
            for keys in list(X2tX2.keys()):
                assert type(X2tX2[keys]) in {np.ndarray, sp.sparse.csr_matrix}
            X.update({
                      'X1tX2_'+db : X1tX2, 
                      'X2tX1_'+db : X2tX1, 
                      'X2tX2_'+db : X2tX2,
                      })
        if db == 'training':
            X.update({
                      'dikt_nodes12'  : dikt_nodes12, 
                      'dikt_func12'   : dikt_func12,
                      })
    print('start gso')
    # Print the size of the covariates when they are large
    # to look after the memory usage
    for k in X:
        if type(X[k]) == dict:
            for j in X[k]:
                if sys.getsizeof(X[k][j]) > big_size_to_print:
                    print('    ', j, ' : ', '{0:.3f}'.format(sys.getsizeof(X[k][j])/10**9), 'Gb')
                assert type(X[k][j]) != dict
    print('end gso')
    return X


def save_list_files(list_files, path_data, hprm):
    # Intermediary function to format the data and save it with the module primitives
    print('start saving')
    for inputs, data_name, prefix, opt in list_files:
        print('\r    ', data_name, ' '*20, end = '')
        try:
            assert save_data
            tools.batch_save(path_data,
                             prefix    = prefix,
                             data      = inputs,
                             data_name = data_name,
                             data_type = opt,
                             )
            print('end saving')
        except (CalledProcessError, ValueError, TimeoutExpired, OSError, AssertionError) as e:
            print('failed') 
            print(e)


def make_dikt_func(d_nodes, hprm):
    # Compute for each covariate the spline functions from the set of nodes
    d_func = {}
    for inpt, trsfm, prm, location in d_nodes:
        cyclic      = hprm['inputs.cyclic'].get(inpt, False)
        d_func[(inpt, trsfm, prm, location)] = tuple([f 
                                                      for e in d_nodes[(inpt, trsfm, prm, location)]
                                                      for f in make_func(e, cyclic, hprm['afm.features.order_splines'])
                                                      ])
    return d_func


def make_dikt_nodes(inputs, hprm, formula):
    dikt_nodes = {}
    # For each covariate, compute the nodes from the number of nodes chosen on the interval [0, 1]
    for (coef, (inpt, trsfm, prm)), (nb_itv, *_) in formula.iterrows():
        for location in inputs.xs((inpt, trsfm, prm), axis = 1).columns:
            if bool(hprm['inputs.cyclic'].get(inpt, False)):
                min_value = 0
                max_value = hprm['inputs.cyclic'][inpt]
            else:
                min_value = inputs.loc[:,(inpt, trsfm, prm, location)].min()
                max_value = inputs.loc[:,(inpt, trsfm, prm, location)].max()
            dikt_nodes[(inpt, trsfm, prm, location)] = [make_nodes(e,
                                                                   hprm['inputs.cyclic'].get(inpt, False),
                                                                   hprm['afm.features.order_splines'],
                                                                   min_value = min_value,
                                                                   max_value = max_value, 
                                                                   ) 
                                                        for e in (nb_itv 
                                                                  if type(nb_itv) == tuple 
                                                                  else 
                                                                  (nb_itv,)
                                                                  )
                                                        ]
    return dikt_nodes


def make_func(nodes, cyclic, order_splines):
    funcs = []
    # Compute the functions (eg triplets of nodes for degree 1-splines)
    # from the list of nodes (eg [0, 0.25, 0.5, 0.75, 1] if a granularity of 0.25 is chosen)
    # for each covariate 
    if type(nodes) == tuple:
        funcs += [e for e in nodes]
    elif type(nodes) == np.ndarray:
        assert nodes.ndim == 1
        for i in range(nodes.shape[0] - (1+order_splines)*(1-bool(cyclic))):
            if cyclic:
                tt = tuple([nodes[(i+j) % len(nodes)] 
                            for j in range((2+order_splines))
                            ])
            else:
                tt = tuple([nodes[(i+j)] 
                            for j in range((2+order_splines))
                            ])
            funcs.append(tt)
    else:
        raise ValueError
    return tuple(funcs)


def make_nodes(nb_itv,
               cyclic,
               order_splines,
               min_value = None,
               max_value = None,
               ):
    # Given the number of nodes, the indicator of a cyclic variable and the degree of the wanted splines
    # Compute the associated nodes
    ######
    if type(nb_itv) == str and nb_itv[0] == 'p': 
        # Identity, Indicator or polynomials
        assert len(nb_itv) == 2
        nodes = tuple([tuple([e
                              for e in range(1, int(nb_itv[1])+1)
                              ])
                       ])
    elif type(nb_itv) == int: 
        def affine_trsfm(x):
            return min_value + x*(max_value - min_value)
        # Normal
        assert nb_itv != 0
        if cyclic:
            nodes = np.array([affine_trsfm(i/nb_itv)
                              for i in range(nb_itv)
                              ])  
        else:
            nodes = np.array([affine_trsfm(i/nb_itv)
                              for i in range(- order_splines,
                                             nb_itv + 1 + order_splines,
                                             )
                              ])
    else:
        raise ValueError
    return nodes


def make_X1(inputs,
            d_func,
            hprm,
            sparse_x1 = False,
            ):
    X1 = {((inpt, trsfm, prm), (location,)):make_cov(d_func[(inpt, trsfm, prm, location)], 
                                                     inputs[[(inpt, trsfm, prm, location)]], 
                                                     sparse_x1,
                                                     hprm,
                                                     inpt,
                                                     )
          for inpt, trsfm, prm, location in inputs.columns
          if (inpt, trsfm, prm, location) in d_func 
          }
    # Compute the covariates from the list of functions and the input inputs
    size1 = {k : v.shape[1]
             for k, v in X1.items()
             }
    return X1, size1


def make_cov(list_funcs,
             inpt_data,
             sparse_x1,
             hprm,
             inpt_name,
             ):
    # Compute for one category (eg the hour, the temperatures or the delayed temperatures)
    # the covariates with the associated list of functions
    cov = np.concatenate([func1d(inpt_data, 
                                 func, 
                                 index_func = ii,
                                 nb_funcs   = len(list_funcs),
                                 cyclic     = hprm['inputs.cyclic'].get(inpt_name, False), 
                                 )
                          for ii, func in enumerate(list_funcs)
                          ], 
                         axis = 1,
                         )
    if sparse_x1:
        cov = sp.sparse.csc_matrix(cov)
    return cov


def make_X2(X12,
            hprm,
            ):
    X2      = {}
    size2   = {} 
    size_tensor_bivariate = {}
    formula_biv  = hprm['afm.formula'].loc[hprm['afm.formula']['nb_intervals'].apply(lambda x : type(x) == tuple)]
    for (inpt1, (location1,)), v1 in X12.items():
        for (inpt2, (location2,)), v2 in X12.items():
            if (inpt1, inpt2) in formula_biv.index.get_level_values('input'):
                #Compute interactions
                if hprm['afm.features.bivariate.combine_function'] == np.min:
                    n, p = v1.shape
                    _, q = v2.shape
                    xleft  = np.repeat(v1[:,:,None], q, axis = 2)
                    xright = np.repeat(v2[:,None,:], p, axis = 1)
                    interaction = np.min([xleft, xright], axis = 0)
                else:
                    interaction = np.einsum('np,nq->npq', 
                                            v1, 
                                            v2,
                                            )
                siz2             = interaction.shape[1:]
                assert len(siz2) == 2
                size_tensor_bivariate[(inpt1, inpt2),(location1, location2)] = siz2
                size2                [(inpt1, inpt2),(location1, location2)] = np.prod(siz2)
                # If the first covariate is associated to p functions and the 
                # second to q functions and there are n observations
                # Reshape the interaction of size (n,p,q) into a matrix of size (n, p*q)
                # It is indeed important to use matrix instead of tensors to 
                # have faster computations, in particular for sparse matrices
                X2[(inpt1, inpt2),(location1, location2)] = interaction.reshape((-1, siz2[0]*siz2[1]))
                if hprm['afm.features.sparse_x2']:
                    X2[(inpt1, inpt2),(location1, location2)] = sp.sparse.csc_matrix(X2[(inpt1, inpt2),(location1, location2)])
                if (    ((inpt2, inpt1),(location2, location1)) not in X12 
                    and ((*inpt2, *inpt1) in formula_biv.index.get_level_values('input')
                         or 'Cbm' in hprm['afm.formula'].index.get_level_values('coefficient')
                         )
                    ):
                    X2[(inpt2, inpt1),(location2, location1)] = interaction.transpose(0,2,1).reshape((-1, siz2[0]*siz2[1]))
                    size_tensor_bivariate[(inpt2, inpt1),(location2, location1)] = siz2[::-1]
                    size2                [(inpt2, inpt1),(location2, location1)] = np.prod(siz2[::-1])
                    if hprm['afm.features.sparse_x2']:
                        X2[(inpt2, inpt1),(location2, location1)] = sp.sparse.csc_matrix(X2[(inpt2, inpt1),(location2, location1)])
    print()
    return X2, size2, size_tensor_bivariate
    

def precomputations_1(X1, mask_univariate, all_products = 0):
    # Compute the products between univariate covariates
    print('Start X1tX1')
    X1tX1 = {}
    for ii, key1 in enumerate(X1):
        for jj, key2 in enumerate(X1):
            print('\r{0:{wid}} / {1} - {2:{wid}} / {3} - '.format(jj,
                                                                  len(X1),
                                                                  ii,
                                                                  len(X1),
                                                                  wid = len(str(len(X1))),
                                                                  ),
                  end = '',
                  )
            if str(key1) <= str(key2):
                if masks.cross_mask(mask_univariate.get(key1), mask_univariate.get(key2)) or all_products:
                    X1tX1[key1,key2] = X1[key1].T @ X1[key2]
                    if type(X1tX1[key1,key2]) in {sp.sparse.csc_matrix, sp.sparse.csr_matrix}:
                        X1tX1[key1,key2] = sp.sparse.csr_matrix(X1tX1[key1,key2])
                    assert type(X1tX1[key1,key2]) in {np.ndarray, sp.sparse.csr_matrix}
    print()
    return X1tX1


def precomputations_2(X1, X2, mask_univariate, mask_bivariate, hprm, all_products = 0):
    # Compute the products between univariate/bivariate covariates
    # We only have to compute the product between pairs of covariates that are accessed a same sustation if the models are independent
    # but have to compute all the product for the sum consistent loss
    if all_products:
        # all_products is an indicator that all products should be compuetd
        assert hprm['afm.features.sparse_x2']
        print(colored('\n \n X2 will be very large with all interactions ie might be too big \n \n', 'red', 'on_cyan'))
    print('Start X1tX2')
    X1tX2 = {}
    # Products between univariate covariates and bivariate covariates
    for ii, key1 in enumerate(X1):
        for jj, key2 in enumerate(X2):
            print('\r{0:{wid2}} / {1} - {2:{wid1}} / {3} - '.format(jj,
                                                                    len(X2),
                                                                    ii,
                                                                    len(X1),
                                                                    wid1 = len(str(len(X1))),
                                                                    wid2 = len(str(len(X2))),
                                                                    ),
                  end = '',
                  )
            if masks.cross_mask(mask_univariate.get(key1), mask_bivariate.get(key2)) or all_products:
                X1tX2[key1,key2] = X1[key1].T @ X2[key2]
                if type(X1tX2[key1,key2]) in {sp.sparse.csc_matrix, sp.sparse.csr_matrix}:
                    X1tX2[key1,key2] = sp.sparse.csr_matrix(X1tX2[key1,key2])
                assert type(X1tX2[key1,key2]) in {np.ndarray, sp.sparse.csr_matrix}
    print()
    print('Start X2tX2')
    X2tX2 = {}
    # Products between bivariate covariates
    for ii, key1 in enumerate(X2):
        for jj, key2 in enumerate(X2):
            print('\r{0:{wid}} / {1} - {2:{wid}} / {3} - '.format(jj,
                                                                  len(X2),
                                                                  ii,
                                                                  len(X2),
                                                                  wid = len(str(len(X2))),
                                                                  ),
                  end = '',
                  )
            if str(key1) <= str(key2):
                if masks.cross_mask(mask_bivariate.get(key1), mask_bivariate.get(key2)) or all_products:
                    X2tX2[key1,key2] = X2[key1].T @ X2[key2]
                    if type(X2tX2[key1,key2]) in {sp.sparse.csc_matrix, sp.sparse.csr_matrix}:
                        X2tX2[key1,key2] = sp.sparse.csr_matrix(X2tX2[key1,key2])
                    assert type(X2tX2[key1,key2]) in {np.ndarray, sp.sparse.csr_matrix}
    print()
    return X1tX2, X2tX2


def func1d(x,
           func,
           index_func = None,
           nb_funcs   = None,
           cyclic     = None,
           ):
    # Computet for the different values stored in x the values of the spline function (defined by a set of nodes) stored in func
    if len(func) == 1:
        assert func[0] == int(func[0])
        return x**func[0]
    elif len(func) == 3:
        aa, bb, dd = func
        assert (aa <= bb <= dd) or cyclic
        #y  = x
        z  = x  + (x <=aa).astype(int)*cyclic
        cc = bb + (bb<=aa).astype(int)*cyclic
        ee = dd + (dd<=aa).astype(int)*cyclic
        assert aa <= cc <= ee
        if aa == ee:
            assert len(np.unique(x)) == 1 # Constant input (e.g. nebulosity Milllau)
            cc = aa + 1
            ee = cc + 1
        del bb, dd
        f1 = 0
        f2 = (z-aa)/(cc-aa)
        f3 = (z-ee)/(cc-ee)
        if not cyclic:
            if index_func == 0:
                v = np.clip(f3,
                            a_min = f1,
                            a_max = None,
                            )
            elif index_func == 1:
                v  = np.clip(np.clip(f2,
                                     a_min = None,
                                     a_max = f3,
                                     ), 
                             a_max = None,
                             a_min = np.clip(f1,
                                             a_min = None, 
                                             a_max = f2, 
                                             ),
                             )
            elif nb_funcs - index_func == -2:
                v  = np.clip(np.clip(f3,
                                     a_min = None,
                                     a_max = f2,
                                     ), 
                             a_max = None,
                             a_min = np.clip(f1,
                                             a_min = None, 
                                             a_max = f3, 
                                             ),
                             )
            elif nb_funcs - index_func == -1:
                v = np.clip(f2,
                            a_min = f1,
                            a_max = None,
                            )
            else:
                v  = np.clip(np.clip(f2,
                                     a_min = None,
                                     a_max = f3,
                                     ), 
                             a_min = f1, 
                             a_max = None,
                             )
        else:
            v  = np.clip(np.clip((z-aa)/(cc-aa),
                         a_min = None,
                         a_max = (z-ee)/(cc-ee),
                         ),
                 a_min = 0,
                 a_max = None,
                 )
    else:
        raise NotImplementedError
    return v
        
