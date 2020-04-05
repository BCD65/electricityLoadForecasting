
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
This script is used to compute from the input data all the covariates based on spline features
"""


import numpy as np
import scipy as sp
import os
import re
import sys
from subprocess import CalledProcessError, TimeoutExpired
from termcolor import colored
#
import electricityLoadForecasting.tools  as tools
import electricityLoadForecasting.paths  as paths


path_data =  os.path.join(paths.outputs,
                          'Saved',
                          'Data',
                          ) 
save_data         = True
big_size_to_print = 1e8


#@profile
def compute_design(inputs, 
                   hprm, 
                   #path, replace with os.path.join(paths.outputs, 'Data')
                   dikt_file_names, 
                   mask_univariate  = None,
                   mask_bivariate   = None,
                   size1            = None, 
                   size2            = None, 
                   dikt_func        = None,       
                   dikt_func12      = None,  
                   db               = None,
                   size_tensor2     = None, 
                   ):
    # This is the main script where the functions for the univariate and the bivariate covariates are called
    # From the number of nodes defined in the parameters, 
    # We compute sequentially the nodes, the corresponding splines functions, 
    # and the transformation of the data by these functions.

    """
    Checks and Prints
    """
    if   db == 'training':
        assert dikt_func   == None
        assert dikt_func12 == None
    elif db == 'validation':
        assert bool(dikt_func)
    else:
        raise ValueError
    if hprm['afm.features.sparse_x1']: # Format to store the univariate covariates
        print(colored('X1_SPARSE', 'green'))
          
    prefix_features_univariate = dikt_file_names['features.{}.univariate'.format(db)] # string used to save/load the covariates
    lbfgs   = (hprm['afm.algorithm'] == 'L-BFGS')
    
    try:
        # We try to load from the local machine or from a distant machine (in the module primitives)
        if db == 'training':
            dikt_nodes = tools.batch_load(path_data, prefix_features_univariate, data_name = 'dikt_nodes',     data_type = 'np')
            dikt_func  = tools.batch_load(path_data, prefix_features_univariate, data_name = 'dikt_func',      data_type = 'np')
            size1      = tools.batch_load(path_data, prefix_features_univariate, data_name = 'size_univariate',data_type = 'np')
        X1             = tools.batch_load(path_data, prefix_features_univariate, data_name = 'X1_'+db,         data_type = ('dict_sp'
                                                                                                                            if hprm['afm.features.sparse_x1']
                                                                                                                            else
                                                                                                                            'dict_np'
                                                                                                                            ))
    except Exception as e:
        print(e)
        print('Start X1 '+db)
        if db == 'training':
            dikt_nodes   = make_dikt_nodes(inputs,
                                           hprm,
                                           hprm['afm.formula'].loc[hprm['afm.formula']['nb_intervals'].apply(lambda x : type(x) != tuple)],
                                           ) # Compute the nodes from the number of nodes for each covariate
            dikt_func    = make_dikt_func(dikt_nodes, hprm) # Compute the associated functions        
#            plot_1d_functions(hprm, 
#                              {(k,''):v for k, v in dikt_func.items()}, 
#                              'X1',
#                              )
        X1, size1 = make_X1(inputs, dikt_func, hprm, sparse_x1 = hprm['afm.features.sparse_x1']) # Compute the covariates and the size of the corresponding vectors
        for key in sorted(X1.keys()):
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
        print('make X1 done')
        #################
        ### SAVE 
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
    X = {
         'X_'+db     : X1,
         }
    if not lbfgs:
        # Compute XtX
        # lbfgs and mfilter are two possible algorithms that do not require the precomputations of XtX
        # since it is done later for lbfgs and it is not done for mfilter.
        try:
            X1tX1 = tools.batch_load(path_data,
                                     prefix = prefix_features_univariate,
                                     data_name = 'X1tX1_'+db,
                                     data_type = ('dict_sp' 
                                                  if hprm.get('sparse_x1')
                                                  else
                                                  'dict_np'
                                                  ),
                                     )
        except Exception as e:
            print(e)
            print('Start XtX1 '+db)
            X1tX1 = precomputations_1(X1, mask_univariate, all_products = hprm.get('gp_pen',0) > 0)
            for key, value in {**X1tX1}.items():
                assert type(value) in {np.ndarray, sp.sparse.csr_matrix}
            if len (X1tX1) < 1e4:
                list_files = [(X1tX1, 
                               'X1tX1_'+db,
                               prefix_features_univariate,
                               ('dict_sp'
                                if hprm.get('sparse_x1')
                                else
                                'dict_np'),
                               )]
            else:
                print('X1tX1 too large to save : len (X1tX1) = {0}'.format(len(X1tX1)))
                list_files = []
            save_list_files(list_files, path_data, hprm)        
        for e in list(X1tX1.keys()):
            assert type(X1tX1[e]) in {np.ndarray, sp.sparse.csr_matrix}
            j, k = e.split('@')
            if j!=k:
                X1tX1.update({k+'@'+j : sp.sparse.csr_matrix(X1tX1[e].T) if type(X1tX1[e]) == sp.sparse.csr_matrix else X1tX1[e].T})
        
        X.update({
                  'X1tX1_'+db : X1tX1 ,
                  })
    if db == 'training':
        X.update({
                  'dikt_nodes'       : dikt_nodes, 
                  'dikt_func'        : dikt_func, 
                  'mask_univariate'  : mask_univariate, 
                  'size_univariate'  : size1, 
                  })
    bool_bivariate = not hprm['afm.formula'].loc[hprm['afm.formula']['nb_intervals'].apply(lambda x : (type(x) == tuple and len(x) == 2))].empty
    if bool_bivariate: # ie if there are interactions
        prefix_features_bivariate = dikt_file_names['features.{}.bivariate'.format(db)] # string used to save/load the covariates
        prefix_features_all       = dikt_file_names['features.{}.all'.format(db)] # string used to save/load the covariates
        if hprm['sparse_x2']:
            print(colored('X2_SPARSE', 'green'))
        try:
            # Try to load the covariates corresponding to the interactions
            # Important because they may demand many more computations
            if db == 'training':
                dikt_nodes12  = tools.batch_load(path_data, prefix_features_bivariate, hprm, obj_name = 'dikt_nodes12')
                dikt_func12   = tools.batch_load(path_data, prefix_features_bivariate, hprm, obj_name = 'dikt_func12')
                size2         = tools.batch_load(path_data, prefix_features_bivariate, hprm, obj_name = 'size_bivariate')
                size_tensor2  = tools.batch_load(path_data, prefix_features_bivariate, hprm, obj_name = 'size_tensor2')
            X2 = tools.batch_load(path_data,
                                  prefix    = prefix_features_bivariate,
                                  data_name = 'X2_' + db,
                                  data_type = ('dict_sp'
                                               if hprm['sparse_x2']
                                               else
                                               'dict_np'
                                               ),
                                  )
        except Exception:
            if db == 'training':
                # Compute the nodes for the univariate functions used to compute the interactions
                # They may be different from the nodes used for the univariate covariates
                dikt_nodes12  = make_dikt_nodes_biv(hprm, hprm['approx_nb_itv'])
                dikt_func12   = make_dikt_func_biv(dikt_nodes12,  hprm)
#                plot_1d_functions(hprm, 
#                                  {(k,'left' ):v[0] for k, v in dikt_func12.items()}, 
#                                  'X12_left', 
#                                  )
#                plot_1d_functions(hprm, 
#                                  {(k,'right'):v[1] for k, v in dikt_func12.items()}, 
#                                  'X12_right', 
#                                  )
            else:
                assert dikt_func12
            print('start X12')
            # Compute the transformations of the data with univariate functions later used to compute the interactions
            X12, size12 = make_X12(inputs, 
                                   dikt_func12, 
                                   hprm, 
                                   )
            print('finished X12')
            print('start X2')
            # Compute the product for the interactions
            X2, size2, size_tensor2 = make_X2(X12, hprm)
            for k in sorted(X2.keys()):
                ord_key = '#'.join(sorted(k.split('#')))
                if ord_key in mask_bivariate:
                    mm = mask_bivariate[ord_key]
                    if mm.ndim == 1 and len(mm) == 0:
                        # Discard interactions that are used by zero substations
                        del X2[k], size2[k], size_tensor2[k]
            print('finished X2') 
            list_files = [(X2, 'X2_'+db, prefix_features_bivariate, 'dict_sp' if hprm['sparse_x2'] else 'dict_np'), 
                          ] if len (X2) < 1e4 else []
            if db == 'training':
                list_files += [ 
                               (dikt_nodes12, 'dikt_nodes12',   prefix_features_bivariate, 'np'), 
                               (dikt_func12,  'dikt_func12',    prefix_features_bivariate, 'np'), 
                               (size2,        'size_bivariate', prefix_features_bivariate, 'np'),
                               (size_tensor2, 'size_tensor2',   prefix_features_bivariate, 'np'),
                               ]
            save_list_files(list_files, path_data, hprm)
        for k in X2:
            hprm['data_cat'][k] = re.sub(r'\d+', '', k) 
        precompute = not (lbfgs or (db=='validation' and not hprm['tf_precompute_validation']))
        if precompute:
            # Compute the empirical covariance between the univariate/bivariate covariates
            try:
                X1tX2 = tools.batch_load(path_data,
                                         prefix    = prefix_features_all,
                                         data_name = 'X1tX2_'+db,
                                         data_type = ('dict_sp'
                                                      if hprm.get('sparse_x1') and hprm['sparse_x2']
                                                      else
                                                      'dict_np'
                                                      ),
                                         )
                X2tX2 = tools.batch_load(path_data,
                                         prefix    = prefix_features_bivariate,
                                         data_name = 'X2tX2_'+db,
                                         data_type = ('dict_sp'
                                                      if hprm['sparse_x2']
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
                                     if hprm.get('sparse_x1') and hprm['sparse_x2']
                                     else
                                     'dict_np'
                                     ),
                                    )] if len (X1tX2) < 1e4 else []
                if len (X1tX2) < 1e4:    
                    list_files += [(X2tX2, 
                                    'X2tX2_'+db, 
                                    prefix_features_bivariate, 
                                    'dict_sp' if hprm['sparse_x2'] else 'dict_np',
                                    )] if len (X2tX2) < 1e4 else []
                save_list_files(list_files, path_data, hprm) 
                print('finished XtX2')

        if db == 'training':
            X.update({
                      'mask_bivariate' : mask_bivariate,
                      'size_bivariate'          : size2,
                      'size_tensor2'   : size_tensor2,
                      })
        X.update({'X2_'+db    : X2})
        X['X_'+db].update(X2)
        if precompute:
            X2tX1 = {}
            for e in list(X1tX2.keys()):
                j, k  = e.split('@')
                X2tX1.update({k+'@'+j : sp.sparse.csr_matrix(X1tX2[e].T) 
                                        if type(X1tX2[e]) == sp.sparse.csr_matrix 
                                        else 
                                        X1tX2[e].T
                                        })
            for e in list(X2tX2.keys()):
                assert type(X2tX2[e]) == sp.sparse.csr_matrix
                j, k = e.split('@')
                if j!=k:
                    X2tX2.update({k+'@'+j : X2tX2[e].T})
                    if type(X2tX2[e]) == sp.sparse.csr_matrix:
                        X2tX2[k+'@'+j] = sp.sparse.csr_matrix(X2tX2[k+'@'+j])
            for e in list(X2tX2.keys()):
                assert type(X2tX2[e]) in {np.ndarray, sp.sparse.csr_matrix}
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


#def plot_1d_functions(hprm, dikt_func, suffix_name):
#    if hprm['fb_1D_functions_plot']:
#        print('BEGIN plot 1d functions ' + suffix_name)
#        plt.ioff()
#        for ii, ((cat, side), funcs) in enumerate(dikt_func.items()):
#            print('{0:5} / {1:5} : {2}'.format(ii, 
#                                               len(dikt_func), 
#                                               cat, 
#                                               ))
#            cat_plotted = {'left'  : cat[0], 
#                           'right' : cat[1], 
#                           ''      : cat, 
#                           }[side]
#            cyclic    = hprm['qo_modulo'][cat_plotted]
#            if len(funcs[0]) == 1:
#                continue
#            for jj, tt in enumerate(funcs):
#                print('\r'+' '*4+'{0:5} / {1:5}'.format(jj, len(funcs)), end = '')
#                title = ('XX'.join(cat) if type(cat) == tuple else cat) + bool(side)*(' '+side) + ' - ' 
#                for e in tt:
#                    title +=  ' {:.3f}'.format(e).replace('-', 'm')
#                title = title.replace('.', '')
#                aa, bb = tt [:2]
#                if 'tf_orthogonalize' in hprm:
#                    for k, v in hprm['tf_orthogonalize'].items():
#                        assert not v, ('orth given up because it leads to dense matrix and did not improve the results')
#                    if hprm['zone'] != 'nat':
#                        print(colored('NO ORTHOGONALIZATION FOR OTHER THAN NAT', 'yellow', 'on_red'))
#                        center = False
#                    else:
#                        if   cat in hprm.get('tf_orthogonalize', {}):
#                            center = hprm['tf_orthogonalize'][cat]
#                        elif cat in hprm.get('tf_orthogonalize', {}):
#                            center = hprm['tf_orthogonalize'][cat]
#                        else:
#                            center = False
#                else:
#                    center = False  
#                x     = np.arange(-0.5, 
#                                  1.5, 
#                                  (bb+(bb<aa)-aa)/20, 
#                                  )
#                y     = func1d(x, 
#                               tt, 
#                               cyclic  = cyclic, 
#                               bd_scal = hprm['tf_boundary_scaling'],
#                               hrch    = hprm.get('tf_hrch', {}).get(cat,0),
#                               center  = center,
#                               )
#                fig, ax = plt.subplots()
#                plt.plot(x, y, lw = 1)
#                ymax = 2*func1d(bb+(bb<aa), 
#                               tt, 
#                               cyclic  = cyclic, 
#                               bd_scal = hprm['tf_boundary_scaling'],
#                               hrch    = hprm.get('tf_hrch', {}).get(cat,0),
#                               )
#                ymax = ymax
#                plt.ylim(-0.05 - (y<0).any(), 1.05)
#                folder = hprm['path_plots'] + 'functions/' + suffix_name + '/'
#                os.makedirs(folder, 
#                            exist_ok = True,
#                            )
#                fig.savefig(folder + title+'.eps', 
#                            format = 'eps', 
#                            )
#                plt.close(fig)
#            print()


def make_dikt_func(d_nodes, hprm):
    # Compute for each covariate the spline functions from the set of nodes
    d_func = {}
    for inpt, trsfm, prm in d_nodes:
        cyclic      = hprm['inputs.cyclic'][inpt]
        d_func[(inpt, trsfm, prm)] = tuple([f 
                                            for e in d_nodes[(inpt, trsfm, prm)]
                                            for f in make_func(e, cyclic, hprm['afm.features.order_splines'])
                                            ])
    return d_func


def make_dikt_func_biv(d_nodes2, hprm):
    d_func2 = {}
    # Compute pairs of lists of functions for the interactions
    for cat1, cat2 in d_nodes2:
        cyclic1 = hprm['qo_modulo'][cat1]
        cyclic2 = hprm['qo_modulo'][cat2]
        funcs1  = tuple([f 
                         for e in d_nodes2[cat1,cat2][0]
                         for f in make_func(e, cyclic1, hprm.get('order_splines',1))
                         ])
        funcs2  = tuple([f 
                         for e in d_nodes2[cat1,cat2][1]
                         for f in make_func(e, cyclic2, hprm.get('order_splines',1))
                         ])
        d_func2[cat1,cat2] = (
                              funcs1, 
                              funcs2,
                              )              
    return d_func2


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


def make_dikt_nodes(inputs, hprm, formula):
    dikt_nodes = {}
    # For each covariate, compute the nodes from the number of nodes chosen on the interval [0, 1]
    for (inpt, trsfm, prm, location) in inputs.columns:
        nb_itv = formula.xs((inpt, trsfm, prm), level = 'input')['nb_intervals'].item()
        if bool(hprm['inputs.cyclic'][inpt]):
            min_value = 0
            max_value = hprm['inputs.cyclic'][inpt]
        else:
            min_value = inputs.loc[:,(inpt, trsfm, prm, location)].min()
            max_value = inputs.loc[:,(inpt, trsfm, prm, location)].max()
        dikt_nodes[(inpt, trsfm, prm)] = [make_nodes(e,
                                                     hprm['inputs.cyclic'][inpt],
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


def make_dikt_nodes_biv(hprm, dikt_nb_itv):
    # Compute pairs of list of nodes for the interactions
    dikt_nodes = {}
    for cat1,cat2 in list(map(lambda x : x.split('#'), 
                              list(filter(lambda x : '#' in x, 
                                          dikt_nb_itv,
                                          )
                                   ),
                              )
                          ):
        if (cat1,cat2) not in dikt_nodes:
            cyclic1 = hprm['qo_modulo'][cat1]
            cyclic2 = hprm['qo_modulo'][cat2]
            nb_itv_left, nb_itv_right = dikt_nb_itv[cat1+'#'+cat2]
            nodes1 = [make_nodes(e, cyclic1, hprm.get('order_splines',1)) 
                      for e in (nb_itv_left 
                                if type(nb_itv_left) == tuple 
                                else 
                                (nb_itv_left,)
                                )
                      ]
            nodes2 = [make_nodes(e, cyclic2, hprm.get('order_splines',1)) 
                      for e in (nb_itv_right 
                                if type(nb_itv_right) == tuple 
                                else 
                                (nb_itv_right,)
                                )
                      ]
            dikt_nodes[cat1,cat2] = (nodes1, nodes2)
    return dikt_nodes


def make_nodes(nb_itv, cyclic, order_splines, min_value = None, max_value = None):
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
                              for i in range(-1, nb_itv + 1 + order_splines)
                              ])
    else:
        raise ValueError
    return nodes


def make_X1(inputs, d_func, hprm, sparse_x1 = False):
    X1 = {(inpt, trsfm, prm, location):make_cov(d_func [(inpt, trsfm, prm)], 
                                                inputs[[(inpt, trsfm, prm, location)]], 
                                                sparse_x1, 
                                                hprm,
                                                inpt,
                                                )
          for inpt, trsfm, prm, location  in inputs.columns
          if (inpt, trsfm, prm) in d_func 
          }
    # Compute the covariates from the list of functions and the input inputs
    size1 = {k : v.shape[1]
             for k, v in X1.items()
             }
    return X1, size1


def make_X12(inputs, d_func, prm):
    # For the interactions
    # Compûte the univariate transformations later used to compute the interactions
    dikt_uni_cov = {}
    X12 = {}
    for key1 in inputs.keys():
        for key2 in inputs.keys():
            cat1 = prm['data_cat'][key1]
            cat2 = prm['data_cat'][key2]
            if (cat1, cat2) in d_func.keys():
                for ii, key in enumerate([key1,key2]):
                    if (key,d_func[cat1,cat2][ii]) not in dikt_uni_cov:
                        cat = prm['data_cat'][key]
                        if 'tf_orthogonalize' in prm:
                            if prm['zone'] != 'nat':
                                #print(colored('NO ORTHOIGONALIZATION FOR OTHER THAN NAT', 'yellow', 'on_red'))
                                center = False
                            else:
                                if   (cat1,cat2) in prm.get('tf_orthogonalize', {}):
                                    center = prm['tf_orthogonalize'][cat1,cat2]
                                elif (cat2,cat1) in prm.get('tf_orthogonalize', {}):
                                    center = prm['tf_orthogonalize'][cat2,cat1]
                                else:
                                    center = False
                        else:
                            center = False
                        dikt_uni_cov[key,d_func[cat1,cat2][ii]] = make_cov(d_func[cat1,cat2][ii], 
                                                                           inputs[key], 
                                                                           prm['qo_modulo'][cat],
                                                                           prm['tf_boundary_scaling'], 
                                                                           False, # Not sparse because einsum after
                                                                           prm,
                                                                           cat,
                                                                           center = center,
                                                                           )
                X12[key1,key2] = (dikt_uni_cov[key1,d_func[prm['data_cat'][key1],
                                                           prm['data_cat'][key2],
                                                           ][0]],
                                  dikt_uni_cov[key2,d_func[prm['data_cat'][key1],
                                                           prm['data_cat'][key2],
                                                           ][1]],
                                  )
    size12 = {(e,f):(v.shape[1],w.shape[1]) for (e,f), (v,w) in X12.items()}
    return X12, size12


def make_cov(list_funcs, inpt_data, sparse_x1, hprm, inpt_name, center = False):
    #assert type(hprm.get('tf_hrch', {})) == dict
    # Compute for one category (eg the hour, the temperatures or the delayed temperatures)
    # the covariates with the associated list of functions
    cov = np.concatenate([func1d(inpt_data, 
                                 func, 
                                 index_func = ii,
                                 nb_funcs   = len(list_funcs),
                                 cyclic     = hprm['inputs.cyclic'][inpt_name], 
                                 bd_scal    = hprm['afm.features.boundary_scaling'],
                                 hrch       = hprm.get('tf_hrch', {}).get(inpt_name,0),
                                 center     = center, 
                                 )
                          for ii, func in enumerate(list_funcs)
                          ], 
                         axis = 1,
                         )
    if sparse_x1:
        cov = sp.sparse.csc_matrix(cov)
    return cov


                                            
                                            

def make_X2(X12, hprm):
    # For the interactions
    # For each pairs of covariates, pairs of univariate covariates are stored in X12
    # One want to compute their product
    X2    = {}
    size2 = {} 
    size_tensor2 = {} 
    for ii, (e,f) in enumerate(sorted(X12)):
        # Check that the interactions is selected in the hprmeters
        data_cat = (  set([e 
                           for coef, list_keys in hprm['tf_config_coef'].items() 
                           for e in list_keys 
                           if '#' in e
                           ]) 
                    & {hprm['data_cat'][e]+'#'+hprm['data_cat'][f], 
                       hprm['data_cat'][f]+'#'+hprm['data_cat'][e],
                       }
                    )
        if len(data_cat)>1:
            raise ValueError # not impossible but must be dealt with ie check that the interaction is in mask2
        if data_cat:
            print('\r'+'{0:20}'.format(e), ii, '/', len(X12), end = '')
            #Compute the product or the minimum interaction
            if hprm['tf_prod0_min1']:
                assert X12[e,f][0].min() >= 0
                assert X12[e,f][0].max() <= 1
                assert X12[e,f][1].min() >= 0
                assert X12[e,f][1].max() <= 1
                n, p = X12[e,f][0].shape
                _, q = X12[e,f][1].shape
                xleft  = np.repeat(X12[e,f][0][:,:,None], q, axis = 2)
                xright = np.repeat(X12[e,f][1][:,None,:], p, axis = 1)
                interaction = np.min([xleft, xright], axis = 0)
            else:
                interaction = np.einsum('np,nq->npq', 
                                        X12[e,f][0], 
                                        X12[e,f][1],
                                        )
            siz2             = interaction.shape[1:]
            assert len(siz2) == 2
            size_tensor2[e+'#'+f] = siz2
            size2       [e+'#'+f] = np.prod(siz2)
            # IF the first covariate is associated to p functions and the second to q functions and there are n observations
            # Reshape the interaction of size (n,p,q)into a matrix of size (n, p*q)
            # It is indeed important to use matrix instead of tensors to have faster computations with numpy, in particular for sparse matrices
            X2[e+'#'+f] = interaction.reshape((-1, siz2[0]*siz2[1]))
            if hprm['sparse_x2']:
                X2[e+'#'+f] = sp.sparse.csc_matrix(X2[e+'#'+f])
            if (    (f,e) not in X12 
                and (hprm['data_cat'][f]+'#'+hprm['data_cat'][e] in {e for 
                                                                       var, list_keys in hprm['tf_config_coef'].items()
                                                                       for e in list_keys
                                                                       }
                     or 'Cbm' in hprm['tf_config_coef']
                     )
                ):
                X2[f+'#'+e] = interaction.transpose(0,2,1).reshape((-1, siz2[0]*siz2[1]))
                size_tensor2[f+'#'+e] = siz2[::-1]
                size2       [f+'#'+e] = np.prod(siz2[::-1])
                if hprm['sparse_x2']:
                    X2[f+'#'+e] = sp.sparse.csc_matrix(X2[f+'#'+e])
    print()
    return X2, size2, size_tensor2
    

def precomputations_1(X1, mask, all_products = 0):
    raise NotImplementedError
    # Compute the products between univariate covariates
#    print('Start X1tX1')
#    X1tX1 = {}
#    for ii, key1 in enumerate(X1):
#        for jj, key2 in enumerate(X1):
#            print('\r{0:{wid}}'.format(jj, wid = len(str(len(X1)))), '/', len(X1), ' - ', '{0:{wid}}'.format(ii, wid = len(str(len(X1)))), '/', len(X1), end = '\r')
#            if key1 <= key2:
#                if approx_tf.cross_mask(mask, key1, mask, key2) or all_products:
#                    X1tX1[key1+'@'+key2] = X1[key1].T @ X1[key2]
#                    if type(X1tX1[key1+'@'+key2]) in {sp.sparse.csc_matrix, sp.sparse.csr_matrix}:
#                        X1tX1[key1+'@'+key2] = sp.sparse.csr_matrix(X1tX1[key1+'@'+key2])
#                    assert type(X1tX1[key1+'@'+key2]) in {np.ndarray, sp.sparse.csr_matrix}
#    print()
#    return X1tX1


def precomputations_2(X1, X2, mask1, mask2, hprm, all_products = 0):
    raise NotImplementedError
#    # Compute the products between univariate/bivariate covariates
#    # We only have to compute the product between pairs of covariates that are accessed a same sustation if the models are independent
#    # but have to compute all the product for the sum consistent loss
#    if all_products:
#        # all_products is an indicator that all products should be compuetd
#        assert hprm['sparse_x2']
#        print(colored('\n \n X2 will be very large with all interactions ie might be too big \n \n', 'red', 'on_cyan'))
#    print('Start X1tX2')
#    X1tX2 = {}
#    # Products between univariate covariates and bivariate covariates
#    for ii, key1 in enumerate(X1):
#        for jj, key2 in enumerate(X2):
#            if approx_tf.cross_mask(mask1, key1, mask2, key2) or all_products:
#                print('\r{0:{wid}}'.format(jj, wid = len(str(len(X2)))), '/', len(X2), ' - ', '{0:{wid}}'.format(ii, wid = len(str(len(X1)))), '/', len(X1), end = '\r')
#                X1tX2[key1+'@'+key2] = X1[key1].T @ X2[key2]
#                if type(X1tX2[key1+'@'+key2]) in {sp.sparse.csc_matrix, sp.sparse.csr_matrix}:
#                    X1tX2[key1+'@'+key2] = sp.sparse.csr_matrix(X1tX2[key1+'@'+key2])
#                assert type(X1tX2[key1+'@'+key2]) in {np.ndarray, sp.sparse.csr_matrix}
#    print()
#    print('Start X2tX2')
#    X2tX2 = {}
#    # Products between bivariate covariates
#    for ii, key1 in enumerate(X2):
#        for jj, key2 in enumerate(X2):
#            if key1 <= key2:
#                if approx_tf.cross_mask(mask2, key1, mask2, key2) or all_products:
#                    print('\r{0:{wid}}'.format(jj, wid = len(str(len(X2)))), '/', len(X2), ' - ', '{0:{wid}}'.format(ii, wid = len(str(len(X2)))), '/', len(X2), end = '\r')
#                    X2tX2[key1+'@'+key2] = X2[key1].T @ X2[key2]
#                    if type(X2tX2[key1+'@'+key2]) in {sp.sparse.csc_matrix, sp.sparse.csr_matrix}:
#                        X2tX2[key1+'@'+key2] = sp.sparse.csr_matrix(X2tX2[key1+'@'+key2])
#                    assert type(X2tX2[key1+'@'+key2]) in {np.ndarray, sp.sparse.csr_matrix}
#    print()
#    return X1tX2, X2tX2


def func1d(x, func, index_func = None, nb_funcs = None, cyclic = None, bd_scal = 0, hrch = None, center = False):
    # Computet for the different values stored in x the values of the spline function (defined by a set of nodes) stored in func
    assert hrch != None
    if len(func) == 1:
        assert func[0] == int(func[0])
        return x**func[0]
    elif len(func) == 3:
        aa, bb, dd = func
        assert (dd >= bb and bb >= aa) or cyclic
        assert not bd_scal # Procedure should be checked
        #y  = x
        z  = x  + (x <=aa).astype(int)*cyclic
        cc = bb + (bb<=aa).astype(int)*cyclic
        ee = dd + (dd<=aa).astype(int)*cyclic
        assert aa < cc
        assert cc < ee
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
        if center:
            v -= (cc - aa)
        ### Checks and factor
        if bd_scal:
            raise NotImplementedError
#            if   aa<0 and bd_scal:
#                assert bb > 0
#                assert dd >0
#                fac = (bb-aa)/bb
#            elif dd>1 and bd_scal:
#                assert aa < 1
#                assert bb < 1
#                fac = (bb-aa)/(1-aa)
        #else:
        if hrch:
            raise NotImplementedError
#            assert cc > aa 
#            assert cc - aa <= 1
#            fac = (1/(cc - aa))*hrch**(-np.log(cc - aa)) # If hrch = 1, they all cost the same thing
#            assert fac > 0
#        else:
#            fac = 1
        fac = 1
    else:
        raise NotImplementedError
    return v*fac 
        
