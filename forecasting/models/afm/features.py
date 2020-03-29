
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
This script is used to compute from the input data all the covariates based on splines
"""


import numpy as np
import scipy as sp
import os
import re
import sys
import matplotlib.pyplot as plt
from subprocess import CalledProcessError, TimeoutExpired
from termcolor import colored
#
import electricityLoadForecasting.paths  as paths
from electricityLoadForecasting.tools.exceptions import custex
from . import masks


save_data         = True
big_size_to_print = 1e8


#@profile
def compute_design(inputs, 
                   hprm, 
                   #path, replace with os.path.join(paths.outputs, 'Data')
                   dikt_files, 
                   masks_univariate = None,
                   masks_bivariate  = None,
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
    if   db == 'train':
        assert dikt_func   == None
        assert dikt_func12 == None
    elif db == 'test':
        assert bool(dikt_func)
    else:
        raise ValueError
    if hprm['afm.sparse_x1']: # Format to store the univariate covariates
        print(colored('X1_SPARSE', 'green'))
        
    prefix1 = dikt['primitives_'+db+'1'] # string used to save/load the covariates
    lbfgs   = np.sum(['lbfgs' in key 
                      for key, value in param['tf_config_coef'].items() 
                      if bool(value)
                      ])
    #mfilter = (param['method'] == 'filter') # Given up
    
    try:
        # We try to load from the local machine or from a distant machine (in the module primitives)
        if db == 'train':
            dikt_nodes = primitives.tf_data_load(path, prefix1, param, obj_name = 'dikt_nodes')
            dikt_func  = primitives.tf_data_load(path, prefix1, param, obj_name = 'dikt_func')
            size1      = primitives.tf_data_load(path, prefix1, param, obj_name = 'size1')
        X1       = primitives.tf_data_load(path, prefix1, param, 'X1_'+db, mod = 'dict_sp' if param.get('sparse_x1') else 'dict_np')
    except Exception as e:
        print(e)
        print('Start X1 '+db)
        if db == 'train':
            dikt_nodes   = make_dikt_nodes(param, param['approx_nb_itv'], data.keys()) # Compute the nodes from the number of nodes for each covariate
            dikt_func    = make_dikt_func(dikt_nodes, param) # Compute the associated functions        
            plot_1d_functions(param, 
                              {(k,''):v for k, v in dikt_func.items()}, 
                              'X1',
                              )
        X1, size1     = make_X1(data, dikt_func, param, sparse_x1 = param.get('sparse_x1')) # Compute the covariates and the size of the corresponding vectors
        for key in sorted(X1.keys()):
            if key in given_mask1:
                mm = given_mask1[key]
                if mm.ndim == 1 and len(mm) == 0: 
                    # If no substation should have access to this covariate, delete it
                    # This can happen when only a region is considered : the weather stations in other regions will not be considered
                    del X1[key], size1[key]
        print('make X1 done')
        #################
        ### SAVE 
        list_files = [
                      (X1, 'X1_'+db , prefix1, 'dict_sp' if param.get('sparse_x1') and param['sparse_x2'] else 'dict_np'), 
                      ]
        if db == 'train':
            list_files += [
                           (dikt_nodes, 'dikt_nodes', prefix1, 'np'), 
                           (dikt_func,  'dikt_func',  prefix1, 'np'), 
                           (size1,      'size1',      prefix1, 'np'), 
                           ]
        save_list_files(list_files, path, param)
        ### FINISHED X1
    X = {
         'X_'+db     : X1,
         }
    if not lbfgs and not mfilter:
        # Compute XtX
        # lbfgs and mfilter are two possible algorithms that do not require the precomputations of XtX
        # since it is done later for lbfgs and it is not done for mfilter.
        try:
            X1tX1 = primitives.tf_data_load(path, prefix1, param, 'X1tX1_'+db, mod = 'dict_sp' if param.get('sparse_x1') else 'dict_np')
        except Exception as e:
            print(e)
            print('Start XtX1 '+db)
            X1tX1 = precomputations_1(X1, given_mask1, all_products = param.get('gp_pen',0) > 0)
            for key, value in {**X1tX1}.items():
                assert type(value) in {np.ndarray, sp.sparse.csr_matrix}
            if len (X1tX1) < 1e4:
                list_files = [(X1tX1, 'X1tX1_'+db, dikt['primitives_'+db+'1'], 'dict_sp' if param.get('sparse_x1') else 'dict_np'),]
            else:
                print('X1tX1 too large to save : len (X1tX1) = {0}'.format(len(X1tX1)))
                list_files = []
            save_list_files(list_files, path, param)        
        for e in list(X1tX1.keys()):
            assert type(X1tX1[e]) in {np.ndarray, sp.sparse.csr_matrix}
            j, k = e.split('@')
            if j!=k:
                X1tX1.update({k+'@'+j : sp.sparse.csr_matrix(X1tX1[e].T) if type(X1tX1[e]) == sp.sparse.csr_matrix else X1tX1[e].T})
        
        X.update({
                  'X1tX1_'+db : X1tX1 ,
                  })
    if db == 'train':
        X.update({
                  'dikt_nodes'  : dikt_nodes, 
                  'dikt_func'   : dikt_func, 
                  'mask1'       : given_mask1, 
                  'size1'       : size1, 
                  })
    w_inter = bool([e 
                    for coef, list_keys in param['tf_config_coef'].items() 
                    for e in list_keys # list_keys contains the covariates (including the interactions)
                    if '#' in e # The name of an interaction between covariates key1 and key2 is key1#key2
                    ])
    if w_inter: # ie if there are interactions
        prefix2     = dikt['primitives_'+db+'2']
        if param['sparse_x2']:
            print(colored('X2_SPARSE', 'green'))
        try:
            # Try to load the covariates corresponding to the interactions
            # Important because they may demand many more computations
            if db == 'train':
                dikt_nodes12  = primitives.tf_data_load(path, prefix2, param, obj_name = 'dikt_nodes12')
                dikt_func12   = primitives.tf_data_load(path, prefix2, param, obj_name = 'dikt_func12')
                size2         = primitives.tf_data_load(path, prefix2, param, obj_name = 'size2')
                size_tensor2  = primitives.tf_data_load(path, prefix2, param, obj_name = 'size_tensor2')
            X2 = primitives.tf_data_load(path, prefix2, param, 'X2_' + db, mod = 'dict_sp' if param['sparse_x2'] else 'dict_np')
        except Exception:
            if db == 'train':
                # Compute the nodes for the univariate functions used to compute the interactions
                # They may be different from the nodes used for the univariate covariates
                dikt_nodes12  = make_dikt_nodes_biv(param, param['approx_nb_itv'])
                dikt_func12   = make_dikt_func_biv(dikt_nodes12,  param)
                plot_1d_functions(param, 
                                  {(k,'left' ):v[0] for k, v in dikt_func12.items()}, 
                                  'X12_left', 
                                  )
                plot_1d_functions(param, 
                                  {(k,'right'):v[1] for k, v in dikt_func12.items()}, 
                                  'X12_right', 
                                  )
            else:
                assert dikt_func12
            print('start X12')
            # Compute the transformations of the data with univariate functions later used to compute the interactions
            X12, size12 = make_X12(data, 
                                   dikt_func12, 
                                   param, 
                                   )
            print('finished X12')
            print('start X2')
            # Compute the product for the interactions
            X2, size2, size_tensor2 = make_X2(X12, param)
            for k in sorted(X2.keys()):
                ord_key = '#'.join(sorted(k.split('#')))
                if ord_key in given_mask2:
                    mm = given_mask2[ord_key]
                    if mm.ndim == 1 and len(mm) == 0:
                        # Discard interactions that are used by zero substations
                        del X2[k], size2[k], size_tensor2[k]
            print('finished X2') 
            list_files = [(X2, 'X2_'+db, prefix2, 'dict_sp' if param['sparse_x2'] else 'dict_np'), 
                          ] if len (X2) < 1e4 else []
            if db == 'train':
                list_files += [ 
                               (dikt_nodes12, 'dikt_nodes12', prefix2, 'np'), 
                               (dikt_func12,  'dikt_func12',  prefix2, 'np'), 
                               (size2,        'size2',        prefix2, 'np'),
                               (size_tensor2, 'size_tensor2', prefix2, 'np'),
                               ]
            save_list_files(list_files, path, param)
        for k in X2:
            param['data_cat'][k] = re.sub(r'\d+', '', k) 
        precompute = not (lbfgs or mfilter or (db=='test' and not param['tf_precompute_test']))
        if precompute:
            # Compute the empirical covariance between the univariate/bivariate covariates
            try:
                X1tX2 = primitives.tf_data_load(path, dikt['primitives_'+db+'12'], param, 'X1tX2_'+db, mod = 'dict_sp' if param.get('sparse_x1') and param['sparse_x2'] else 'dict_np')
                X2tX2 = primitives.tf_data_load(path, dikt['primitives_'+db+'2'],  param, 'X2tX2_'+db, mod = 'dict_sp' if param['sparse_x2'] else 'dict_np')
            except Exception:
                X1tX2, X2tX2 = precomputations_2(X1, X2, given_mask1, given_mask2, param, all_products = (param.get('gp_pen',0) > 0 and param['gp_matrix'] != ''))
                for key, value in {**X1tX2}.items():
                    assert type(value) in {np.ndarray, sp.sparse.csr_matrix}
                for key, value in {**X2tX2}.items():
                    assert type(value) in {np.ndarray, sp.sparse.csr_matrix}
                list_files = []
                if len (X1tX2) < 1e4:
                    list_files += [(X1tX2, 
                                    'X1tX2_'+db, 
                                    dikt['primitives_'+db+'12'], 
                                    'dict_sp' if param.get('sparse_x1') and param['sparse_x2'] else 'dict_np',
                                    )] if len (X1tX2) < 1e4 else []
                if len (X1tX2) < 1e4:    
                    list_files += [(X2tX2, 
                                    'X2tX2_'+db, 
                                    dikt['primitives_'+db+'2' ], 
                                    'dict_sp' if param['sparse_x2'] else 'dict_np',
                                    )] if len (X2tX2) < 1e4 else []
                save_list_files(list_files, path, param) 
                print('finished XtX2')

        if db == 'train':
            X.update({
                      'mask2'        : given_mask2,
                      'size2'        : size2,
                      'size_tensor2' : size_tensor2,
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
        if db == 'train':
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

def save_list_files(list_files, path, param):
    # Intermediary function to format the data and save it with the module primitives
    print('start saving')
    for M, s, pref, opt in list_files:
        print('\r    ', s, ' '*20, end = '')
        try:
            assert save_data
            primitives.tf_data_save(path, pref, param, M, s, mod = opt)
        except (CalledProcessError, custex, TimeoutExpired, OSError, AssertionError):
            pass 
    print('end saving')

def plot_1d_functions(param, dikt_func, suffix_name):
    if param['fb_1D_functions_plot']:
        print('BEGIN plot 1d functions ' + suffix_name)
        plt.ioff()
        for ii, ((cat, side), funcs) in enumerate(dikt_func.items()):
            print('{0:5} / {1:5} : {2}'.format(ii, 
                                               len(dikt_func), 
                                               cat, 
                                               ))
            cat_plotted = {'left'  : cat[0], 
                           'right' : cat[1], 
                           ''      : cat, 
                           }[side]
            cyclic    = param['qo_modulo'][cat_plotted]
            if len(funcs[0]) == 1:
                continue
            for jj, tt in enumerate(funcs):
                print('\r'+' '*4+'{0:5} / {1:5}'.format(jj, len(funcs)), end = '')
                title = ('XX'.join(cat) if type(cat) == tuple else cat) + bool(side)*(' '+side) + ' - ' 
                for e in tt:
                    title +=  ' {:.3f}'.format(e).replace('-', 'm')
                title = title.replace('.', '')
                aa, bb = tt [:2]
                if 'tf_orthogonalize' in param:
                    for k, v in param['tf_orthogonalize'].items():
                        assert not v, ('orth given up because it leads to dense matrix and did not improve the results')
                    if param['zone'] != 'nat':
                        print(colored('NO ORTHOGONALIZATION FOR OTHER THAN NAT', 'yellow', 'on_red'))
                        center = False
                    else:
                        if   cat in param.get('tf_orthogonalize', {}):
                            center = param['tf_orthogonalize'][cat]
                        elif cat in param.get('tf_orthogonalize', {}):
                            center = param['tf_orthogonalize'][cat]
                        else:
                            center = False
                else:
                    center = False  
                x     = np.arange(-0.5, 
                                  1.5, 
                                  (bb+(bb<aa)-aa)/20, 
                                  )
                y     = func1d(x, 
                               tt, 
                               cyclic  = cyclic, 
                               bd_scal = param['tf_boundary_scaling'],
                               hrch    = param.get('tf_hrch', {}).get(cat,0),
                               center  = center,
                               )
                fig, ax = plt.subplots()
                plt.plot(x, y, lw = 1)
                ymax = 2*func1d(bb+(bb<aa), 
                               tt, 
                               cyclic  = cyclic, 
                               bd_scal = param['tf_boundary_scaling'],
                               hrch    = param.get('tf_hrch', {}).get(cat,0),
                               )
                ymax = ymax
                plt.ylim(-0.05 - (y<0).any(), 1.05)
                folder = param['path_plots'] + 'functions/' + suffix_name + '/'
                os.makedirs(folder, 
                            exist_ok = True,
                            )
                fig.savefig(folder + title+'.eps', 
                            format = 'eps', 
                            )
                plt.close(fig)
            print()


##################### 
#   Do not DELETE   #
# This could have been used for the trend-filtering optimization problem, to use a change of basis 
# and substitute the trend filtering proximal operator with a simple lasso proximal operator
# in order to get faster computations and see if the trend filtering problem is relevant.
# It is no longer useful since this lead was given up due to time constraints.
##################### 
#def make_ch_basis(self, basis):
#    ch_basis     = {}
#    if not (basis):
#        ch_basis = {k:np.eye(self.size1[k]) for k in self.keys1al}
#    elif basis == 1:
#        for k in self.keys1al:
#            J = np.tril(np.ones((self.size1[k],self.size1[k])))
#            if self.orth_J:
#                for j in range(1, J.shape[0]):
#                    J[:,j] = J[:,j] - J[:,j].mean()
#            ch_basis[k] = J
#    elif basis == 2:
#        for k in self.keys1al:
#            J = np.array([[max(0, i - j + 1) for j in range(self.size1[k])] for i in range(self.size1[k])], dtype = np.float)
#            J[:,0] = 1
#            if self.orth_J:
#                # orth par rapport à 1ere colonne
#                for j in range(1, J.shape[0]):
#                    J[:,j] = J[:,j] - J[:,j].mean()
#                # orth par rapport à deuxième colonne
#                for j in range(2, J.shape[0]):
#                    J[:,j] = J[:,j] - ((J[:,1].T.dot(J[:,j]))/np.linalg.norm(J[:,1])**2)*J[:,1]
#            ch_basis[k] = J
#    return ch_basis


def make_dikt_func(d_nodes, param):
    d_func = {}
    # Compute for each covariate the spline functions from the set of nodes
    for cat in d_nodes:
        cyclic      = param['qo_modulo'][cat]
        d_func[cat] = tuple([f 
                             for e in d_nodes[cat]
                             for f in make_func(e, cyclic, param.get('order_splines',1))
                             ])
    return d_func


def make_dikt_func_biv(d_nodes2, param):
    d_func2 = {}
    # Compute pairs of lists of functions for the interactions
    for cat1, cat2 in d_nodes2:
        cyclic1 = param['qo_modulo'][cat1]
        cyclic2 = param['qo_modulo'][cat2]
        funcs1  = tuple([f 
                         for e in d_nodes2[cat1,cat2][0]
                         for f in make_func(e, cyclic1, param.get('order_splines',1))
                         ])
        funcs2  = tuple([f 
                         for e in d_nodes2[cat1,cat2][1]
                         for f in make_func(e, cyclic2, param.get('order_splines',1))
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
        for i in range(nodes.shape[0] - (1+order_splines)*(1-cyclic)):
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


def make_dikt_nodes(param, dikt_nb_itv, keys1):
    dikt_nodes = {}
    # For each covariate, compute the nodes from the number of nodes chosen on the interval [0, 1]
    for k in keys1:
        cat = param['data_cat'][k] 
        if (    '#' not in cat
            and cat not in dikt_nodes
            and cat in dikt_nb_itv
            and cat in sorted([e 
                               for coef, list_keys in param['tf_config_coef'].items() 
                               for e in list_keys
                               ])
            ):
            cyclic = param['qo_modulo'][cat]
            nb_itv = dikt_nb_itv[cat]
            dikt_nodes[cat] = [make_nodes(e, cyclic, param.get('order_splines',1)) 
                               for e in (nb_itv 
                                         if type(nb_itv) == tuple 
                                         else 
                                         (nb_itv,)
                                         )
                               ]
    return dikt_nodes


def make_dikt_nodes_biv(param, dikt_nb_itv):
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
            cyclic1 = param['qo_modulo'][cat1]
            cyclic2 = param['qo_modulo'][cat2]
            nb_itv_left, nb_itv_right = dikt_nb_itv[cat1+'#'+cat2]
            nodes1 = [make_nodes(e, cyclic1, param.get('order_splines',1)) 
                      for e in (nb_itv_left 
                                if type(nb_itv_left) == tuple 
                                else 
                                (nb_itv_left,)
                                )
                      ]
            nodes2 = [make_nodes(e, cyclic2, param.get('order_splines',1)) 
                      for e in (nb_itv_right 
                                if type(nb_itv_right) == tuple 
                                else 
                                (nb_itv_right,)
                                )
                      ]
            dikt_nodes[cat1,cat2] = (nodes1, nodes2)
    return dikt_nodes


def make_nodes(nb_itv, cyclic, order_splines):
    # Given the number of nodes, the indicator of a cyclic variable and the degree of the wanted splines
    # Compute the associated nodes
    if type(nb_itv) == str and nb_itv[0] == 'p': 
        # Identity, Indicator or polynomials
        assert len(nb_itv) == 2
        nodes = tuple([tuple([e]) for e in range(1, eval(nb_itv[1])+1)])
    elif type(nb_itv) == int: 
        # Normal
        assert nb_itv != 0
        if cyclic:
            nodes = np.array([i/nb_itv for i in range(nb_itv)])  
        else:
            nodes = np.array([i/nb_itv for i in range(-1, nb_itv + 1 + order_splines)])
    else:
        raise ValueError
    return nodes


def make_X1(data, d_func, prm, sparse_x1 = False):
    X1 = {key:make_cov(d_func[prm['data_cat'][key]], 
                       data[key], 
                       prm['qo_modulo'][prm['data_cat'][key]],
                       prm['tf_boundary_scaling'], 
                       sparse_x1, 
                       prm,
                       prm['data_cat'][key],
                       )
          for key in data.keys() 
          if prm['data_cat'][key] in d_func 
          }
    # Compute the covariates from the list of functions and the input data
    size1 = {k:v.shape[1] for k, v in X1.items()}
    return X1, size1


def make_X12(data, d_func, prm):
    # For the interactions
    # Compûte the univariate transformations later used to compute the interactions
    dikt_uni_cov = {}
    X12 = {}
    for key1 in data.keys():
        for key2 in data.keys():
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
                                                                           data[key], 
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


def make_cov(d_func_cat, data_key, cyclic, bd_scal, sparse_x1, param, cat, center = False):
    assert type(param.get('tf_hrch', {})) == dict
    # Compute for one category (eg the hour, the temperatures or the delayed temperatures)
    # the covariates with the associated list of functions
    cov = np.concatenate([func1d(data_key, 
                                 func, 
                                 cyclic  = cyclic, 
                                 bd_scal = bd_scal,
                                 hrch    = param.get('tf_hrch', {}).get(cat,0),
                                 center  = center, 
                                 )
                          for func in d_func_cat 
                          ], 
                         axis = 1,
                         )
    if sparse_x1:
        cov = sp.sparse.csc_matrix(cov)
    return cov


def make_X2(X12, param):
    # For the interactions
    # For each pairs of covariates, pairs of univariate covariates are stored in X12
    # One want to compute their product
    X2    = {}
    size2 = {} 
    size_tensor2 = {} 
    for ii, (e,f) in enumerate(sorted(X12)):
        # Check that the interactions is selected in the parameters
        data_cat = (  set([e 
                           for coef, list_keys in param['tf_config_coef'].items() 
                           for e in list_keys 
                           if '#' in e
                           ]) 
                    & {param['data_cat'][e]+'#'+param['data_cat'][f], 
                       param['data_cat'][f]+'#'+param['data_cat'][e],
                       }
                    )
        if len(data_cat)>1:
            raise ValueError # not impossible but must be dealt with ie check that the interaction is in mask2
        if data_cat:
            print('\r'+'{0:20}'.format(e), ii, '/', len(X12), end = '')
            #Compute the product or the minimum interaction
            if param['tf_prod0_min1']:
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
            if param['sparse_x2']:
                X2[e+'#'+f] = sp.sparse.csc_matrix(X2[e+'#'+f])
            if (    (f,e) not in X12 
                and (param['data_cat'][f]+'#'+param['data_cat'][e] in {e for 
                                                                       var, list_keys in param['tf_config_coef'].items()
                                                                       for e in list_keys
                                                                       }
                     or 'Cbm' in param['tf_config_coef']
                     )
                ):
                X2[f+'#'+e] = interaction.transpose(0,2,1).reshape((-1, siz2[0]*siz2[1]))
                size_tensor2[f+'#'+e] = siz2[::-1]
                size2       [f+'#'+e] = np.prod(siz2[::-1])
                if param['sparse_x2']:
                    X2[f+'#'+e] = sp.sparse.csc_matrix(X2[f+'#'+e])
    print()
    return X2, size2, size_tensor2
    

def precomputations_1(X1, mask, all_products = 0):
    # Compute the products between univariate covariates
    print('Start X1tX1')
    X1tX1 = {}
    for ii, key1 in enumerate(X1):
        for jj, key2 in enumerate(X1):
            print('\r{0:{wid}}'.format(jj, wid = len(str(len(X1)))), '/', len(X1), ' - ', '{0:{wid}}'.format(ii, wid = len(str(len(X1)))), '/', len(X1), end = '\r')
            if key1 <= key2:
                if approx_tf.cross_mask(mask, key1, mask, key2) or all_products:
                    X1tX1[key1+'@'+key2] = X1[key1].T @ X1[key2]
                    if type(X1tX1[key1+'@'+key2]) in {sp.sparse.csc_matrix, sp.sparse.csr_matrix}:
                        X1tX1[key1+'@'+key2] = sp.sparse.csr_matrix(X1tX1[key1+'@'+key2])
                    assert type(X1tX1[key1+'@'+key2]) in {np.ndarray, sp.sparse.csr_matrix}
    print()
    return X1tX1


def precomputations_2(X1, X2, mask1, mask2, param, all_products = 0):
    # Compute the products between univariate/bivariate covariates
    # We only have to compute the product between pairs of covariates that are accessed a same sustation if the models are independent
    # but have to compute all the product for the sum consistent loss
    if all_products:
        # all_products is an indicator that all products should be compuetd
        assert param['sparse_x2']
        print(colored('\n \n X2 will be very large with all interactions ie might be too big \n \n', 'red', 'on_cyan'))
    print('Start X1tX2')
    X1tX2 = {}
    # Products between univariate covariates and bivariate covariates
    for ii, key1 in enumerate(X1):
        for jj, key2 in enumerate(X2):
            if approx_tf.cross_mask(mask1, key1, mask2, key2) or all_products:
                print('\r{0:{wid}}'.format(jj, wid = len(str(len(X2)))), '/', len(X2), ' - ', '{0:{wid}}'.format(ii, wid = len(str(len(X1)))), '/', len(X1), end = '\r')
                X1tX2[key1+'@'+key2] = X1[key1].T @ X2[key2]
                if type(X1tX2[key1+'@'+key2]) in {sp.sparse.csc_matrix, sp.sparse.csr_matrix}:
                    X1tX2[key1+'@'+key2] = sp.sparse.csr_matrix(X1tX2[key1+'@'+key2])
                assert type(X1tX2[key1+'@'+key2]) in {np.ndarray, sp.sparse.csr_matrix}
    print()
    print('Start X2tX2')
    X2tX2 = {}
    # Products between bivariate covariates
    for ii, key1 in enumerate(X2):
        for jj, key2 in enumerate(X2):
            if key1 <= key2:
                if approx_tf.cross_mask(mask2, key1, mask2, key2) or all_products:
                    print('\r{0:{wid}}'.format(jj, wid = len(str(len(X2)))), '/', len(X2), ' - ', '{0:{wid}}'.format(ii, wid = len(str(len(X2)))), '/', len(X2), end = '\r')
                    X2tX2[key1+'@'+key2] = X2[key1].T @ X2[key2]
                    if type(X2tX2[key1+'@'+key2]) in {sp.sparse.csc_matrix, sp.sparse.csr_matrix}:
                        X2tX2[key1+'@'+key2] = sp.sparse.csr_matrix(X2tX2[key1+'@'+key2])
                    assert type(X2tX2[key1+'@'+key2]) in {np.ndarray, sp.sparse.csr_matrix}
    print()
    return X1tX2, X2tX2


def func1d(x, func, cyclic = None, bd_scal = 0, hrch = None, center = False):
    # Computet for the different values stored in x the values of the spline function (defined by a set of nodes) stored in func
    assert hrch != None
    if len(func) == 1:
        assert func[0] == int(func[0])
        return x**func[0]
    elif len(func) == 3:
        aa, bb, dd = func
        assert (dd >= bb and bb >= aa) or cyclic
        assert not bd_scal # Procedure should be checked
        y  = x
        zz = y  + (y <=aa)*cyclic 
        ee = dd + (dd<=aa)*cyclic 
        cc = bb + (bb<=aa)*cyclic 
        assert ee > cc
        assert cc > aa
        np.allclose(cc - aa, ee - cc)
        del bb, dd, y, x
        f1 = 0
        f2 = (zz-aa)/(cc-aa)
        f3 = (zz-ee)/(cc-ee)
        if not cyclic:
            if cc == 0:
                v = np.clip(f3,
                            a_min = f1,
                            a_max = None,
                            )
            elif aa == 0:
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
            elif cc == 1:
                v = np.clip(f2,
                            a_min = f1,
                            a_max = None,
                            )
            elif ee == 1:
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
            else:
                assert aa > 0 and ee < 1
                v  = np.clip(np.clip(f2,
                                     a_min = None,
                                     a_max = f3,
                                     ), 
                             a_min = f1, 
                             a_max = None,
                             )
        else:
            v  = np.clip(np.clip((zz-aa)/(cc-aa),
                         a_min = None,
                         a_max = (zz-ee)/(cc-ee),
                         ),
                 a_min = 0,
                 a_max = None,
                 )
        if center:
            v -= (cc - aa)
        ### Checks and factor
        if bd_scal:
            if   aa<0 and bd_scal:
                assert bb > 0
                assert dd >0
                fac = (bb-aa)/bb
            elif dd>1 and bd_scal:
                assert aa < 1
                assert bb < 1
                fac = (bb-aa)/(1-aa)
        #else:
        if hrch:
            assert cc > aa 
            assert cc - aa <= 1
            fac = (1/(cc - aa))*hrch**(-np.log(cc - aa)) # If hrch = 1, they all cost the same thing
            assert fac > 0
        else:
            fac = 1
    elif len(func) == 4:
        aa, bb, dd, ff = func
        y  = x#np.clip(x,0,1)
        zz = x  + (x <=aa)*cyclic 
        gg = ff + (ff<=aa)*cyclic 
        ee = dd + (dd<=aa)*cyclic 
        cc = bb + (bb<=aa)*cyclic 
        C  = 1/(cc - aa)
        del bb, dd, ff, y, x
        f1 =    np.clip(C*(zz-aa), a_min = 0, a_max = None)**2
        f2 = -3*np.clip(C*(zz-cc), a_min = 0, a_max = None)**2
        f3 =  3*np.clip(C*(zz-ee), a_min = 0, a_max = None)**2
        f4 = -  np.clip(C*(zz-gg), a_min = 0, a_max = None)**2
        v  = 0.5*(f1+f2+f3+f4)
        ### Checks and factor
        assert (ff >= dd and dd >= bb and bb >= aa) or cyclic
        assert not bd_scal # Procedure should be checked
        fac = 1
    else:
        raise ValueError         
    #return (1/C)*v*fac 
    return v*fac 
        
