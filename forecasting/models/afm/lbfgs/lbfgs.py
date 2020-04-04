#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import scipy as sp
from termcolor import colored
#
from .lbfgs_tools import bfgs_ridge, grad_bfgs_ridge, cat_bound, make_coef


EXTRA_CHECK = 0


"""
This script contains different functions used for optimization with the function sp.optimize.fmin_l_bfgs_b in lbfgs_tools.py
This function requires that all the coefficients are stored in a unique vector (for all the substations)
Most of the code is dedicated to reshaping this vector, managing the covariates shared by all the substations 
and the covariates usef by a subset of the substations.
The evaluation and the gradient computations are also decomposed in three parts : 
the data-fitting term, the sum-consistent term and the regularization
"""


#profile
def start_lbfgs(model, ):
    print('start lbfgs')
    # Check no column update
    for k, v in model.col_upd.items():
        assert not v
    if (model.gp_pen and bool(model.param['gp_matrix'])): # Use old if gp_pen because the structure sharde/owned seems unadapted in this case
        print(colored(' \nGP PEN IN LBFGS\n', 'red', 'on_green'))
    # Remove useless data
    for key in list(model.X_train.keys()):
        if '#' in key:
            if key.split('#') != sorted(key.split('#')):
                         # There is no need to have the interactions key1#key2 and key2#key1 since there is no imposed structure here
                del model.X_train[key], model.X_test[key]
    # Check that the regularizations are defined as ridge or smoothing splines or nothing
    for k, d in model.pen.items():
        for a, b in d.items():
            assert b in {'rsm', 'r1sm', 'r2sm', 'rs2m', '', 'row2sm'}
        
    for key in model.sorted_keys:
        model.keys['lbfgs_coef'].append(key)
    
    # These dictionaries are used to locate the different covariates in the unique long vector     
    model.width_col              = [model.X_train[key].shape[1] for key in model.sorted_keys]
    model.key_col_large_matching = {key:(int(np.sum(model.width_col[:ii])), int(np.sum(model.width_col[:ii+1]))) for ii, key in enumerate(model.sorted_keys)}
    model.col_key_large_matching = []
    
    for key in model.key_col_large_matching:
        model.col_key_large_matching += [key for ii in range(*model.key_col_large_matching[key])]
    model.col_key_large_matching = np.array(model.col_key_large_matching).reshape((-1,1))
    
    for key in model.key_col_large_matching:
        assert model.key_col_large_matching[key][0] < model.key_col_large_matching[key][1], (key, model.key_col_large_matching[key][0], model.key_col_large_matching[key][1])
    
    model.col_large_cat_matching = np.concatenate([[model.param['data_cat'][key] 
                                                    for ii in range(int(model.key_col_large_matching[key][1] - model.key_col_large_matching[key][0]))
                                                    ]
                                                   for key in model.sorted_keys
                                                   ])
    ### masks
    print('start masks')
    model.concat_masks = []
    for pp in range(model.k):
        print('\r'+str(pp), '/', model.k, end = '\r')
        bool_sel = []
        for key in model.sorted_keys:
            mm = model.mask.get(key, slice(None))
            if (type(mm) == slice and mm == slice(None)) or (type(mm) == np.ndarray and pp in mm):
                bool_sel += [1]*model.X_train[key].shape[1]
            else:
                bool_sel += [0]*model.X_train[key].shape[1]
        model.concat_masks.append(sp.sparse.csc_matrix(bool_sel).reshape((-1, 1)))
    model.concat_masks  = sp.sparse.hstack(model.concat_masks, format = 'csc')
    model.nb_covariates = model.concat_masks[:,0].sum()
    for pp in range(model.concat_masks.shape[1]):
        assert model.concat_masks[:,pp].sum() == model.nb_covariates # All posts have the same number of covariates
    
    # There are covariates shared by all the substations and covariates used by a subset of the substations
    # They are separated inn the computations to improve the speed of the algorithm
    model.concat_range_shared   = np.array([not model.cat_owned[model.col_large_cat_matching[ii]] 
                                            for ii in range(model.col_large_cat_matching.shape[0])
                                            ])
    
    assert (model.concat_range_shared[:-1].astype(int) - model.concat_range_shared[1:].astype(int) >= 0).all() # Shared then owned variables
    ind_shared                = np.arange(model.concat_range_shared.shape[0])[model.concat_range_shared]
    model.concat_slice_shared = slice(ind_shared.min(), 
                                      ind_shared.max()+1, 
                                      )
    model.submasks_shared = model.concat_masks[model.concat_slice_shared].toarray().astype(bool) 

    if sp.sparse.issparse(model.concat_masks):
        model.concat_large_masks_owned = sp.sparse.csc_matrix(model.concat_masks.multiply(1 - model.concat_range_shared[:,None]))
        model.concat_large_masks_owned.eliminate_zeros()
        ind_owned  = np.arange(model.concat_large_masks_owned.shape[0])[np.array(model.concat_large_masks_owned.sum(axis = 1).astype(bool))[:,0]]
    else:
        model.concat_large_masks_owned = np.multiply(model.concat_masks, 1 - model.concat_range_shared[:,None]).astype(bool)        
        ind_owned  = np.arange(model.concat_large_masks_owned.shape[0])[model.concat_large_masks_owned.sum(axis = 1).astype(bool)]

    if np.prod(ind_owned.shape)>0:
        assert ind_shared.max() < ind_owned.min()
        model.concat_slice_owned = slice(ind_shared.max()+1,
                                         ind_owned.max()+1, 
                                         )
    else:
        model.concat_slice_owned = slice(0,0)

    ### data
    print('start data')
    model.sparseX = model.param.get('sparse_x1', False) or model.param.get('sparse_x2', False)
    if model.param.get('sparse_x1', False):
        print(colored('X1_SPARSE', 'green'))
    if model.param.get('sparse_x2', False):
        print(colored('X2_SPARSE', 'green'))

    # Concatenation of the covariates in a single matrix
    if model.sparseX:
        model.concat_train_csc = sp.sparse.hstack([model.X_train[key]
                                                   for key in model.sorted_keys
                                                   ], 
                                                  format = 'csc',
                                                  )
        model.concat_train_csr = sp.sparse.csr_matrix(model.concat_train_csc)
        
        model.concat_shared_train = model.concat_train_csc[:,model.concat_slice_shared]
        
        model.concat_test_csc     = sp.sparse.hstack([model.X_test[key]
                                                      for key in model.sorted_keys
                                                      ], 
                                                     format = 'csc',
                                                     )
        model.concat_shared_test  = model.concat_test_csc[:,model.concat_slice_shared]
        
    else:
        model.concat_train        = np.concatenate([model.X_train[key]
                                                    for key in model.sorted_keys
                                                    ], 
                                                   axis = 1, 
                                                   )
        model.concat_shared_train = model.concat_train[:,model.concat_slice_shared]
        model.concat_owned_train  = np.concatenate([model.concat_train[:,model.concat_large_masks_owned[:,k].indices][:,:,None] 
                                                    for k in range(model.concat_large_masks_owned.shape[1])
                                                    ], axis = 2)
        model.concat_test         = np.concatenate([model.X_test[key] 
                                                    for key in model.sorted_keys
                                                    ], 
                                                   axis = 1, 
                                                   )
        model.concat_shared_test  = model.concat_test[:,model.concat_slice_shared]

    model.precompute = model.param['lbfgs_precompute']

    model.ny2_train      = (1/model.n_train)*np.linalg.norm(model.Y_train)**2
    if model.active_gp:
        model.ny2_sum_train  = {tuple_indices : (1/model.n_train)*np.linalg.norm(model.Y_train[:,list(tuple_indices)].sum(axis = 1))**2
                                for tuple_indices in model.partition_tuples
                                }
    
    # Computation of the XtX and XtY part
    print('model.sparseX : ', model.sparseX)
    if model.sparseX:
        print('precompute xshtxsh')
        model.nxtx_sh_train = (1/model.n_train)*model.concat_shared_train.T @ model.concat_shared_train
        print('\n'+'precompute xshtxy')
        model.nxshty_train  = (1/model.n_train)*model.concat_shared_train.T @ model.Y_train
        print('precompute xowtxy')
        model.nxowty_train     = np.concatenate([(1/model.n_train)*model.concat_train_csc[:,model.concat_large_masks_owned[:,k].indices].T@model.Y_train[:,k][:,None]
                                                 for k in range(model.k)
                                                 ], 
                                                axis = 1,
                                                )
        if model.precompute or True: # The question of precomputations is mainly relevant for gp_pen
            print('precompute xowtxow')
            model.nxtx_ow_train = {}
            for k in range(model.k):
                print('\r    '+'{0:5} / {1:5}'.format(k, model.k), end = '')
                model.nxtx_ow_train[k] = (1/model.n_train) * model.concat_train_csc[:,model.concat_large_masks_owned[:,k].indices].T @ model.concat_train_csc[:,model.concat_large_masks_owned[:,k].indices]

            print('\n'+'precompute xowtxsh')
            model.nxtx_owsh_train         = {}
            for k in range(model.k):
                print('\r    '+'{0:5} / {1:5}'.format(k, model.k), end = '')
                model.nxtx_owsh_train[k] = (1/model.n_train)*model.concat_train_csc[:,model.concat_large_masks_owned[:,k].indices].T@(model.concat_shared_train if model.active_gp else model.concat_shared_train.multiply(model.submasks_shared[:,k].reshape((1,-1))))
    
            print('\n'+'precompute xshtxow')
            model.nxtx_show_train = {}
            for k in range(model.k):
                print('\r    '+'{0:5} / {1:5}'.format(k, model.k), end = '')
                if   type(model.nxtx_owsh_train[k]) == sp.sparse.csr_matrix :
                    model.nxtx_show_train[k] = sp.sparse.csr_matrix(model.nxtx_owsh_train[k].T)
                elif type(model.nxtx_owsh_train[k]) == sp.sparse.csc_matrix :
                    model.nxtx_show_train[k] = sp.sparse.csc_matrix(model.nxtx_owsh_train[k].T)
                else:
                    model.nxtx_show_train[k] = model.nxtx_owsh_train[k].T
        
        # Computations of the XtX and XtY parts for the sum-consistent model        
        if model.active_gp:            
            print('\n'+'precompute xshty_sum')
            model.nxshty_sum_train = {tuple_indices : model.nxshty_train[:,list(tuple_indices)].sum(axis = 1)
                                      for tuple_indices in model.partition_tuples
                                      }
            
            print('\n'+'precompute xowtxy_large_sum')

            model.nxowty_large_sum_train = {}
            for ii, tuple_indices in enumerate(model.partition_tuples):
                model.nxowty_large_sum_train[tuple_indices] = np.zeros((model.concat_train_csc[:,model.concat_large_masks_owned[:,0].indices].shape[1], 
                                                                        model.k, 
                                                                        ))

                for pp in tuple_indices:
                    print('\r' + '{0:5} / {1:5} - {2:5} / {3:5}'.format(ii, 
                                                                        len(model.partition_tuples), 
                                                                        pp, 
                                                                        model.k, 
                                                                        ), end = '')
                    model.nxowty_large_sum_train[tuple_indices][:,pp] = (1/model.n_train) * (  model.concat_train_csc[:,model.concat_large_masks_owned[:,pp].indices].T
                                                                                             @ model.Y_train[:,list(tuple_indices)].sum(axis = 1)
                                                                                             )
            model.part_xowty = np.concatenate([np.sum([(len(model.partition_tuples_to_posts[tuple_indices])/len(tuple_indices)**2) * 
                                                       model.nxowty_large_sum_train[tuple_indices][:,pp]
                                                       for tuple_indices in model.partition_tuples
                                                       if pp in tuple_indices
                                                       ], 
                                                      axis = 0,
                                                      )[:,None]
                                                for pp in range(model.k)
                                                ], 
                                               axis = 1, 
                                               )
            
            if model.precompute:
                print('\n'+'precompute xowtxow_large')
                model.nxtx_ow_large_train = {}
                counter_xowtxow = 0
                for ii, tuple_indices in enumerate(model.partition_tuples):
                    for k in tuple_indices:
                        for l in tuple_indices:
                            print('\r    '+'{0:5} / {1:5} - {2:5} / {3:5} - {4:5} / {5:5} - counter = {6:5}'.format(
                                                                                  ii,  len(model.partition_tuples), 
                                                                                  k,  model.k, 
                                                                                  l, model.k,
                                                                                  counter_xowtxow,
                                                                                  ), end = '') 
                            if (k,l) not in model.nxtx_ow_large_train:
                                counter_xowtxow += 1
                                model.nxtx_ow_large_train[k,l] = (1/model.n_train)*(
                                                                                      model.concat_train_csc[:,model.concat_large_masks_owned[:,k].indices].T
                                                                                    @ model.concat_train_csc[:,model.concat_large_masks_owned[:,l].indices]
                                                                                    )
                                model.nxtx_ow_large_train[k,l] = sp.sparse.csr_matrix(model.nxtx_ow_large_train[k,l])
            else:
                pass

    else:
        print('Precomputations')
        model.nxtx_sh_train = (1/model.n_train)*model.concat_shared_train.T @ model.concat_shared_train
        print('precompute xowtxow')
        model.nxtx_ow_train       = (1/model.n_train)*np.einsum('npk,nqk->pqk', 
                                                                model.concat_owned_train, 
                                                                model.concat_owned_train,
                                                                optimize = True
                                                                )
        if model.active_gp:
            model.nxtx_ow_large_train       = (1/model.n_train)*np.einsum('npk,nql->pqkl', 
                                                                          model.concat_owned_train, 
                                                                          model.concat_owned_train,
                                                                          optimize = True
                                                                          )
        print('precompute xowtxsh')
        model.nxtx_owsh_train = (1/model.n_train)*np.einsum('npk,nq->pqk', 
                                                            model.concat_owned_train, 
                                                            model.concat_shared_train
                                                            )
        model.nxtx_show_train  = model.nxtx_owsh_train.transpose(1,0,2)
        print('precompute xshtxy')
        model.nxshty_train     = (1/model.n_train)* model.concat_shared_train.T @ model.Y_train
        print('precompute xshty_sum')
        model.nxshty_sum_train = (1/model.n_train)* model.concat_shared_train.T @ model.Y_train.sum(axis = 1)
        print('precompute xowtxy')
        model.nxowty_train     = (1/model.n_train)*np.einsum('npk,nk->pk', 
                                                             model.concat_owned_train, 
                                                             model.Y_train
                                                             )
        if model.active_gp:
            print('precompute xowtxy_large')
            model.nxowty_large_train = (1/model.n_train)*np.einsum('npk,nl->pkl', 
                                                                   model.concat_owned_train, 
                                                                   model.Y_train
                                                                   )
            print('precompute xowtxy_large_sum')
            model.nxowty_large_sum_train = model.nxowty_large_train.sum(axis = 2)
            
            model.part_xowty = np.concatenate([np.sum([(len(model.partition_tuples_to_posts[tuple_indices])/len(tuple_indices)**2) * 
                                                       model.nxowty_large_sum_train[tuple_indices][:,pp]
                                                       for tuple_indices in model.partition_tuples
                                                       if pp in tuple_indices
                                                       ], 
                                                      axis = 0,
                                                      )[:,None]
                                                for pp in range(model.k)
                                                ], 
                                               axis = 1, 
                                               )
        del model.concat_owned_train
    
    # Locate the covariates in the concatenated matrices and vectors
    print('start col_cat')    
    model.col_cat_matching = list(model.col_large_cat_matching[model.concat_slice_shared]) + list(model.col_large_cat_matching[model.concat_large_masks_owned[:,0].indices])
    for k in range(model.k):
        assert model.col_cat_matching == list(model.col_large_cat_matching[model.concat_slice_shared]) + list(model.col_large_cat_matching[model.concat_large_masks_owned[:,k].indices])
    model.col_cat_matching   = np.array(model.col_cat_matching)
    model.cat_bound_matching = cat_bound(model.col_cat_matching)
    
    ###
    model.idx_key_matching = np.concatenate([model.col_key_large_matching[model.concat_masks[:,k].indices] 
                                             for k in range(model.concat_masks.shape[1])
                                             ], 
                                            axis = 1,
                                            )
    
    model.key_slice_matching_zero = {}
    for key in model.sorted_keys:
        for k in range(1):
            idx = np.where(model.idx_key_matching[:,k]==key)[0]
            assert idx.ndim == 1
            assert np.all(idx[1:] - idx[:-1] == 1)
            if np.prod(idx.shape[0]):
                model.key_slice_matching_zero[key,k] = slice(
                                                             np.where(model.idx_key_matching[:,k]==key)[0].min(), 
                                                             np.where(model.idx_key_matching[:,k]==key)[0].max()+1, 
                                                             )
    
    vec_coef_0 = np.zeros(((model.concat_shared_train.shape[1] + model.concat_large_masks_owned[:,0].sum())*model.k,1)).reshape((-1, 1)).copy()
    
    # Test functions
    if False:
        _ = loss(vec_coef_0, model)
        _ = grad_loss(vec_coef_0, model)
    ###
 
    # All concatenations and precomputations are done
    # Begin the descent
    model.ans_lbfgs, final_grad, info_lbfgs = make_coef(model, loss, grad_loss, vec_coef_0)

    # Reshape the results in a matrix with different columns for different substations
    ans_reshaped = model.ans_lbfgs.reshape((-1, model.k))
    
    sparse_coef = 0 # Not useful if memory is not a problem
    if sparse_coef:
        model.bfgs_long_coef = sp.sparse.csc_matrix(model.concat_masks.shape)
    else:
        model.bfgs_long_coef = np.zeros(model.concat_masks.shape)
    model.bfgs_long_coef[model.concat_slice_shared] = ans_reshaped[model.concat_slice_shared]

    # Recast the computed coefficients within a larger matrix so that a line corresponds 
    # to one covariate. For instance, the matrix model.bfgs_long_coef has a subset of rows corresponding to 
    # a given weather station and the corresponding coefficients will be nonzero only for the columns corresponding
    # to substations that have access to this weather station
    for k in range( model.bfgs_long_coef.shape[1]):
        model.bfgs_long_coef[model.concat_large_masks_owned[:,k].indices,k] = ans_reshaped[model.concat_slice_owned,k]
    del ans_reshaped
    
    if type(model.bfgs_long_coef) in {sp.sparse.csc_matrix, sp.sparse.csr_matrix}:
        assert sp.sparse.linalg.norm(model.bfgs_long_coef[:model.submasks_shared.shape[0]][~model.submasks_shared]) == 0
        assert len((((model.bfgs_long_coef!=0).astype(int) - sp.sparse.csc_matrix(model.concat_masks).astype(int))>0).data) == 0
    else:
        assert np.linalg.norm(model.bfgs_long_coef[:model.submasks_shared.shape[0]][~model.submasks_shared]) == 0
        assert np.linalg.norm(model.bfgs_long_coef[~model.concat_masks.toarray().astype(bool)]) == 0
    
 
    del vec_coef_0, 
    

#profile    
def loss(vec_coef, model):
    # Compute the loss (including the sum-consistent term and the differentiable regularization)
    coef = vec_coef.reshape((-1, model.k))
    if model.active_gp:
        return bfgs_mse_precomp(model, coef) + bfgs_mse_mean_precomp(model, coef) + bfgs_ridge(model, coef, model.normalized_alphas)
    else:
        return bfgs_mse_precomp(model, coef) + bfgs_ridge(model, coef, model.normalized_alphas)
    

#profile            
def grad_loss(vec_coef, model):
    # Compute the gradient (including the sum-consistent term and the differentiable regularization)
    coef = vec_coef.reshape((-1, model.k))
    grad_mse      = grad_bfgs_mse     (model, coef)
    grad_mse_mean = grad_bfgs_mse_mean(model, grad_mse, coef)
    grad_ridge    = grad_bfgs_ridge   (model, coef, model.normalized_alphas)
    grad_mse     [:model.submasks_shared.shape[0]][~model.submasks_shared] = 0
    if type(grad_mse_mean) != int:
        grad_mse_mean[:model.submasks_shared.shape[0]][~model.submasks_shared] = 0
    if model.active_gp:
        ans = (grad_mse + grad_mse_mean + grad_ridge).reshape(-1)
    else:
        ans = (grad_mse + grad_ridge).reshape(-1)
    if EXTRA_CHECK:
        assert np.linalg.norm(grad_mse[:model.submasks_shared.shape[0]][~model.submasks_shared]) == 0
        assert grad_mse_mean == 0 or np.linalg.norm(grad_mse_mean[:model.submasks_shared.shape[0]][~model.submasks_shared]) == 0
        assert np.linalg.norm(grad_ridge[:model.submasks_shared.shape[0]][~model.submasks_shared]) == 0
    return ans
    
##profile            
def bfgs_pred(model, coef, data = 'train'):
    # Compute the prediction from coef, paying attention to the fact that the substations have access to different covariates
    assert data in {'train', 'test'}
    if data == 'train':
        X_sh = model.concat_shared_train
        if model.sparseX:
            concat = model.concat_train_csc
        else:
            concat = model.concat_train
    elif data == 'test':
        X_sh = model.concat_shared_test
        if model.sparseX:
            concat = model.concat_test_csc
        else:
            concat = model.concat_test
    pred_sh = X_sh @ model.bfgs_long_coef[model.concat_slice_shared]
    if model.sparseX:
        if sp.sparse.issparse(model.concat_large_masks_owned):
            pred_ow = np.concatenate([concat[:,model.concat_large_masks_owned[:,k].indices] @ model.bfgs_long_coef[model.concat_large_masks_owned[:,k].indices,k][:,None] 
                                      for k in range(model.bfgs_long_coef.shape[1])
                                      ], axis = 1)
        else:
            pred_ow = np.concatenate([concat[:,model.concat_large_masks_owned[:,k]] @ model.bfgs_long_coef[model.concat_large_masks_owned[:,k],k][:,None] 
                                      for k in range(model.bfgs_long_coef.shape[1])
                                      ], axis = 1)
    else:
        if sp.sparse.issparse(model.concat_large_masks_owned):
            pred_ow = np.concatenate([concat[:,model.concat_large_masks_owned[:,k].indices]@model.bfgs_long_coef[model.concat_large_masks_owned[:,k].indices,k][:,None] 
                                      for k in range(model.bfgs_long_coef.shape[1])
                                      ], 
                                     axis = 1, 
                                     )
        else:
            pred_ow = np.concatenate([concat[:,model.concat_large_masks_owned[:,k]]@model.bfgs_long_coef[model.concat_large_masks_owned[:,k],k][:,None] 
                                      for k in range(model.bfgs_long_coef.shape[1])
                                      ], 
                                     axis = 1, 
                                     )        
    return pred_sh + pred_ow

#profile            
def bfgs_mse_precomp(model, coef, data = 'train'):
    # Compute the dat-fitting term
    print('In bfgs_mse_precomp - ', end = '')
    assert data == 'train'

    nxtx_sh   = model.nxtx_sh_train
    nxtx_ow   = model.nxtx_ow_train
    nxtx_show = model.nxtx_show_train
    nxshty    = model.nxshty_train
    nxowty    = model.nxowty_train
    ny2       = model.ny2_train

    coef_sh = coef[model.concat_slice_shared]
    coef_ow = coef[model.concat_slice_owned]  
        
    csh_xshtxsh_csh = np.einsum('pk,pk->',
                                coef_sh, 
                                nxtx_sh @ coef_sh,
                                )
    ###
    if model.sparseX:
        csh_xshtxow_cow = 2 * np.sum([coef_sh[:,k].T @ nxtx_show[k] @ coef_ow[:,k] 
                                      for k in range(coef.shape[1])
                                      ])
        ###
        cow_xowtxow_cow = np.sum([coef_ow[:,k].T @ nxtx_ow[k] @ coef_ow[:,k] 
                                  for k in range(coef.shape[1])
                                  ])
        ###
        ytxsh_csh = 2*np.einsum('pk,pk->', 
                                nxshty, 
                                coef_sh,
                                )
        ytxow_cow = 2*np.sum([nxowty[:,k].T @ coef_ow[:,k] 
                              for k in range(coef.shape[1])
                              ]) 
    ###

    else:
        xshtxow_cow = np.einsum('pqk,qk->pk', 
                                nxtx_show,
                                coef_ow
                                ) 
        csh_xshtxow_cow = 2 * np.einsum('pk,pk->',
                                        coef_sh, 
                                        xshtxow_cow,
                                        optimize = True
                                        )
        ###
        xtx_cow = np.einsum('pqk,qk->pk', 
                            nxtx_ow, 
                            coef_ow,)
        cow_xowtxow_cow = np.einsum('pk,pk',
                                    coef_ow, 
                                    xtx_cow,
                                    )
        ###
        ytxsh_csh = 2*np.einsum('pk,pk->', 
                                nxshty, 
                                coef_sh,
                                )
        ytxow_cow = 2*np.einsum('pk,pk->', 
                                nxowty, 
                                coef_ow,
                                )
    print('finished')
    return (0.5)* (  ny2
                   + csh_xshtxsh_csh + cow_xowtxow_cow + csh_xshtxow_cow
                   - ytxsh_csh - ytxow_cow
                   )

#profile            
def bfgs_mse_mean_precomp(model, coef, data = 'train'):
    # Compute the sum-consistent term
    print('In bfgs_mse_mean_precomp - ', end = '')
    if not model.active_gp:
        return 0
    assert data == 'train'
    nxtx_sh          = model.nxtx_sh_train
    nxtx_show        = model.nxtx_show_train
    ny2_sum          = model.ny2_sum_train
    nxshty_sum       = model.nxshty_sum_train
    nxowty_large_sum = model.nxowty_large_sum_train

    coef_ow       = coef[model.concat_slice_owned]

    
    if model.precompute:
        nxtx_ow_large    = model.nxtx_ow_large_train
    else:
        assert model.sparseX
        xow_cow = {k : model.concat_train_csr[:,model.concat_large_masks_owned[:,k].indices] @ coef_ow[:,k]
                   for k in range(coef.shape[1])
                   }
    
    # Coef
        
    """
    model.nxtx_ow_large_train[k,l] = (1/model.n_train)*model.concat_train_csc[:,model.concat_large_masks_owned[:,k].indices].T@ model.concat_train_csc[:,model.concat_large_masks_owned[:,l].indices]
    """
    
    coef_sh_sum     = {}
    csh_xshtxsh_csh = {} 
    csh_xshtxow_cow = {} 
    cow_xowtxow_cow = {} 
    ytxsh_csh       = {} 
    ytxow_cow       = {}
    
    for tuple_indices in model.partition_tuples:
        
        coef_sh_sum    [tuple_indices] = coef[model.concat_slice_shared][:,list(tuple_indices)].sum(axis = 1)
        
        
        csh_xshtxsh_csh[tuple_indices] = np.einsum('p,p->',
                                                   coef_sh_sum[tuple_indices], 
                                                   nxtx_sh @ coef_sh_sum[tuple_indices],
                                                   )

        if model.sparseX:
            ###        
            csh_xshtxow_cow[tuple_indices] = coef_sh_sum[tuple_indices].T @ np.sum([nxtx_show[k] @ coef_ow[:,k]
                                                                for k in tuple_indices
                                                                ], 
                                                                axis = 0,
                                                                )
            ###
            if model.precompute:
                cow_xowtxow_cow[tuple_indices] = np.sum([coef_ow[:,k].T @ (nxtx_ow_large[k,l] @ coef_ow[:,l])
                                                         for k in tuple_indices
                                                         for l in tuple_indices
                                                         ])
            else:
                cow_xowtxow_cow[tuple_indices] = (1/model.n_train)*np.sum([xow_cow[k].T @ xow_cow[l]
                                                                           for k in tuple_indices
                                                                           for l in tuple_indices
                                                                           ])
            ###
            ytxsh_csh[tuple_indices] = nxshty_sum[tuple_indices].T @ coef_sh_sum[tuple_indices]
            ytxow_cow[tuple_indices] = np.sum([nxowty_large_sum[tuple_indices][:,k].T @ coef_ow[:,k] 
                                               for k in tuple_indices
                                               ])      
        else:
            ### 
            xshtxow_cow = np.einsum('pqk,qk->p', 
                                    nxtx_show,
                                    coef_ow
                                    ) 
            csh_xshtxow_cow = np.einsum('p,p',
                                            coef_sh_sum, 
                                            xshtxow_cow,
                                            )
            ###
            nxtx_cow = np.einsum('pqkl,ql->pk', 
                                nxtx_ow_large, 
                                coef_ow,
                                )
            cow_xowtxow_cow = np.einsum('pk,pk->',
                                        coef_ow, 
                                        nxtx_cow,
                                        )
            ###
            ytxsh_csh = np.einsum('p,p->', 
                                  nxshty_sum, 
                                  coef_sh_sum,
                                  )
            ytxow_cow = np.einsum('pk,pk->', 
                                   nxowty_large_sum, 
                                   coef_ow,
                                   )   
        
    return np.sum([(0.5*len(model.partition_tuples_to_posts[tuple_indices])/len(tuple_indices)**2
                    )*(  ny2_sum[tuple_indices]
                       + csh_xshtxsh_csh[tuple_indices]
                       + cow_xowtxow_cow[tuple_indices] 
                       + 2 * csh_xshtxow_cow[tuple_indices]
                       - 2 * ytxsh_csh[tuple_indices]
                       - 2 * ytxow_cow[tuple_indices]
                       )
                   for tuple_indices in model.partition_tuples
                   ])

                    

#profile            
def grad_bfgs_mse(model, coef):
    # Compute the gradient of the data-fitting term
    print('In grad_bfgs_mse - ', end = '')
    nxtx_sh = model.nxtx_sh_train
    nxtx_ow = model.nxtx_ow_train
    nxtx_owsh = model.nxtx_owsh_train
    nxtx_show = model.nxtx_show_train
    nxshty = model.nxshty_train
    nxowty = model.nxowty_train
    coef_sh = coef[model.concat_slice_shared]
    coef_ow = coef[model.concat_slice_owned]
    xtx_csh = nxtx_sh @ coef_sh
    if model.sparseX:
        xtx_cow     = np.concatenate([nxtx_ow[k]@coef_ow[:,k][:,None] 
                                      for k in range(nxshty.shape[1])
                                      ], axis = 1)
        xowtxsh_csh = np.concatenate([nxtx_owsh[k]@coef_sh[:,k][:,None] 
                                      for k in range(nxshty.shape[1])
                                      ], axis = 1)
        xshtxow_cow = np.concatenate([nxtx_show[k]@coef_ow[:,k][:,None]
                                      for k in range(nxshty.shape[1])
                                      ], axis = 1)
    else:
        xtx_cow = np.einsum('pqk,qk->pk', 
                            nxtx_ow, 
                            coef_ow,
                            )
        xowtxsh_csh = np.einsum('pqk,qk->pk', 
                                nxtx_owsh,
                                coef_sh,
                                )
        xshtxow_cow = np.einsum('pqk,qk->pk', 
                                nxtx_show,
                                coef_ow
                                )

    grad = np.concatenate([
                           xtx_csh + xshtxow_cow - nxshty, 
                           xtx_cow + xowtxsh_csh - nxowty, 
                           ], axis = 0)
    print('finished')
    return grad

#profile            
def grad_bfgs_mse_mean(model, grad_mse, coef):
    # Compute the gradient of the sum-consistent term
    if not model.active_gp:
        return 0
    else:
        print('In grad_bfgs_mse_mean - ')#, end = '')
        
        grad_sh_partition = {tuple_indices : grad_mse[model.concat_slice_shared,list(tuple_indices)].sum(axis = 1)
                             for tuple_indices in model.partition_tuples
                             }   
        grad_sh          = np.concatenate([np.sum([(len(model.partition_tuples_to_posts[tuple_indices])/len(tuple_indices)**2) * grad_sh_partition[tuple_indices] 
                                                   for tuple_indices in model.partition_tuples
                                                   if pp in tuple_indices
                                                   ], 
                                                   axis = 0,
                                                   )[:,None]
                                           for pp in range(model.k)
                                           ], 
                                           axis = 1,
                                           )
                
        
        nxtx_owsh        = model.nxtx_owsh_train
        # Coef
        coef_sh          = coef[model.concat_slice_shared]
        coef_ow          = coef[model.concat_slice_owned]
        
        
        if model.precompute:
            print('model.precompute = {0}'.format(model.precompute))
            nxtx_ow_large    = model.nxtx_ow_large_train
        else:
            assert model.sparseX
            xow_cow = {k : model.concat_train_csr[:,model.concat_large_masks_owned[:,k].indices] @ coef_ow[:,k]
                       for k in range(coef.shape[1])
                       }
            
            
        if model.sparseX:
            xowtxsh_csh = np.concatenate([np.sum([ (len(model.partition_tuples_to_posts[tuple_indices])/len(tuple_indices)**2)
                                                  * np.sum([nxtx_owsh[k] @ coef_sh[:,l]
                                                            for l in tuple_indices
                                                            ], 
                                                           axis = 0,
                                                           )
                                                      for tuple_indices in model.partition_tuples
                                                      if k in tuple_indices
                                                      ], 
                                                     axis = 0,
                                                     )[:,None]
                                              for k in range(coef_ow.shape[1])
                                              ], 
                                              axis = 1,
                                              )
            if model.precompute:
                xowtxow_cow = np.concatenate([np.sum([ (len(model.partition_tuples_to_posts[tuple_indices])/len(tuple_indices)**2)
                                                      * np.sum([nxtx_ow_large[k,l] @ coef_ow[:,l] 
                                                                for l in tuple_indices
                                                                ], 
                                                               axis = 0,
                                                               )
                                                      for tuple_indices in model.partition_tuples
                                                      if k in tuple_indices
                                                      ], 
                                                     axis = 0,
                                                     )[:,None]
                                              for k in range(coef_ow.shape[1])
                                              ], 
                                              axis = 1,
                                              )
            else:
                xowtxow_cow = np.concatenate([np.sum([ (len(model.partition_tuples_to_posts[tuple_indices])/len(tuple_indices)**2)
                                                      * np.sum([model.concat_train_csc[:,model.concat_large_masks_owned[:,k].indices].T @ xow_cow[l]
                                                                for l in tuple_indices
                                                                ], 
                                                               axis = 0,
                                                               )
                                                      for tuple_indices in model.partition_tuples
                                                      if k in tuple_indices
                                                      ], 
                                                     axis = 0,
                                                     )[:,None]
                                              for k in range(coef_ow.shape[1])
                                              ], 
                                              axis = 1,
                                              )
                
        else:
            xowtxow_cow = np.einsum('pqkl,ql->pk', 
                                    nxtx_ow_large, 
                                    coef_ow,
                                    )
            xowtxsh_csh = np.einsum('pqk,ql->pk', 
                                    nxtx_owsh,
                                    coef_sh,
                                    )   
        grad_ow = (  xowtxsh_csh 
                   + xowtxow_cow
                   - model.part_xowty
                   )
        
        grad = model.gp_pen * np.concatenate([
                                              grad_sh, 
                                              grad_ow, 
                                              ], 
                                             axis = 0,
                                             )
        print('finished')
        return grad
