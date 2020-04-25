
import os
import numpy as np
import scipy.signal  as sig
import scipy.ndimage as spim
import scipy as sp
from termcolor import colored
from numbers   import Number
#
from electricityLoadForecasting import tools, paths

path_betas = os.path.join(paths.outputs, 'Betas')

def cat_bound(col_cat_matching):
    # Location of the different categories of variables in the design matrix
    cat_bound_matching = {cat:(np.min([ii for ii, row in enumerate(col_cat_matching) if tuple(row) == cat]), 
                               np.max([ii for ii, row in enumerate(col_cat_matching) if tuple(row) == cat])+1,
                               ) 
                          for cat in sorted(set([e for e in col_cat_matching]),
                                            key = lambda x : str(x),
                                            )
                          }
    # Check that the rows of the design matrix are well ordered
    for cat1, v1 in cat_bound_matching.items():
        for cat2, v2 in cat_bound_matching.items():
            if cat1 != cat2:
                assert (   (v1[0] < v2[0] and v1[1] < v2[1]) 
                        or (v1[0] > v2[0] and v1[1] > v2[1]) 
                        )
    return cat_bound_matching

            
def bfgs_regularization(model,
                        coef,
                        alphas,
                        ):
    reg = 0
    # Compute the differentiable penalizations
    # Simple computations for Ridge
    # Computations of the convolutions first for the smoothing splines penalizations
    for var in model.formula.index.get_level_values('coefficient'):
        for inpt, location in model.col_key_large_matching.loc[model.concat_masks[:,0].indices 
                                                               if sp.sparse.issparse(model.concat_masks) 
                                                               else 
                                                               model.concat_masks[:,0],
                                                               0
                                                               ].unique():
            if (inpt, location,0) not in model.key_slice_matching_zero:
                continue
            import ipdb; ipdb.set_trace()
            key_slice      = model.key_slice_matching_zero[(inpt, location),0]
            alpha          = alphas[var].get(inpt,0)
            reshape_tensor = (type(inpt[0]) == tuple) 
            cc             = coef[key_slice]
            if reshape_tensor:
                inpt1, inpt2 = inpt
                cc           = cc.reshape(*model.size_tensor_bivariate[inpt,location], -1)
            if (   type(alpha) == tuple and alpha[0] !=0
                or isinstance(alpha, Number) and alpha != 0
                ):
                ##### Ridge #####
                if   model.pen[var].get(inpt) == 'ridge':
                    conv  = cc
                    reg += 0.5 * alpha * np.linalg.norm(conv)**2
                ##### smoothing-splines regularization #####
                elif model.pen[var].get(inpt) == 'smoothing_reg':
                    if not reshape_tensor:
                        if cc.shape[0] > 2:
                            ker  = np.array([[1],[-2],[1]])
                            if model.hprm['inputs.cyclic'].get(inpt):
                                conv = spim.filters.convolve(cc,ker,mode = 'wrap')
                            else:
                                conv = sig.convolve(cc, ker, mode = 'valid')
                            reg += 0.5 * alpha * np.linalg.norm(conv)**2
                    else:
                        
                        if cc.shape[0] > 2:
                            ker1  = np.array([[[1]],[[-2]],[[1]]]) 
                            if model.hprm['inputs.cyclic'].get(inpt1):
                                conv1 = spim.filters.convolve(cc,ker1,mode = 'wrap')
                            else:
                                conv1 = sig.convolve(cc, ker1, mode = 'valid')
                            reg += 0.5 * alpha * np.linalg.norm(conv1)**2
                        if cc.shape[1] > 2:
                            ker2  = np.array([[[1 ],[ -2 ],[ 1]]])
                            if model.hprm['inputs.cyclic'].get(inpt2):
                                conv2 = spim.filters.convolve(cc,ker2,mode = 'wrap')
                            else:
                                conv2 = sig.convolve(cc, ker2, mode = 'valid')
                            reg += 0.5 * alpha * np.linalg.norm(conv2)**2  
                ##### block smoothing-splines regularization #####
                elif model.pen[var][inpt]   == 'block_smoothing_reg':
                    assert type(model.alpha[var].get(inpt)) == tuple
                    assert len(model.alpha[var][inpt]) == 2
                    aa, pp = model.alpha[var][inpt]
                    if not reshape_tensor:
                        if cc.shape[0] > 2:
                            ker  = np.array([[1],[-2],[1]])
                            if model.hprm['inputs.cyclic'].get(inpt):
                                conv = spim.filters.convolve(cc,ker,mode = 'wrap')
                            else:
                                conv = sig.convolve(cc, ker, mode = 'valid')
                            reg += 0.5 * aa * (np.linalg.norm(conv, axis = 1)**pp).sum()
                    else:
                        if cc.shape[0] > 2:
                            ker1  = np.array([[[1]],[[-2]],[[1]]]) 
                            if model.hprm['inputs.cyclic'].get(inpt1):
                                conv1 = spim.filters.convolve(cc,ker1,mode = 'wrap')
                            else:
                                conv1 = sig.convolve(cc, ker1, mode = 'valid')
                            reg += 0.5 * aa * (np.linalg.norm(conv1, axis = 2)**pp).sum()
                        if cc.shape[1] > 2:
                            ker2  = np.array([[[1 ],[ -2 ],[ 1]]])
                            if model.hprm['inputs.cyclic'].get(inpt2):
                                conv2 = spim.filters.convolve(cc,ker2,mode = 'wrap')
                            else:
                                conv2 = sig.convolve(cc, ker2, mode = 'valid')
                            reg += 0.5 * aa * (np.linalg.norm(conv2, axis = 2)**pp).sum()
                else:
                    assert model.pen.get(inpt,'') == ''
                    conv = 0
                    reg += 0
    return reg
                 

def grad_bfgs_regularization(model, coef, alphas):
    # Compute the gradient of the differentiable plenalizations
    # The slmoothing spline regularizations require the computations of the convolutions first
    grad = np.zeros(coef.shape)
    for var in model.formula.index.get_level_values('coefficient'):
        for inpt, location in model.col_key_large_matching.loc[model.concat_masks[:,0].indices 
                                                               if sp.sparse.issparse(model.concat_masks) 
                                                               else 
                                                               model.concat_masks[:,0],
                                                               0
                                                               ].unique():
            if ((inpt,location),0) not in model.key_slice_matching_zero:
                continue
            key_slice      = model.key_slice_matching_zero[(inpt,location),0]
            alpha          = alphas[var].get(inpt,0)
            reshape_tensor = (type(inpt[0]) == tuple) 
            cc             = coef[key_slice]
            if reshape_tensor:
                inpt1, inpt2 = inpt
                cc           = cc.reshape(*model.size_tensor_bivariate[inpt,location], -1)
                grad_tmp     = np.zeros(cc.shape)
            if (   type(alpha) == tuple and alpha[0] !=0
                or isinstance(alpha, Number) and alpha != 0
                ):
                ##### Ridge #####
                if model.pen[var].get(inpt, 0) == 'ridge':
                    grad[key_slice] = alpha * cc.reshape((grad[key_slice].shape))
                ##### smoothing-splines regularization #####
                elif model.pen[var].get(inpt) == 'smoothing_reg':
                    assert key_slice.stop >= key_slice.start+2
                    if not reshape_tensor:
                        if cc.shape[0] > 2:
                            if model.hprm['inputs.cyclic'].get(inpt):
                                ker             = np.array([[1],[-4],[6],[-4],[1]])
                                conv            = spim.filters.convolve(cc,ker,mode = 'wrap')
                                grad[key_slice] = alpha * conv
                            else:
                                
                                ker  = np.array([[1],[-2],[1]])
                                conv = sig.convolve(cc, ker, mode = 'valid')
                                grad[slice(key_slice.start,   key_slice.stop-2)] += alpha * conv
                                grad[slice(key_slice.start+1, key_slice.stop-1)] -= alpha * 2*conv
                                grad[slice(key_slice.start+2, key_slice.stop  )] += alpha * conv
                    else:
                        if cc.shape[0] > 2:
                            if model.hprm['inputs.cyclic'].get(inpt1):
                                ker1      = np.array([[[1],[-4],[6],[-4],[1]]])
                                conv1     = spim.filters.convolve(cc,ker1,mode = 'wrap')
                                grad_tmp += alpha * conv1
                            else:
                                ker1              = np.array([[[1]],[[-2]],[[ 1]]])
                                conv1             = sig.convolve(cc, ker1, mode = 'valid')
                                grad_tmp[ :-2,:] += alpha * conv1
                                grad_tmp[1:-1,:] -= alpha * 2*conv1
                                grad_tmp[2:  ,:] += alpha * conv1
                        if cc.shape[1] > 2: 
                            if model.hprm['inputs.cyclic'].get(inpt2):
                                ker2      = np.array([[[1],[-4],[6],[-4],[1]]])
                                conv2     = spim.filters.convolve(cc,ker2,mode = 'wrap')
                                grad_tmp += alpha * conv2
                            else:
                                ker2              = np.array([[[1],  [-2],  [1]]])
                                conv2             = sig.convolve(cc, ker2, mode = 'valid')
                                grad_tmp[:, :-2] += alpha *   conv2
                                grad_tmp[:,1:-1] -= alpha * 2*conv2
                                grad_tmp[:,2:  ] += alpha *   conv2
                        grad[key_slice] = grad_tmp.reshape((grad[key_slice].shape))
                ##### block smoothing-splines regularization #####
                elif model.pen[var].get(inpt) == 'block_smoothing_reg':
                    assert len(alphas[var][inpt])  == 2, alphas[var][inpt] 
                    assert type(alphas[var][inpt]) == tuple, alphas[var][inpt] 
                    aa, pp = alphas[var][inpt]
                    assert key_slice.stop >= key_slice.start+2
                    if not reshape_tensor:
                        if cc.shape[0] > 2:
                            if model.hprm['inputs.cyclic'].get(inpt):
                                ker           = np.array([[1],[-2],[1]])
                                conv          = spim.filters.convolve(cc,ker,mode = 'wrap')
                                norm_conv     = np.linalg.norm(conv, axis = 1)
                                norm_conv_pm2 = np.where(norm_conv, 
                                                         norm_conv**(pp-2), 
                                                         np.zeros(norm_conv.shape),
                                                         )[:,None]
                                grad[key_slice] = (0.5 * aa * pp) * (- 2 * conv * norm_conv_pm2
                                                                     + np.concatenate([conv[1:],   conv[[0]]], axis = 0) * np.concatenate([norm_conv_pm2[ 1:], 
                                                                                                                                           norm_conv_pm2[[0]], 
                                                                                                                                           ], 
                                                                                                                                           axis = 0, 
                                                                                                                                           )
                                                                     + np.concatenate([conv[[-1]], conv[:-1]], axis = 0) * np.concatenate([norm_conv_pm2[[-1]], 
                                                                                                                                           norm_conv_pm2[:-1]], 
                                                                                                                                           axis = 0, 
                                                                                                                                           )
                                                                     )
                            else:
                                ker           = np.array([[1],[-2],[1]])
                                conv          = sig.convolve(cc, ker, mode = 'valid')
                                norm_conv     = np.linalg.norm(conv, axis = 1)
                                norm_conv_pm2 = np.where(norm_conv, 
                                                         norm_conv**(pp-2), 
                                                         np.zeros(norm_conv.shape),
                                                         )[:,None]
                                grad[slice(key_slice.start,   key_slice.stop-2)] += (0.5 * aa * pp) *   conv * norm_conv_pm2
                                grad[slice(key_slice.start+1, key_slice.stop-1)] -= (0.5 * aa * pp) * 2*conv * norm_conv_pm2
                                grad[slice(key_slice.start+2, key_slice.stop  )] += (0.5 * aa * pp) *   conv * norm_conv_pm2
                    else:
                        if cc.shape[0] > 2:
                            if model.hprm['inputs.cyclic'].get(inpt1):
                                ker1           = np.array([[[ 1],
                                                            [-4],
                                                            [ 6],
                                                            [-4],
                                                            [ 1], 
                                                            ]])
                                conv1          = spim.filters.convolve(cc,ker1,mode = 'wrap')
                                norm_conv1     = np.linalg.norm(conv1, axis = 2)
                                norm_conv1_pm2 = np.where(norm_conv1, 
                                                          norm_conv1**(pp-2), 
                                                          np.zeros(norm_conv1.shape),
                                                          )[:,:,None]
                                grad_tmp += (0.5 * aa * pp) * (- 2 * conv1 * norm_conv1_pm2
                                                               + np.concatenate([conv1[ 1:], 
                                                                                 conv1[[0]], 
                                                                                 ], 
                                                                                axis = 0, 
                                                                                ) * np.concatenate([norm_conv1_pm2[ 1:], 
                                                                                                    norm_conv1_pm2[[0]], 
                                                                                                    ], 
                                                                                                   axis = 0, 
                                                                                                   ) 
                                                               + np.concatenate([conv1[[-1]], 
                                                                                 conv1[ :-1],
                                                                                 ], 
                                                                                 axis = 0,
                                                                                 ) * np.concatenate([norm_conv1_pm2[[-1]], 
                                                                                                     norm_conv1_pm2[ :-1], 
                                                                                                     ], 
                                                                                                    axis = 0, 
                                                                                                    )
                                                               )
                            else:
                                ker1           = np.array([[[ 1]],
                                                           [[-2]],
                                                           [[ 1]],
                                                           ])
                                conv1          = sig.convolve(cc, ker1, mode = 'valid')
                                norm_conv1     = np.linalg.norm(conv1, axis = 2)
                                norm_conv1_pm2 = np.where(norm_conv1, 
                                                          norm_conv1**(pp-2), 
                                                          np.zeros(norm_conv1.shape),
                                                          )[:,:,None]
                                grad_tmp[ :-2,:] += (0.5 * aa * pp) *   conv1 * norm_conv1_pm2
                                grad_tmp[1:-1,:] -= (0.5 * aa * pp) * 2*conv1 * norm_conv1_pm2
                                grad_tmp[2:  ,:] += (0.5 * aa * pp) *   conv1 * norm_conv1_pm2
                        if cc.shape[1] > 2: 
                            if model.hprm['inputs.cyclic'].get(inpt2):
                                ker2           = np.array([[[ 1],
                                                            [-4],
                                                            [ 6],
                                                            [-4],
                                                            [ 1]
                                                            ]])
                                conv2          = spim.filters.convolve(cc,ker2,mode = 'wrap')
                                norm_conv2     = np.linalg.norm(conv2, axis = 2)
                                norm_conv2_pm2 = np.where(norm_conv2, 
                                                          norm_conv2**(pp-2), 
                                                          np.zeros(norm_conv2.shape),
                                                          )[:,:,None]
                                grad_tmp += (0.5 * aa * pp) * (- 2 * conv2 * norm_conv2_pm2
                                                               + np.concatenate([conv2[:, 1:], 
                                                                                 conv2[:,[0]], 
                                                                                 ], 
                                                                                axis = 1, 
                                                                                ) * np.concatenate([norm_conv2_pm2[:, 1:], 
                                                                                                    norm_conv2_pm2[:,[0]], 
                                                                                                    ], 
                                                                                                   axis = 1, 
                                                                                                   ) 
                                                               + np.concatenate([conv2[:,[-1]], 
                                                                                 conv2[:, :-1], 
                                                                                 ], 
                                                                                 axis = 1, 
                                                                                 ) * np.concatenate([norm_conv2_pm2[:,[-1]], 
                                                                                                     norm_conv2_pm2[:, :-1], 
                                                                                                     ], 
                                                                                                    axis = 1, 
                                                                                                    )
                                                               )
                            else:
                                ker2           = np.array([[[ 1],  
                                                            [-2],  
                                                            [ 1], 
                                                            ]])
                                conv2          = sig.convolve(cc, ker2, mode = 'valid')
                                norm_conv2     = np.linalg.norm(conv2, axis = 2)
                                norm_conv2_pm2 = np.where(norm_conv2, 
                                                          norm_conv2**(pp-2), 
                                                          np.zeros(norm_conv2.shape),
                                                          )[:,:,None]
                                grad_tmp[:, :-2] += (0.5 * aa * pp) *     conv2 * norm_conv2_pm2
                                grad_tmp[:,1:-1] -= (0.5 * aa * pp) * 2 * conv2 * norm_conv2_pm2
                                grad_tmp[:,2:  ] += (0.5 * aa * pp) *     conv2 * norm_conv2_pm2
                        grad[key_slice] = grad_tmp.reshape((grad[key_slice].shape))
                else:
                    assert model.pen.get(inpt,'') == ''            
    return grad                    


def optimize_coef(model,
                  loss,
                  grad_loss,
                  vec_coef_0,
                  ):
    # Optimization of the differentiable objectives with the function sp.optimize.fmin_l_bfgs_b
    try:
        # Try to load
        ans_lbfgs = tools.batch_load(path_betas, 
                                     model.dikt['experience.whole'], 
                                     obj_name = 'ans_lbfgs', 
                                     mod      = 'np',
                                     )
        print(colored('ans_lbfgs loaded', 'green'))
    except:
        # Compute and save
        print(colored('ans_lbfgs not loaded', 'red'))
        model.machine_precision = np.finfo(float).eps
        model.tol_lbfgs         = model.hprm['afm.algorithm.lbfgs.tol']
        model.factr             = model.k*model.tol_lbfgs/model.machine_precision
        print('{0:20.20} : {1:.3e}\n{2:20.20} : {3:.3e}\n{4:20.20} : {5:.3e}'.format('machine_precision',       model.machine_precision, 
                                                                                     'afm.algorithm.lbfgs.tol', model.tol_lbfgs, 
                                                                                     'model.factr',             model.factr,
                                                                                     )) 
        ans_lbfgs  = sp.optimize.fmin_l_bfgs_b(
                                               loss, 
                                               vec_coef_0, 
                                               fprime  = grad_loss, 
                                               args    = (model,),
                                               iprint  = 99, 
                                               factr   = model.factr,
                                               maxfun  = model.hprm['afm.algorithm.lbfgs.maxfun'],
                                               maxiter = model.hprm['afm.algorithm.lbfgs.maxiter'],
                                               pgtol   = model.hprm['afm.algorithm.lbfgs.pgtol'],
                                               )
        try:
            tools.batch_save(
                             path_betas, 
                             data      = ans_lbfgs, 
                             prefix    = model.dikt['experience.whole'], 
                             data_name = 'ans_lbfgs', 
                             data_type = 'np',
                             )      
            print(colored('ans_lbfgs saved', 'green'))
        except Exception as e:
            print(e)
    return ans_lbfgs
        
 
