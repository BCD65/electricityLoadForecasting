
import numpy as np
import scipy.signal  as sig
import scipy.ndimage as spim
import scipy as sp
from termcolor import colored
from numbers   import Number
#
import electricityLoadForecasting.tools  as tools


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
                assert (v1[0] < v2[0] and v1[1] < v2[1]) or (v1[0] > v2[0] and v1[1] > v2[1]) 
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
            key_slice = model.key_slice_matching_zero[(inpt, location),0]
            alpha     = alphas[var].get(inpt,0)
            if (   type(alpha) == tuple and alpha[0] !=0
                or isinstance(alpha, Number) and alpha != 0
                ):
                cc   = coef[key_slice]                      
                ##### rsm #####
                if   model.pen[var].get(inpt) == 'rsm':
                    conv  = cc
                    reg += 0.5 * alpha * np.linalg.norm(conv)**2
                elif model.pen[var].get(inpt) == 'r1sm':
                    reshape_tensor = '#' in inpt and np.all([e > 1 for e in model.size_tensor2[inpt,location]]) 
                    if reshape_tensor:
                        cc = cc.reshape(*model.size_tensor2[inpt], -1)                        
                    if not reshape_tensor:
                        ker  = np.array([[1],[-1]])
                    else:
                        ker = np.array([
                                        [[ 2],[ -1]], 
                                        [[-1],[  0]], 
                                        ])
                    if not reshape_tensor and model.hprm['qo_modulo'].get(inpt):
                        conv = spim.filters.convolve(cc,ker,mode = 'wrap')
                    else:
                        conv = sig.convolve(cc, ker, mode = 'valid')
                    reg += 0.5 * alpha * np.linalg.norm(conv)**2
                ##### r2sm #####
                elif model.pen[var].get(inpt) == 'r2sm':
                    reshape_tensor = (type(inpt[0]) == tuple) 
                    if not reshape_tensor:
                        if cc.shape[0] > 2:
                            ker  = np.array([[1],[-2],[1]])
                            if model.hprm['qo_modulo'].get(inpt):
                                conv = spim.filters.convolve(cc,ker,mode = 'wrap')
                            else:
                                conv = sig.convolve(cc, ker, mode = 'valid')
                            reg += 0.5 * alpha * np.linalg.norm(conv)**2
                    else:
                        cc = cc.reshape(*model.size_tensor2[inpt,location], -1)  
                        inpt1, inpt2 = inpt
                        if cc.shape[0] > 2:
                            ker1  = np.array([[[1]],[[-2]],[[1]]]) 
                            if model.hprm['qo_modulo'].get(inpt1):
                                conv1 = spim.filters.convolve(cc,ker1,mode = 'wrap')
                            else:
                                conv1 = sig.convolve(cc, ker1, mode = 'valid')
                            reg += 0.5 * alpha * np.linalg.norm(conv1)**2
                        if cc.shape[1] > 2:
                            ker2  = np.array([[[1 ],[ -2 ],[ 1]]])
                            if model.hprm['qo_modulo'].get(inpt2):
                                conv2 = spim.filters.convolve(cc,ker2,mode = 'wrap')
                            else:
                                conv2 = sig.convolve(cc, ker2, mode = 'valid')
                            reg += 0.5 * alpha * np.linalg.norm(conv2)**2  
                ##### group r2sm #####
                elif model.pen[var][inpt]   == 'row2sm':
                    assert type(model.alpha[var].get(inpt)) == tuple
                    assert len(model.alpha[var][inpt]) == 2
                    aa, pp = model.alpha[var][inpt]
                    reshape_tensor = (type(inpt[0]) == tuple)
                    if not reshape_tensor:
                        if cc.shape[0] > 2:
                            ker  = np.array([[1],[-2],[1]])
                            if model.hprm['qo_modulo'].get(inpt):
                                conv = spim.filters.convolve(cc,ker,mode = 'wrap')
                            else:
                                conv = sig.convolve(cc, ker, mode = 'valid')
                            reg += 0.5 * aa * (np.linalg.norm(conv, axis = 1)**pp).sum()
                    else:
                        cc = cc.reshape(*model.size_tensor2[inpt,location], -1)  
                        inpt1, inpt2 = inpt
                        if cc.shape[0] > 2:
                            ker1  = np.array([[[1]],[[-2]],[[1]]]) 
                            if model.hprm['qo_modulo'].get(inpt1):
                                conv1 = spim.filters.convolve(cc,ker1,mode = 'wrap')
                            else:
                                conv1 = sig.convolve(cc, ker1, mode = 'valid')
                            reg += 0.5 * aa * (np.linalg.norm(conv1, axis = 2)**pp).sum()
                        if cc.shape[1] > 2:
                            ker2  = np.array([[[1 ],[ -2 ],[ 1]]])
                            if model.hprm['qo_modulo'].get(inpt2):
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
            key_slice = model.key_slice_matching_zero[(inpt,location),0]
            alpha     = alphas[var].get(inpt,0)
            if (   type(alpha) == tuple and alpha[0] !=0
                or isinstance(alpha, Number) and alpha != 0
                ):
                cc = coef[key_slice]
                ##### rsm #####
                if model.pen[var].get(inpt, 0) == 'rsm':
                    grad[key_slice] = alpha * cc
                ##### r1sm #####
                elif model.pen[var].get(inpt) == 'r1sm':
                    reshape_tensor = (type(inpt[0]) == tuple)  and np.all([e > 1 for e in model.size_tensor2[inpt,location]]) 
                    if reshape_tensor:
                        cc = cc.reshape(*model.size_tensor2[inpt,location], -1)
                    if not reshape_tensor and model.hprm['qo_modulo'].get(inpt):
                        ker  = np.array([[-1],[2],[-1]])
                        conv = spim.filters.convolve(cc,ker,mode = 'wrap')
                        grad[key_slice] = alpha * conv
                    else:
                        assert key_slice.stop >= key_slice.start+1
                        if not reshape_tensor:
                            ker  = np.array([[1],[-1]])
                            conv = sig.convolve(cc, ker, mode = 'valid')
                            grad[slice(key_slice.start,   key_slice.stop-1)] += alpha * conv
                            grad[slice(key_slice.start+1, key_slice.stop  )] -= alpha * conv
                        else:
                            cc = cc.reshape(*model.size_tensor2[inpt,location], -1)
                            ker  = np.array(
                                             [[[ 2],[-1]], 
                                             [[-1],[ 0]], 
                                             ])
                            conv = sig.convolve(cc, ker, mode = 'valid')
                            grad_tmp = np.zeros(cc.shape)
                            grad_tmp[:-1,   :] += alpha * conv
                            grad_tmp[  :, :-1] += alpha * conv
                            grad[slice(key_slice.start, key_slice.stop  )] = grad_tmp.reshape((grad[slice(key_slice.start, key_slice.stop  )].shape))
                ##### r2sm #####
                elif model.pen[var].get(inpt) == 'r2sm':
                    reshape_tensor = (type(inpt[0]) == tuple) 
                    assert key_slice.stop >= key_slice.start+2
                    if not reshape_tensor:
                        if cc.shape[0] > 2:
                            if model.hprm['qo_modulo'].get(inpt):
                                ker  = np.array([[1],[-4],[6],[-4],[1]])
                                conv = spim.filters.convolve(cc,ker,mode = 'wrap')
                                grad[key_slice] = alpha * conv
                            else:
                                
                                ker  = np.array([[1],[-2],[1]])
                                conv = sig.convolve(cc, ker, mode = 'valid')
                                grad[slice(key_slice.start,   key_slice.stop-2)] += alpha * conv
                                grad[slice(key_slice.start+1, key_slice.stop-1)] -= alpha * 2*conv
                                grad[slice(key_slice.start+2, key_slice.stop  )] += alpha * conv
                    else:
                        cc = cc.reshape(*model.size_tensor2[inpt,location], -1)
                        grad_tmp = np.zeros(cc.shape)
                        inpt1, inpt2 = inpt.split('#')
                        if cc.shape[0] > 2:
                            if model.hprm['qo_modulo'].get(inpt1):
                                ker1  = np.array([[[1],[-4],[6],[-4],[1]]])
                                conv1     = spim.filters.convolve(cc,ker1,mode = 'wrap')
                                grad_tmp += alpha * conv1
                            else:
                                ker1  = np.array([[[1]],[[-2]],[[ 1]]])
                                conv1     = sig.convolve(cc, ker1, mode = 'valid')
                                grad_tmp[ :-2,:] += alpha * conv1
                                grad_tmp[1:-1,:] -= alpha * 2*conv1
                                grad_tmp[2:  ,:] += alpha * conv1
                        if cc.shape[1] > 2: 
                            if model.hprm['qo_modulo'].get(inpt2):
                                ker2  = np.array([[[1],[-4],[6],[-4],[1]]])
                                conv2 = spim.filters.convolve(cc,ker2,mode = 'wrap')
                                grad_tmp += alpha * conv2
                            else:
                                ker2  = np.array([[[1],  [-2],  [1]]])
                                conv2     = sig.convolve(cc, ker2, mode = 'valid')
                                grad_tmp[:, :-2] += alpha *   conv2
                                grad_tmp[:,1:-1] -= alpha * 2*conv2
                                grad_tmp[:,2:  ] += alpha *   conv2
                        grad[slice(key_slice.start, key_slice.stop)] = grad_tmp.reshape((grad[slice(key_slice.start, key_slice.stop)].shape))
                ##### group r2sm #####
                elif model.pen[var].get(inpt) == 'row2sm':
                    assert len(alphas[var][inpt])  == 2, alphas[var][inpt] 
                    assert type(alphas[var][inpt]) == tuple, alphas[var][inpt] 
                    aa, pp = alphas[var][inpt]
                    reshape_tensor = (type(inpt[0]) == tuple) 
                    assert key_slice.stop >= key_slice.start+2
                    if not reshape_tensor:
                        if cc.shape[0] > 2:
                            if model.hprm['qo_modulo'].get(inpt):
                                ker  = np.array([[1],[-2],[1]])
                                conv = spim.filters.convolve(cc,ker,mode = 'wrap')
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
                        cc = cc.reshape(*model.size_tensor2[inpt,location], -1)
                        grad_tmp = np.zeros(cc.shape)
                        inpt1, inpt2 = inpt
                        if cc.shape[0] > 2:
                            if model.hprm['qo_modulo'].get(inpt1):
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
                            if model.hprm['qo_modulo'].get(inpt2):
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
                        grad[slice(key_slice.start, key_slice.stop)] = grad_tmp.reshape((grad[slice(key_slice.start, key_slice.stop)].shape))
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
        ans_lbfgs = tools.batch_load(model.path_betas, 
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
        print('{0:20.20}{1}{2:.3e}\n{3:20.20}{4}{5:.3e}\n{6:20.20}{7}{8:.3e}'.format('machine_precision',       ' : ', model.machine_precision, 
                                                                                     'afm.algorithm.lbfgs.tol', ' : ', model.tol_lbfgs, 
                                                                                     'model.factr',             ' : ', model.factr,
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
                             model.path_betas, 
                             data      = ans_lbfgs, 
                             prefix    = model.dikt['experience.whole'], 
                             data_name = 'ans_lbfgs', 
                             data_type = 'np',
                             )      
            print(colored('ans_lbfgs saved', 'green'))
        except Exception as e:
            print(e)
    return ans_lbfgs
        
        
def sort_keys(keys, masks):
    cat_owned = {}
    # Sort the categories of covariates
    keys_shared = []
    keys_owned  = []
    for inpt, location in keys: 
        cond = (    (inpt,location) in masks
                and not (type(masks[inpt,location]) == type(slice(None)) and masks[inpt,location] == slice(None))
                )
        if cond:
            keys_owned.append((inpt, location))
            cat_owned[inpt] = True
        else:
            keys_shared.append((inpt, location))
            if inpt not in cat_owned:
                cat_owned[inpt] = False
    keys_shared = sorted(keys_shared, key = lambda x : str(x))
    keys_owned  = sorted(keys_owned,  key = lambda x : str(x))
    print('    {0} cats shared - {1} cats owned'.format(len(keys_shared), len(keys_owned)))
    # The shared variables are in the top rows of the design matrix
    # The individual covariates come after
    return keys_shared + keys_owned, cat_owned

  
