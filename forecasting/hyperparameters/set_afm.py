

"""
Parameters for the main model of this work
"""

from . import default
import copy as cp
from termcolor import colored
import numpy as np
import electricityLoadForecasting.forecasting.config as config



def set_afm(hprm):
    
    # Design and optimization problem
    assert hprm['learning.model'] == 'afm'
    hprm.update({
                  'afm.features.boundary_scaling'           : False, # Activate to select a different scaling of the splines near the boundary (does not improve the results)
                  #'tf_masks'                  : cp.deepcopy(default.active_mask), # Choose the number of weather stations a substations should have access to
                  'afm.features.natural_splines'            : True, # Activate to extrapolate linearly
                  'afm.features.order_splines'              : 1,    # 1 for piecewise linear
                  'afm.features.bivariate.combine_function' : np.prod,    # Use product or minimum of splines for interactions
                   })

#    hprm['afm.tf_orthogonalize'] = { # Not used anymore since it removes the sparsity and did not improve the performances
#                                    ('targetlag', 'wh') : False, 
#                                    ('meteo',     'yd') : False, 
#                                    ('wh',        'yd') : False, 
#                                    }
    
    hprm.update({
                  'afm.formula'             : cp.deepcopy(default.afm.dikt_formula.get((hprm['database'], 
                                                                                        hprm['sites.aggregation'],
                                                                                        ),
                                                                                      None, #default.afm.dikt_formula['nat'], 
                                                                                      )['tmp']
                                                          ), # Choose the granularity for the different inputs ie the number of splines
#                  'afm.formula'            : {var:tuple([key 
#                                                            for key in list_keys 
#                                                            #if 'targetlag' not in key  # Uncomment for Middle-Term models
#                                                            ]) 
#                                                 for var, list_keys in cp.deepcopy(default.afm.dikt_formula.get((hprm['database'], 
#                                                                                                                 hprm['sites.aggregation'],
#                                                                                                                 ), 
#                                                                                                                default.afm.dikt_formula['nat'], 
#                                                                                                                )['tmp']).items()
#                                                                                                                  # Possible values
#                                                                                                                  # 'lowrank'
#                                                                                                                  # 'univariate'
#                                                                                                                  # 'lbfgs'
#                                                                                                                  # 'interUV'
#                                                                                                                  # 'sesq'
#                                                 }, 
#
#                 
#                  'afm.regularization.functions'                    : cp.deepcopy(default.afm.regularization_func.get((hprm['database'], 
#                                                                                                                       hprm['sites.aggregation'],
#                                                                                                                       ), 
#                                                                                                                      default.afm.regularization_func['nat'], # Default 
#                                                                                                                      )),                                                                                                               # 'lbfgs'
#                  'afm.regularization.coefficients'                  : cp.deepcopy(default.afm.regularization_coef.get((hprm['database'], 
#                                                                                                                        hprm['sites.aggregation'],
#                                                                                                                        ), 
#                                                                                                                       default.afm.regularization_coef['nat'], # Default
#                                                                                                                       )),
                            
                            # Choose which variables should be part of the low-rank components and ot the unstructured components
                            # Blr        for low-rank (over the substations) formulation with a first-order descent algorithm
                            # Bsp        for independent models and unstructured coedfficients with a first-order descent algorithm
                            # lbfgs_coef for independent models and unstructured coedfficients with a first-order descent algorithm
                            # Cuv        for independent models and low-rank interactions
                            # Cb         for independent models and the univariate part of the sesquivariate constraint
                            # Cbm        for independent models and the sesquivariate constraint for the interactions
                  })
    
    #regularization_setting = 'row2sm'
    # Choose the regularization   
    # Possible values
    # hprm['zone'] 
    # 'nat' 
    # 'full'
    # 'row2sm'                                                                                                                
    # 'lbfgs'
    # 'hrch' 
    

                  
    hprm.update({
                                                                                           
                  #'tf_ch_basis1'              : 0,
                  #'tf_ch_basis2'              : 0,
                  'afm.constraints.ranks_B'     : {  # Rank of the low-rank (substations wise) components for each input
                                                 'wh' : 4, 
                                                 'yd' : 4, 
                                                 },
                  'afm.constraints.ranks_UV'    : 1,    # Rank of the coefficient matrices for the interactions (individually per post) 
                  'afm.regularization.share_enet'                : 0.95, # Coefficient for the elastic-net penalty
                  })
            
            
    hprm.update({
                  'afm.regularization.gp_matrix'          : '', # Form of the sum-consistent matrix
                                                 # '' not to consider the sum-consistent model
                                                 # 'cst' penalize the nationally aggregated predictions
                                                 # 'nat' penalize the nationally aggregated predictions
                                                 # 'rteReg' penalize the aggregated predictions in each RTE regions
                                                 # 'admReg' penalize the aggregated predictions in each administrative regions
                                                 # 'districts' penalize the aggregated predictions in each districts
                  'afm.regularization.gp_pen'             : 1, # Coefficient for the sum-consistent model
                  })
    # Algorithm
    hprm.update({
                  'afm.algorithm'                  : 'L-BFGS',
                  'afm.algorithm.lbfgs_precompute' : False, # Precompute XtX and XtY for LbfgS
                  'afm.algorithm.try_warmstart'    : True and config.load_model, # True warm-start
                  'afm.algorithm.bcd'              : True, # Use block coordinate descent
                  'afm.algorithm.active_set'       : True, # Use an active set strategy
                  'afm.algorithm.prop_active_set'  : 0.2,
                  'afm.algorithm.column_update'    : {**{k:False  
                                                         for k in set([e 
                                                           for coef, list_keys in hprm['afm.formula'].items() 
                                                           for e in list_keys
                                                           ])
                                            }, # Ine the BCD procedure, activate to have column update instead of larger block updates
                                          #'wh':True,
                                          #'yd':True,
                                         },
                  'afm.features.sparse_x1'          : True, # Store the univariate covariates in a sparse format
                  'afm.features.sparse_x2'          : True, # Store the interaction covariates in a sparse format
                  'afm.algorithm.sparse_coef'           : False, # Memory size of coef does not seem to be a problem, at least when there is no interactions
                  'afm.algorithm.sparse_coef_after_optim' : True,  # After the optimization, store the coefficients in a sparse format
                  #'tf_init_B'                 : False,
                  #'trunc_svd'                 : -1,
                  #'tol_B'                     : 1e-10,
                  #'tf_init_C'                 : False,
                  #'tol_C'                     : 1e-10,
                  #'tf_sesq'                   : False, # Optimization in two steps to introduce 2nd order interactions
                  #'tf_method_init_UV'         : 'w_uni', # 'haar' 'random' 'w_uni'
                  #'freeze_B'             : False, # 
                  })
        
    hprm.update({ # EARLY STOPPING CRITERIA
                  'afm.precompute.precompute_test'       : True,  
                  'afm.algorithm.early_stop_test'       : False, # Use performance on test basis as a stopping criteria 
                  'afm.algorithm.early_stop_ind_test'   : False, # Use performance on test basis as a stopping criteria 
                  'afm.algorithm.early_small_decrease'  : True,  # Use decrease of objective as a stopping criteria 
                  'afm.algorithm.early_stop_small_grad' : True,  # Use norm of grad as a stopping criteria 
                  'afm.algorithm.tol'                   : 1e-8,  # Is compared with the decrease of the objective
                  'afm.algorithm.norm_grad_min'         : 1e-8, 
                  'afm.algorithm.dist_min'              : 1e-8,
                  'afm.algorithm.max_iter'              : 1e6, 
                  'afm.algorithm.early_stop_period'     : 10 ,
                  ### LBFGS
                  **cp.deepcopy(default.afm.stopping_criteria.get((hprm['database'], 
                                                                   hprm['sites.aggregation'],
                                                                   ), 
                                                                  default.afm.stopping_criteria['nat'], 
                                                                  )
                                ),
                  })
    
    if (hprm['database'], 
        hprm['sites.aggregation'],
        ) not in default.afm.stopping_criteria:
        print(colored('NO PARAMETRIZATION SET FOR {0}'.format(hprm['zone']), 'red', 'on_cyan'))


###############################################################################
###                       PREDEFEDINED SETTINGS                             ###
###############################################################################
    
    
#    # hprmeters for specific tests (all are deactivated in define_database.py)
#    if hprm['spec_hrch']:
#        hprm['tf_config_coef']       = cp.deepcopy(default.dikt_config['lowrank'])                    
#        hprm['approx_nb_itv']        = cp.deepcopy(default.nb_itv     ['hrch'])
#        hprm['tf_pen']               = cp.deepcopy(default.pen_base   ['hrch'])
#        hprm['tf_alpha']             = cp.deepcopy(default.alphas_base['hrch'])
#        ccc = 0.2
#        hprm['tf_hrch']              = {
#                                         'wh' : ccc, 
#                                         'yd' : ccc, 
#                                         }
#        hprm['any_plot']             = True 
#        hprm['exp_plot']             = True 
#        hprm['exp_plot_1d_effect']   = True
#        hprm['fb_1D_functions_plot'] = False
#        
#                    
#    elif hprm['spec_gp_pen_full']:
#        assert hprm['zone'] == 'full'
#        hprm['gp_matrix']        = 'districts'
#        hprm['gp_pen']           = 1
#        hprm['lbfgs_precompute'] = True # It is faster to precompute at least for rteReg and admReg
#        hprm['save_pred']        = False
#        #hprm['tf_sparse_coef'] = True
#        # Change to self.active_gp and True in approx_tf
#        
#    elif hprm['spec_just_load_data']:
#        hprm['just_load_data'] = True
#        
#    elif hprm['self_reg']:
#        assert hprm['zone'] == 'full'  
#        hprm['tf_masks']            = {}
#        hprm['load_perf']           = False
#        hprm['load_pred']           = False
#        hprm['selected_variables']  = ('target',)
#        hprm['tf_config_coef']      = {'Bsp'    : ('target',)}
#        hprm['approx_nb_itv']       = {'target' : 'p1'}
#        hprm['tf_pen']              = {'Bsp'    : {'target':'rlasso'}}
#        hprm['tf_alpha']            = {'Bsp'    : {'target':1e-2}}
#        
#    elif hprm['row2sm']:
#        hprm['tf_pen']         = default.pen_base   ['row2sm']
#        hprm['tf_alpha']       = default.alphas_base['row2sm']


            
    return hprm