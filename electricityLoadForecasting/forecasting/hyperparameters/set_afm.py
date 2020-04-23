

"""
Parameters for the main additive features model
"""

import copy as cp
import numpy as np
#
from . import default


def set_afm(hprm):
    
    # Design and optimization problem
    assert hprm['learning.model'] == 'afm'
    hprm.update({
                  'afm.features.boundary_scaling'           : False, # Activate to select a different scaling of the splines near the boundary (does not improve the results)
                  'afm.features.natural_splines'            : True, # Activate to extrapolate linearly
                  'afm.features.order_splines'              : 1,    # 1 for piecewise linear
                  'afm.features.bivariate.combine_function' : np.prod, # Use product or minimum of splines for interactions
                  'afm.features.sparse_x1'                  : True, # Store the univariate covariates in a sparse format
                  'afm.features.sparse_x2'                  : True, # Store the interaction covariates in a sparse format
                   })

    
    hprm.update({
                  'afm.formula'             : cp.deepcopy(default.afm.dikt_formula.get((hprm['database'], 
                                                                                        hprm['sites.aggregation'],
                                                                                        ),
                                                                                       None, #default.afm.dikt_formula['nat'], 
                                                                                       )['bivariate']
                                                          ), # Choose the granularity for the different inputs ie the number of splines
                  })
    
    # Low-rank structure
    hprm.update({
                  'afm.constraints.ranks_B'     : {# Rank of the low-rank (substations wise) components for each input
                                                   'wh' : 4, 
                                                   'yd' : 4, 
                                                    },
                  'afm.constraints.ranks_UV'      : 1,    # Rank of the coefficient matrices for the interactions (individually per post) 
                  'afm.regularization.share_enet' : 0.95, # Coefficient for the elastic-net penalty
                  })
            
    # Sum-consistent model        
    hprm.update({
                  'afm.sum_consistent.gp_pen'    : 1, # Coefficient for the sum-consistent model
                  'afm.sum_consistent.gp_matrix' : '', # Form of the sum-consistent matrix
                                                 # '' not to consider the sum-consistent model
                                                 # 'cst' penalize the nationally aggregated predictions
                                                 # 'nat' penalize the nationally aggregated predictions
                                                 # 'rteReg' penalize the aggregated predictions in each RTE regions
                                                 # 'admReg' penalize the aggregated predictions in each administrative regions
                                                 # 'districts' penalize the aggregated predictions in each districts
                  })
    # Algorithm
    hprm.update({
                  'afm.algorithm'                  : 'L-BFGS',
                  'afm.algorithm.lbfgs_precompute' : False, # Precompute XtX and XtY for LbfgS
                  'afm.algorithm.try_warmstart'    : False, # True warm-start
                  'afm.algorithm.bcd'              : True, # Use block coordinate descent
                  'afm.algorithm.active_set'       : True, # Use an active set strategy
                  'afm.algorithm.prop_active_set'  : 0.2,
                  'afm.algorithm.column_update'    : {**{(coef, keys) : False  
                                                         for coef, keys in hprm['afm.formula'].index
                                                         }
                                                      }, # In the BCD procedure, activate to have column update instead of larger block updates
                                                      #'wh':True,
                                                      #'yd':True,
                  'afm.algorithm.sparse_coef'               : False, # Memory size of coef does not seem to be a problem, at least when there is no interactions
                  'afm.algorithm.sparse_coef_after_optim'   : True,  # After the optimization, store the coefficients in a sparse format
                  'afm.algorithm.frozen_variables'          : {}, # 
                  'afm.algorithm.early_stop_validation'     : False, # Use performance on validation set as a stopping criteria 
                  'afm.algorithm.early_stop_ind_validation' : False, # Use performance on validation set as a stopping criteria 
                  'afm.algorithm.early_small_decrease'      : True,  # Use decrease of objective as a stopping criteria 
                  'afm.algorithm.early_stop_small_grad'     : True,  # Use norm of grad as a stopping criteria 
                  'afm.algorithm.tol'                       : 1e-8,  # Is compared with the decrease of the objective
                  'afm.algorithm.norm_grad_min'             : 1e-8, 
                  'afm.algorithm.dist_min'                  : 1e-8,
                  'afm.algorithm.max_iter'                  : 1e6, 
                  'afm.algorithm.early_stop_period'         : 10 ,
                  ### LBFGS
                  **cp.deepcopy(default.afm.stopping_criteria.get((hprm['database'], 
                                                                   hprm['sites.aggregation'],
                                                                   ), 
                                                                  default.afm.stopping_criteria['eCO2mix.France',
                                                                                                None,
                                                                                                ], 
                                                                  )
                                ),
                  #'tf_init_B'                 : False,
                  #'trunc_svd'                 : -1,
                  #'tol_B'                     : 1e-10,
                  #'tf_init_C'                 : False,
                  #'tol_C'                     : 1e-10,
                  #'tf_sesq'                   : False, # Optimization in two steps to introduce 2nd order interactions
                  #'tf_method_init_UV'         : 'w_uni', # 'haar' 'random' 'w_uni'
                  })
            
    return hprm