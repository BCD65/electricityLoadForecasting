

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
                  'afm.features.natural_splines'            : True,    # Activate to extrapolate linearly
                  'afm.features.order_splines'              : 1,       # 1 for piecewise linear
                  'afm.features.bivariate.combine_function' : np.prod, # Use product or minimum of splines for interactions
                  'afm.features.sparse_x1'                  : True,    # Store the univariate covariates in a sparse format
                  'afm.features.sparse_x2'                  : True,    # Store the interaction covariates in a sparse format
                   })

    
    hprm.update({
                  'afm.formula'             : cp.deepcopy(default.afm.dikt_formula.get((hprm['database'], 
                                                                                        hprm['sites.aggregation'],
                                                                                        ),
                                                                                       None, #default.afm.dikt_formula['nat'], 
                                                                                       )['unconstrained_bivariate']
                                                          ), # Choose the granularity for the different inputs ie the number of splines
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
                  'afm.algorithm'                           : 'FirstOrder', # 'L-BFGS' # 'FirstOrder'
                  'afm.algorithm.sparse_coef_after_optim'   : True,  # After the optimization, store the coefficients in a sparse format
                  #'afm.algorithm.sparse_coef'               : False, # Memory size of coef does not seem to be a problem, at least when there is no interactions
                  # First-order descent algorithm
                  'afm.algorithm.first_order.frozen_variables'          : {},
                  'afm.algorithm.first_order.early_stop_validation'     : False, # Use performance on validation set as a stopping criteria 
                  'afm.algorithm.first_order.early_stop_ind_validation' : False, # Use performance on validation set as a stopping criteria 
                  'afm.algorithm.first_order.max_iter'                  : 1e6,  # Is compared with the decrease of the objective
                  'afm.algorithm.first_order.try_warmstart'             : False, # True warm-start
                  'afm.algorithm.first_order.bcd'                       : True, # Use block coordinate descent
                  'afm.algorithm.first_order.active_set'                : True, # Use an active set strategy
                  'afm.algorithm.first_order.prop_active_set'           : 0.2,
                  'afm.algorithm.first_order.column_update'             : {**{(coef, keys) : False  
                                                                              for coef, keys in hprm['afm.formula'].index
                                                                              }
                                                                           }, # In the BCD procedure, activate to have column update instead of larger block updates
                                                                           #'wh':True,
                                                                           #'yd':True,
                  'afm.algorithm.first_order.tol'                       : 1e-8,  # Is compared with the decrease of the objective
                  'afm.algorithm.first_order.early_small_decrease'      : True,  # Use decrease of objective as a stopping criteria 
                  'afm.algorithm.first_order.early_stop_small_grad'     : True,  # Use norm of grad as a stopping criteria 
                  'afm.algorithm.first_order.norm_grad_min'             : 1e-8, 
                  'afm.algorithm.first_order.dist_min'                  : 1e-8,
                  #'afm.algorithm.first_order.early_stop_period'         : 10 ,
                  ### LBFGS
                  'afm.algorithm.lbfgs.precompute' : False, # Precompute XtX and XtY for LbfgS
                  **cp.deepcopy(default.afm.stopping_criteria.get((hprm['database'], 
                                                                   hprm['sites.aggregation'],
                                                                   ), 
                                                                  default.afm.stopping_criteria['eCO2mix.France',
                                                                                                None,
                                                                                                ], 
                                                                  )
                                ),
                  })
            
    return hprm